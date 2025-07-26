import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import cv2
import yaml
import json
import pickle
import numpy as np
from tqdm import tqdm
from PIL import Image
from deepface import DeepFace
from torchvision import transforms
from accelerate import Accelerator
from argparse import ArgumentParser
from modules.util import center_crop
from accelerate.utils import set_seed
from torch.utils.data import DataLoader
from moviepy.editor import ImageSequenceClip, VideoFileClip, AudioFileClip

from modules.model.generator import *
from modules.model.base_model import *
from modules.dataloader.stage_one_dataloader import HumanTalkingJsonDataset

from motion_diffusion import DiffusionPriorTrainer
from motion_diffusion.dataloaders import MotionDataset, VASAMotionDataset
from motion_diffusion.motion_diffusion import DitBlock, DiTNetwork, DiffusionPrior, DiTNetworkE5, DiTNetworkE7, DiTNetworkE8

from preprocess_models.wav2vec_exactor import MyWav2Vec, down_sample_for_audio


def make_model(config, device, accelerator):
    # dit_net = DiTNetwork(**config["prior"]["net"])
    dit_net = create_instance_by_name(config["train"]["network"])(**config["prior"]["net"])
    
    config["prior"].pop("net")
    diffusion_prior = DiffusionPrior(
        net=dit_net, 
        **config["prior"]
        # feature_embed_dim=config["prior"]["feature_embed_dim"],
        # timesteps=config["prior"]["timesteps"],
        # cond_drop_prob=config["prior"]["cond_drop_prob"],
        # loss_type=config["prior"]["loss_type"],
        # predict_x_start=config["prior"]["predict_x_start"],
        # beta_schedule=config["prior"]["beta_schedule"],
        # feature_embed_scale=config["prior"]["feature_embed_scale"]
        )
    # instantiate the trainer
    trainer = DiffusionPriorTrainer(
        diffusion_prior=diffusion_prior,
        lr=config["train"]["lr"],
        wd=config["train"]["wd"],
        max_grad_norm=config["train"]["max_grad_norm"],
        amp=config["train"]["amp"],
        use_ema=config["train"]["use_ema"],
        device=device,
        accelerator=accelerator,
        warmup_steps=config["train"]["warmup_steps"],
        cosine_decay_max_steps=config["train"]["cosine_decay_max_steps"]
    )

    return trainer


def create_instance_by_name(class_name: str):
    # 使用globals()获取全局命名空间中的所有定义
    class_defs = globals()
    
    # 使用getattr()根据字符串名称获取对应的类
    class_def = class_defs.get(class_name)
    
    # 检查是否找到了对应的类
    if class_def is None or not isinstance(class_def, type):
        raise ValueError(f"No such class: {class_name}")
    
    # 创建类的实例
    # instance = class_def()
    return class_def

@torch.no_grad()
def extract_feature_source(source_image, generator, pose_model, device='cuda', img_size=256):
    transform = transforms.Compose(
        [
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize([0], [1]),
        ]
    )
    # source_image = cv2.imread(image_path)
    faces = DeepFace.analyze(source_image[:, :, ::-1], actions=['emotion'], enforce_detection=False)
    
    facial_area = faces[0]['region']
    face_bbox = [facial_area['x'], facial_area['y'], facial_area['w'], facial_area['h']]
    face_bbox[2] += face_bbox[0]
    face_bbox[3] += face_bbox[1]
    source_face_image, _ = center_crop(source_image, face_bbox)

    source_image = cv2.resize(source_image, (img_size, img_size))
    source_image_pil = Image.fromarray(source_image[:, :, ::-1])
    source_image_tensor = transform(source_image_pil)

    source_face_image = cv2.resize(source_face_image, (img_size, img_size))
    source_face_image_pil = Image.fromarray(source_face_image[:, :, ::-1])
    source_face_image_tensor = transform(source_face_image_pil)

    source_image = (source_image_tensor.clone().permute((1, 2, 0)).numpy() * 255).astype(np.uint8)[:, :, ::-1]

    source_image_tensor = source_image_tensor.to(device).unsqueeze(0)
    source_face_image_tensor = source_face_image_tensor.to(device).unsqueeze(0)
    source_pose = pose_model(source_image_tensor * 2 - 1)
    vs, global_descriptor, z_s = generator.extract_feature(source_image_tensor, source_face_image_tensor, source_pose)

    return vs, global_descriptor, z_s, source_pose, source_image

def split_audio(video_path, file_path): 
    cmd = f"ffmpeg -i {video_path} -f wav -ar 16000 {file_path} -y -loglevel quiet"
    os.system(cmd)


def get_audio_feature(data, model):
    img = cv2.imread(data["img_path"])
    mask = np.load(data["mask_path"]) / 255
    mask = cv2.resize(mask, img.shape[:2])
    img = img * mask[..., np.newaxis]
    img = img.astype(np.uint8)

    audio_path = data["audio_path"]
    audio_list = model.load_audio(audio_path)
    feature = model(audio_list)
    feature = feature.cpu().detach().numpy()
    feature = down_sample_for_audio(feature)
    return feature, img

@torch.no_grad()
def motion_diffusion_inference(trainer, config, motion_pre_cond, audio_feature_return, cond_scale, device, convert_trans = True):
    index = 0
    
    num_feature_embeds = 16
    num_audio_embeds = 24
    pre_feature_embeddings = motion_pre_cond.reshape(4, 518)

    audio_feat = torch.from_numpy(audio_feature_return).reshape(-1, 1024).float().to(device)
    length = len(audio_feat)

    padd_length = ((length - num_audio_embeds) // num_feature_embeds + 1) * num_feature_embeds + num_audio_embeds - length
    audio_feat = torch.cat([audio_feat, torch.zeros((padd_length, audio_feat.shape[1])).to(device)], dim=0)

    length = len(audio_feat)
    shape = (1, num_feature_embeds, 518)
    
    while index <= length - num_audio_embeds:
        input_pre = pre_feature_embeddings[-4:].clone()
        if convert_trans:
            input_pre[:, 3:6] = input_pre[:, 3:6] - input_pre[0, 3:6][None]
 
        signal_cond = dict(
            motion_pre_cond=input_pre.unsqueeze(0),
            audio_cond=audio_feat[index:index+num_audio_embeds].unsqueeze(0)
        )

        predicted_feature_embeddings = trainer.p_sample_loop(
                shape,   ## feature形状（b, num_feature_embeds, 1030）
                signal_cond,  ## 驱动信号，为视频前四帧，需要不断迭代，音频特征
                timesteps=50,
                cond_scale=cond_scale  ## cfg系数
            )
        predicted_feature_embeddings = predicted_feature_embeddings.reshape((num_feature_embeds, 518))

        if convert_trans:
            predicted_feature_embeddings[:, 3:6] = predicted_feature_embeddings[:, 3:6] + pre_feature_embeddings[-4, 3:6][None]
        pre_feature_embeddings = torch.cat([pre_feature_embeddings, predicted_feature_embeddings], dim=0)

        index = index + num_feature_embeds
    return pre_feature_embeddings

def main():
    parser = ArgumentParser()
    parser.add_argument("--config", default="config/eval.yaml", help="path to config")
    parser.add_argument("--save_dir", default="./results/tmp", help="path to train data")
    parser.add_argument("--convert_trans", action="store_true")
    parser.add_argument("--json_list_path", default="")
    parser.add_argument("--num", default=20, type=int)
    parser.add_argument("--cond_scale", default=1., type=float)
    parser.add_argument("--stage_one_model_path", default="", help="path to model")
    parser.add_argument("--stage_two_model_path", default="", help="path to model")
    opt = parser.parse_args()

    accelerator = Accelerator()
    device = accelerator.device

    set_seed(84513)

    ### 加载vasa-stage-one模型
    with open(opt.config) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    img_size = config['dataset_params']['img_size']
    generator = create_instance_by_name(config['model_params']['common_params']['generator_type'])(config, device)
    pose_model = HeadPose_train().to(device)

    checkpoint = torch.load(opt.stage_one_model_path, map_location='cpu')
    generator.load_state_dict(checkpoint['generator'], strict=True)
    pose_model.load_state_dict(checkpoint['pose_model'], strict=True)
    generator.eval()
    pose_model.eval()

    ### 加载motion-diffusion模型
    config = config["motion_diffusion"]
    trainer = make_model(config, device, accelerator).to(device)
    trainer.load(opt.stage_two_model_path, overwrite_lr=True, strict=True)


    ### 加载音频特征提取模型
    audio_model = MyWav2Vec(device)


    ### inference
    out_dir = os.path.join(opt.save_dir, os.path.splitext(os.path.basename(opt.stage_two_model_path))[0])

    eval_data = json.load(open(opt.json_list_path, 'r'))
    
    if opt.num > 0:
        eval_data = eval_data[:opt.num]
    
    for index, data in enumerate(eval_data):
        # try:
        video_name = data["img_path"].split('/')[-1][:-4] + '.mp4'
        save_video_path = os.path.join(out_dir, video_name)

        audio_feature_return, img = get_audio_feature(data, audio_model)

        vs, global_descriptor, z_s, source_pose, source_image = extract_feature_source(img, generator, pose_model, device=device, img_size=img_size)

        pose_emotion = torch.cat([source_pose['rotation'], source_pose['translation'], z_s], dim=1)
        motion_pre_cond = pose_emotion.repeat((4, 1))
        result_features = motion_diffusion_inference(trainer, config, motion_pre_cond, audio_feature_return, opt.cond_scale, device, convert_trans=opt.convert_trans)

        ## 平滑后处理 start
        pose_feat_list = result_features[:, :6]
        pose_feat_smooth_list = []
        for i in range(len(result_features)):
            if i == 0:
                pose_feat_smooth_list.append(pose_feat_list[i:i+2].mean(0)[None,:])
            elif i == len(result_features) - 1:
                pose_feat_smooth_list.append(pose_feat_list[i-1:].mean(0)[None,:])
            else:
                pose_feat_smooth_list.append(pose_feat_list[i-1:i+2].mean(0)[None,:])
        pose_feat_smooth_list = torch.cat(pose_feat_smooth_list, 0)

        ### ori-trans #####
        result_features[:, :1] = (pose_feat_smooth_list[:, :1] - result_features[:, :1]) * 0.3 + result_features[:1, :1][None]
        result_features[:, 5:6] = (pose_feat_smooth_list[:, 5:6] - result_features[:, 5:6]) * 0.2 + result_features[:1, 5:6][None]
        result_features[:, 1:5] = pose_feat_smooth_list[:, 1:5]
        ### ori-trans #####

        def center_trans(result, scale=0.5):
            refer = result[0]
            result = result - result[0][None]
            result = result * scale
            ss = max(torch.abs(min(result)), torch.abs(max(result))) 
            if ss > 0.2:
                scale_1 = 0.2 / ss
            else:
                scale_1 = 1
            result = result * scale_1
            result = result + refer
            return result
        def crop_rotations(result, scale=0.8):
            refer = result[0]
            result = result - result[0][None]
            result = torch.clip(result, min(result) * scale, max(result) * scale)
            result = result + refer
            return result
            
        # result_features[:, 0] = crop_rotations(result_features[:, 0], scale=0.8)
        # result_features[:, 1] = crop_rotations(result_features[:, 1], scale=0.8)
        # result_features[:, 2] = crop_rotations(result_features[:, 2], scale=0.8)
        result_features[:, 3] = center_trans(result_features[:, 3]) 
        result_features[:, 4] = center_trans(result_features[:, 4]) 
        result_features[:, 5] = center_trans(result_features[:, 5]) 

        ## 平滑后处理 done

        save_paths = []
        for index, driven_feature in enumerate(result_features):
            rotation = driven_feature[:3].unsqueeze(0)
            translation = driven_feature[3:6].unsqueeze(0)
            driven_pose = {'rotation': rotation, 'translation': translation}
            exp = driven_feature[6:].unsqueeze(0)
            output = generator.drive_feature_vasa(vs.clone(), global_descriptor.clone(), exp, driven_pose)
            output_image = (output.permute((0, 2, 3, 1)).cpu().detach().numpy() * 255).clip(0, 255).astype(np.uint8)[:, :, :, ::-1][0]

            img = np.concatenate([source_image, output_image], axis=1)
            img_path = os.path.join(os.path.splitext(save_video_path)[0], '%05d.png' % index)
            index += 1
            os.makedirs(os.path.dirname(img_path), exist_ok=True)
            cv2.imwrite(img_path, img)
            save_paths.append(img_path)

        audio_clip = AudioFileClip(data["audio_path"])
        clip = ImageSequenceClip(save_paths, fps=25)
        clip = clip.set_audio(audio_clip)
        clip.write_videofile(save_video_path, codec='libx264', audio_codec="mp3")
            # break
        # except:
        #     continue


if __name__ == "__main__":
    main()




