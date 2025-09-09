import argparse
import os
import warnings
import torch
import torch.utils.checkpoint
from PIL import Image
import numpy as np
from omegaconf import OmegaConf
import glob
from tqdm import tqdm
import json
import cv2
import pdb
from diffusers import AutoencoderKLTemporalDecoder
from diffusers.schedulers import EulerDiscreteScheduler
from transformers import WhisperModel, CLIPVisionModelWithProjection
import importlib
from src.utils.util import import_filename, save_videos_grid, seed_everything
from src.dataset.test_preprocess import preprocess_resize_shortedge as preprocess
from src.dataset.test_preprocess import get_custom_affine_transform_512, DEFAULT_CROP_SIZE
from src.models.base.unet_spatio_temporal_condition import UNetSpatioTemporalConditionModel, add_ip_adapters, load_adapter_states
from src.pipelines.pipeline_svd_audio_adapter_motionexp_idembed_vasa_two_ip import Pose2VideoLongSVDPipeline
from src.models.audio_adapter.pose_guider import PoseGuider
from src.models.audio_adapter.audio_proj import AudioProjModel, IDProjModel, VasaProjModel
from src.utils.RIFE.RIFE_HDv3 import RIFEModel
from src.utils.face_align import AlignImage
from src.utils.enhance_teeth.enhance_teeth_pnnx import Model as TeethModel
from src.dataset.vasa_feature_v2 import HeadExpression, HeadPose_train
from einops import rearrange
import torch.nn.functional as F
warnings.filterwarnings("ignore")

def main(config, args):
    cfg = config
    exp_name = cfg.exp_name
    save_dir = f"{cfg.output_dir}/{exp_name}"
    if cfg.seed is not None:
        seed_everything(cfg.seed)

    vae = AutoencoderKLTemporalDecoder.from_pretrained(
        config.pretrained_model_name_or_path, 
        subfolder="vae",
        variant="fp16")
    
    val_noise_scheduler = EulerDiscreteScheduler.from_pretrained(
        config.pretrained_model_name_or_path, 
        subfolder="scheduler")
    
    # image_encoder = CLIPVisionModelWithProjection.from_pretrained(
    #     config.pretrained_model_name_or_path, 
    #     subfolder="image_encoder",
    #     variant="fp16")
    module, cls = cfg.unet_cls.rsplit(".", 1)
    unet_cls = getattr(importlib.import_module(module, package=None), cls)
    unet = unet_cls.from_pretrained(
        cfg.pretrained_model_name_or_path,
        subfolder="unet",
        variant="fp16",
        low_cpu_mem_usage=False,
        device_map=None,
    )
    # unet = UNetSpatioTemporalConditionModel.from_pretrained(
    #     config.pretrained_model_name_or_path,
    #     subfolder="unet",
    #     variant="fp16",
    #     low_cpu_mem_usage=False,
    #     device_map=None)
    print('cfg.ip_audio_scale', cfg.ip_audio_scale)
    adapter_modules = add_ip_adapters(unet, [32,32], [cfg.ip_audio_scale, cfg.ip_audio_scale])
    
    pose_guider = PoseGuider(
        conditioning_embedding_channels=320, 
        block_out_channels=(16, 32, 96, 256)
    ).to(device="cuda")
    audio_linear = AudioProjModel(seq_len=10, blocks=5, channels=384, intermediate_dim=1024, output_dim=1024, context_tokens=32).to(device="cuda")
    id_proj_model = IDProjModel(input_dim=512, output_dim=1024, intermediate_dim=1024).to(device="cuda")
    vasa_linear = VasaProjModel(input_dim=512, output_dim=cfg.vasa_expression_dim).to(device="cuda")

    if cfg.resume_from_checkpoint is True:
        resume_dir = save_dir
        # Get the most recent checkpoint
        global_step = 0
        if os.path.isdir(resume_dir):
            dirs = os.listdir(resume_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            if len(dirs) > 0:
                path = dirs[-1]
                global_step = int(path.split("-")[1])
    else:
        global_step = cfg.resume_from_checkpoint
        
    print('loading from ', global_step)

    if global_step > 0:
        pose_guider_checkpoint_path = os.path.join(save_dir, f'pose_guider-{global_step}.pth')
        unet_checkpoint_path = os.path.join(save_dir, f'unet-{global_step}.pth')
        audio_linear_checkpoint_path = os.path.join(save_dir, f'audio_linear-{global_step}.pth')
        adapter_module_checkpoint_path = os.path.join(save_dir, f'adapter_module-{global_step}.pth')
        id_proj_checkpoint_path = os.path.join(save_dir, f'id_proj_model-{global_step}.pth')
        vasa_linear_checkpoint_path = os.path.join(save_dir, f'vasa_linear-{global_step}.pth')
    else:
        pose_guider_checkpoint_path = config.pose_guider_checkpoint_path
        unet_checkpoint_path = config.unet_checkpoint_path
        audio_linear_checkpoint_path = config.audio_linear_checkpoint_path
        adapter_module_checkpoint_path = config.adapter_module_checkpoint_path
        id_proj_checkpoint_path = config.id_proj_checkpoint_path
        vasa_linear_checkpoint_path = config.vasa_linear_checkpoint_path
    # src_ckpt = os.path.join(save_dir, f'*-{global_step}.pth')
    # dst_ckpt = os.path.join(save_dir, 'model_pick')
    # os.makedirs(dst_ckpt, exist_ok=True)
    # os.system(f"ls {src_ckpt}; cp -n {src_ckpt} {dst_ckpt}")

    # load pretrained weights
    load_adapter_states(adapter_modules, [torch.load(adapter_module_checkpoint_path, map_location="cpu")])

    pose_guider.load_state_dict(
        torch.load(pose_guider_checkpoint_path, map_location="cpu"),
        strict=True,
    )


    unet.load_state_dict(
        torch.load(unet_checkpoint_path, map_location="cpu"),
        strict=True,
    )
    
    audio_linear.load_state_dict(
        torch.load(audio_linear_checkpoint_path, map_location="cpu"),
        strict=True,
    )
    
    id_proj_model.load_state_dict(
        torch.load(id_proj_checkpoint_path, map_location="cpu"),
        strict=True,
    )

    vasa_linear.load_state_dict(
        torch.load(vasa_linear_checkpoint_path, map_location="cpu"),
        strict=True,
    )

    # 加载表情模型
    expression_model = HeadExpression(512).to(device="cuda")
    checkpoint = torch.load(cfg.vasa_checkpoint_path, map_location='cpu')
    generator = checkpoint['generator']
    expression_data = {}
    for key in generator:
        if 'expression_model.' in key:
            expression_data[key[len('expression_model.'):]] = generator[key]
    expression_model.load_state_dict(expression_data, strict=True)
    expression_model.to(device="cuda")
    expression_model.eval()
    expression_model.requires_grad_(False)  
    
    pose_model = HeadPose_train()
    pose_model.load_state_dict(checkpoint['pose_model'])
    pose_model.to(device="cuda")
    pose_model.eval()
    pose_model.requires_grad_(False) 

    if cfg.weight_dtype == "fp16":
        weight_dtype = torch.float16
    elif cfg.weight_dtype == "fp32":
        weight_dtype = torch.float32
    elif cfg.weight_dtype == "bf16":
        weight_dtype = torch.bfloat16
    else:
        raise ValueError(
            f"Do not support weight dtype: {cfg.weight_dtype} during training"
        )

    wav2vec = WhisperModel.from_pretrained(cfg.model_paths.whisper_model).to(device="cuda").eval()
    
    wav2vec.requires_grad_(False)

    if cfg.use_interframe:
        rife = RIFEModel()
        rife.load_model(cfg.model_paths.rife_model)
        
    device_id = 0
    device = 'cuda:{}'.format(device_id) if device_id > -1 else 'cpu'
   
    if cfg.use_bfr:
        from src.utils.enhance import bfr_enhance
        enhance_instance = bfr_enhance.test_pipeline()
        enhance_instance.init_model(cfg.model_paths.bfr_enhance_model, device)
        print('can do enhance!')
    else:
        enhance_instance = None

    # image_encoder.to(weight_dtype)
    vae.to(weight_dtype)
    pose_guider.to(weight_dtype)
    unet.to(weight_dtype)
    # Cast mamba parameters to float32
    
    exp_name = cfg.exp_name
    if config.get('save_dir', None):
        save_dir = f"{config.save_dir}/{exp_name}"
    else:
        save_dir = f"{config.output_dir}/{exp_name}"
    
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # print(unet, file=open('unet.network', 'w'))

    image_size = config.image_size
    
    image_path = args.ref
    drive_audio_path = args.audio
    vasa_vido = args.video
    print(image_path)
    image_name = os.path.basename(image_path)
    # image_name = image_path.replace(demo_image_dir, '')
    prefix = f"{save_dir}/visuals/{global_step:06d}-{image_size}-{config.frame_num}-{cfg.min_appearance_guidance_scale}-{cfg.max_appearance_guidance_scale}-{config.audio_guidance_scale}-{config.vasa_guidance_scale}-{config.ip_audio_scale}-motion{config.motion_bucket_id}-motion{config.motion_bucket_id_exp}-area{config.area}-overlap{config.overlap}-shift{config.shift_offset}-noise{config.i2i_noise_strength}-crop{config.crop}-{config.expand_ratio}-bfr{config.use_bfr}-interframe{config.use_interframe}/{config.save_prefix}/{image_name}"
    video_path = prefix  + ".mp4"
    audio_video_path = prefix + "_audio.mp4"
    audio_name = os.path.basename(drive_audio_path)
    print('crop:', config.crop)
    test_data = preprocess(image_path, drive_audio_path,video_path=vasa_vido, limit=config.frame_num, image_size=image_size, area=config.area, crop=config.crop, expand_ratio=config.expand_ratio, enhance_instance=enhance_instance, aspect_type=config.aspect_type, with_id_encoder=True)
    height, width = test_data['ref_img'].shape[-2:]
    print('height, width: ', height, width)
    video = test(
        vae=vae,
        unet=unet,
        wav_enc=wav2vec,
        audio_pe=audio_linear,
        pose_guider=pose_guider,
        id_proj_model=id_proj_model,
        scheduler=val_noise_scheduler,
        width=width,
        height=height,
        batch=test_data,
        vasa_linear=vasa_linear,
        expression_model=expression_model,
        pose_model=pose_model,
        )
    # import ipdb
    # ipdb.set_trace()


    if cfg.use_interframe:
        out = video.to(device)
        results = []
        video_len = out.shape[2]
        for idx in tqdm(range(video_len-1), ncols=0):
            I1 = out[:, :, idx]
            I2 = out[:, :, idx+1]
            middle = rife.inference(I1, I2).clamp(0, 1).detach()
            results.append(out[:, :, idx])
            results.append(middle)
        results.append(out[:, :, video_len-1])
        video = torch.stack(results, 2).cpu()
    # prefix = f"{save_dir}/visuals/{global_step:06d}-{image_size}-{config.frame_num}-{cfg.min_appearance_guidance_scale}-{cfg.max_appearance_guidance_scale}-{config.audio_guidance_scale}-{config.ip_audio_scale}-motion{config.motion_bucket_id}-{config.motion_bucket_id_exp}-area{config.area}-overlap{config.overlap}-shift{config.shift_offset}-noise{config.i2i_noise_strength}-crop{config.crop}-{config.expand_ratio}-{config.aspect_type}-bfr{config.use_bfr}-teeth{config.use_teeth_enhance}-interframe{config.use_interframe}/{config.save_prefix}/{image_name}_{audio_name}"
    # video_path = prefix  + ".mp4"
    # audio_video_path = prefix + "_audio.mp4"
    save_videos_grid(video, video_path, n_rows=video.shape[0], fps=cfg.fps * 2 if cfg.use_interframe else cfg.fps)
    print(f"Successfully saved videio to {video_path}, {os.path.isfile(video_path)}")
    
    os.system(f"ffmpeg -i '{video_path}' -i '{drive_audio_path}' -vcodec: libx264 -c:v libx264 -c:a aac -shortest '{audio_video_path}' -y")


def test(
    vae,
    unet,
    wav_enc,
    audio_pe,
    pose_guider,
    id_proj_model,
    scheduler,
    width,
    height,
    batch=None,
    vasa_linear=None,
    expression_model=None,
    pose_model=None,
):
    if config.seed is not None:
        seed_everything(config.seed)

    pipe = Pose2VideoLongSVDPipeline(
        vae=vae,
        unet=unet,
        id_proj_model=id_proj_model,
        pose_guider=pose_guider,
        scheduler=scheduler,
        feature_extractor=None
    )
    pipe = pipe.to("cuda", dtype=unet.dtype)
    for name, param in pipe.unet.named_parameters():
        if any(x in name for x in ['A_logs', 'Ds', 'dt_projs_bias']):
            param.data = param.data.to(torch.float32)
            print(name, param.data.dtype)
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            batch[k] = v.unsqueeze(0).to(device="cuda").float()
            print(batch[k].shape)
    ref_img = batch['ref_img']
    clip_img = batch['clip_images']

    # import torchvision
    # torchvision.utils.save_image(clip_img, 'clip_img.png')
    
    audio_feature = batch['audio_feature']
    audio_len = batch['audio_len']
    
    step = int(config.step)

    
    window = 3000
    audio_prompts = []
    for i in range(0, audio_feature.shape[-1], window):
        audio_prompt = wav_enc.encoder(audio_feature[:,:,i:i+window], output_hidden_states=True).hidden_states
        audio_prompt = torch.stack(audio_prompt, dim=2)
        audio_prompts.append(audio_prompt)
    audio_prompts = torch.cat(audio_prompts, dim=1)
    audio_prompts = audio_prompts[:,:audio_len*2]

    # import ipdb
    # ipdb.set_trace()
    audio_prompts = torch.cat([torch.zeros_like(audio_prompts[:,:4]), audio_prompts, torch.zeros_like(audio_prompts[:,:6])], 1)

    pose_tensor_list = []
    ref_tensor_list = []
    audio_tensor_list = []
    uncond_audio_tensor_list = []
    exp_mask_tensor_list = []
    mouth_mask_tensor_list = []
    if args.mode == 0:
        gate = [1,0]
    elif args.mode == 1:
        gate = [0,1]
    elif args.mode == 2:
        gate = [1,1]
    else:
        gate = [1,1]
    # get vasa
    if args.mode == 0 or batch['vasa_face_image'] is None or batch['vasa_pose_image'] is None:
        # Audio-only mode – skip expensive VASA computations and create dummy placeholders
        dummy_vasa_dim = config.vasa_expression_dim + 6  # expression embedding + 6 pose dims (rot, trans)
        vasa_prompts = torch.zeros(1, dummy_vasa_dim, device="cuda")
        uncond_vasa_prompts = torch.zeros_like(vasa_prompts)
        vasa_pose_img = torch.zeros(1, 3, 256, 256, device="cuda")
    else:
        crop_face = batch['vasa_face_image']
        crop_face = rearrange(crop_face, 'b f c h w -> (b f) c h w')
        # if gate[0] == 1 and gate[1] == 1:
        #     crop_face[:,:,128:]=0
        vasa_feature = expression_model(crop_face) # bs*f, 512
       
        # get pose
        vasa_pose_img = batch['vasa_pose_image']
        vasa_pose_img = rearrange(vasa_pose_img, 'b f c h w -> (b f) c h w')
        vasa_pose_fea = pose_model(vasa_pose_img * 2 - 1.0)

        # (1,3) / (1,3)
        rot, trans = vasa_pose_fea['rotation'], vasa_pose_fea['translation']
        vasa_prompts = torch.cat([vasa_feature, rot, trans * 0.], dim=-1) # bs*f, 518
        vasa_prompts, vasa_pose_fea = vasa_prompts[...,:-6], vasa_prompts[..., -6:]
        
        uncond_vasa_prompts = vasa_linear(torch.zeros_like(vasa_prompts))

        vasa_prompts = vasa_linear(vasa_prompts)
        vasa_prompts = torch.cat([vasa_prompts, vasa_pose_fea], dim=-1)
        uncond_vasa_prompts = torch.cat([uncond_vasa_prompts, torch.zeros_like(vasa_pose_fea)], dim=-1)
        
    vasa_prompts_list = []
    uncond_vasa_prompts_list = []
    gt_vasa = []
    if args.mode == 0:
        max_len = audio_len
    elif args.mode == 1:
        max_len = len(vasa_prompts)
    elif args.mode == 2:
        max_len = min(audio_len, len(vasa_prompts))
    else:
        max_len = min(audio_len, len(vasa_prompts))
    for i in tqdm(range(max_len//step)):
        pixel_values_pose = batch["img_pose"]
        pixel_values_exp_mask = batch["exp_mask"] if args.mode !=2 else 1 - batch["mouth_mask"]
        pixel_values_mouth_mask = batch["mouth_mask"]
        audio_clip = audio_prompts[:,i*2*step:i*2*step+10].unsqueeze(0)

        cond_audio_clip = audio_pe(audio_clip).squeeze(0)
        uncond_audio_clip = audio_pe(torch.zeros_like(audio_clip)).squeeze(0)

        pose_tensor_list.append(pixel_values_pose[0])
        exp_mask_tensor_list.append(pixel_values_exp_mask[0])
        mouth_mask_tensor_list.append(pixel_values_mouth_mask[0])
        ref_tensor_list.append(ref_img[0])
        audio_tensor_list.append(cond_audio_clip[0])
        uncond_audio_tensor_list.append(uncond_audio_clip[0])
        if args.mode == 0:
            vasa_prompts_list.append(vasa_prompts[0])
            uncond_vasa_prompts_list.append(uncond_vasa_prompts[0])
            gt_vasa.append(vasa_pose_img[0])
        else:
            vasa_prompts_list.append(vasa_prompts[i*step])
            uncond_vasa_prompts_list.append(uncond_vasa_prompts[i*step])
            gt_vasa.append(vasa_pose_img[i*step])
    if gate[0] == 0 and gate[1] == 1:
        exp_mask_tensor_list = [torch.ones_like(ele) for ele in exp_mask_tensor_list]
    if gate[0] == 1 and gate[1] == 0:
        mouth_mask_tensor_list = [torch.zeros_like(ele) for ele in mouth_mask_tensor_list]
    exp_mask_tensor_list = [torch.ones_like(exp_mask_tensor_list[i]) for i in range(len(exp_mask_tensor_list))]
    mouth_mask_tensor_list = [torch.ones_like(mouth_mask_tensor_list[i]) for i in range(len(mouth_mask_tensor_list))]
    video = pipe(
        ref_img,
        clip_img,
        pose_tensor_list,
        exp_mask_tensor_list,
        mouth_mask_tensor_list,
        audio_tensor_list,
        uncond_audio_tensor_list,
        vasa_prompts_list,
        uncond_vasa_prompts_list,
        height=height,
        width=width,
        num_frames=len(pose_tensor_list),
        decode_chunk_size=config.decode_chunk_size,
        motion_bucket_id=config.motion_bucket_id,
        motion_bucket_id_exp=config.motion_bucket_id_exp,
        fps=config.fps,
        noise_aug_strength=config.noise_aug_strength,
        min_guidance_scale1=config.min_appearance_guidance_scale, # 1.0,
        max_guidance_scale1=config.max_appearance_guidance_scale,
        min_guidance_scale2=config.audio_guidance_scale, # 1.0,
        max_guidance_scale2=config.audio_guidance_scale,
        min_guidance_scale3=config.vasa_guidance_scale,
        max_guidance_scale3=config.vasa_guidance_scale,
        overlap=config.overlap,
        shift_offset=config.shift_offset,
        frames_per_batch=config.data.n_sample_frames,
        num_inference_steps=config.num_inference_steps,
        i2i_noise_strength=config.i2i_noise_strength,
        gate=gate
    ).frames

    # import ipdb
    # ipdb.set_trace()

    # Concat it with pose tensor
    video = (video*0.5 + 0.5).clamp(0, 1)
    video = torch.cat([video.to(device="cuda")], dim=0).cpu()

    gt_vasa = torch.stack(gt_vasa, 1)
    gt_vasa_resized = F.interpolate(gt_vasa, size=video.shape[-2:], mode='bilinear', align_corners=False).unsqueeze(0).cpu()
    video = torch.cat([video, gt_vasa_resized], dim=-1)
    # del tmp_denoising_unet
    del pipe
    torch.cuda.empty_cache()

    return video


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="./configs/training/stage2.yaml")
    parser.add_argument("--batch", action='store_true')
    parser.add_argument("--ref", type=str)
    parser.add_argument("--audio",type=str)
    parser.add_argument('--video', type=str)
    parser.add_argument('--mode', type=int, default=0, help='0: only auido, 1: only vasa, 2: only exp, 3: both')
    args = parser.parse_args()

    if args.config[-5:] == ".yaml":
        config = OmegaConf.load(args.config)
    elif args.config[-3:] == ".py":
        config = import_filename(args.config).cfg
    else:
        raise ValueError("Do not support this format config file")
    main(config, args)