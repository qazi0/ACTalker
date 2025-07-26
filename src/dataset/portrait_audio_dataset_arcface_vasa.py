import os
import numpy as np
import random
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from transformers import CLIPImageProcessor, AutoFeatureExtractor
from torch.utils.data import Dataset, ConcatDataset

import librosa
import soundfile as sf

import os
import json
import decord
import einops
from ..utils.motion_estimation_service import get_motion_score
from src.utils.util import import_filename, save_videos_grid, seed_everything
from src.utils.face_align.utils import get_pts5
import cv2
import math
from einops import rearrange


def get_bbox_by_aspect(bbox_s, aspect_type, w, h):
    x1, y1, x2, y2 = bbox_s
    ww = x2 - x1
    hh = y2 - y1
    cc_x = (x1 + x2)/2
    cc_y = (y1 + y2)/2
    if aspect_type == '1:1':
        # 1:1
        ww = hh = min(ww, hh)
        x1, x2 = round(cc_x - ww/2), round(cc_x + ww/2)
        y1, y2 = round(cc_y - hh/2), round(cc_y + hh/2)
    elif aspect_type == '16:9':
        # 16:9
        ww = hh / 9 * 16
        x1, x2 = round(cc_x - ww/2), round(cc_x + ww/2)
        if x1 < 0:
            x2 -= x1
            x1 = 0
        if x2 > w:
            x1 = max(0, x1-(x2-w))
            x2 = w
    elif aspect_type == '9:16':
        # 9:16
        hh = ww / 9 * 16
        y1, y2 = y1, round(y1 + hh)
        if y2 > h:
            y1 = max(0, y1-(y2-h))
            y2 = h
    else:
        print('aspect_type: ', aspect_type)
        raise NotImplementedError
    
    return [x1, y1, x2, y2]

eps = 0.01
def smart_width(d):
    if d<5:
        return 1
    elif d<10:
        return 2
    elif d<20:
        return 3
    elif d<40:
        return 4
    elif d<80:
        return 5
    elif d<160:
        return 6  
    elif d<320:
        return 7 
    else:
        return 8



def draw_bodypose(canvas, candidate, subset, random_select, length_scale = 1, draw_points = False, drop_head=0.0):
    H, W, C = canvas.shape
    candidate = np.array(candidate)
    subset = np.array(subset)

    limbSeq = [[2, 3], [2, 6], [3, 4], [4, 5], [6, 7], [7, 8], [2, 9], [9, 10], \
               [10, 11], [2, 12], [12, 13], [13, 14], [2, 1], [1, 15], [15, 17], \
               [1, 16], [16, 18], [3, 17], [6, 18]]

    colors = [[255, 0, 0], [255, 85, 0], [255, 170, 0], [255, 255, 0], [170, 255, 0], [85, 255, 0], [0, 255, 0], \
              [0, 255, 85], [0, 255, 170], [0, 255, 255], [0, 170, 255], [0, 85, 255], [0, 0, 255], [85, 0, 255], \
              [170, 0, 255], [255, 0, 255], [255, 0, 170], [255, 0, 85]]
    head_ratio = random.random()
    for i in range(17):
        for n in range(len(subset)):
            if head_ratio < drop_head and i in [12, 13, 14, 15, 16, 17, 19]:
                continue
            index = subset[n][np.array(limbSeq[i]) - 1]
            if -1 in index:
                continue
            ratio = random.random()
            if ratio < random_select:
                # print(f"skipping {index} due to ratio({ratio}) ...")
                continue
            Y = candidate[index.astype(int), 0] * float(W)
            X = candidate[index.astype(int), 1] * float(H)
            Y = Y * length_scale
            X = X * length_scale
            mX = np.mean(X)
            mY = np.mean(Y)
            length = ((X[0] - X[1]) ** 2 + (Y[0] - Y[1]) ** 2) ** 0.5            
            angle = math.degrees(math.atan2(X[0] - X[1], Y[0] - Y[1]))

            width = smart_width(length)
            polygon = cv2.ellipse2Poly((int(mY), int(mX)), (int(length / 2), width), int(angle), 0, 360, 1)
            cv2.fillConvexPoly(canvas, polygon, colors[i])       

    canvas = (canvas * 0.6).astype(np.uint8)

    if draw_points:
        for i in range(18):
            for n in range(len(subset)):
                index = int(subset[n][i])
                if index == -1:
                    continue
                ratio = random.random()
                if ratio < random_select:
                    # print(f"skipping {index} due to ratio({ratio}) ...")
                    continue                
                x, y = candidate[index][0:2]
                x = int(x * W)
                y = int(y * H)
                radius = 4
                cv2.circle(canvas, (int(x), int(y)), radius, colors[i], thickness=-1)

    return canvas


def draw_handpose(canvas, all_hand_peaks, random_select, length_scale=1, draw_points=False):
    import matplotlib
    
    H, W, C = canvas.shape

    edges = [[0, 1], [1, 2], [2, 3], [3, 4], [0, 5], [5, 6], [6, 7], [7, 8], [0, 9], [9, 10], \
             [10, 11], [11, 12], [0, 13], [13, 14], [14, 15], [15, 16], [0, 17], [17, 18], [18, 19], [19, 20]]
    
    # (person_number*2, 21, 2)
    for i in range(len(all_hand_peaks)):
        peaks = all_hand_peaks[i]
        peaks = np.array(peaks)
        
        for ie, e in enumerate(edges):

            x1, y1 = peaks[e[0]]
            x2, y2 = peaks[e[1]]
            
            x1 = int(x1 * W * length_scale)
            y1 = int(y1 * H * length_scale)
            x2 = int(x2 * W * length_scale)
            y2 = int(y2 * H * length_scale)
            if x1 > eps and y1 > eps and x2 > eps and y2 > eps:
                length = ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5                
                width = smart_width(length * length_scale)
                cv2.line(canvas, (x1, y1), (x2, y2), matplotlib.colors.hsv_to_rgb([ie / float(len(edges)), 1.0, 1.0]) * 255, thickness=width)

        if draw_points:
            for _, keyponit in enumerate(peaks):
                x, y = keyponit

                x = int(x * W)
                y = int(y * H)
                if x > eps and y > eps:
                    radius = 3
                    cv2.circle(canvas, (x, y), radius, (0, 0, 255), thickness=-1)
    return canvas


def draw_facepose(canvas, all_lmks):
    H, W, C = canvas.shape
    for lmks in all_lmks:
        lmks = np.array(lmks)
        for lmk in lmks:
            x, y = lmk
            x = int(x * W)
            y = int(y * H)
            if x > eps and y > eps:
                radius = 3
                cv2.circle(canvas, (x, y), radius, (255, 255, 255), thickness=-1)
    return canvas

# Calculate the resolution 
def size_calculate(h, w, resolution):
    
    H = float(h)
    W = float(w)

    # resize the short edge to the resolution
    k = float(resolution) / min(H, W) # short edge
    H *= k
    W *= k

    # resize to the nearest integer multiple of 64
    H = int(np.round(H / 64.0)) * 64
    W = int(np.round(W / 64.0)) * 64
    return H, W

def warpAffine_kps(kps, M):
    a = M[:,:2]
    t = M[:,2]
    kps = np.dot(kps, a.T) + t
    return kps

def draw_pose(pose, H, W, draw_face, threshold=0.3, random_select=0, length_scale=1, drop_head=0.0):
    bodies = pose['bodies']
    faces = pose['faces']
    hands = pose['hands']
    candidate = bodies['candidate']
    subset = bodies['subset']
    score = bodies['score'][0]
    hand_score = pose['hands_score']

    # only the most significant person
    faces = pose['faces'][:1]
    hands = pose['hands'][:2]
    candidate = bodies['candidate'][:18]
    subset = bodies['subset'][:1]

    un_visible = score<threshold
    candidate[un_visible[0]] = -1    
    # TODO: check here (some pkl have (4,x)/(6,x)/(8,x) instead of (2,x))
    # only the most significant person?
    hand_score = hand_score[:2]
    hands[hand_score<threshold] = -1   

    # draw
    canvas = np.zeros(shape=(H, W, 3), dtype=np.uint8)
    canvas = draw_bodypose(canvas, candidate, subset, random_select, length_scale, drop_head=drop_head)
    canvas = draw_handpose(canvas, hands, random_select, length_scale)
    if draw_face == True:
        canvas = draw_facepose(canvas, faces)

    return canvas

def affine_align_3landmarks(landmarks, M):
    new_landmarks = np.concatenate([landmarks, np.ones((3, 1))], 1)
    affined_landmarks = np.matmul(new_landmarks, M.transpose())
    return affined_landmarks

def get_affine(src):
    dst = np.array([[87,  59],
                    [137,  59],
                    [112, 120]], dtype=np.float32)
    tform = trans.SimilarityTransform()
    tform.estimate(src, dst)
    M = tform.params[0:2, :]
    return M

def get_mouth_bias(three_points):
    bias = np.array([112, 120]) - three_points[2]
    return bias

def get_eyes_mouths(landmark):
    three_points = np.zeros((3, 2))
    three_points[0] = landmark[32:56].mean(0)
    three_points[1] = landmark[56:80].mean(0)
    three_points[2] = landmark[102:174].mean(0)

    return three_points

def affine_align_img(img, M, crop_size=224):
    warped = cv2.warpAffine(img, M, (crop_size, crop_size), borderValue=0.0)
    return warped

def align_face(image, landmark, output_shape=(112, 112)):
    landmark = np.asarray(landmark)
    if len(landmark) == 256:
        left_eye = landmark[32:56]  
        left_eye = left_eye.mean(axis=0, keepdims=True)
        right_eye = landmark[56:80]  
        right_eye = right_eye.mean(axis=0, keepdims=True)
        nose = landmark[80:81] 
        mouth_left = landmark[102:103] 
        mouth_right = landmark[156:157]  
        points = np.concatenate([left_eye, right_eye, nose, mouth_left, mouth_right], axis=0)
    elif len(landmark) == 68:
        left_eye = landmark[36:42]  
        left_eye = left_eye.mean(axis=0, keepdims=True)
        right_eye = landmark[42:48]  
        right_eye = right_eye.mean(axis=0, keepdims=True)
        nose = landmark[30:31] 
        mouth_left = landmark[48:49] 
        mouth_right = landmark[54:55]  
        points = np.concatenate([left_eye, right_eye, nose, mouth_left, mouth_right], axis=0)
    else:
        raise ValueError
    # 定义目标五点位置
    dst_points = np.array([
        (30.2946, 51.6963),
        (65.5318, 51.5014),
        (48.0252, 71.7366),
        (33.5493, 92.3655),
        (62.7299, 92.2041)
    ], dtype=np.float32)
    dst_points[:, 0] += 8.0
    # 计算仿射变换矩阵
    tform = tf.SimilarityTransform()
    tform.estimate(np.array(points), dst_points)

    # 应用变换
    aligned_image = tf.warp(image, tform.inverse, output_shape=output_shape)
    return aligned_image

def center_crop(img_driven, face_bbox, scale=1.0):
    h, w = img_driven.shape[:2]
    x0, y0, x1, y1 = face_bbox[:4]
    center = (int((x0 + x1) / 2), int((y0 + y1) / 2))
    crop_size = int(max(x1 - x0, y1 - y0)) // 2
    crop_size = int(crop_size * scale)
    new_x0, new_y0, new_x1, new_y1 = center[0] - crop_size, center[1] - crop_size, center[0] + crop_size, center[1] + crop_size
    pad_left, pad_top, pad_right, pad_bottom = 0, 0, 0, 0
    if new_x0 < 0:
        pad_left, new_x0 = -new_x0, 0
    if new_y0 < 0:
        pad_top, new_y0 = -new_y0, 0
    if new_x1 > w:
        pad_right, new_x1 = new_x1 - w, w
    if new_y1 > h:
        pad_bottom, new_y1 = new_y1 - h, h
    img_mtn = img_driven[new_y0:new_y1, new_x0:new_x1]
    img_mtn = cv2.copyMakeBorder(img_mtn, pad_top, pad_bottom, pad_left, pad_right, cv2.BORDER_CONSTANT, value=(0, 0, 0))
    return img_mtn

def img2tensor(image, transform=None):
    if transform is None:
        output_tensor = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0)
        output_tensor = output_tensor / 255.
    else:
        image = Image.fromarray(image)
        output_tensor = transform(image).unsqueeze(0)
    return output_tensor


def process_bbox(bbox, expand_radio, height, width):
    """
    raw_vid_path:
    bbox: format: x1, y1, x2, y2
    radio: expand radio against bbox size
    height,width: source image height and width
    """

    def expand(bbox, ratio, height, width):
        
        bbox_h = bbox[3] - bbox[1]
        bbox_w = bbox[2] - bbox[0]
        
        expand_x1 = max(bbox[0] - ratio * bbox_w, 0)
        expand_y1 = max(bbox[1] - ratio * bbox_h, 0)
        expand_x2 = min(bbox[2] + ratio * bbox_w, width)
        expand_y2 = min(bbox[3] + ratio * bbox_h, height)

        return [expand_x1,expand_y1,expand_x2,expand_y2]

    def to_square(bbox_src, bbox_expend, height, width):

        h = bbox_expend[3] - bbox_expend[1]
        w = bbox_expend[2] - bbox_expend[0]
        c_h = (bbox_expend[1] + bbox_expend[3]) / 2
        c_w = (bbox_expend[0] + bbox_expend[2]) / 2

        c = min(h, w) / 2

        c_src_h = (bbox_src[1] + bbox_src[3]) / 2
        c_src_w = (bbox_src[0] + bbox_src[2]) / 2

        s_h, s_w = 0, 0
        if w < h:
            d = abs((h - w) / 2)
            s_h = min(d, abs(c_src_h-c_h))
            s_h = s_h if  c_src_h > c_h else s_h * (-1)
        else:
            d = abs((h - w) / 2)
            s_w = min(d, abs(c_src_w-c_w))
            s_w = s_w if  c_src_w > c_w else s_w * (-1)


        c_h = (bbox_expend[1] + bbox_expend[3]) / 2 + s_h
        c_w = (bbox_expend[0] + bbox_expend[2]) / 2 + s_w

        square_x1 = c_w - c
        square_y1 = c_h - c
        square_x2 = c_w + c
        square_y2 = c_h + c 

        return [round(square_x1), round(square_y1), round(square_x2), round(square_y2)]


    bbox_expend = expand(bbox, expand_radio, height=height, width=width)
    processed_bbox = to_square(bbox, bbox_expend, height=height, width=width)

    return processed_bbox


def get_motion_bucketid(bboxs, max_value=128):
    bbox_init = bboxs[0]
    x1, y1, x2, y2 = bbox_init
    init_length = np.sqrt((x2-x1)*(y2-y1))
    bboxs = np.array(bboxs)
    
    # motion = 0
    # for step in range(1, 2):
    diff = (bboxs[1:] - bboxs[:-1]) ** 2
    diff = np.sqrt(diff.sum(1))
    motion = np.mean(diff / init_length) * 1024
        # motion += motion_step
    motion = int(motion)
    motion = max(motion, 0)
    motion = min(motion, max_value)
    return motion

def get_head_exp_motion_bucketid(lmks, max_value=128):
    
    exp_lmks = np.array([lmk[:102] - lmk[80] for lmk in lmks])
    # print(exp_lmks.shape)
    init_lmk = exp_lmks[0]
    scale = np.sqrt(((init_lmk.max(0) - init_lmk.min(0))**2).sum())
    exp_var = np.sqrt(((exp_lmks - exp_lmks.mean(0))**2).sum(2))
    exp_var = exp_var.mean()
    exp_var = exp_var/scale * 1024
    # print(exp_var)

    exp_var = int(exp_var)
    exp_var = max(exp_var, 0)
    exp_var = min(exp_var, max_value)


    head_poses = np.array([lmk[80] for lmk in lmks])
    # print(exp_lmks.shape)
    head_var = np.sqrt(((head_poses - head_poses.mean(0))**2).sum(1))
    head_var = head_var.mean()/scale  * 256
    # print(head_var)

    head_var = int(head_var)
    head_var = max(head_var, 0)
    head_var = min(head_var, max_value)

    return head_var, exp_var

def check_lmk(lmks):
    lmks = [get_pts5(lmk) for lmk in lmks]
    init_lmk = lmks[0]
    scale = np.sqrt(((init_lmk.max(0) - init_lmk.min(0))**2).sum())
    lmks = np.array(lmks)
    diff = lmks[1:] - lmks[:-1]
    v = np.sqrt((diff ** 2).sum(2)).mean(1) / scale
    return round(v.max()/v.mean()*32)


def get_head_preprocessed_img(image, head_bbox):
    image_transform = transforms.Compose(
        [
            transforms.Resize((112, 112), interpolation=transforms.InterpolationMode.BILINEAR), 
            transforms.ToTensor()
        ]
    )

    x_min, y_min, x_max, y_max = head_bbox
    face_image = image[int(y_min):int(y_max), int(x_min):int(x_max)]
    face_image = Image.fromarray(face_image)
    face_image = image_transform(face_image).float()
    face_image = (face_image * 2.) - 1.
    return face_image

def draw_landmarks_with_indices(image, landmarks, image_size):
    """
    Draws facial landmarks on an image and annotates them with their indices.

    Parameters:
    - landmarks: List of tuples [(x1, y1), (x2, y2), ..., (xn, yn)] representing facial landmarks.
    - image_size: Tuple (width, height) representing the size of the image.
    """
    # Create a blank white image
    # image = np.ones((image_size[1], image_size[0], 3), dtype=np.uint8) * 0

    # Iterate over the landmarks and draw them on the image
    for idx, (x, y) in enumerate(landmarks[102:136]):
        # Draw the landmark as a small circle
        cv2.circle(image, (int(x), int(y)), 2, (0, 0, 255), -1)
        # Annotate the landmark with its index
        cv2.putText(image, str(idx), (int(x) + 5, int(y) - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    return image
class ParentDataset(Dataset):
    """
    Dataset of videos, each video can be represented as:
      - an image of concatenated frames
      - '.mp4' or '.gif'
      - folder with all frames
    """

    def __init__(self, 
                 cfg, 
                 list_paths, 
                 root_path='/apdcephfs_jn/share_302243908/0_public_datasets/talkinghead/all/',
                 repeats=None):
        
        meta_paths = []
        pkl_paths = {}
        if repeats is None:
            repeats = [1] * len(list_paths)
        assert len(repeats) == len(list_paths)
        
        for list_path, repeat_time in zip(list_paths, repeats):
            with open(list_path, 'r') as f:
                num = 0
                f.readline()
                for line in f.readlines():
                    line_info = line.strip()
                    meta, valid_clip, _, av_offset = line_info.split()
                    if meta and int(valid_clip) >= cfg['T'] and abs(int(av_offset)) <= 1:
                        for _ in range(repeat_time):
                            meta_paths.append(os.path.join(root_path, line_info.split()[0]))
                        num += 1
                print(f'{list_path}: {num} x {repeat_time} = {num*repeat_time} samples')
        self.meta_paths = meta_paths
        self.root_path = root_path
        self.image_size = cfg['image_size']
        self.T = cfg['T']
        self.drop_head = cfg['drop_head']
        self.color_jitter = cfg['color_jitter']
        self.to_tensor = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        self.pose_to_tensor = transforms.Compose([
            transforms.ToTensor(),
        ])
        # hack here (xzn)
        self.vasa_image_size = 256
        self.vasa_transform = transforms.Compose(
            [
                transforms.Resize(self.vasa_image_size),
                transforms.ToTensor(),
                transforms.Normalize([0], [1]),
            ]
        )    
        # self.clip_image_processor = CLIPImageProcessor()
        self.feature_extractor = AutoFeatureExtractor.from_pretrained("enhance_model/whisper-tiny/")
    def _color_transfer(self, img):
        transfer_c = np.random.uniform(0.3, 1.6)
        start_channel = np.random.randint(0, 2)
        end_channel = np.random.randint(start_channel + 1, 4)
        img2 = img.copy()

        img2[:, :, start_channel:end_channel] = np.minimum(np.maximum(img[:, :, start_channel:end_channel] * transfer_c, np.zeros(img[:, :, start_channel:end_channel].shape)),
                                    np.ones(img[:, :, start_channel:end_channel].shape) * 255)
        return img2
    def _blur_and_sharp(self, img):
        blur = np.random.randint(0, 2)
        img2 = img.copy()
        if blur:
            ksize = np.random.choice([3, 5, 7, 9])
            output = cv2.medianBlur(img2, ksize)
        else:
            kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
            output = cv2.filter2D(img2, -1, kernel)
        return output
    def augmentation_mtn_pcavs(self, img_mtn):
        img_mtn = self._color_transfer(img_mtn)
        img_mtn = self._blur_and_sharp(img_mtn)
        return img_mtn
    def get_union_bbox(self, bboxes):
        bboxes = np.array(bboxes)
        min_x = np.min(bboxes[:, 0])
        min_y = np.min(bboxes[:, 1])
        max_x = np.max(bboxes[:, 2])
        max_y = np.max(bboxes[:, 3])
        return np.array([min_x, min_y, max_x, max_y])
    def get_low_high_half_bbox(self, bbox):
        x1, y1, x2, y2 = bbox
        x_mid = (x1 + x2) / 2
        y_mid = (y1 + y2) / 2
        return np.array([x1, y1, x2, y_mid]), np.array([x1, y_mid, x2, y2])
    def get_face_mask(self, img, bbox):
        mask = np.zeros_like(np.array(img))
        min_x, min_y, max_x, max_y = bbox
        min_x = max(min_x, 0)
        min_y = max(min_y, 0)
        mask[round(min_y):round(max_y), round(min_x):round(max_x)] = 255
        return mask, Image.fromarray(mask)

    def crop_resize_img(self, img, bbox, image_size):
        x1, y1, x2, y2 = bbox
        img = img.crop((x1, y1, x2, y2))
        w, h = img.size
        scale = np.sqrt(image_size ** 2 / (h * w))
        new_w = int(w * scale) // 64 * 64
        new_h = int(h * scale) // 64 * 64
        img = img.resize((new_w, new_h), Image.LANCZOS)
        return img

    def crop_face_vasa(self, image, landmark):
        # face_landmark_image = draw_landmarks_with_indices(landmark, image.shape[:2])
        # cv2.imwrite('face_landmark_image.png', face_landmark_image)
        # import pdb;pdb.set_trace()
        if len(landmark) == 256:
            face_landmark = np.asarray(landmark[:174])
        elif len(landmark) == 68:
            face_landmark = np.asarray(landmark[16:])
        else:
            raise ValueError
       
        face_x_min, face_x_max = min(face_landmark[:, 0]), max(face_landmark[:, 0])
        face_y_min, face_y_max = min(face_landmark[:, 1]), max(face_landmark[:, 1])
        lmk_face_bbox = np.asarray([face_x_min, face_y_min, face_x_max, face_y_max])   
        vasa_crop_face = center_crop(image, lmk_face_bbox)
        vasa_crop_face = Image.fromarray(vasa_crop_face)    
        vasa_crop_face_tensor = self.vasa_transform(vasa_crop_face)
        return vasa_crop_face_tensor
    
    def crop_face_pdfgc(self, image, landmark):
        three_points = get_eyes_mouths(np.asarray(landmark))
        avg_points = three_points
        M = get_affine(avg_points)
        affined_3landmarks = affine_align_3landmarks(three_points, M)
        bias = get_mouth_bias(affined_3landmarks)
        M_i = M.copy()
        M_i[:, 2] = M[:, 2] + bias
        pdfgc_img = affine_align_img(image.copy(), M_i)
        pdfgc_img = cv2.resize(pdfgc_img[:-32, 16:-16], (224, 224))
        driven_face_images_tensor = self.to_tensor(pdfgc_img)
        return driven_face_images_tensor
    
    def get_audio_file(self, wav_path, start_index):
        if not os.path.exists(wav_path):
            return None
        audio_input, sampling_rate = librosa.load(wav_path, sr=16000)
        assert sampling_rate == 16000

        while start_index >= 25 * 30:
            audio_input = audio_input[16000*30:]
            start_index -= 25 * 30
        if start_index + 2 * 25 >= 25 * 30:
            start_index -= 4 * 25
            audio_input = audio_input[16000*4:16000*34]
        else:
            audio_input = audio_input[:16000*30]

        assert 2 * (start_index) >= 0
        assert 2 * (start_index + 2 * 25) <= 1500

        audio_feature = self.feature_extractor(audio_input, 
                                             return_tensors="pt",
                                             sampling_rate=sampling_rate
                                             ).input_features
        return audio_input, audio_feature, start_index
    
    def get_mouth_boxes(self, landmark_list):
        mouth_bboxes = []
        for landmark in landmark_list:
            mouth_lmks = landmark[102:136]
            minx = min(mouth_lmks[:][0])
            miny = min(mouth_lmks[:][1])
            maxx = max(mouth_lmks[:][0])
            maxy = max(mouth_lmks[:][1])
            mouth_bbox = np.array([minx, miny, maxx, maxy])
            mouth_bboxes.append(mouth_bbox)
        return mouth_bboxes
    def __len__(self):
        return len(self.meta_paths)

    def __getitem__(self, idx):
        try:  
            meta_path = self.meta_paths[idx]
            # meta_path = random.sample(self.meta_paths, k=1)[0]

            with open(meta_path, 'r') as f:
                meta_data = json.load(f)

            image_id = meta_data["mp4_path"][:-4].replace('/', '_')
            video_path = os.path.join(self.root_path, meta_data["mp4_path"])
            wav_path = os.path.join(self.root_path, meta_data["wav_path"])
            bbox_list = meta_data["face_list"]
            h,w = meta_data["video_size"]
            s,e = meta_data["valid_clip"]
            T = self.T
            len_valid_clip = e - s

            
            landmark_list = meta_data["landmark_list"]
            quality_score_list = meta_data["quality_score_list"]
            pre_frame_similarity_list = meta_data["pre_frame_similarity_list"]
            pre_face_similarity_list = meta_data["pre_face_similarity_list"]

            # import pdb;pdb.set_trace()
            cap = decord.VideoReader(video_path, fault_tol=1)
            raw_fps = cap.get_avg_fps()

            total_frames = len(cap)
            assert total_frames==len(landmark_list)
            assert total_frames==len(quality_score_list)
            assert total_frames==len(pre_frame_similarity_list)
            
            
            
            assert len_valid_clip >= T
            if len_valid_clip < 2 * T:
                step = 1
            else:
                step = 2
            drive_idx_start = random.randint(s, e - T * step)
            drive_idx_list = list(range(drive_idx_start, drive_idx_start + T * step, step))
            assert len(drive_idx_list) == T

            # src
            src_idx = random.randint(drive_idx_list[0]-T, drive_idx_list[-1]+T)
            src_idx = min(src_idx, e-1)
            src_idx = max(src_idx, s)

            imSrc = Image.fromarray(cap[src_idx].asnumpy())

            """ 获取头部图片，用于id保持 """
            head_img = get_head_preprocessed_img(image=cap[src_idx].asnumpy(), head_bbox=bbox_list[src_idx])
            """ 获取头部图片，用于id保持 """

            img_width, img_height = imSrc.size
            bbox = self.get_union_bbox(bbox_list[s:e])
            mouth_bboxes = self.get_mouth_boxes(landmark_list[s:e])
            
            mouth_bbox = self.get_union_bbox(mouth_bboxes)
            
            # upper_half_bbox, lower_half_bbox = self.get_low_high_half_bbox(bbox)
            face_arr, face_mask = self.get_face_mask(imSrc, bbox)
            mouth_arr, mouth_mask = self.get_face_mask(imSrc, mouth_bbox)
            exp_arr = face_arr - mouth_arr
            exp_mask = Image.fromarray(exp_arr)
            # upper_half_face_mask = self.get_face_mask(imSrc, upper_half_bbox)
            # lower_half_face_mask = self.get_face_mask(imSrc, lower_half_bbox)


            scale = 2 * np.random.rand()
            bbox_s = process_bbox(bbox, expand_radio=scale, height=img_height, width=img_width)

            aspect_type = random.sample(['1:1', '9:16', '16:9'], k=1)[0]
            bbox_aspect  = get_bbox_by_aspect(bbox_s, aspect_type, w, h)
            
            image_size = 512 + (self.image_size - 512) * np.random.rand()


            imSrc = self.crop_resize_img(imSrc, bbox_aspect, image_size)

            face_mask = self.crop_resize_img(face_mask, bbox_aspect, image_size)
            mouth_mask = self.crop_resize_img(mouth_mask, bbox_aspect, image_size)
            exp_mask = self.crop_resize_img(exp_mask, bbox_aspect, image_size)
            imSameIDs = []
            bboxes  = []
            lmks = []

            for drive_idx in drive_idx_list:
                imSameID = Image.fromarray(cap[drive_idx].asnumpy())
                imSameID = self.crop_resize_img(imSameID, bbox_aspect, image_size)
                imSameIDs.append(imSameID)
                bboxes.append(bbox_list[drive_idx])
                lmks.append(np.array(landmark_list[drive_idx]))



            audio_offset = drive_idx_list[0]
            audio_step = step
            
            fps = raw_fps / step
            # motion_bucket_id = get_motion_bucketid(bboxes)
            motion_bucket_id_head, motion_bucket_id_exp = get_head_exp_motion_bucketid(lmks)
            outlier = check_lmk(lmks)
            if outlier > 128:
                return self.__getitem__(random.randint(0, len(self)))


            audio_input, audio_feature, audio_offset = self.get_audio_file(wav_path, audio_offset)


            vid = torch.stack([self.to_tensor(imSameID) for imSameID in imSameIDs], dim=0)
            _, _, h, w = vid.shape
            vid_small = torch.nn.functional.interpolate(vid, (h//4, w//4))
            motion_bucket_id_flow = get_motion_score(einops.rearrange((vid_small * 0.5 + 0.5)*255, 't c h w -> t h w c'))

            if motion_bucket_id_flow > 128:
                return self.__getitem__(random.randint(0, len(self)))


            # clip_image = self.clip_image_processor(
            #     images=imSrc.resize((224, 224), Image.LANCZOS), return_tensors="pt"
            # ).pixel_values[0]

            # import torchvision
            # torchvision.utils.save_image(clip_image, 'clip_image.png')
            driven_dwpose_images_list = []
            driven_face_images_list = []
            driven_pose_images_list = []
           
            for i, drive_idx in enumerate(drive_idx_list):
                target_face = cap[drive_idx].asnumpy()
                H, W = target_face.shape[:2]
                face_bbox = bbox_list[drive_idx]
                target_face_landmark = landmark_list[drive_idx] 

                vasa_face_image = target_face.copy()
                if self.color_jitter:
                    vasa_face_image = self.augmentation_mtn_pcavs(vasa_face_image)
                    
                driven_face_images_tensor = self.crop_face_vasa(vasa_face_image, target_face_landmark)
                driven_face_images_list.append(driven_face_images_tensor)

                driven_pose_image = center_crop(target_face.copy(), face_bbox, scale=1.7)
                driven_pose_image = Image.fromarray(driven_pose_image)
                driven_pose_images_tensor = self.vasa_transform(driven_pose_image) # 3, 256, 256
                driven_pose_images_list.append(driven_pose_images_tensor)

                       
        
            sample = dict(
                image_id=image_id,
                pixel_values_vid=vid,
                pixel_values_ref_img=self.to_tensor(imSrc),
                pixel_values_face_mask=self.pose_to_tensor(face_mask),
                pixel_values_mouth_mask=self.pose_to_tensor(mouth_mask),
                pixel_values_exp_mask=self.pose_to_tensor(exp_mask),
                head_img=head_img,
                audio_feature=audio_feature[0],
                audio_input=audio_input,
                audio_offset=audio_offset,
                audio_step=audio_step,
                fps=fps,
                motion_bucket_id_head=motion_bucket_id_head,
                motion_bucket_id_exp=motion_bucket_id_exp,
                vasa_face_image=torch.stack(driven_face_images_list, dim=0),
                vasa_pose_image=torch.stack(driven_pose_images_list, dim=0),
            )

            return sample

        except KeyboardInterrupt:
            exit()
        except Exception as e:
            return self.__getitem__(random.randint(0, len(self)))

        

class SingDataset(ParentDataset):
    def __init__(self, cfg):
        # Please specify your own dataset root path
        root_path = './data/talkinghead/all/'
        list_paths = [
          
        ]
        repeats = [5,8,2]

        super().__init__(cfg, list_paths, root_path, repeats)
        print('SingDataset: ', len(self))

class ChineseTalkDataset(ParentDataset):
    def __init__(self, cfg):
        # Please specify your own dataset root path
        root_path = './data/talkinghead/all/'
        list_paths = [
          
        ]
        repeats = [1,4,2]
        super().__init__(cfg, list_paths, root_path, repeats)
        print('ChineseTalkDataset: ', len(self))

class EnglishTalkDataset(ParentDataset):
    def __init__(self, cfg):
        # Please specify your own dataset root path
        root_path = './data/talkinghead/all/'
        list_paths = [
          
        ]
        repeats = [50,1,2,2,1,1]

        super().__init__(cfg, list_paths, root_path, repeats)
        print('EnglishTalkDataset: ', len(self))

def PortraitDataset(cfg=None):
    return ConcatDataset([SingDataset(cfg), ChineseTalkDataset(cfg), EnglishTalkDataset(cfg)])
    # return ConcatDataset([SingDataset(cfg)])


def main():
    num_frames = 25

    dataset = PortraitDataset(cfg={
                        'image_size':640,
                        'T': num_frames})
    dataset[0]
    
    import tqdm
    print(len(dataset))
    # exit()
    dataloader = torch.utils.data.DataLoader(
        dataset, 
        batch_size=1, 
        shuffle=True, 
        num_workers=0
    )
    import tqdm
    import torchvision
    idx = 0
    for batch in tqdm.tqdm(dataloader):
        print(batch['motion_bucket_id_head'].item(), batch['motion_bucket_id_exp'].item())
        pixel_values_ref_img = batch['pixel_values_ref_img'].unsqueeze(1).repeat(1,num_frames,1,1,1)
        pixel_values_ref_img = rearrange(
            pixel_values_ref_img, "b f c h w -> (b f) c h w"
        )

        pixel_values_face_mask = batch['pixel_values_face_mask'].unsqueeze(1).repeat(1,num_frames,1,1,1)
        pixel_values_face_mask = rearrange(
            pixel_values_face_mask, "b f c h w -> (b f) c h w"
        )
        
        pixel_values = rearrange(batch["pixel_values_vid"], 'b f c h w-> (b f) c h w')
        
        concat = torch.cat([pixel_values_ref_img*0.5+0.5, pixel_values*0.5+0.5, pixel_values_face_mask], dim=-1)
        # video = concat.reshape((1, num_frames, 3, concat.shape[-2], concat.shape[-1]))
        video = rearrange(concat, "(b f) c h w -> b c f h w ", b=1)


        video_path = f'debug_video/{idx}.mp4'
        # cv2.imwrite(f'{save_dir}/samples/sample_{global_step}_{accelerator.device}.jpg',   torch.cat([concat[i] for i in range(concat.shape[0])], dim=-1).permute(1,2,0).cpu().numpy()[:,:,[2,1,0]]*255)
        save_videos_grid(video, video_path, n_rows=video.shape[0], fps=batch['fps'].item())

        
        

        audio_input = batch['audio_input'][0]
        start_t = batch['audio_offset'][0]
        step_t = batch['audio_step'][0]
        audio_clip = audio_input[start_t*640:(start_t+25*step_t)*640]

        wav_path = f'debug_video/{idx}.wav'

        # librosa.output.write_wav(wav_path, audio_clip.numpy(), 16000, norm=False)
        sf.write(wav_path, audio_clip.numpy(), 16000)

        audio_video_path = f"debug_video/{idx}_{batch['image_id'][0]}_{batch['fps'].item()}_{batch['motion_bucket_id_head'].item()}_{batch['motion_bucket_id_exp'].item()}.mp4"
        os.system(f'ffmpeg -i {video_path} -i {wav_path} -shortest {audio_video_path} -y; rm {video_path}; rm {wav_path}')




        # import ipdb
        # ipdb.set_trace()


        idx+=1

        if idx > 1000:
            break



        

        
    
if __name__ == '__main__':
    main()


