import os
import sys
import cv2
import pdb
import time
import glob
import pickle
import argparse
import subprocess
import numpy as np
from shutil import rmtree
import fnmatch
import copy
import scenedetect
from tqdm import tqdm
from scenedetect.video_manager import VideoManager
from scenedetect.scene_manager import SceneManager
from scenedetect.frame_timecode import FrameTimecode
from scenedetect.stats_manager import StatsManager
from scenedetect.detectors import ContentDetector

from scipy import signal
from scipy.io import wavfile
from scipy.interpolate import interp1d

from detectors import S3FD
from sync.SyncNetInstance import *


def bb_intersection_over_union(boxA, boxB):
  
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    
    interArea = max(0, xB - xA) * max(0, yB - yA)
    
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    
    iou = interArea / float(boxAArea + boxBArea - interArea)
    
    return iou

def track_shot(opt, scenefaces):

    iouThres  = 0.5     # Minimum IOU between consecutive face detections
    tracks    = []
    while True:
        track     = []
        for framefaces in scenefaces:
            for face in framefaces:
                if track == []:
                    track.append(face)
                    framefaces.remove(face)
                elif face['frame'] - track[-1]['frame'] <= opt.num_failed_det:
                    iou = bb_intersection_over_union(face['bbox'], track[-1]['bbox'])
                    if iou > iouThres:
                        track.append(face)
                        framefaces.remove(face)
                        continue
                else:
                    break

        if track == []:
            break
        elif len(track) > opt.min_track:
        
            framenum = np.array([ f['frame'] for f in track ])
            bboxes = np.array([np.array(f['bbox']) for f in track])

            frame_i = np.arange(framenum[0],framenum[-1]+1)

            bboxes_i = []
            for ij in range(0,4):
                interpfn  = interp1d(framenum, bboxes[:,ij])
                bboxes_i.append(interpfn(frame_i))
            bboxes_i  = np.stack(bboxes_i, axis=1)

            if max(np.mean(bboxes_i[:,2]-bboxes_i[:,0]), np.mean(bboxes_i[:,3]-bboxes_i[:,1])) > opt.min_face_size:
                tracks.append({'frame':frame_i,'bbox':bboxes_i})

    return tracks

def crop_video(opt, track, cropfile):

    flist = glob.glob(os.path.join(opt.frames_dir,opt.reference,'*.jpg'))
    flist.sort()

    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    vOut = cv2.VideoWriter(cropfile+'t.avi', fourcc, opt.frame_rate, (224,224))

    dets = {'x':[], 'y':[], 's':[]}

    for det in track['bbox']:
        dets['s'].append(max((det[3]-det[1]),(det[2]-det[0]))/2) 
        dets['y'].append((det[1]+det[3])/2) # crop center x 
        dets['x'].append((det[0]+det[2])/2) # crop center y

    # Smooth detections
    dets['s'] = signal.medfilt(dets['s'],kernel_size=13)   
    dets['x'] = signal.medfilt(dets['x'],kernel_size=13)
    dets['y'] = signal.medfilt(dets['y'],kernel_size=13)

    for fidx, frame in enumerate(track['frame']):
        cs  = opt.crop_scale
        bs  = dets['s'][fidx]   # Detection box size
        bsi = int(bs*(1+2*cs))  # Pad videos by this amount 
        image = cv2.imread(flist[frame])
        frame = np.pad(image,((bsi,bsi),(bsi,bsi),(0,0)), 'constant', constant_values=(110,110))
        my  = dets['y'][fidx]+bsi  # BBox center Y
        mx  = dets['x'][fidx]+bsi  # BBox center X
        face = frame[int(my-bs):int(my+bs*(1+2*cs)),int(mx-bs*(1+cs)):int(mx+bs*(1+cs))]
        vOut.write(cv2.resize(face,(224,224)))

    audiotmp    = os.path.join(opt.tmp_dir,opt.reference,'audio.wav')
    audiostart  = (track['frame'][0])/opt.frame_rate
    audioend    = (track['frame'][-1]+1)/opt.frame_rate

    vOut.release()

    command = ("ffmpeg -y -i %s -ss %.3f -to %.3f %s -loglevel quiet" % (os.path.join(opt.avi_dir,opt.reference,'audio.wav'),audiostart,audioend,audiotmp)) 
    output = subprocess.call(command, shell=True, stdout=None)

    if output != 0:
        pdb.set_trace()

    command = ("ffmpeg -y -i %st.avi -i %s -c:v copy -c:a copy %s.avi -loglevel quiet" % (cropfile,audiotmp,cropfile))
    output = subprocess.call(command, shell=True, stdout=None)

    if output != 0:
        pdb.set_trace()
    os.remove(cropfile+'t.avi')
    return {'track':track, 'proc_track':dets}


class Evaluation:
    def __init__(self, opt):
        self.opt = opt
        self.DET = S3FD(device='cuda')

        self.s = SyncNetInstance();

        self.s.loadParameters(opt.initial_model);

    def inference_video_for_fid(self, videofile1, videofile2):
        process_imgs_dir1 = os.path.join(self.opt.frames_dir, "dir1")
        process_imgs_dir2 = os.path.join(self.opt.frames_dir, "dir2")
        os.makedirs(process_imgs_dir1)
        os.makedirs(process_imgs_dir2)
        command = ("ffmpeg -y -i %s -qscale:v 2 -threads 1 -f image2 %s -loglevel quiet" % (videofile1, os.path.join(process_imgs_dir1,'%06d.jpg'))) 
        output = subprocess.call(command, shell=True, stdout=None) 
        command = ("ffmpeg -y -i %s -qscale:v 2 -threads 1 -f image2 %s -loglevel quiet" % (videofile2, os.path.join(process_imgs_dir2,'%06d.jpg'))) 
        output = subprocess.call(command, shell=True, stdout=None) 

        command = f'python -m pytorch_fid {process_imgs_dir1}  {process_imgs_dir2}'
        output = subprocess.call(command, shell=True, stdout=None) 

    def inference_video_for_sync(self, videofile=None, is_raw=True):
        self.prepare_data_dir()
        offset, conf, dist = None, None, None
        if videofile is not None:
            self.opt.videofile = videofile

        if is_raw:
            self.prepare_video()
            faces = self.det_for_video()
            scene = self.scene_detect()

            alltracks = []
            vidtracks = []
            for shot in scene:
                if shot[1].frame_num - shot[0].frame_num >= self.opt.min_track :
                    alltracks.extend(track_shot(self.opt,faces[shot[0].frame_num:shot[1].frame_num]))

            for ii, track in enumerate(alltracks):
                vidtracks.append(crop_video(self.opt,track,os.path.join(self.opt.crop_dir,self.opt.reference,'%05d'%ii)))

            rmtree(os.path.join(self.opt.tmp_dir,self.opt.reference))
            
            flist = glob.glob(os.path.join(self.opt.crop_dir, self.opt.reference, '0*.avi'))
            flist.sort()
            dists = []
            for idx, fname in enumerate(flist):
                offset, conf, dist = self.s.evaluate(self.opt, videofile=fname)
                dists.append(dist)
        else:
            offset, conf, dist = self.s.evaluate(self.opt, videofile=self.opt.videofile)
        return offset, conf, dist

    def prepare_data_dir(self):
        ## 清空工作目录，若没有，则创建
        if os.path.exists(os.path.join(self.opt.work_dir,self.opt.reference)):
            rmtree(os.path.join(self.opt.work_dir,self.opt.reference))

        if os.path.exists(os.path.join(self.opt.crop_dir,self.opt.reference)):
            rmtree(os.path.join(self.opt.crop_dir,self.opt.reference))

        if os.path.exists(os.path.join(self.opt.avi_dir,self.opt.reference)):
            rmtree(os.path.join(self.opt.avi_dir,self.opt.reference))

        if os.path.exists(os.path.join(self.opt.frames_dir,self.opt.reference)):
            rmtree(os.path.join(self.opt.frames_dir,self.opt.reference))

        if os.path.exists(os.path.join(self.opt.tmp_dir,self.opt.reference)):
            rmtree(os.path.join(self.opt.tmp_dir,self.opt.reference))

        os.makedirs(os.path.join(self.opt.work_dir,self.opt.reference))
        os.makedirs(os.path.join(self.opt.crop_dir,self.opt.reference))
        os.makedirs(os.path.join(self.opt.avi_dir,self.opt.reference))
        os.makedirs(os.path.join(self.opt.frames_dir,self.opt.reference))
        os.makedirs(os.path.join(self.opt.tmp_dir,self.opt.reference))
    
    def prepare_video(self):
        ## 对输入视频进行帧率转换，并解帧，其中转换后的视频和音频保存在avi_dir，解帧保存在frames_dir，分别命名为video.avi和audio.wav
        command = ("ffmpeg -y -i %s -qscale:v 2 -async 1 -r 25 %s -loglevel quiet" % (self.opt.videofile,os.path.join(self.opt.avi_dir,self.opt.reference,'video.avi')))
        output = subprocess.call(command, shell=True, stdout=None)

        command = ("ffmpeg -y -i %s -qscale:v 2 -threads 1 -f image2 %s -loglevel quiet" % (os.path.join(self.opt.avi_dir,self.opt.reference,'video.avi'),os.path.join(self.opt.frames_dir,self.opt.reference,'%06d.jpg'))) 
        output = subprocess.call(command, shell=True, stdout=None)

        command = ("ffmpeg -y -i %s -ac 1 -vn -acodec pcm_s16le -ar 16000 %s -loglevel quiet" % (os.path.join(self.opt.avi_dir,self.opt.reference,'video.avi'),os.path.join(self.opt.avi_dir,self.opt.reference,'audio.wav'))) 
        output = subprocess.call(command, shell=True, stdout=None)

    def det_for_video(self):
        ## 调用S3FD进行人脸检测
        flist = glob.glob(os.path.join(self.opt.frames_dir,self.opt.reference,'*.jpg'))
        flist.sort()

        dets = []
            
        for fidx, fname in enumerate(flist):
            image = cv2.imread(fname)
            h, w, c = image.shape
            image = image[:, w//2:]
            cv2.imwrite(fname, image)
            # print(image.shape)
            image_np = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            bboxes = self.DET.detect_faces(image_np, conf_th=0.9, scales=[self.opt.facedet_scale])

            dets.append([]);
            for bbox in bboxes:
                dets[-1].append({'frame':fidx, 'bbox':(bbox[:-1]).tolist(), 'conf':bbox[-1]})

        return dets

    def scene_detect(self):

        video_manager = VideoManager([os.path.join(self.opt.avi_dir, self.opt.reference,'video.avi')])
        stats_manager = StatsManager()
        scene_manager = SceneManager(stats_manager)

        scene_manager.add_detector(ContentDetector())
        base_timecode = video_manager.get_base_timecode()
        video_manager.set_downscale_factor()
        video_manager.start()
        scene_manager.detect_scenes(frame_source=video_manager)
        scene_list = scene_manager.get_scene_list(base_timecode)
        if scene_list == []:
            scene_list = [(video_manager.get_base_timecode(),video_manager.get_current_timecode())]

        return scene_list
    
def run_video(opt, video_path):
    opt.videofile = video_path
    setattr(opt,'avi_dir',os.path.join(opt.data_dir, 'pyavi'))
    setattr(opt,'tmp_dir',os.path.join(opt.data_dir, 'pytmp'))
    setattr(opt,'work_dir',os.path.join(opt.data_dir, 'pywork'))
    setattr(opt,'crop_dir',os.path.join(opt.data_dir, 'pycrop'))
    setattr(opt,'frames_dir',os.path.join(opt.data_dir, 'pyframes'))

    eva = Evaluation(opt)
    offset, conf, dist = eva.inference_video_for_sync(is_raw=True)
    rmtree(opt.data_dir)
    return offset, conf, dist

def find_data_files(directory, prefix):
    mp4_files = []
    for root, dirnames, filenames in os.walk(directory):
        for filename in fnmatch.filter(filenames, prefix):
            mp4_files.append(os.path.join(root, filename))
    return mp4_files

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = "FaceTracker");
    parser.add_argument('--data_dir',       type=str, default='output', help='Output direcotry');
    parser.add_argument('--save_dir',       type=str, default=None);
    parser.add_argument('--model_path',       type=str, default=None);
    parser.add_argument('--videofile',      type=str, default='/apdcephfs_jn/share_302243908/zhentaoyu/Driving/00_VASA/01_exp/vasa_02/checkpoint/EP03/test_self/EP03_checkpoint_epoch_3_step_9999/id00154/geNO_h-jZOc/00145.mp4',   help='Input video file');
    parser.add_argument('--reference',      type=str, default='tests2',   help='Video name');
    parser.add_argument('--facedet_scale',  type=float, default=0.25, help='Scale factor for face detection');
    parser.add_argument('--crop_scale',     type=float, default=0.40, help='Scale bounding box');
    parser.add_argument('--min_track',      type=int, default=100,  help='Minimum facetrack duration');
    parser.add_argument('--frame_rate',     type=int, default=25,   help='Frame rate');
    parser.add_argument('--num_failed_det', type=int, default=25,   help='Number of missed detections allowed before tracking is stopped');
    parser.add_argument('--min_face_size',  type=int, default=100,  help='Minimum face size in pixels');
    parser.add_argument('--initial_model', type=str, default="data/syncnet_v2.model", help='');
    parser.add_argument('--batch_size', type=int, default='20', help='');
    parser.add_argument('--vshift', type=int, default='15', help='');
    opt = parser.parse_args();
    print('--------------------Start Sync--------------------')

    if opt.model_path is not None:
        model_name = os.path.splitext(os.path.basename(opt.model_path))[0]
        opt.save_dir = os.path.join(opt.save_dir, model_name)
        opt.data_dir = os.path.join(opt.data_dir, model_name)
    video_paths = find_data_files(opt.save_dir, '*.mp4')[:]
    counter = [0., 0., 0]
    for video_path in tqdm(video_paths):
        print(f'{video_path} is running')
        offset, conf, dist = run_video(copy.deepcopy(opt), video_path)
        if offset is None:
            continue
        counter[0] += conf 
        counter[1] += dist 
        counter[2] += 1
    
    sync_c = counter[0] / counter[2]
    sync_d = counter[1] / counter[2]
    print('Eval Results:', opt.save_dir)
    print('Sync_C: %.3f' % sync_c)
    print('Sync_D: %.3f' % sync_d)
    f_w = open(os.path.join(opt.save_dir, 'eval.txt'), 'a')
    f_w.write('Sync_C: %.3f\n' % sync_c)
    f_w.write('Sync_D: %.3f\n' % sync_d)
    f_w.flush()
    f_w.close()
    print('--------------------End Sync--------------------')
    print()

    
    # eva.inference_video_for_fid(videofile1, videofile2)
