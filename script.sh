training:
CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7 accelerate launch --main_process_port 12359 train.py  --config config/train.yaml --exp_name ACTalker

Testing:
CUDA_VISIBLE_DEVICES=7 python Inference.py --config config/inference.yaml --ref assets/ref.jpg --audio assets/audio.mp3 --video assets/video.mp4 --mode 2