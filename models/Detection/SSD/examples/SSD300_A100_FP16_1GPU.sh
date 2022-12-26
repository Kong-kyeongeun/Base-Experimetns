# This script launches SSD300 training in FP16 on 1 GPUs using 256 batch size
# Usage bash SSD300_FP16_1GPU.sh <path to this repository> <path to dataset> <additional flags>
# !/bin/bashrc
CUDA_VISIBLE_DEVICES=5 python3 /home/ex9845/ws/SSD/main.py --backbone resnet50 --backbone-path /home/ex9845/ws/NewReprune/logs/resnet/resnet50_reprune_imagenet/init/0.5/model_best.pth.tar --warmup 300 --bs 256 --amp --data /home/ex9845/ws/coco --save /home/ex9845/ws/SSD/logs/0.5