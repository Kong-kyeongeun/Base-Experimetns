# !/bin/bashrc


# CUDA_VISIBLE_DEVICES=4 python3 experiment2.py --dataset=cifar10 --arch=vgg --cuda=True --seed=3 --prune_ratio=0.1 --save=./logs/vgg/vgg16_new_reprune_cifar10/new_thr_3
# CUDA_VISIBLE_DEVICES=4 python3 experiment2.py --dataset=cifar10 --arch=vgg --cuda=True --seed=3 --prune_ratio=0.3 --save=./logs/vgg/vgg16_new_reprune_cifar10/new_thr_3
# CUDA_VISIBLE_DEVICES=4 python3 experiment2.py --dataset=cifar10 --arch=vgg --cuda=True --seed=3 --prune_ratio=0.5 --save=./logs/vgg/vgg16_new_reprune_cifar10/new_thr_3
# CUDA_VISIBLE_DEVICES=4 python3 experiment2.py --dataset=cifar10 --arch=vgg --cuda=True --seed=3 --prune_ratio=0.7 --save=./logs/vgg/vgg16_new_reprune_cifar10/new_thr_3
# CUDA_VISIBLE_DEVICES=4 python3 experiment2.py --dataset=cifar10 --arch=vgg --cuda=True --seed=3 --prune_ratio=0.9 --save=./logs/vgg/vgg16_new_reprune_cifar10/new_thr_3

# CUDA_VISIBLE_DEVICES=5 python3 experiment2.py --dataset=cifar10 --arch=resnet56 --prune_ratio=0.1 --save=./logs/resnet/resnet56_new_reprune_cifar10/new_thr
# CUDA_VISIBLE_DEVICES=5 python3 experiment2.py --dataset=cifar10 --arch=resnet56 --prune_ratio=0.3 --save=./logs/resnet/resnet56_new_reprune_cifar10/new_thr
# CUDA_VISIBLE_DEVICES=5 python3 experiment2.py --dataset=cifar10 --arch=resnet56 --prune_ratio=0.5 --save=./logs/resnet/resnet56_new_reprune_cifar10/new_thr
# CUDA_VISIBLE_DEVICES=5 python3 experiment2.py --dataset=cifar10 --arch=resnet56 --prune_ratio=0.7 --save=./logs/resnet/resnet56_new_reprune_cifar10/new_thr
# CUDA_VISIBLE_DEVICES=5 python3 experiment2.py --dataset=cifar10 --arch=resnet56 --prune_ratio=0.9 --save=./logs/resnet/resnet56_new_reprune_cifar10/new_thr

# CUDA_VISIBLE_DEVICES=5 python3 experiment2.py --dataset=cifar10 --arch=resnet110 --prune_ratio=0.1 --save=./logs/resnet/resnet110_new_reprune_cifar10/new_thr
# CUDA_VISIBLE_DEVICES=5 python3 experiment2.py --dataset=cifar10 --arch=resnet110 --prune_ratio=0.3 --save=./logs/resnet/resnet110_new_reprune_cifar10/new_thr
# CUDA_VISIBLE_DEVICES=5 python3 experiment2.py --dataset=cifar10 --arch=resnet110 --prune_ratio=0.5 --save=./logs/resnet/resnet110_new_reprune_cifar10/new_thr
# CUDA_VISIBLE_DEVICES=5 python3 experiment2.py --dataset=cifar10 --arch=resnet110 --prune_ratio=0.7 --save=./logs/resnet/resnet110_new_reprune_cifar10/new_thr
# CUDA_VISIBLE_DEVICES=5 python3 experiment2.py --dataset=cifar10 --arch=resnet110 --prune_ratio=0.9 --save=./logs/resnet/resnet110_new_reprune_cifar10/new_thr

# CUDA_VISIBLE_DEVICES=5 python3 experiment2.py --dataset=imagenet --arch=resnet50 --cuda=True --prune_ratio=0.7 --save=./logs/resnet/resnet50_new_reprune_imagenet
# CUDA_VISIBLE_DEVICES=5 python3 experiment2.py --dataset=imagenet --arch=resnet50 --cuda=True --prune_ratio=0.1 --save=./logs/resnet/resnet50_new_reprune_imagenet
# CUDA_VISIBLE_DEVICES=5 python3 experiment2.py --dataset=imagenet --arch=resnet50 --cuda=True --prune_ratio=0.3 --save=./logs/resnet/resnet50_new_reprune_imagenet
# CUDA_VISIBLE_DEVICES=5 python3 experiment2.py --dataset=imagenet --arch=resnet50 --cuda=True --prune_ratio=0.5 --save=./logs/resnet/resnet50_new_reprune_imagenet
# CUDA_VISIBLE_DEVICES=5 python3 experiment2.py --dataset=imagenet --arch=resnet50 --cuda=True --prune_ratio=0.9 --save=./logs/resnet/resnet50_new_reprune_imagenet
# !/bin/bashrc

# Mincheol Park CVPR Frameworks
# Main training shell scripts

# Datasets:   cifar10, cifar100, imagenet
# Benchmarks: vgg, resnet
# Epochs: 160
# optims: SGD
# Batch_size: 64 (cifar10), 256 (imagenet)
# Learning_rate: 0.1 (start) *0.1 (50% ~ 75% epochs) * 0.1 (75% ~ 100% epochs) (cifar10)
# weight_decay: 1e-4(L2)

LOGPATH='./logs/'
LENETPATH=${LOGPATH}'lenet/'
VGGPATH=${LOGPATH}'vgg/'
RESPATH=${LOGPATH}'resnet/'
WRNPATH=${LOGPATH}'wideresnet/'
MOBILEPATH=${LOGPATH}'mobilenet/'
IMAGENETPATH='/disk/imagenet'

PROGRAM_NAME=`/usr/bin/basename "$0"`
echo shell arg 0: $0
echo USING BASENAME: ${PROGRAM_NAME}
arg_data=default
arg_arch=default
arg_lambd=default

function print_usage(){
/bin/cat << EOF
Usage:
    ${PROGRAM_NAME} [-d arg_data] [-a arg_arch] [-s seed] [-c arg_cuda]
Option:
    -d, dataset
    -a, model
    -s, seed
    -c, cuda
EOF
}
if [ $# -eq 0 ];
then
    print_usage
    exit 1
fi

while getopts "d:a:s:c:h" opt
do
    case $opt in
        d) arg_data=$OPTARG; echo "ARG DATA: $arg_data";;
        a) arg_arch=$OPTARG; echo "ARG ARCH: $arg_arch";;
        s) arg_seed=$OPTARG; echo "ARG SEED: $arg_seed";;
        c) arg_cuda=$OPTARG; echo "ARG CUDA: $arg_cuda";;
        h) print_usage;;
    esac
done

#                --weight_path="./logs/vgg/vgg16_pruned_cifar10/bench/l1/finetuning1/0.7/model_best.pth.tar" \
# Cifar10
if [ "$arg_data" = "cifar10" ];
then
    if [ "$arg_arch" = "vgg" ]
    then
        python3 train.py \
                --dataset=$arg_data  \
                --arch=$arg_arch  \
                --batch_size=128 \
                --test_batch_size=256 \
                --epochs=200 \
                --lr=0.1 \
                --wd=5e-4 \
                --milestones 100 150 \
                --save=${VGGPATH}'vgg16_baseline_cifar10_bnx' \
                --ngpu=$arg_cuda \
                --seed=$arg_seed 
    fi 
    if [ "$arg_arch" = "resnet56" ]
    then
        python3 train.py \
                --dataset=$arg_data  \
                --arch=$arg_arch  \
                --batch_size=128 \
                --test_batch_size=256 \
                --epochs=200 \
                --lr=0.1 \
                --wd=5e-4 \
                --milestones 100 150 \
                --save=${RESPATH}'resnet56_baseline_cifar10' \
                --ngpu=$arg_cuda \
                --weight_path="./logs/resnet/resnet56_baseline_cifar10/model_best.pth.tar" \
                --extract \
                --seed=$arg_seed 
    fi
    if [ "$arg_arch" = "resnet110" ]
    then
        python3 train.py \
                --dataset=cifar10 \
                --arch=$arg_arch \
                --depth=110 \
                --batch-size=128 \
                --test-batch-size=256 \
                --lr=0.01 \
                --wd=0.001 \
                --dr=0.1 \
                --save=${RESPATH}'resnet110_forcereg_cifar10' \
                --resume=${RESPATH}'resnet110_forcereg_cifar10/model_best.pth.tar' \
                --evaluate \
                --flops \
                --ngpu='cuda:0'
    fi
fi

# Cifar100
if [ "$arg_data" = "cifar100" ];
then
    if [ "$arg_arch" = "vgg" ]
    then
        python3 train.py \
                --dataset=$arg_data  \
                --arch=$arg_arch  \
                --batch_size=128 \
                --test_batch_size=256 \
                --epochs=200 \
                --lr=0.1 \
                --wd=5e-4 \
                --milestones 100 150 \
                --save=${VGGPATH}'vgg16_baseline_cifar100' \
                --ngpu=$arg_cuda \
                --weight_path="./logs/vgg/vgg16_baseline_cifar100/model_best.pth.tar" \
                --extract \
                --seed=$arg_seed 
    fi
    if [ "$arg_arch" = "resnet56" ]
    then
        python3 train.py \
                --dataset=$arg_data  \
                --arch=$arg_arch  \
                --batch_size=128 \
                --test_batch_size=256 \
                --epochs=200 \
                --lr=0.1 \
                --wd=5e-4 \
                --milestones 100 150 \
                --save=${RESPATH}'resnet56_baseline_cifar100' \
                --ngpu=$arg_cuda \
                --weight_path="./logs/resnet/resnet56_baseline_cifar100/model_best.pth.tar" \
                --extract \
                --seed=$arg_seed 
    fi
    if [ "$arg_arch" = "resnet110" ]
    then
        python3 train.py \
                --dataset=cifar100 \
                --arch=$arg_arch \
                --batch-size=128 \
                --test-batch-size=256 \
                --save=${RESPATH}'resnet110_original_cifar100' \
                --resume=${RESPATH}'resnet110_original_cifar100/model_best.pth.tar' \
                --evaluate \
                --flops \
                --ngpu='cuda:0'
    fi
fi
# ImageNet
if [ "$arg_data" = "imagenet" ];
then
    if [ "$arg_arch" = "vgg16" ] || [ "$arg_arch" = "vgg16_bn" ]
    then
        python3 train.py \
                --dataset=imagenet \
                --arch=vgg16_bn \
                --batch_size=256 \
                --test_batch_size=256 \
                --epochs=90 \
                --lr=0.1 \
                --wd=1e-4 \
                --milestones 30 60 \
                --save=${VGGPATH}'vgg16_baseline_imagenet' \
                --ngpu=$arg_cuda \
                --seed=$arg_seed 
    fi

    if [ "$arg_arch" = "resnet18" ]
    then
        python3 train.py \
                --dataset=$arg_data \
                --arch=$arg_arch \
                --lr=0.1 \
                --wd=1e-4 \
                --extract \
                --epochs=100 \
                --milestones 30 60 90 \
                --seed=$arg_seed \
                --batch_size=256 \
                --test_batch_size=256 \
                --ngpu=$arg_cuda \
                --weight_path='./logs/resnet/resnet18_baseline_imagenet/model_best.pth.tar' \
                --ngpu=$arg_cuda
    fi

    if [ "$arg_arch" = "resnet50" ]
    then
        python3 train.py \
                --dataset=$arg_data \
                --arch=$arg_arch \
                --extract \
                --lr=0.1 \
                --wd=1e-4 \
                --epochs=100 \
                --milestones 30 60 90 \
                --seed=$arg_seed \
                --batch_size=256 \
                --test_batch_size=256 \
                --ngpu=$arg_cuda \
                --weight_path='./logs/resnet/resnet50_baseline_imagenet/model_best.pth.tar' \
                --ngpu=$arg_cuda
    fi
fi
