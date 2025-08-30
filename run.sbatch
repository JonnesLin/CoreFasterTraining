#!/bin/bash

#SBATCH --job-name=baseline
#SBATCH --ntasks-per-node=1
#SBATCH --time=165:00:00
#SBATCH -G 8
#SBATCH --cpus-per-task=128
#SBATCH --partition morgadolab
#SBATCH --output=basaeline.out
#SBATCH -N 1

module purge
conda init zsh
conda activate py10

torchrun --nproc_per_node=8 train.py \
    --data-dir /srv/home/jlin398/data/imagenet \
    --model vit_base_patch16_224 \
    --num-classes 1000 --img-size 224 \
    --batch-size 64 --epochs 300 --warmup-epochs 40 \
    --aa rand-m9-mstd0.5-inc1 --reprob 0.25 --remode pixel --drop-path 0.1 \
    --min-lr 1e-5 --amp \
    --log-wandb --wandb-project lowrank-vit \
    --spec-monitor --spec-every 25 --spec-topk 8 \
    --spec-targets "attn.qkv,attn.proj,mlp.fc1,mlp.fc2" \
    --spec-no-ema-log \
    --spec-interval 8
# --aa rand-m9-mstd0.5-inc1 --mixup 0.8 --cutmix 1.0 --smoothing 0.1 \