# 在 CIFAR‑100 / ImageNet‑100 / ImageNet‑1k 上训练 ViT‑Small/Base 指南

本指南基于本仓库的 `train.py`/`validate.py` 脚本，给出数据准备、推荐超参与完整命令示例，覆盖 ViT‑Small 与 ViT‑Base 在 CIFAR‑100、ImageNet‑100、ImageNet‑1k 上的训练与验证。

## 环境与基础命令
- 安装依赖（建议新环境）：
  ```bash
  python -m pip install -r requirements.txt
  python -m pip install -r requirements-dev.txt
  python -m pip install -e .
  ```
- 运行测试（可选）：
  ```bash
  pytest -k "vit" -n 4 tests/
  ```

## 模型与常用配置
- 模型名称：
  - ViT‑Small: `vit_small_patch16_224`
  - ViT‑Base: `vit_base_patch16_224`
- 常用训练配置（ViT 通用强增广设置）：
  - 优化器与调度：`--opt adamw --lr 5e-4 --weight-decay 0.05 --sched cosine --warmup-epochs 20 --min-lr 1e-5`
  - 增广与正则：`--aa rand-m9-mstd0.5-inc1 --mixup 0.8 --cutmix 1.0 --smoothing 0.1 --reprob 0.25 --remode pixel`
  - DropPath：ViT‑S 用 `--drop-path 0.1`，ViT‑B 用 `0.2~0.3`
  - 其它：`--epochs 300 --amp`（混精度）；显存紧张时加 `--grad-accum-steps`
  - 输入尺寸：224（CIFAR‑100 也建议升采样至 224，可显式 `--img-size 224`）
- 预训练：对于 ImageNet‑100/1k，建议加 `--pretrained` 加速收敛；微调时可把 `--epochs` 减到 ~100、`--lr` 低 10x。

## 数据准备
- CIFAR‑100：使用内置数据集无需手动准备，`--dataset torch/cifar100 --dataset-download` 会自动下载到默认缓存目录。
- ImageNet‑100：将 100 个类别整理为 ImageFolder 结构：
  ```
  imagenet-100/
    train/<class_name>/*.JPEG
    val/<class_name>/*.JPEG
  ```
  注意：脚本默认验证集 split 名称为 `validation`，但会自动搜索同义目录（含 `val`），因此 `val/` 亦可。
- ImageNet‑1k：标准 ImageFolder 结构：
  ```
  imagenet/
    train/<1000 classes>/*
    val/<1000 classes>/*
  ```

## 训练命令示例

### CIFAR‑100（ViT‑Small/Base）
- ViT‑Small（单卡示例）
  ```bash
  python train.py \
    --dataset torch/cifar100 --dataset-download \
    --model vit_small_patch16_224 \
    --num-classes 100 --img-size 224 \
    --batch-size 128 \
    --opt adamw --lr 5e-4 --weight-decay 0.05 \
    --sched cosine --epochs 300 --warmup-epochs 20 \
    --aa rand-m9-mstd0.5-inc1 --mixup 0.8 --cutmix 1.0 --smoothing 0.1 \
    --reprob 0.25 --remode pixel --drop-path 0.1 \
    --min-lr 1e-5 --amp
  ```
- ViT‑Base（显存更大，或下调 batch）
  ```bash
  python train.py \
    --dataset torch/cifar100 --dataset-download \
    --model vit_base_patch16_224 \
    --num-classes 100 --img-size 224 \
    --batch-size 64 \
    --opt adamw --lr 3e-4 --weight-decay 0.05 \
    --sched cosine --epochs 300 --warmup-epochs 20 \
    --aa rand-m9-mstd0.5-inc1 --mixup 0.8 --cutmix 1.0 --smoothing 0.1 \
    --reprob 0.25 --remode pixel --drop-path 0.2 \
    --min-lr 1e-5 --amp
  ```
- 多卡分布式（以 4 卡为例）
  ```bash
  torchrun --nproc_per_node=4 train.py [同上参数]
  ```

### ImageNet‑100（ViT‑Small/Base）
- ViT‑Small（推荐启用预训练）
  ```bash
  python train.py \
    --data-dir /path/imagenet-100 \
    --model vit_small_patch16_224 \
    --num-classes 100 \
    --batch-size 128 \
    --opt adamw --lr 5e-4 --weight-decay 0.05 \
    --sched cosine --epochs 300 --warmup-epochs 20 \
    --aa rand-m9-mstd0.5-inc1 --mixup 0.8 --cutmix 1.0 --smoothing 0.1 \
    --reprob 0.25 --remode pixel --drop-path 0.1 \
    --min-lr 1e-5 --amp --pretrained
  ```
- ViT‑Base
  ```bash
  python train.py \
    --data-dir /path/imagenet-100 \
    --model vit_base_patch16_224 \
    --num-classes 100 \
    --batch-size 64 \
    --opt adamw --lr 3e-4 --weight-decay 0.05 \
    --sched cosine --epochs 300 --warmup-epochs 20 \
    --aa rand-m9-mstd0.5-inc1 --mixup 0.8 --cutmix 1.0 --smoothing 0.1 \
    --reprob 0.25 --remode pixel --drop-path 0.2 \
    --min-lr 1e-5 --amp --pretrained
  ```

### ImageNet‑1k（ViT‑Small/Base）
- ViT‑Small（从头训练）
  ```bash
  python train.py \
    --data-dir /path/imagenet \
    --model vit_small_patch16_224 \
    --batch-size 128 \
    --opt adamw --lr 5e-4 --weight-decay 0.05 \
    --sched cosine --epochs 300 --warmup-epochs 20 \
    --aa rand-m9-mstd0.5-inc1 --mixup 0.8 --cutmix 1.0 --smoothing 0.1 \
    --reprob 0.25 --remode pixel --drop-path 0.1 \
    --min-lr 1e-5 --amp
  ```
- ViT‑Base（可加层衰减）
  ```bash
  torchrun --nproc_per_node=8 train.py \
  --data-dir /srv/home/jlin398/data/imagenet/imagenet-100 \
  --model vit_base_patch16_224 \
  --num-classes 100 --img-size 224 \
  --batch-size 64 \
  --opt adamw --lr 3e-4 --weight-decay 0.05 \
  --sched cosine --epochs 120 --warmup-epochs 20 \
  --aa rand-m9-mstd0.5-inc1 --mixup 0.8 --cutmix 1.0 --smoothing 0.1 \
  --reprob 0.25 --remode pixel --drop-path 0.1 \
  --min-lr 1e-5 --amp \
  --log-wandb --wandb-project lowrank-vit \
  --spec-monitor --spec-every 10 --spec-topk 8 --spec-interval 2 \
  --spec-targets "attn.qkv,attn.proj,mlp.fc1,mlp.fc2"
  ```
- 微调已预训练权重（更快收敛）
  ```bash
  python train.py \
    --data-dir /path/imagenet \
    --model vit_base_patch16_224 \
    --pretrained --epochs 100 --lr 5e-5 \
    --batch-size 64 \
    --opt adamw --weight-decay 0.05 \
    --sched cosine --warmup-epochs 20 \
    --aa rand-m9-mstd0.5-inc1 --mixup 0.8 --cutmix 1.0 --smoothing 0.1 \
    --reprob 0.25 --remode pixel --drop-path 0.2 \
    --min-lr 1e-5 --amp
  ```

- 多卡分布式（8 卡示例，ViT‑Base + W&B + 谱监控）
  ```bash
  torchrun --nproc_per_node=8 train.py \
    --data-dir /srv/home/jlin398/data/imagenet \
    --model vit_base_patch16_224 \
    --batch-size 64 \
    --opt adamw --lr 3e-4 --weight-decay 0.05 \
    --sched cosine --epochs 300 --warmup-epochs 20 \
    --aa rand-m9-mstd0.5-inc1 --mixup 0.8 --cutmix 1.0 --smoothing 0.1 \
    --reprob 0.25 --remode pixel --drop-path 0.3 \
    --layer-decay 0.65 \
    --min-lr 1e-5 --amp \
    --log-wandb --wandb-project lowrank-vit \
    --spec-monitor --spec-every 10 --spec-topk 8 --spec-interval 2 \
    --spec-targets "attn.qkv,attn.proj,mlp.fc1,mlp.fc2"
  ```

## 验证与推理
- 验证：
  ```bash
  python validate.py \
    --model vit_base_patch16_224 \
    --data-dir /path/imagenet \
    --checkpoint /path/to/best.pth.tar \
    --batch-size 256
  ```
- 单图推理：
  ```bash
  python inference.py \
    --model vit_base_patch16_224 \
    --checkpoint /path/to/best.pth.tar \
    --input /path/to/img.jpg
  ```

## 启用 W&B 与谱监控（可选）
- 本仓库提供“权重谱监控”并支持将指标写入 W&B。以下参数可叠加到任意训练命令：
  - `--log-wandb --wandb-project <项目名>`：开启 W&B 日志；
  - `--spec-monitor`：开启谱监控；
  - `--spec-every 100`：每 100 次参数更新计算一次 SVD/子空间主余弦；
  - `--spec-topk 8`：记录前 k 个与后 k 个（last-k）奇异值/向量；
  - `--spec-interval 2`：按间隔选取奇异索引（如取 2，则记录第 1、3、5... 以及倒数第 1、3、5...）。
  - `--spec-targets "attn.qkv,attn.proj,mlp.fc1,mlp.fc2"`：指定监控的线性层模块名匹配；
  - `--spec-on-cpu`：在 CPU 上做 SVD（减轻加速器负担）。
  - 子空间相似度：记录逐项主余弦（canonical correlations），键为 `cos_u1..cos_uk`、`cos_v1..cos_vk`（前 k），以及 `cos_u_last1..cos_u_lastk`、`cos_v_last1..cos_v_lastk`（后 k）；另附 `cos_u_max/mean`、`cos_v_max/mean` 汇总（针对前 k）。数值范围 0–1 越大越相似。
  - EMA 监控（默认开启）：默认启用模型 EMA，并同步记录 EMA 的谱指标，使用 `spec_ema/` 前缀；可用 `--model-ema-decay`、`--model-ema-warmup` 做 EMA 超参调整。

- 示例：CIFAR‑100（ViT‑Small）+ W&B + 谱监控
  ```bash
  python train.py \
    --dataset torch/cifar100 --dataset-download \
    --model vit_small_patch16_224 \
    --num-classes 100 --img-size 224 \
    --batch-size 128 \
    --opt adamw --lr 5e-4 --weight-decay 0.05 \
    --sched cosine --epochs 300 --warmup-epochs 20 \
    --aa rand-m9-mstd0.5-inc1 --mixup 0.8 --cutmix 1.0 --smoothing 0.1 \
    --reprob 0.25 --remode pixel --drop-path 0.1 \
    --min-lr 1e-5 --amp \
    --log-wandb --wandb-project lowrank-vit \
    --spec-monitor --spec-every 100 --spec-topk 8 --spec-interval 2 \
    --spec-targets "attn.qkv,attn.proj,mlp.fc1,mlp.fc2"
  ```

- 多卡分布式（4 卡示例）：
  ```bash
  torchrun --nproc_per_node=4 train.py \
    --dataset torch/cifar100 --dataset-download \
    --model vit_small_patch16_224 \
    --num-classes 100 --img-size 224 \
    --batch-size 128 \
    --opt adamw --lr 5e-4 --weight-decay 0.05 \
    --sched cosine --epochs 300 --warmup-epochs 20 \
    --aa rand-m9-mstd0.5-inc1 --mixup 0.8 --cutmix 1.0 --smoothing 0.1 \
    --reprob 0.25 --remode pixel --drop-path 0.1 \
    --min-lr 1e-5 --amp \
    --log-wandb --wandb-project lowrank-vit \
    --spec-monitor --spec-every 100 --spec-topk 8 --spec-interval 2 \
    --spec-targets "attn.qkv,attn.proj,mlp.fc1,mlp.fc2"
  ```

提示：在 W&B 中可直接绘制以下曲线进行稳定性/机理分析（以模块名 m 为例）：
- 基模型：`spec/m/sigma_max`、`spec/m/sv1..svK`、`spec/m/sv_last1..sv_lastK`、`spec/m/delta_sv_rel`、`spec/m/cos_u1..cos_uK`、`spec/m/cos_v1..cos_vK`、`spec/m/cos_u_last1..cos_u_lastK`、`spec/m/cos_v_last1..cos_v_lastK`、`spec/m/cos_u_max/mean`、`spec/m/cos_v_max/mean`；
- EMA 模型：对应的 `spec_ema/m/...` 指标（如 `spec_ema/m/sigma_max`、`spec_ema/m/cos_u_mean`）。

## 离线谱分析（analyze_spectrum.py）
- 预训练权重：
  ```bash
  python analyze_spectrum.py --model vit_base_patch16_224 --pretrained --topk 8
  ```
- 本地权重：
  ```bash
  python analyze_spectrum.py --model vit_base_patch16_224 --checkpoint your.ckpt.pth --svd-on-cpu --topk 8
  ```
- 切分 fused QKV 并保存向量：
  ```bash
  python analyze_spectrum.py --model vit_base_patch16_224 --pretrained --split-qkv --topk 8 --save-vectors --format npz --save-dir out
  ```

输出说明

- 终端打印每层摘要与按类型聚合统计。
- 若指定 `--save-dir out`：
  - `out/<model>_spectral.json`：meta、sv_topk、sv_lastk、sigma_max/min 等。
  - 选 `--save-vectors`：另存 U/V（npz 或 pt；JSON 不含大数组）。

常用选项

- `--targets "attn.qkv,attn.proj,mlp.fc1,mlp.fc2"`：筛选层。
- `--svd-on-cpu`：在 CPU 上做 SVD，减轻显存。
- `--dtype float64`：更高精度（更慢）。
- `--full`：保存全谱及（可选）向量。

## 训练小贴士
- 显存不足：减小 `--batch-size` 或使用 `--grad-accum-steps`；开启 `--amp` 可显著节省显存。
- 学习率与 batch 大小：脚本默认会按全局 batch 大小缩放学习率；若显式设置 `--lr`，则以显式值为准。
- 数据目录：
  - ImageFolder 模式下默认 `train/` 与 `validation/`，脚本会自动匹配 `val/` `valid/` 等同义目录。
  - 使用 `torch/<name>`（torchvision）数据集且未显式传 `--data-dir` 时，脚本将自动使用 `~/.cache/torch/datasets`（若可写）或回退到 `./data` 作为根目录，并在日志中提示。建议手动指定 `--data-dir` 到你期望的位置以便管理数据。
- 多卡训练：推荐 `torchrun --nproc_per_node=<gpus>` 启动，日志/保存路径请自行设置（如 `--output`、`--experiment`）。
- 预训练权重：使用 `--pretrained` 微调可更快收敛，适当降低 `--lr` 与 `--epochs`。
