# Agent Notes

This document summarizes recent instrumentation changes relevant to training scripts and logging.

## Spectral Monitor

- Cosine-only metrics: Spectral subspace drift is logged via principal cosines only. Angle metrics have been removed.
  - Per-index cosines: `spec/<module>/cos_u1..cos_uK`, `spec/<module>/cos_v1..cos_vK` (U = input-side, V = output-side).
  - Summaries: `spec/<module>/cos_u_max`, `spec/<module>/cos_u_mean`, `spec/<module>/cos_v_max`, `spec/<module>/cos_v_mean`.
  - Also logged: `sigma_max`, `sv1..svK`, `delta_sv_rel`.
- Targets and cadence:
  - `--spec-monitor` enables logging; `--spec-every` controls update cadence; `--spec-topk` sets K; `--spec-targets` filters `nn.Linear` by name; `--spec-on-cpu` moves SVD to CPU.
- W&B integration: With `--log-wandb`, metrics are logged under `spec/...`. EMA metrics use the `spec_ema/...` prefix.

Implementation details:

- For each target linear weight `W`, compute `U,S,Vh = torch.linalg.svd(W, full_matrices=False)` and keep top-K bases.
- Principal cosines are singular values of `U_prev.T @ U_cur` (and similarly for V); values in [0, 1].

## CLI & Defaults

- Removed: `--spec-metric` (only cosine mode remains).
- EMA default: `--model-ema` is enabled by default. EMA spectral metrics are logged automatically if spectral monitor is enabled.

Migration notes:

- Prior angle-based fields (`angle_u_*`, `angle_v_*`) are no longer emitted.
- Existing dashboards should be updated to use `cos_*` fields.

Example:

```
torchrun --nproc_per_node=1 train.py \
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
  --spec-monitor --spec-every 100 --spec-topk 8 \
  --spec-targets "attn.qkv,attn.proj,mlp.fc1,mlp.fc2"
```

