#!/usr/bin/env bash
# v0.4.2-fast：分块编码器 + static 形状 + torch.compile。
#
# 替换掉 2026-09-20 首次跑的那版（compile=false）：实测 0.396 s/步，
# 比 v0.4.1 的 0.244 s/步慢 1.62 倍。根因与修法见 configs/udet_v042.yaml 头部注释。
#
# 流程与之前一致：训练 2 epoch -> 在 best.pt 上评估。
set -u
cd /root/autodl-tmp/u-det

PY=/root/miniconda3/envs/udet/bin/python
TAG=v0.4.2
LOG=/tmp/v042_fast.log

echo "[fast] $(date +%H:%M:%S) 启动训练（static+compile，2 epoch）"
OMP_NUM_THREADS=8 nohup "$PY" train.py --config configs/udet_v042.yaml \
  --tag "$TAG" --epochs 2 > "$LOG" 2>&1 &
wait $!

echo "[fast] $(date +%H:%M:%S) 训练结束，评估 best.pt"
OMP_NUM_THREADS=8 "$PY" train.py --config configs/udet_v042.yaml \
  --tag "$TAG" --eval --ckpt "runs/$TAG/best.pt" > /tmp/v042_fast_eval.log 2>&1
echo "[fast] $(date +%H:%M:%S) 全部完成"
