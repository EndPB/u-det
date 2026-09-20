#!/usr/bin/env bash
# v0.4.3：单任务消融（只保留样本级分类），2 epoch。
#
# 目的：确认「后面的跳连（上采样路径）+ token 级分类」是不是在拖累样本级表现。
# 与 v0.4.2 的差异与可比性说明见 configs/udet_v043.yaml 头部注释。
set -u
cd /root/autodl-tmp/u-det

PY=/root/miniconda3/envs/udet/bin/python
TAG=v0.4.3
CFG=configs/udet_v043.yaml

echo "[v043] $(date +%H:%M:%S) 启动训练（单任务，2 epoch）"
OMP_NUM_THREADS=8 nohup "$PY" train.py --config "$CFG" --tag "$TAG" --epochs 2 > /tmp/v043.log 2>&1 &
wait $!

echo "[v043] $(date +%H:%M:%S) 训练结束，评估 best.pt"
OMP_NUM_THREADS=8 "$PY" train.py --config "$CFG" --tag "$TAG" \
  --eval --ckpt "runs/$TAG/best.pt" > /tmp/v043_eval.log 2>&1
echo "[v043] $(date +%H:%M:%S) 全部完成"
