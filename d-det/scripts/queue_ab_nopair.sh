#!/usr/bin/env bash
# d-det 消融 A：纯 s1 单流（--no-pair），与 v0.1.0 双流对照。
# 目的：回答「pair 流是否干扰 s1 学习 / 双流是否有害」。
#
# 用法（有卡）：
#   bash scripts/queue_ab_nopair.sh > /tmp/ddet_ab_nopair.log 2>&1 &
set -u
cd "$(dirname "$0")/.."
PY=/root/miniconda3/envs/udet/bin/python
export OMP_NUM_THREADS=8
TAG=v0.1.0_ab_nopair

echo "[abA] $(date '+%F %T') 开始训练（--no-pair 纯 s1）tag=$TAG"
"$PY" train.py --config configs/ddet_base.yaml --tag "$TAG" --no-pair \
  || { echo "[abA] $(date '+%F %T') 训练失败，终止"; exit 1; }

echo "[abA] $(date '+%F %T') 训练完成，开始评测（--dump-scores）"
"$PY" train.py --config configs/ddet_base.yaml --tag "$TAG" --eval \
  --ckpt "runs/$TAG/best.pt" --dump-scores \
  || { echo "[abA] $(date '+%F %T') 评测失败"; exit 1; }

echo "[abA] $(date '+%F %T') 全部完成：runs/$TAG/（eval.json + scores_*.pt）"
