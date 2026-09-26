#!/usr/bin/env bash
# d-det v0.1.0 接力：训练 → 自动评测（- -dump-scores 落盘逐样本分数）。
#
# 用法（有卡）：
#   bash scripts/queue_v010.sh [TAG]        # 默认 TAG=v0.1.0
#
# 串行原因：训练与评测都吃满 GPU，必须一个跑完再下一个；
# 日志建议：bash scripts/queue_v010.sh > /tmp/ddet_queue_v010.log 2>&1 &
set -u
cd "$(dirname "$0")/.."
PY=/root/miniconda3/envs/udet/bin/python
export OMP_NUM_THREADS=8
TAG=${1:-v0.1.0}

echo "[queue] $(date '+%F %T') 开始训练 tag=$TAG"
"$PY" train.py --config configs/ddet_base.yaml --tag "$TAG" \
  || { echo "[queue] $(date '+%F %T') 训练失败，终止"; exit 1; }

echo "[queue] $(date '+%F %T') 训练完成，开始评测（--dump-scores）"
"$PY" train.py --config configs/ddet_base.yaml --eval --ckpt "runs/$TAG/best.pt" --dump-scores \
  || { echo "[queue] $(date '+%F %T') 评测失败"; exit 1; }

echo "[queue] $(date '+%F %T') 全部完成：runs/$TAG/（eval.json + scores_*.pt）"
