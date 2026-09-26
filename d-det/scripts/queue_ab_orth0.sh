#!/usr/bin/env bash
# d-det 消融 B：关掉正交正则（loss.orth=0），与 v0.1.0 双流正交版对照。
# 目的：检验设计文档 §4 的核心论断——「两维正交是结构性的（监督信号自然诱导），
#       不靠人工正则」。若 orth=0 时 |cos| 依然很小 ⇒ 论断成立；
#       若 |cos| 明显变大 ⇒ λc 是必要的保险。
#
# 用法（有卡）：
#   bash scripts/queue_ab_orth0.sh > /tmp/ddet_ab_orth0.log 2>&1 &
set -u
cd "$(dirname "$0")/.."
PY=/root/miniconda3/envs/udet/bin/python
export OMP_NUM_THREADS=8
TAG=v0.1.0_ab_orth0

echo "[abB] $(date '+%F %T') 开始训练（loss.orth=0）tag=$TAG"
"$PY" train.py --config configs/ddet_base.yaml --tag "$TAG" --set loss.orth=0.0 \
  || { echo "[abB] $(date '+%F %T') 训练失败，终止"; exit 1; }

echo "[abB] $(date '+%F %T') 训练完成，开始评测（--dump-scores）"
"$PY" train.py --config configs/ddet_base.yaml --tag "$TAG" --eval \
  --ckpt "runs/$TAG/best.pt" --dump-scores \
  || { echo "[abB] $(date '+%F %T') 评测失败"; exit 1; }

echo "[abB] $(date +'%F %T') 全部完成：runs/$TAG/（eval.json + scores_*.pt）"
