#!/usr/bin/env bash
# v0.4.4 追加 2 个 epoch（累计 4 轮），与 v0.4.2 @4ep / CodeT5 滑窗基线 @4ep 对齐预算。
#
# 已核实的起点状态（2026-09-20 17:37 完成第一轮 2 epoch）：
#   metrics.csv 含 epoch 0 与 1，每轮 81 条记录、最后 step 均为 16069
#   [val 0] m4 0.9610 / [val 1] m4 0.9686；eval.json(2ep) = 0.9745 / 0.7808 / 0.4885 / 0.8130
#
# ★ 先保存 2 轮的评估快照：后面的评估会**覆盖** eval.json，而 2 轮那份要用来和
#   v0.4.2 @2ep 做同预算对照，丢了就得去翻日志。
set -u
cd /root/autodl-tmp/u-det

PY=/root/miniconda3/envs/udet/bin/python
TAG=v0.4.4
CFG=configs/udet_v044.yaml

if [ -f "runs/$TAG/eval.json" ] && [ ! -f "runs/$TAG/eval_2ep.json" ]; then
  cp "runs/$TAG/eval.json" "runs/$TAG/eval_2ep.json"
  echo "[ext] $(date +%H:%M:%S) 已保存 2 轮评估快照 -> runs/$TAG/eval_2ep.json"
fi

echo "[ext] $(date +%H:%M:%S) 追加 2 轮（累计 4 轮）：--resume last.pt"
OMP_NUM_THREADS=8 nohup "$PY" train.py --config "$CFG" --tag "$TAG" \
  --resume "runs/$TAG/last.pt" --epochs 2 > /tmp/v044_ext.log 2>&1 &
wait $!

echo "[ext] $(date +%H:%M:%S) 重新评估 best.pt（覆盖 eval.json）"
OMP_NUM_THREADS=8 "$PY" train.py --config "$CFG" --tag "$TAG" \
  --eval --ckpt "runs/$TAG/best.pt" > /tmp/v044_ext_eval.log 2>&1
echo "[ext] $(date +%H:%M:%S) 全部完成（累计 4 轮）"
