#!/usr/bin/env bash
# v0.4.5 追加 2 个 epoch（累计 4 轮），与 v0.4.4 @4ep 对齐预算。
#
# 为什么必须补齐：v0.4.4 的 raw_hybrid_test.pt 现在来自它的 **4 轮** best.pt，
# 而 v0.4.5 的来自 **2 轮** —— 直接比会变成 4ep vs 2ep，对 v0.4.5 不公平。
# 2ep 的同预算对比只能用整体指标（v0.4.4 的 2 轮 ckpt 已被 4 轮覆盖，拿不到 raw 了）。
#
# 先保存 2 轮评估快照，避免被最终评估覆盖。
set -u
cd /root/autodl-tmp/u-det

PY=/root/miniconda3/envs/udet/bin/python
TAG=v0.4.5
CFG=configs/udet_v045.yaml

if [ -f "runs/$TAG/eval.json" ] && [ ! -f "runs/$TAG/eval_2ep.json" ]; then
  cp "runs/$TAG/eval.json" "runs/$TAG/eval_2ep.json"
  cp "runs/$TAG/raw_hybrid_test.pt" "runs/$TAG/raw_hybrid_test_2ep.pt" 2>/dev/null || true
  echo "[ext] $(date +%H:%M:%S) 已保存 2 轮快照（eval + hybrid raw）"
fi

echo "[ext] $(date +%H:%M:%S) 追加 2 轮（累计 4 轮）：--resume last.pt"
OMP_NUM_THREADS=8 nohup "$PY" train.py --config "$CFG" --tag "$TAG" \
  --resume "runs/$TAG/last.pt" --epochs 2 > /tmp/v045_ext.log 2>&1 &
wait $!

echo "[ext] $(date +%H:%M:%S) 重新评估 best.pt（覆盖 eval.json）"
OMP_NUM_THREADS=8 "$PY" train.py --config "$CFG" --tag "$TAG" \
  --eval --ckpt "runs/$TAG/best.pt" --dump-raw > /tmp/v045_ext_eval.log 2>&1
echo "[ext] $(date +%H:%M:%S) 全部完成（累计 4 轮）"
