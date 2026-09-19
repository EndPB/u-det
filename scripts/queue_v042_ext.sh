#!/usr/bin/env bash
# v0.4.2 追加 2 个 epoch（累计 4 轮），与 v0.4.1 @4ep / CodeT5 基线 @4ep 对齐预算。
#
# 用户指令：「跑完后再跑2epoch」。
#
# 等待条件用 **PID 存活**（唯一不会"早已为真"的信号），并校验 metrics.csv 的轮数 ——
# 如果在训练没完成时就去 resume，epoch 计数会算错、且会在错误的权重上继续训。
set -u
cd /root/autodl-tmp/u-det

PID="${1:?用法: queue_v042_ext.sh <要等待的PID>}"
PY=/root/miniconda3/envs/udet/bin/python
TAG=v0.4.2
CFG=configs/udet_v042.yaml

echo "[ext] $(date +%H:%M:%S) 等待 PID $PID（v0.4.2 前 2 轮 + 评估）结束…"
while kill -0 "$PID" 2>/dev/null; do sleep 30; done

ep=$(awk -F, 'NR>1{print $1}' "runs/$TAG/metrics.csv" 2>/dev/null | sort -un | wc -l)
echo "[ext] $(date +%H:%M:%S) PID 已退出；metrics.csv 里已完成 $ep 轮"
if [ "$ep" -ne 2 ]; then
  echo "[ext] 轮数不是 2（实际 $ep），放弃追加 —— 请人工检查 runs/$TAG 与 /tmp/v042_fast.log"
  exit 1
fi

echo "[ext] $(date +%H:%M:%S) 追加 2 轮（累计 4 轮）：--resume last.pt"
OMP_NUM_THREADS=8 nohup "$PY" train.py --config "$CFG" --tag "$TAG" \
  --resume "runs/$TAG/last.pt" --epochs 2 > /tmp/v042_ext.log 2>&1 &
wait $!

echo "[ext] $(date +%H:%M:%S) 重新评估 best.pt（覆盖 eval.json）"
OMP_NUM_THREADS=8 "$PY" train.py --config "$CFG" --tag "$TAG" \
  --eval --ckpt "runs/$TAG/best.pt" > /tmp/v042_ext_eval.log 2>&1
echo "[ext] $(date +%H:%M:%S) 全部完成（累计 4 轮）"
