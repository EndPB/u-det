#!/usr/bin/env bash
# 等 CodeT5 窗口基线第 1 轮（2 epoch）真正结束，再追加第 2 轮 2 epoch（累计 4 epoch），
# 最后在 best.pt 上做一次完整评估。
#
# 防重复启动的两个条件（缺一不可）：
#   1) 等待条件 = 指定 PID 退出（PID 是唯一不会「早已为真」的信号）
#   2) 放行条件 = metrics.csv 里的 epoch 数恰好等于期望值
# 历史上曾用「metrics.jsonl 行数 >= N」做等待条件，结果条件在启动瞬间就为真，
# 于是并发起了第二个 --resume，差点把产物目录写坏。这里用 PID + 计数双保险。
set -u
cd /root/autodl-tmp/u-det

PID="${1:?用法: queue_baseline.sh <pid> [期望epoch数]}"
WANT="${2:-2}"
TAG=base_codet5_tok
PY=/root/miniconda3/envs/udet/bin/python
LOG="/tmp/${TAG}_c2.log"

echo "[queue] $(date +%H:%M:%S) 等待 PID $PID 结束（第 1 轮 2 epoch）…"
while kill -0 "$PID" 2>/dev/null; do sleep 60; done

GOT=$(awk -F, 'NR>1{print $1}' "runs/$TAG/metrics.csv" 2>/dev/null | sort -un | wc -l)
echo "[queue] $(date +%H:%M:%S) PID 已退出；metrics.csv 中 epoch 数=$GOT（期望 $WANT）"
if [ "$GOT" -ne "$WANT" ]; then
  echo "[queue] 轮次不符，放弃第 2 轮（多半是中途报错）。请人工检查 runs/$TAG 与 /tmp/${TAG}.log"
  exit 1
fi

echo "[queue] $(date +%H:%M:%S) 启动第 2 轮：--resume 追加 2 epoch（累计 4 epoch）"
OMP_NUM_THREADS=8 nohup "$PY" train.py --config configs/baseline_codet5_tok.yaml \
  --tag "$TAG" --resume "runs/$TAG/last.pt" --epochs 2 > "$LOG" 2>&1 &
wait $!

echo "[queue] $(date +%H:%M:%S) 第 2 轮结束，评估 best.pt"
OMP_NUM_THREADS=8 "$PY" train.py --config configs/baseline_codet5_tok.yaml \
  --tag "$TAG" --eval --ckpt "runs/$TAG/best.pt" 2>&1 | tee "/tmp/${TAG}_eval.log"
echo "[queue] $(date +%H:%M:%S) 全部完成"
