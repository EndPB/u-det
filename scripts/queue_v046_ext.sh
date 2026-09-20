#!/usr/bin/env bash
# v0.4.6 追加 2 轮（累计 4 轮），与 v0.4.5 @4ep 对齐预算。
#
# 用法（**现在就启动**，它会等当前 2 轮跑完再动手）：
#   setsid nohup bash scripts/queue_v046_ext.sh > /tmp/queue_v046_ext.log 2>&1 &
#
# ★★ 快照必须早于任何写 eval.json 的动作：
#    2 轮跑完时产出的 eval.json / raw_*.pt / best.pt 会被 4 轮的结果**覆盖**，
#    而 2 轮的同预算对比数据只此一份（v0.4.4 / v0.4.5 都为此吃过亏）。
#    所以先另存 eval_2ep.json / raw_*_2ep.pt / best_2ep.pt。
#
# ★ 放行条件（lessons D8）：用 last.pt 的 epoch 字段判已完成轮数，
#   **不能数 metrics.csv 的 distinct epoch** —— csv 每个 log_every 写一行，
#   进行中的 epoch 也会被写进去，会把轮数**多算**。
#
# ★ 等待条件用「没有 queue_v046.sh 且没有 train.py」。注意本脚本叫 queue_v046_ext.sh，
#   正则 `bash scripts/queue_v046\.sh` 不会误匹配自己（"queue_v046" 后面是 "_ext.sh"）。
set -u
cd /root/autodl-tmp/u-det

PY=/root/miniconda3/envs/udet/bin/python
TAG=v0.4.6
CFG=configs/udet_v046.yaml
TARGET=4

say() { echo "[ext] $(date +%H:%M:%S) $*"; }

epochs_done() {   # 已完成轮数 = last.pt 的 epoch + 1
  local e
  e=$(OMP_NUM_THREADS=1 "$PY" -c "
import torch
c = torch.load('runs/$TAG/last.pt', map_location='cpu', weights_only=False)
print(int(c.get('epoch', -1)) + 1)
" 2>/dev/null | tail -1)
  case "$e" in ''|*[!0-9]*) e=-1 ;; esac
  echo "$e"
}

say "已登记：等当前 2 轮跑完（含它自己的评测）后，追加到 $TARGET 轮"
while pgrep -f 'bash scripts/queue_v046\.sh' >/dev/null 2>&1 \
   || pgrep -f 'python train\.py' >/dev/null 2>&1; do
  sleep 60
done
say "当前任务已结束，开始接力"

ep=$(epochs_done)
if [ "$ep" -lt 0 ]; then
  say "✗ 读不到 runs/$TAG/last.pt 的 epoch，无法判断进度 ⇒ 放弃（不盲跑）"
  exit 1
fi
say "last.pt 显示已完成 $ep 轮"

# ---------------- 快照（必须在任何覆盖之前）---------------- #
if [ -f "runs/$TAG/eval.json" ] && [ ! -f "runs/$TAG/eval_2ep.json" ]; then
  cp "runs/$TAG/eval.json" "runs/$TAG/eval_2ep.json"
  for split in val test; do
    for stream in m4 hybrid; do
      src="runs/$TAG/raw_${stream}_${split}.pt"
      [ -f "$src" ] && cp "$src" "runs/$TAG/raw_${stream}_${split}_2ep.pt"
    done
  done
  [ -f "runs/$TAG/best.pt" ] && cp "runs/$TAG/best.pt" "runs/$TAG/best_2ep.pt"
  say "✓ 已存 2 轮快照：eval_2ep.json + raw_*_2ep.pt + best_2ep.pt"
else
  say "⚠ 未存快照（eval.json 不存在，或 eval_2ep.json 已存在）"
fi

if [ "$ep" -ge "$TARGET" ]; then
  say "已完成 $ep 轮 ≥ 目标 $TARGET ⇒ 无需续跑"
  exit 0
fi

left=$((TARGET - ep)); [ "$left" -lt 1 ] && left=1
say "续跑 $left 轮（目标累计 $TARGET 轮）：--resume last.pt --epochs $left"
OMP_NUM_THREADS=8 "$PY" train.py --config "$CFG" --tag "$TAG" \
  --resume "runs/$TAG/last.pt" --epochs "$left" > /tmp/v046_ext.log 2>&1
rc=$?
say "训练退出码 $rc"
if [ "$rc" -ne 0 ]; then
  say "✗ 续跑失败，跳过评测（详见 /tmp/v046_ext.log）"
  exit "$rc"
fi

say "重新评测 best.pt（--dump-raw，会覆盖 eval.json；2 轮快照已另存）"
OMP_NUM_THREADS=8 "$PY" train.py --config "$CFG" --tag "$TAG" \
  --eval --ckpt "runs/$TAG/best.pt" --dump-raw > /tmp/v046_ext_eval.log 2>&1
say "✓ 全部完成（累计 $TARGET 轮）"
