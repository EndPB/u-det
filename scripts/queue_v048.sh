#!/usr/bin/env bash
# v0.4.8：文档级头的池化 mean -> ABMIL。**2 epoch**，排在 v0.4.7 之后跑。
#
# 用法（现在就启动，它会等 v0.4.7 跑完再动手）：
#   setsid nohup bash scripts/queue_v048.sh > /tmp/queue_v048.log 2>&1 &
#
# 等待判据 = 「没有 queue_v047.sh 且没有 train.py」。
#   正则 `bash scripts/queue_v047\.sh` 不会误匹配本脚本（queue_v048.sh），
#   而且 `python train\.py` 那一项已经覆盖了"v0.4.7 正在训练"的情形。
#
# 动手前先给 v0.4.7 存一份 2 轮快照（它可能以后还要续到 4 轮）。
set -u
cd /root/autodl-tmp/u-det

PY=/root/miniconda3/envs/udet/bin/python
TAG=v0.4.8
CFG=configs/udet_v048.yaml
PREV=v0.4.7

say() { echo "[q48] $(date +%H:%M:%S) $*"; }

say "已登记：等 $PREV 跑完后再训练 $TAG 2 epoch（ABMIL 池化）"
while pgrep -f 'bash scripts/queue_v047\.sh' >/dev/null 2>&1 \
   || pgrep -f 'python train\.py' >/dev/null 2>&1; do
  sleep 60
done
say "$PREV 已结束"

# ---- 给 v0.4.7 存 2 轮快照（防它以后续跑到 4 轮时被覆盖）----
if [ -f "runs/$PREV/eval.json" ] && [ ! -f "runs/$PREV/eval_2ep.json" ]; then
  cp "runs/$PREV/eval.json" "runs/$PREV/eval_2ep.json"
  for split in val test; do
    for stream in m4 hybrid; do
      src="runs/$PREV/raw_${stream}_${split}.pt"
      [ -f "$src" ] && cp "$src" "runs/$PREV/raw_${stream}_${split}_2ep.pt"
    done
  done
  [ -f "runs/$PREV/best.pt" ] && cp "runs/$PREV/best.pt" "runs/$PREV/best_2ep.pt"
  say "✓ 已存 $PREV 的 2 轮快照"
else
  say "⚠ 未存 $PREV 快照（eval.json 不存在，或 eval_2ep.json 已存在）"
fi

if [ ! -f "runs/$PREV/eval.json" ]; then
  say "✗ $PREV 没有 eval.json（可能训练失败）⇒ 放弃，不盲跑"
  exit 1
fi

say "开始训练 $TAG（配置 $CFG）"
OMP_NUM_THREADS=8 "$PY" train.py --config "$CFG" --tag "$TAG" > /tmp/v048.log 2>&1
rc=$?
say "训练退出码 $rc"
if [ "$rc" -ne 0 ]; then
  say "✗ 训练失败，跳过评测（详见 /tmp/v048.log）"
  exit "$rc"
fi

say "评测 best.pt（--dump-raw）"
OMP_NUM_THREADS=8 "$PY" train.py --config "$CFG" --tag "$TAG" \
  --eval --ckpt "runs/$TAG/best.pt" --dump-raw > /tmp/v048_eval.log 2>&1
say "✓ 全部完成（$TAG 2 轮）"
