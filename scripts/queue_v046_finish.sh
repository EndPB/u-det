#!/usr/bin/env bash
# v0.4.6 收尾：从 last.pt 续跑到 4 轮 + 评测落盘。
#
# 背景（2026-09-21 的事故）：08:30 实例被关机、09:03:36 容器重启（PID 1 = /init/boot/boot.sh）。
# 队列 / 接力脚本 / 看门狗**三个兄弟进程一起死了**，训练停在 epoch 3 的 45%。
# ⇒ 容器内的看门狗**防不住实例级关机**；真正的连续性来自「定期 ckpt + resume」，
#    以及接力脚本里那份**提前存的 2 轮快照**（eval_2ep.json / raw_*_2ep.pt / best_2ep.pt）。
#
# 现状：last.pt 的 epoch=2 ⇒ 已完成 3 轮；目标 4 轮 ⇒ 只需再跑 1 轮。
#
# 用法：setsid nohup bash scripts/queue_v046_finish.sh > /tmp/queue_v046_finish.log 2>&1 &
set -u
cd /root/autodl-tmp/u-det

PY=/root/miniconda3/envs/udet/bin/python
TAG=v0.4.6
CFG=configs/udet_v046.yaml
TARGET=4

say() { echo "[fin] $(date +%H:%M:%S) $*"; }

epochs_done() {   # 已完成轮数 = last.pt 的 epoch + 1（lessons D8：不要数 metrics.csv）
  local e
  e=$(OMP_NUM_THREADS=1 "$PY" -c "
import torch
c = torch.load('runs/$TAG/last.pt', map_location='cpu', weights_only=False)
print(int(c.get('epoch', -1)) + 1)
" 2>/dev/null | tail -1)
  case "$e" in ''|*[!0-9]*) e=-1 ;; esac
  echo "$e"
}

# 已有训练在跑就别抢 GPU
if pgrep -f 'python train\.py' >/dev/null 2>&1; then
  say "已有 train.py 在跑，本次退出（避免抢显卡）"
  exit 0
fi

ep=$(epochs_done)
if [ "$ep" -lt 0 ]; then
  say "✗ 读不到 runs/$TAG/last.pt 的 epoch ⇒ 放弃"
  exit 1
fi
say "已完成 $ep 轮（目标 $TARGET）"

if [ "$ep" -ge "$TARGET" ]; then
  say "轮数已达标，只补评测"
else
  left=$((TARGET - ep))
  say "续跑 $left 轮：--resume last.pt --epochs $left"
  OMP_NUM_THREADS=8 "$PY" train.py --config "$CFG" --tag "$TAG" \
    --resume "runs/$TAG/last.pt" --epochs "$left" > /tmp/v046_finish.log 2>&1
  rc=$?
  say "训练退出码 $rc"
  if [ "$rc" -ne 0 ]; then
    say "✗ 训练失败，跳过评测（详见 /tmp/v046_finish.log）"
    exit "$rc"
  fi
fi

say "评测 best.pt（--dump-raw；会覆盖 eval.json，2 轮快照 eval_2ep.json 已另存）"
OMP_NUM_THREADS=8 "$PY" train.py --config "$CFG" --tag "$TAG" \
  --eval --ckpt "runs/$TAG/best.pt" --dump-raw > /tmp/v046_finish_eval.log 2>&1
say "✓ 全部完成（累计 $TARGET 轮）"
