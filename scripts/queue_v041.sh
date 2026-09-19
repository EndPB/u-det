#!/bin/bash
# v0.4.1 = v0.4.0 的结构（门控）+ 三项损失改造（A cons_pool / B2 cons_pos / B3 lse 目标）。
# 与 v0.4.0 用**完全相同的 6-epoch 协议**（2 epoch 一轮 × 3 轮热重启），保证同预算可比。
export OMP_NUM_THREADS=8
PY=/root/miniconda3/envs/udet/bin/python
CFG=configs/udet_v04_cons.yaml
cd /root/autodl-tmp/u-det
FAIL='out of memory|CUDA error|RuntimeError|Killed'

cycle () {   # $1=说明  $2=空(从头) / resume
  echo "[queue] $(date +%H:%M:%S) v0.4.1 $1"
  if [ -z "$2" ]; then
    $PY train.py --config $CFG --tag v0.4.1 --epochs 2 > /tmp/v041.log 2>&1
  else
    $PY train.py --config $CFG --tag v0.4.1 --resume runs/v0.4.1/last.pt --epochs 2 >> /tmp/v041.log 2>&1
  fi
  if grep -qE "$FAIL" /tmp/v041.log; then
    echo "[queue] !! 失败：$(grep -m1 -oE "$FAIL" /tmp/v041.log)"
  fi
  return 0
}

cycle "周期 1（e0/e1，从头）" ""
cycle "周期 2（e2/e3，热重启）" resume
cycle "周期 3（e4/e5，热重启）" resume

echo "[queue] $(date +%H:%M:%S) 最终评测"
$PY train.py --config $CFG --tag v0.4.1 --eval --ckpt runs/v0.4.1/best.pt >> /tmp/v041.log 2>&1
echo "[queue] $(date +%H:%M:%S) 全部完成"
