#!/bin/bash
# 第三轮：把 v0.4.0（门控）推到 6 epoch，对齐 v0.3.1 判停时的长度。
# 判停规则同 §7.3.1：看周期末端 mean 是否再创新高（当前 e3 = 0.8037，基线已到顶于 0.8021）。
export OMP_NUM_THREADS=8
PY=/root/miniconda3/envs/udet/bin/python
cd /root/autodl-tmp/u-det
FAIL='out of memory|CUDA error|RuntimeError|Killed'

echo "[queue] $(date +%H:%M:%S) ⑥ v0.4.0 第 3 个周期（e4/e5）"
$PY train.py --config configs/udet_v04.yaml --tag v0.4.0 --resume runs/v0.4.0/last.pt \
    --epochs 2 > /tmp/v040z.log 2>&1
if grep -qE "$FAIL" /tmp/v040z.log; then
  echo "[queue] !! 训练疑似失败：$(grep -m1 -oE "$FAIL" /tmp/v040z.log)"
fi
$PY train.py --config configs/udet_v04.yaml --tag v0.4.0 --eval --ckpt runs/v0.4.0/best.pt \
    >> /tmp/v040z.log 2>&1
echo "[queue] $(date +%H:%M:%S) ⑥ 完成"
echo "[queue] 全部完成"
