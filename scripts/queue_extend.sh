#!/bin/bash
export OMP_NUM_THREADS=8
PY=/root/miniconda3/envs/udet/bin/python
cd /root/autodl-tmp/u-det
FAIL='out of memory|CUDA error|RuntimeError|Killed'

echo "[queue] $(date +%H:%M:%S) 等 v0.4.0 结束"
while ! grep -qE "结果已写入|$FAIL" /tmp/v040.log 2>/dev/null; do sleep 60; done
if grep -qE "$FAIL" /tmp/v040.log; then
  echo "[queue] !! v0.4.0 疑似失败： $(grep -m1 -oE "$FAIL" /tmp/v040.log)"
else
  echo "[queue] v0.4.0 完成"
fi

echo "[queue] $(date +%H:%M:%S) ② v0.3.2 续跑 +2 epoch（codet5tok 冻结）"
$PY train.py --config configs/udet_v03.yaml --tag v0.3.2 --resume runs/v0.3.2/best.pt \
    --epochs 2 --encoder codet5tok --freeze-encoder > /tmp/v032x.log 2>&1
$PY train.py --config configs/udet_v03.yaml --tag v0.3.2 --eval --ckpt runs/v0.3.2/best.pt \
    --encoder codet5tok --freeze-encoder >> /tmp/v032x.log 2>&1
echo "[queue] $(date +%H:%M:%S) ② 完成"

echo "[queue] $(date +%H:%M:%S) ③ v0.3.1 续跑 +2 epoch（LoRA）"
$PY train.py --config configs/udet_v03.yaml --tag v0.3.1 --resume runs/v0.3.1/best.pt \
    --epochs 2 > /tmp/v031x.log 2>&1
$PY train.py --config configs/udet_v03.yaml --tag v0.3.1 --eval --ckpt runs/v0.3.1/best.pt >> /tmp/v031x.log 2>&1
echo "[queue] $(date +%H:%M:%S) ③ 完成"
echo "[queue] 全部完成"
