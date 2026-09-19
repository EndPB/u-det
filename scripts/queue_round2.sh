#!/bin/bash
# 第二轮队列：把 v0.3.1 推到第 3 个 LR 周期（e4/e5），再把 v0.4.0 对齐到 4 epoch。
# 两组都用"热重启续跑"这同一协议，因此可与 v0.3.1 @4ep 做同协议对照。
export OMP_NUM_THREADS=8
PY=/root/miniconda3/envs/udet/bin/python
cd /root/autodl-tmp/u-det

echo "[queue] $(date +%H:%M:%S) ④ v0.3.1 第 3 个周期（e4/e5）"
$PY train.py --config configs/udet_v03.yaml --tag v0.3.1 --resume runs/v0.3.1/last.pt \
    --epochs 2 > /tmp/v031y.log 2>&1
$PY train.py --config configs/udet_v03.yaml --tag v0.3.1 --eval --ckpt runs/v0.3.1/best.pt \
    >> /tmp/v031y.log 2>&1
echo "[queue] $(date +%H:%M:%S) ④ 完成"

echo "[queue] $(date +%H:%M:%S) ⑤ v0.4.0 对齐到 4 epoch（同协议）"
$PY train.py --config configs/udet_v04.yaml --tag v0.4.0 --resume runs/v0.4.0/best.pt \
    --epochs 2 > /tmp/v040y.log 2>&1
$PY train.py --config configs/udet_v04.yaml --tag v0.4.0 --eval --ckpt runs/v0.4.0/best.pt \
    >> /tmp/v040y.log 2>&1
echo "[queue] $(date +%H:%M:%S) ⑤ 完成"
echo "[queue] 全部完成"
