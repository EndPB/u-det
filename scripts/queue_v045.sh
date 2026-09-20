#!/usr/bin/env bash
# v0.4.5：token 头旁路（最细尺度拼接编码器全长特征），2 epoch。
#
# 依据：§8.10.2 的长度分桶显示 U-Det 的弱项恰好集中在**短文档的 token 级指标**
# （[0,512) 桶 line −5.22 / chunk −6.23），与「主干各尺度都在 L/64 瓶颈之后」吻合；
# 而同一篇短文档在**样本级**上 U-Det 已 +0.40 领先（v0.4.4 的多尺度样本头已绕过瓶颈）。
# 所以这次把「绕过瓶颈」推广到 token 路径。
set -u
cd /root/autodl-tmp/u-det

PY=/root/miniconda3/envs/udet/bin/python
TAG=v0.4.5
CFG=configs/udet_v045.yaml

echo "[v045] $(date +%H:%M:%S) 启动训练（token 头旁路，2 epoch）"
OMP_NUM_THREADS=8 nohup "$PY" train.py --config "$CFG" --tag "$TAG" --epochs 2 > /tmp/v045.log 2>&1 &
wait $!

echo "[v045] $(date +%H:%M:%S) 训练结束，评估 best.pt"
OMP_NUM_THREADS=8 "$PY" train.py --config "$CFG" --tag "$TAG" \
  --eval --ckpt "runs/$TAG/best.pt" --dump-raw > /tmp/v045_eval.log 2>&1
echo "[v045] $(date +%H:%M:%S) 全部完成"
