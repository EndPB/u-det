#!/usr/bin/env bash
# v0.4.4：样本头读全部 5 个尺度（跨度 64→1），2 epoch。
#
# 动机：样本头以前只读瓶颈 levels[0]（跨度 64），中位 293 token 的 m4 被压成
# ceil(293/64)=5 个向量；而 v0.2 的零训练诊断显示最细层（L/1）的样本级线性可达
# 0.832 ≫ 瓶颈 0.669 ≈ 样本头实测 0.673（当时判为"头已到顶"）。
#
# 与 v0.4.2 的唯一变量是 heads.sample_levels（1 -> 5），其余（含 token 头与
# A/B2/B3 损失）逐字保留 —— §8.8 已证明去掉 token 级监督会让样本级掉 1.76 分。
set -u
cd /root/autodl-tmp/u-det

PY=/root/miniconda3/envs/udet/bin/python
TAG=v0.4.4
CFG=configs/udet_v044.yaml

echo "[v044] $(date +%H:%M:%S) 启动训练（样本头多尺度，2 epoch）"
OMP_NUM_THREADS=8 nohup "$PY" train.py --config "$CFG" --tag "$TAG" --epochs 2 > /tmp/v044.log 2>&1 &
wait $!

echo "[v044] $(date +%H:%M:%S) 训练结束，评估 best.pt"
OMP_NUM_THREADS=8 "$PY" train.py --config "$CFG" --tag "$TAG" \
  --eval --ckpt "runs/$TAG/best.pt" > /tmp/v044_eval.log 2>&1
echo "[v044] $(date +%H:%M:%S) 全部完成"
