#!/usr/bin/env bash
# v0.4.7：**2 epoch**（与前几版对齐预算）+ 自动评测落盘。
#
# 相对 v0.4.6 的**唯一改动**：loss.token_weight_nmin_rel: 0.125（即 N_min = L/8）
#   ⇒ 修掉固定锚点下"L ≥ 2048 权重退化成均匀"的饱和（§8.12.11）。
#   短文档形状不变（保住 v0.4.6 的短桶收益），长文档回到细尺度主导。
#
# ★ --dump-raw 必须带：本次的核心判据就是**长桶是否收回**。
set -u
cd /root/autodl-tmp/u-det

PY=/root/miniconda3/envs/udet/bin/python
TAG=v0.4.7
CFG=configs/udet_v047.yaml

echo "[q] $(date +%H:%M:%S) 训练 $TAG 2 epoch（配置 $CFG）"
OMP_NUM_THREADS=8 "$PY" train.py --config "$CFG" --tag "$TAG" > /tmp/v047.log 2>&1
rc=$?
echo "[q] $(date +%H:%M:%S) 训练退出码 $rc"
if [ "$rc" -ne 0 ]; then
  echo "[q] ✗ 训练失败，跳过评测（详见 /tmp/v047.log）"
  exit "$rc"
fi

echo "[q] $(date +%H:%M:%S) 评测 best.pt（--dump-raw）"
OMP_NUM_THREADS=8 "$PY" train.py --config "$CFG" --tag "$TAG" \
  --eval --ckpt "runs/$TAG/best.pt" --dump-raw > /tmp/v047_eval.log 2>&1
echo "[q] $(date +%H:%M:%S) 全部完成"
