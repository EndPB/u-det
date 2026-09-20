#!/usr/bin/env bash
# v0.4.6：**2 epoch**（与 v0.4.5 / v0.4.4 的 2ep 对齐预算）+ 自动评测落盘。
#
# 两项改动一起跑（用户已确认）：
#   loss.token_weight_mode: length   (N_min=64, τ=0.5)   ← 只动 token 级
#   report.mode: vector + heads.report_dim: 7            ← 只动文档级头
#
# ★ 归因：两项改动的**前提**都已由零训练探针验证（docx/u-det-v0.4.md §8.12.5/§8.12.7），
#   所以这次只跑**一个**组合版；P2 还给了改动 1 的离线隔离估计。
#
# ★ 评测必须带 --dump-raw：否则做不了长度分桶（本次的关键判据就是短桶）。
set -u
cd /root/autodl-tmp/u-det

PY=/root/miniconda3/envs/udet/bin/python
TAG=v0.4.6
CFG=configs/udet_v046.yaml

echo "[q] $(date +%H:%M:%S) 训练 $TAG 2 epoch（配置 $CFG）"
OMP_NUM_THREADS=8 "$PY" train.py --config "$CFG" --tag "$TAG" > /tmp/v046.log 2>&1
rc=$?
echo "[q] $(date +%H:%M:%S) 训练退出码 $rc"

if [ "$rc" -ne 0 ]; then
  echo "[q] ✗ 训练失败，跳过评测（详见 /tmp/v046.log）"
  exit "$rc"
fi

echo "[q] $(date +%H:%M:%S) 评测 best.pt（--dump-raw）"
OMP_NUM_THREADS=8 "$PY" train.py --config "$CFG" --tag "$TAG" \
  --eval --ckpt "runs/$TAG/best.pt" --dump-raw > /tmp/v046_eval.log 2>&1
echo "[q] $(date +%H:%M:%S) 全部完成"
