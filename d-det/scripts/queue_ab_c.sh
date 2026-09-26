#!/usr/bin/env bash
# d-det 消融 C 队列（v0.2 实验组）：--no-m4 → s2_rank=4 → pool=abmil，各自训练+评测。
#
# 用法（有卡）：
#   bash scripts/queue_ab_c.sh > /tmp/ddet_ab_c.log 2>&1 &
#
# 说明：
# - 预算纪律：C2/C3 与 v0.1.0 完全对齐（16069 step × 2ep）；
#   C1（--no-m4）是能力探针（s2 单独判别力），训练步数自动变为 pair 流长度（1048/ep）。
# - 评测命令带上与训练一致的 --set（模型段一致性检查要求）。
set -u
cd "$(dirname "$0")/.."
PY=/root/miniconda3/envs/udet/bin/python
export OMP_NUM_THREADS=8

step() { echo "[abC] $(date '+%F %T') $1"; }

TAG=v0.2.0_ab_nom4
step "开始训练（--no-m4 纯 s2）tag=$TAG"
"$PY" train.py --config configs/ddet_base.yaml --tag "$TAG" --no-m4 || { step "$TAG 训练失败，终止"; exit 1; }
step "训练完成，评测 tag=$TAG"
"$PY" train.py --config configs/ddet_base.yaml --tag "$TAG" --eval --ckpt "runs/$TAG/best.pt" --dump-scores || { step "$TAG 评测失败"; exit 1; }
step "$TAG 全部完成"

TAG=v0.2.1_ab_s2r4
step "开始训练（s2_rank=4）tag=$TAG"
"$PY" train.py --config configs/ddet_base.yaml --tag "$TAG" --set model.s2_rank=4 || { step "$TAG 训练失败，终止"; exit 1; }
step "训练完成，评测 tag=$TAG"
"$PY" train.py --config configs/ddet_base.yaml --tag "$TAG" --eval --ckpt "runs/$TAG/best.pt" --dump-scores --set model.s2_rank=4 || { step "$TAG 评测失败"; exit 1; }
step "$TAG 全部完成"

TAG=v0.2.2_ab_abmil
step "开始训练（pool=abmil）tag=$TAG"
"$PY" train.py --config configs/ddet_base.yaml --tag "$TAG" --set model.pool=abmil || { step "$TAG 训练失败，终止"; exit 1; }
step "训练完成，评测 tag=$TAG"
"$PY" train.py --config configs/ddet_base.yaml --tag "$TAG" --eval --ckpt "runs/$TAG/best.pt" --dump-scores --set model.pool=abmil || { step "$TAG 评测失败"; exit 1; }
step "$TAG 全部完成"

step "队列全部完成（3/3）"
