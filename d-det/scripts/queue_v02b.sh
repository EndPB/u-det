#!/usr/bin/env bash
# v0.2b 队列：基线对照三件套（用户门坎：「不能显著超越 codet5 基线都是免谈」）
#   1) 片段基线对照：abA（CodeT5 单头基线）片段探针（n=500）
#   2) 片段最强 d-det：C3（abmil）片段探针（n=500）
#   3) 跨规模检测对照：v0.1.0 与 C3 的 s1 / s2 / 融合（1.5B 生成 vs 人类）
#   4) abmil 单流对照（--no-pair + abmil）：隔离「d-det 双维机制」在最佳配置下的检测贡献
set -u
cd "$(dirname "$0")/.."
PY=/root/miniconda3/envs/udet/bin/python
export OMP_NUM_THREADS=8
step() { echo "[v02b] $(date '+%F %T') $1"; }

step "1/5 片段对照：abA（单头基线）"
"$PY" scripts/probe_fragments.py --ckpt runs/v0.1.0_ab_nopair/best.pt --n 500 || step "abA 片段失败(不阻塞)"

step "2/5 片段对照：C3（abmil）"
"$PY" scripts/probe_fragments.py --ckpt runs/v0.2.2_ab_abmil/best.pt --n 500 --set model.pool=abmil || step "C3 片段失败(不阻塞)"

step "3/5 跨规模对照：v0.1.0"
"$PY" scripts/probe_scale_detect.py --ckpt runs/v0.1.0/best.pt --out runs/v0.1.0/scale_detect.json || step "scale v0.1.0 失败(不阻塞)"

step "4/5 跨规模对照：C3（abmil）"
"$PY" scripts/probe_scale_detect.py --ckpt runs/v0.2.2_ab_abmil/best.pt --set model.pool=abmil --out runs/v0.2.2_ab_abmil/scale_detect.json || step "scale C3 失败(不阻塞)"

TAG=v0.2.5_ab_abmil_nopair
step "5/5 单流对照训练：abmil + --no-pair（tag=$TAG，对齐预算）"
"$PY" train.py --config configs/ddet_base.yaml --tag "$TAG" --no-pair --set model.pool=abmil || { step "$TAG 训练失败，终止"; exit 1; }
step "$TAG 评测"
"$PY" train.py --config configs/ddet_base.yaml --tag "$TAG" --eval --ckpt "runs/$TAG/best.pt" --dump-scores --set model.pool=abmil || step "$TAG 评测失败"
step "队列完成（5/5）"
