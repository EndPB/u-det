#!/usr/bin/env bash
# v0.2 修复版队列（2026-09-23 下午场）：
#   1) C2-fixed：s2_rank=4（W2 小随机初始化修复）训练+评测
#   2) C4a：qwen2.5-coder-1.5b 配对生成（跨模型族/规模泛化数据）
#   3) C4b：用 v0.1.0 的 s2 对新族配对报 dir_acc（scripts/probe_pair_gen.py）
#   4) C3：pool=abmil 训练+评测
#
# 用法: nohup bash scripts/queue_v02_fixC.sh > /tmp/ddet_ab_fixC.log 2>&1 &
set -u
cd "$(dirname "$0")/.."
PY=/root/miniconda3/envs/udet/bin/python
export OMP_NUM_THREADS=8
step() { echo "[fixC] $(date '+%F %T') $1"; }

# ---- 1) C2 修复版：s2_rank=4 ----
TAG=v0.2.1b_ab_s2r4
step "开始训练（s2_rank=4 修复版）tag=$TAG"
"$PY" train.py --config configs/ddet_base.yaml --tag "$TAG" --set model.s2_rank=4 || { step "$TAG 训练失败，终止"; exit 1; }
step "训练完成，评测 tag=$TAG"
"$PY" train.py --config configs/ddet_base.yaml --tag "$TAG" --eval --ckpt "runs/$TAG/best.pt" --dump-scores --set model.s2_rank=4 || { step "$TAG 评测失败"; exit 1; }
step "$TAG 全部完成"

# ---- 2/3) 跨模型族泛化：1.5B 配对 + 泛化探针 ----
P15B=checkpoints/qwen2.5-coder-1.5b-base
P15I=checkpoints/qwen2.5-coder-1.5b-instruct
if [ -d "$P15B" ] && [ -d "$P15I" ]; then
  step "开始 1.5B 配对生成（limit=400）"
  if "$PY" scripts/gen_pairs.py --pair-name qwen2.5-coder-1.5b --out data/processed/pairs_qwen15.parquet --limit 400 --batch-size 8; then
    step "生成完成，开始泛化探针（v0.1.0 的 s2 × 1.5B 配对）"
    "$PY" scripts/probe_pair_gen.py --ckpt runs/v0.1.0/best.pt --pair-file data/processed/pairs_qwen15.parquet --out runs/v0.2.4_pairgen15/probe.json || step "泛化探针失败（不阻塞）"
    step "泛化探针完成"
  else
    step "1.5B 生成失败，跳过探针"
  fi
else
  step "1.5B 模型未就绪（检查下载），跳过生成与探针"
fi

# ---- 4) C3：pool=abmil ----
TAG=v0.2.2_ab_abmil
step "开始训练（pool=abmil）tag=$TAG"
"$PY" train.py --config configs/ddet_base.yaml --tag "$TAG" --set model.pool=abmil || { step "$TAG 训练失败，终止"; exit 1; }
step "训练完成，评测 tag=$TAG"
"$PY" train.py --config configs/ddet_base.yaml --tag "$TAG" --eval --ckpt "runs/$TAG/best.pt" --dump-scores --set model.pool=abmil || { step "$TAG 评测失败"; exit 1; }
step "$TAG 全部完成"

step "修复版队列全部完成（4/4）"
