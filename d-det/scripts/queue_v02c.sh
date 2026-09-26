#!/usr/bin/env bash
# v0.2c 队列：真·异族对照（DeepSeek-Coder-1.3B 系）——回答「s2 的第二检测轴是否跨族成立」
#   1) 下载 deepseek-coder-1.3b{,-instruct}（hf-mirror）
#   2) 等 abmil 单流（v0.2.5）评测完成（GPU 互斥）
#   3) DeepSeek 配对生成（limit=400）
#   4) 跨族核查：v0.1.0（双流）与 abA（单头基线），npz 落盘供交叉融合分析
set -u
cd "$(dirname "$0")/.."
PY=/root/miniconda3/envs/udet/bin/python
export OMP_NUM_THREADS=8
export HF_ENDPOINT=https://hf-mirror.com
step() { echo "[v02c] $(date '+%F %T') $1"; }

step "1/4 下载 DeepSeek-Coder-1.3B 对（hf-mirror）"
"$PY" scripts/fetch_pair.py deepseek-coder-1.3b || { step "下载失败，终止"; exit 1; }

step "2/4 等待 abmil 单流（v0.2.5）评测完成（GPU 互斥）"
for i in $(seq 1 300); do [ -f runs/v0.2.5_ab_abmil_nopair/eval.json ] && break; sleep 30; done

step "3/4 DeepSeek 配对生成（limit=400）"
"$PY" scripts/gen_pairs.py --pair-name deepseek-coder-1.3b --out data/processed/pairs_ds13.parquet --limit 400 --batch-size 8 || { step "生成失败，终止"; exit 1; }

step "4/4 跨族核查：v0.1.0（双流）"
"$PY" scripts/probe_scale_verify.py --ckpt runs/v0.1.0/best.pt --tag v0.1.0_ds13 --pair-file data/processed/pairs_ds13.parquet --npz runs/v0.1.0/scale_scores_ds13.npz || step "v0.1.0 核查失败"

step "4/4 跨族核查：abA（单头基线）"
"$PY" scripts/probe_scale_verify.py --ckpt runs/v0.1.0_ab_nopair/best.pt --tag v0.1.0_ab_nopair_ds13 --pair-file data/processed/pairs_ds13.parquet --npz runs/v0.1.0_ab_nopair/scale_scores_ds13.npz || step "abA 核查失败"

step "队列完成（4/4）"
