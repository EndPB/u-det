#!/usr/bin/env bash
# v0.3.0 续跑（+2 epoch，累计 4）：2ep 读数仍在爬升（s1/mlm/token 都在降；hybrid 每 epoch +1.1~1.2pt）
#   1) --resume last.pt +2ep   2) 评测 best.pt   3) 探针（片段 + 跨规模 qwen15/ds13）
set -u
cd "$(dirname "$0")/.."
PY=/root/miniconda3/envs/udet/bin/python
export OMP_NUM_THREADS=8
export HF_ENDPOINT=https://hf-mirror.com
step() { echo "[v030x] $(date '+%F %T') $1"; }

step "1/3 续跑 +2 epoch（--resume last.pt，累计 4）"
$PY train.py --config configs/ddet_v030.yaml --tag v0.3.0_fullft \
    --resume runs/v0.3.0_fullft/last.pt --epochs 2 > /tmp/ddet_v030_r2.log 2>&1 \
    || { step "续跑失败（见 /tmp/ddet_v030_r2.log）"; exit 1; }

step "2/3 评测 best.pt"
$PY train.py --config configs/ddet_v030.yaml --eval --ckpt runs/v0.3.0_fullft/best.pt --dump-scores \
    > /tmp/ddet_v030_eval2.log 2>&1 || step "评测失败"

step "3/3 探针（片段 + 跨规模）"
$PY scripts/probe_fragments.py --config configs/ddet_v030.yaml \
    --ckpt runs/v0.3.0_fullft/best.pt --n 500 --dump-samples \
    > /tmp/ddet_v030_frag2.log 2>&1 || step "片段探针失败"
$PY scripts/probe_scale_verify.py --config configs/ddet_v030.yaml \
    --ckpt runs/v0.3.0_fullft/best.pt --tag v0.3.0_fullft \
    --pair-file data/processed/pairs_qwen15.parquet --npz runs/v0.3.0_fullft/scale_scores.npz \
    > /tmp/ddet_v030_scale2.log 2>&1 || step "跨规模探针失败"
if [ -f data/processed/pairs_ds13.parquet ]; then
  $PY scripts/probe_scale_verify.py --config configs/ddet_v030.yaml \
      --ckpt runs/v0.3.0_fullft/best.pt --tag v0.3.0_fullft_ds13 \
      --pair-file data/processed/pairs_ds13.parquet --npz runs/v0.3.0_fullft/scale_scores_ds13.npz \
      > /tmp/ddet_v030_scale_ds13_2.log 2>&1 || step "ds13 探针失败"
fi

step "===== 汇总（4ep best.pt）====="
$PY - <<'PY'
import json
try:
    m = json.load(open('runs/v0.3.0_fullft/eval.json'))['metrics']
    for k, v in m.items():
        row = {kk: (round(vv, 4) if isinstance(vv, float) else vv) for kk, vv in v.items()}
        print('[summary]', k, row)
except Exception as exc:
    print('[summary] 读取失败：', exc)
PY
step "续跑完成（累计 4 epoch）"
