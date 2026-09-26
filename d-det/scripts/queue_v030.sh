#!/usr/bin/env bash
# v0.3.0 单实验（先跑 2 epoch 版）：全参微调 + 四路监督（m4/hybrid/pair/MLM）
#   0) 等 v02c（PID 39408）结束（GPU 互斥，最长等 3h）
#   1) 冒烟：--limit 100（失败即终止）
#   2) 正式训练 2 epoch
#   3) 评测 best.pt（m4 + hybrid + pair）
#   4) 探针：片段 + 跨规模（qwen1.5；ds13 有就带）
set -u
cd "$(dirname "$0")/.."
PY=/root/miniconda3/envs/udet/bin/python
export OMP_NUM_THREADS=8
export HF_ENDPOINT=https://hf-mirror.com
step() { echo "[v030] $(date '+%F %T') $1"; }

step "0/5 等待 v02c（PID 39408）结束"
for i in $(seq 1 360); do kill -0 39408 2>/dev/null || break; sleep 30; done
sleep 20

step "1/5 冒烟（--limit 100 --epochs 1）"
$PY train.py --config configs/ddet_v030.yaml --tag v0.3.0_smoke --limit 100 --epochs 1 \
    > /tmp/ddet_v030_smoke.log 2>&1 || { step "冒烟失败，终止（见 /tmp/ddet_v030_smoke.log）"; exit 1; }
grep -q "\[train\] 完成" /tmp/ddet_v030_smoke.log || { step "冒烟未收尾，终止"; exit 1; }
step "冒烟通过：$(grep -E '^\[model\]' /tmp/ddet_v030_smoke.log | head -1)"

step "2/5 正式训练：2 epoch（tag=v0.3.0_fullft）"
$PY train.py --config configs/ddet_v030.yaml --tag v0.3.0_fullft > /tmp/ddet_v030_r1.log 2>&1 \
    || { step "训练失败（见 /tmp/ddet_v030_r1.log）"; exit 1; }

step "3/5 评测 best.pt"
$PY train.py --config configs/ddet_v030.yaml --eval --ckpt runs/v0.3.0_fullft/best.pt --dump-scores \
    > /tmp/ddet_v030_eval.log 2>&1 || step "评测失败"

step "4/5 探针：片段"
$PY scripts/probe_fragments.py --config configs/ddet_v030.yaml \
    --ckpt runs/v0.3.0_fullft/best.pt --n 500 --dump-samples \
    > /tmp/ddet_v030_frag.log 2>&1 || step "片段探针失败"

step "5/5 探针：跨规模（qwen1.5；ds13 若有则带）"
$PY scripts/probe_scale_verify.py --config configs/ddet_v030.yaml \
    --ckpt runs/v0.3.0_fullft/best.pt --tag v0.3.0_fullft \
    --pair-file data/processed/pairs_qwen15.parquet --npz runs/v0.3.0_fullft/scale_scores.npz \
    > /tmp/ddet_v030_scale.log 2>&1 || step "跨规模探针失败"
if [ -f data/processed/pairs_ds13.parquet ]; then
  $PY scripts/probe_scale_verify.py --config configs/ddet_v030.yaml \
      --ckpt runs/v0.3.0_fullft/best.pt --tag v0.3.0_fullft_ds13 \
      --pair-file data/processed/pairs_ds13.parquet --npz runs/v0.3.0_fullft/scale_scores_ds13.npz \
      > /tmp/ddet_v030_scale_ds13.log 2>&1 || step "ds13 探针失败"
fi

step "===== 汇总 ====="
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
step "全部完成（2 epoch 版）"
