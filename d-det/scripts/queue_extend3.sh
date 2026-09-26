#!/usr/bin/env bash
# 扩族收尾（chain4）：HF_HUB_DISABLE_XET=1 补 qwen3b → Δ → 6 族重训。
set -u
cd "$(dirname "$0")/.."
PY=/root/miniconda3/envs/udet/bin/python
export OMP_NUM_THREADS=8
export HF_ENDPOINT=https://hf-mirror.com
export HF_HUB_DISABLE_XET=1
export PATH=/root/miniconda3/envs/udet/bin:$PATH
MDIR=/dev/shm/models

step(){ echo "[ext3 $(date '+%F %T')] $*"; }
shm_gb(){ df -BG /dev/shm | awk 'NR==2{print $4}' | tr -d 'G'; }

step "chain4 启动（/dev/shm 剩 $(shm_gb)GB）"

if [ ! -f "runs/v0.5_disc/d_qwen3b.npz" ]; then
  if [ ! -f "$MDIR/qwen3b-base/config.json" ]; then
    hf download Qwen/Qwen2.5-Coder-3B --local-dir "$MDIR/qwen3b-base" \
      > /tmp/ddet_ext3_dl_q3b.log 2>&1 || step "qwen3b base 下载失败"
  fi
  if [ ! -f "$MDIR/qwen3b-instruct/config.json" ]; then
    hf download Qwen/Qwen2.5-Coder-3B-Instruct --local-dir "$MDIR/qwen3b-instruct" \
      > /tmp/ddet_ext3_dli_q3b.log 2>&1 || step "qwen3b instruct 下载失败"
  fi
  if [ -f "$MDIR/qwen3b-base/config.json" ] && [ -f "$MDIR/qwen3b-instruct/config.json" ]; then
    if [ ! -f "data/processed/pairs_qwen3b.parquet" ]; then
      $PY scripts/gen_pairs.py --pair-name qwen3b --base-model "$MDIR/qwen3b-base" \
        --instruct-model "$MDIR/qwen3b-instruct" --out "data/processed/pairs_qwen3b.parquet" \
        --batch-size 8 > /tmp/ddet_ext3_gen_q3b.log 2>&1 || step "qwen3b 生成失败"
    fi
    if [ -f "data/processed/pairs_qwen3b.parquet" ]; then
      $PY scripts/extract_delta_feat.py --pairs "data/processed/pairs_qwen3b.parquet" \
        --out "runs/v0.5_disc/d_qwen3b.npz" > /tmp/ddet_ext3_feat_q3b.log 2>&1 \
        || step "qwen3b Δ 失败"
    fi
    rm -rf "$MDIR/qwen3b-base" "$MDIR/qwen3b-instruct"
  fi
  step "qwen3b 结束（/dev/shm 剩 $(shm_gb)GB）"
else
  step "qwen3b 已有"
fi

$PY scripts/disc_v05_train6.py > /tmp/ddet_ext3_train6.log 2>&1 || step "6 族训练失败"
step "训练完成（若 qwen3b 就位则为 6 族）"
tail -40 /tmp/ddet_ext3_train6.log
step "=====EXT3_DONE====="
