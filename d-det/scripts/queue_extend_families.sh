#!/usr/bin/env bash
# v0.5 扩族（③）：逐族「下载 → 生成配对 → Δ 特征 → 删模型」，fail-soft。
# 计划三族：Qwen2.5-Coder-3B / Yi-Coder-1.5B / Granite-3.1-2B（均非 gated）。
# 产物：data/processed/pairs_<name>.parquet + runs/v0.5_disc/d_<name>.npz
set -u
cd "$(dirname "$0")/.."
PY=/root/miniconda3/envs/udet/bin/python
export OMP_NUM_THREADS=8
export HF_ENDPOINT=https://hf-mirror.com
export PATH=/root/miniconda3/envs/udet/bin:$PATH
MDIR=/tmp/models

step(){ echo "[ext $(date '+%F %T')] $*"; }
free_tmp(){ df -BG /tmp | awk 'NR==2{print $4}' | tr -d 'G'; }

run_family(){  # $1=name $2=base_repo $3=instruct_repo
  local NAME=$1 B=$2 I=$3
  if [ -f "runs/v0.5_disc/d_$NAME.npz" ]; then step "$NAME 已有特征，跳过"; return; fi
  step "$NAME 开始（/tmp 剩 $(free_tmp)GB）"
  if [ ! -f "$MDIR/$NAME-base/config.json" ]; then
    hf download "$B" --local-dir "$MDIR/$NAME-base" \
      > "/tmp/ddet_ext_dl_$NAME.log" 2>&1 || { step "$NAME 下载失败($B)，跳过"; return; }
  fi
  if [ ! -f "$MDIR/$NAME-instruct/config.json" ]; then
    hf download "$I" --local-dir "$MDIR/$NAME-instruct" \
      > "/tmp/ddet_ext_dli_$NAME.log" 2>&1 || { step "$NAME 下载失败($I)，跳过"; rm -rf "$MDIR/$NAME-base"; return; }
  fi
  if [ ! -f "data/processed/pairs_$NAME.parquet" ]; then
    $PY scripts/gen_pairs.py --pair-name "$NAME" --base-model "$MDIR/$NAME-base" \
      --instruct-model "$MDIR/$NAME-instruct" --out "data/processed/pairs_$NAME.parquet" \
      --batch-size 8 > "/tmp/ddet_ext_gen_$NAME.log" 2>&1 \
      || { step "$NAME 生成失败（见日志），跳过"; return; }
  fi
  $PY scripts/extract_delta_feat.py --pairs "data/processed/pairs_$NAME.parquet" \
    --out "runs/v0.5_disc/d_$NAME.npz" > "/tmp/ddet_ext_feat_$NAME.log" 2>&1 \
    || step "$NAME Δ 特征失败（保留模型）"
  rm -rf "$MDIR/$NAME-base" "$MDIR/$NAME-instruct"
  step "$NAME 完成（模型已删，/tmp 剩 $(free_tmp)GB）"
}

run_family qwen3b   Qwen/Qwen2.5-Coder-3B            Qwen/Qwen2.5-Coder-3B-Instruct
run_family yi15     01-ai/Yi-Coder-1.5B              01-ai/Yi-Coder-1.5B-Chat
run_family granite2b ibm-granite/granite-3.1-2b-base  ibm-granite/granite-3.1-2b-instruct

step "=====EXTEND_DONE====="
ls -la runs/v0.5_disc/ | grep "d_" || true
