#!/usr/bin/env bash
# 扩族第 6 族（chain5）：SmolLM2-1.7B（全新谱系；Qwen3B 双模型超 12GB 显存，放弃）。
set -u
cd "$(dirname "$0")/.."
PY=/root/miniconda3/envs/udet/bin/python
export OMP_NUM_THREADS=8
export HF_ENDPOINT=https://hf-mirror.com
export HF_HUB_DISABLE_XET=1
export PATH=/root/miniconda3/envs/udet/bin:$PATH
MDIR=/dev/shm/models
NAME=smollm2
BASE=HuggingFaceTB/SmolLM2-1.7B
INST=HuggingFaceTB/SmolLM2-1.7B-Instruct

step(){ echo "[ext4 $(date '+%F %T')] $*"; }
shm_gb(){ df -BG /dev/shm | awk 'NR==2{print $4}' | tr -d 'G'; }

step "chain5 启动（/dev/shm 剩 $(shm_gb)GB）"

if [ ! -f "runs/v0.5_disc/d_$NAME.npz" ]; then
  # 注：SmolLM2 Instruct 仓库含 ~19GB onnx/ 导出文件（曾撑爆 15GB /dev/shm），必须过滤
  hf download "$BASE" --local-dir "$MDIR/$NAME-base" \
    --exclude "onnx/*" --exclude "runs/*" --exclude "*.md" \
    > /tmp/ddet_ext4_dl.log 2>&1 || step "下载失败 base"
  hf download "$INST" --local-dir "$MDIR/$NAME-instruct" \
    --exclude "onnx/*" --exclude "runs/*" --exclude "*.md" \
    > /tmp/ddet_ext4_dli.log 2>&1 || step "下载失败 instruct"
  if [ -f "$MDIR/$NAME-base/model.safetensors" ] && [ -f "$MDIR/$NAME-instruct/model.safetensors" ] \
     && [ -f "$MDIR/$NAME-instruct/tokenizer.json" ]; then
    [ -f "data/processed/pairs_$NAME.parquet" ] || \
      $PY scripts/gen_pairs.py --pair-name "$NAME" --base-model "$MDIR/$NAME-base" \
        --instruct-model "$MDIR/$NAME-instruct" --out "data/processed/pairs_$NAME.parquet" \
        --batch-size 8 > /tmp/ddet_ext4_gen.log 2>&1 || step "生成失败"
    [ -f "data/processed/pairs_$NAME.parquet" ] && \
      $PY scripts/extract_delta_feat.py --pairs "data/processed/pairs_$NAME.parquet" \
        --out "runs/v0.5_disc/d_$NAME.npz" > /tmp/ddet_ext4_feat.log 2>&1 || step "Δ 失败"
  else
    step "模型文件不完整，跳过生成"
  fi
  rm -rf "$MDIR/$NAME-base" "$MDIR/$NAME-instruct"
  step "$NAME 结束（/dev/shm 剩 $(shm_gb)GB）"
fi

$PY scripts/disc_v05_train6.py > /tmp/ddet_ext4_train6.log 2>&1 || step "训练失败"
step "训练完成（含则 6 族）"
tail -40 /tmp/ddet_ext4_train6.log
step "=====EXT4_DONE====="
