#!/usr/bin/env bash
# E22 生成队列（坍缩度前置；串行、断点续跑）：
#   逐族「（如缺）下载 instruct 模型 → 主实验 8×329 题 @T=0.7 → 温度消融 0.2/1.0 × 前 50 题 → 删模型」
# 预注册协议在此写死：N=8、top_p=0.95、任务集=6 族共同 329 题（排序后前 50 题用于消融）。
set -u
cd "$(dirname "$0")/.."
PY=/root/miniconda3/envs/udet/bin/python
export OMP_NUM_THREADS=8
export HF_ENDPOINT=https://hf-mirror.com
export HF_HUB_DISABLE_XET=1
export PATH=/root/miniconda3/envs/udet/bin:$PATH
MDIR=/dev/shm/models
mkdir -p "$MDIR"

step(){ echo "[e22 $(date '+%F %T')] $*"; }

gen_family(){  # $1=name $2=model_dir
  local NAME=$1 M=$2
  if [ ! -f "data/processed/multisample_${NAME}_t0.7.parquet" ]; then
    step "$NAME 主实验 T=0.7 开始"
    $PY scripts/gen_multi_samples.py --family "$NAME" --model "$M" \
      --out "data/processed/multisample_${NAME}_t0.7.parquet" --n 8 --temperature 0.7 \
      > "/tmp/ddet_e22_gen_${NAME}_07.log" 2>&1 || { step "$NAME t0.7 失败"; return 1; }
  fi
  if [ ! -f "data/processed/multisample_${NAME}_t0.2.parquet" ]; then
    step "$NAME 消融 T=0.2 开始"
    $PY scripts/gen_multi_samples.py --family "$NAME" --model "$M" \
      --out "data/processed/multisample_${NAME}_t0.2.parquet" --n 8 --temperature 0.2 --limit 50 \
      > "/tmp/ddet_e22_gen_${NAME}_02.log" 2>&1 || step "$NAME t0.2 失败"
  fi
  if [ ! -f "data/processed/multisample_${NAME}_t1.0.parquet" ]; then
    step "$NAME 消融 T=1.0 开始"
    $PY scripts/gen_multi_samples.py --family "$NAME" --model "$M" \
      --out "data/processed/multisample_${NAME}_t1.0.parquet" --n 8 --temperature 1.0 --limit 50 \
      > "/tmp/ddet_e22_gen_${NAME}_10.log" 2>&1 || step "$NAME t1.0 失败"
  fi
}

dl_gen(){  # $1=name $2=repo（/dev/shm 中转；大仓库过滤 onnx/runs/md）
  local NAME=$1 REPO=$2
  [ -f "data/processed/multisample_${NAME}_t1.0.parquet" ] && { step "$NAME 已完成，跳过"; return; }
  if [ -z "$(ls "$MDIR/$NAME-instruct/"*.safetensors 2>/dev/null)" ]; then
    step "$NAME 下载 $REPO（/dev/shm 剩 $(df -BG /dev/shm | awk 'NR==2{print $4}')）"
    hf download "$REPO" --local-dir "$MDIR/$NAME-instruct" \
      --exclude "onnx/*" --exclude "runs/*" --exclude "*.md" \
      > "/tmp/ddet_e22_dl_$NAME.log" 2>&1 || { step "$NAME 下载失败，跳过"; return; }
  fi
  gen_family "$NAME" "$MDIR/$NAME-instruct"
  rm -rf "$MDIR/$NAME-instruct"
  step "$NAME 模型已删（/dev/shm 剩 $(df -BG /dev/shm | awk 'NR==2{print $4}')）"
}

step "队列启动"
# ① 本地已有模型的三族
gen_family qwen05 checkpoints/qwen2.5-coder-0.5b-instruct
gen_family qwen15 checkpoints/qwen2.5-coder-1.5b-instruct
gen_family ds13   checkpoints/deepseek-coder-1.3b-instruct
# ② 需下载的三族
dl_gen yi15      01-ai/Yi-Coder-1.5B-Chat
dl_gen granite2b ibm-granite/granite-3.1-2b-instruct
dl_gen smollm2   HuggingFaceTB/SmolLM2-1.7B-Instruct

step "=====E22_GEN_DONE====="
ls -la data/processed/ | grep multisample || true
