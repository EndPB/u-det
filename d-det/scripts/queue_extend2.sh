#!/usr/bin/env bash
# 扩族接力（chain3）：等 chain2（granite2b）完成 → 补 Δ×4 → 补 qwen3b → 6 族判别头训练。
# 背景：granite 生成占满 GPU 时 Δ 提取会 OOM（已实测），故一切 GPU 任务排队等到空闲。
set -u
cd "$(dirname "$0")/.."
PY=/root/miniconda3/envs/udet/bin/python
export OMP_NUM_THREADS=8
export HF_ENDPOINT=https://hf-mirror.com
export PATH=/root/miniconda3/envs/udet/bin:$PATH
MDIR=/dev/shm/models

step(){ echo "[ext2 $(date '+%F %T')] $*"; }
shm_gb(){ df -BG /dev/shm | awk 'NR==2{print $4}' | tr -d 'G'; }

step "chain3 启动（/dev/shm 剩 $(shm_gb)GB）"

# ---------- S0：等 chain2 结束（EXTEND_DONE 或进程消失；上限 90min） ----------
for i in $(seq 1 90); do
  grep -q "EXTEND_DONE" /tmp/ddet_ext_chain2.log 2>/dev/null && break
  pgrep -f "bash scripts/queue_extend_families.sh" >/dev/null 2>&1 || break
  sleep 60
done
step "S0 chain2 结束（/dev/shm 剩 $(shm_gb)GB）"

# ---------- S1：补 Δ 特征（qwen05/qwen15/ds13；granite2b 若缺） ----------
for spec in "qwen05 data/processed/pairs.parquet" \
            "qwen15 data/processed/pairs_qwen15.parquet" \
            "ds13 data/processed/pairs_ds13.parquet" \
            "granite2b data/processed/pairs_granite2b.parquet"; do
  set -- $spec
  NAME=$1; P=$2
  if [ -f "runs/v0.5_disc/d_$NAME.npz" ]; then step "S1 $NAME 已有，跳过"; continue; fi
  [ -f "$P" ] || { step "S1 $NAME 缺配对文件，跳过"; continue; }
  $PY scripts/extract_delta_feat.py --pairs "$P" --out "runs/v0.5_disc/d_$NAME.npz" \
    > "/tmp/ddet_ext2_feat_$NAME.log" 2>&1 || step "S1 $NAME Δ 失败"
  step "S1 $NAME Δ 完成"
done

# ---------- S2：补 qwen3b（下载→生成→Δ→删模型） ----------
if [ ! -f "runs/v0.5_disc/d_qwen3b.npz" ]; then
  step "S2 qwen3b 开始（/dev/shm 剩 $(shm_gb)GB）"
  if [ ! -f "$MDIR/qwen3b-base/config.json" ]; then
    hf download Qwen/Qwen2.5-Coder-3B --local-dir "$MDIR/qwen3b-base" \
      > /tmp/ddet_ext2_dl_q3b.log 2>&1 || { step "S2 qwen3b base 下载失败"; }
  fi
  if [ ! -f "$MDIR/qwen3b-instruct/config.json" ]; then
    hf download Qwen/Qwen2.5-Coder-3B-Instruct --local-dir "$MDIR/qwen3b-instruct" \
      > /tmp/ddet_ext2_dli_q3b.log 2>&1 || { step "S2 qwen3b instruct 下载失败"; }
  fi
  if [ -f "$MDIR/qwen3b-base/config.json" ] && [ -f "$MDIR/qwen3b-instruct/config.json" ]; then
    if [ ! -f "data/processed/pairs_qwen3b.parquet" ]; then
      $PY scripts/gen_pairs.py --pair-name qwen3b --base-model "$MDIR/qwen3b-base" \
        --instruct-model "$MDIR/qwen3b-instruct" --out "data/processed/pairs_qwen3b.parquet" \
        --batch-size 8 > /tmp/ddet_ext2_gen_q3b.log 2>&1 || step "S2 qwen3b 生成失败"
    fi
    if [ -f "data/processed/pairs_qwen3b.parquet" ]; then
      $PY scripts/extract_delta_feat.py --pairs "data/processed/pairs_qwen3b.parquet" \
        --out "runs/v0.5_disc/d_qwen3b.npz" > /tmp/ddet_ext2_feat_q3b.log 2>&1 \
        || step "S2 qwen3b Δ 失败"
    fi
    rm -rf "$MDIR/qwen3b-base" "$MDIR/qwen3b-instruct"
  fi
  step "S2 qwen3b 结束（/dev/shm 剩 $(shm_gb)GB）"
else
  step "S2 qwen3b 已有，跳过"
fi

# ---------- S3：6 族判别头训练 ----------
$PY scripts/disc_v05_train6.py > /tmp/ddet_ext2_train6.log 2>&1 || step "S3 训练失败"
step "S3 训练完成"

# ---------- S4：汇总 ----------
tail -30 /tmp/ddet_ext2_train6.log
step "=====EXT2_DONE====="
