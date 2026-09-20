#!/usr/bin/env bash
# 对三个关键 run 各做一次带 `--dump-raw` 的评测，产出离线分析所需的逐样本原始数据。
#
# 产出：runs/<tag>/raw_{m4,hybrid}_{val,test}.pt
#   → 之后用 scripts/analyze_raw.py 就能离线做**阈值扫描**与**长度分桶**，不再占 GPU。
#
# 注意：run_eval 会把 eval.json 写到 ckpt 所在目录，所以这会**覆盖**各 run 原有的 eval.json。
# 评测是确定性的（无 dropout、argmax），同一 config + 同一 ckpt 复跑结果应当逐位一致 ——
# 跑完会逐项核对，正好顺便验证这一点。
set -u
cd /root/autodl-tmp/u-det

PY=/root/miniconda3/envs/udet/bin/python

dump() {  # $1=tag  $2=config
  echo "[dump] $(date +%H:%M:%S) 开始 $1  <-  $2"
  OMP_NUM_THREADS=8 "$PY" train.py --config "$2" --tag "$1" \
    --eval --ckpt "runs/$1/best.pt" --dump-raw > "/tmp/dump_$1.log" 2>&1
  local rc=$?
  echo "[dump] $(date +%H:%M:%S) $1 完成（退出码 $rc）"
}

dump v0.4.4          configs/udet_v044.yaml
dump v0.4.2          configs/udet_v04_cons.yaml
dump base_codet5_tok configs/baseline_codet5_tok.yaml

echo "[dump] $(date +%H:%M:%S) 全部完成"
