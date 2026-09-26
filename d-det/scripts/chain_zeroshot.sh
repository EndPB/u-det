#!/usr/bin/env bash
# 等 SemEval 主队列（PID 见 /tmp/ddet_sev.pid）结束后，自动运行冻结零样本/线性探针。
set -u
PID_FILE=/tmp/ddet_sev.pid
cd "$(dirname "$0")/.."
while kill -0 "$(cat "$PID_FILE" 2>/dev/null || echo 0)" 2>/dev/null; do sleep 60; done
echo "[chain] 主队列已结束 $(date '+%F %T')，启动冻结探针…"
OMP_NUM_THREADS=8 HF_ENDPOINT=https://hf-mirror.com \
  /root/miniconda3/envs/udet/bin/python scripts/semeval_zeroshot.py \
  > /tmp/ddet_sev_zeroshot.log 2>&1
echo "ZS_EXIT=$?" >> /tmp/ddet_sev_zeroshot.log
echo "[chain] 探针完成 $(date '+%F %T')" >> /tmp/ddet_sev_zeroshot.log
