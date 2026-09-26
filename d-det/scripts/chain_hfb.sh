#!/usr/bin/env bash
# 等 A 诊断（chain2）结束后，在 HF 500K test 抽样上运行 B 臂模型。
set -u
cd "$(dirname "$0")/.."
for i in $(seq 1 150); do
  if grep -q "ADIAG_EXIT=" /tmp/ddet_sev_adiag.log 2>/dev/null; then break; fi
  if [ "$i" -gt 2 ] && ! kill -0 "$(cat /tmp/ddet_sev_chain2.pid 2>/dev/null || echo 0)" 2>/dev/null; then break; fi
  sleep 60
done
echo "[chain3] A 诊断结束/超时 $(date '+%F %T')，运行 HF test 推理…"
OMP_NUM_THREADS=8 HF_ENDPOINT=https://hf-mirror.com \
  /root/miniconda3/envs/udet/bin/python scripts/semeval_hf_b.py \
  > /tmp/ddet_hf_b.log 2>&1
echo "HFB_EXIT=$?" >> /tmp/ddet_hf_b.log
echo "[chain3] HF-B 完成 $(date '+%F %T')" >> /tmp/ddet_hf_b.log
