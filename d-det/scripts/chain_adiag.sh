#!/usr/bin/env bash
# 等零样本探针链结束（ZS_EXIT 出现或探针进程退出，上限 150 分钟）后，运行 A 臂塌陷诊断。
set -u
cd "$(dirname "$0")/.."
for i in $(seq 1 150); do
  if grep -q "ZS_EXIT=" /tmp/ddet_sev_zeroshot.log 2>/dev/null; then break; fi
  if [ "$i" -gt 2 ] && ! kill -0 "$(cat /tmp/ddet_sev_chain.pid 2>/dev/null || echo 0)" 2>/dev/null; then break; fi
  sleep 60
done
echo "[chain2] 探针结束/超时 $(date '+%F %T')，运行 A 诊断…"
OMP_NUM_THREADS=8 HF_ENDPOINT=https://hf-mirror.com \
  /root/miniconda3/envs/udet/bin/python scripts/semeval_a_diag.py \
  > /tmp/ddet_sev_adiag.log 2>&1
echo "ADIAG_EXIT=$?" >> /tmp/ddet_sev_adiag.log
echo "[chain2] A 诊断完成 $(date '+%F %T')" >> /tmp/ddet_sev_adiag.log
