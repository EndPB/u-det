#!/usr/bin/env bash
# v0.3 归因-修复队列：回答「指标为什么下降 + 怎么修」
#   由 scripts/v03_tasks.txt 逐行驱动（每行一条 bash 命令；日志 /tmp/ddet_v03_<n>.log）。
#   前提：等 v02c（PID 39408，DeepSeek 跨族核查）结束，避免 GPU 互斥。
set -u
cd "$(dirname "$0")/.."
export OMP_NUM_THREADS=8
export HF_ENDPOINT=https://hf-mirror.com
export PY=/root/miniconda3/envs/udet/bin/python
step() { echo "[v03] $(date '+%F %T') $1"; }

step "等待 v02c（PID 39408）结束"
for i in $(seq 1 360); do
  kill -0 39408 2>/dev/null || break
  sleep 30
done
sleep 20
grep -q "完成" /tmp/ddet_v02c.log && step "v02c 正常完成" || step "警告：v02c 未检测到完成标记（见日志）"
step "开始任务列表 scripts/v03_tasks.txt"

i=0
while IFS= read -r line; do
  case "$line" in ''|\#*) continue ;; esac
  i=$((i+1))
  step "task $i: $line"
  if bash -c "$line" > "/tmp/ddet_v03_${i}.log" 2>&1; then
    step "task $i 完成（/tmp/ddet_v03_${i}.log）"
  else
    step "task $i 失败（见 /tmp/ddet_v03_${i}.log）——继续后续任务"
  fi
done < scripts/v03_tasks.txt
step "全部任务结束（共 $i 条）"
