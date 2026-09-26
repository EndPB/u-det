#!/usr/bin/env bash
# Qwen2.5-Coder-1.5B 骨干接入队列：等夜链释放 GPU → 重分词 → 冒烟 → B 主臂（2ep）
set -u
cd "$(dirname "$0")/.."
PY=/root/miniconda3/envs/udet/bin/python
export OMP_NUM_THREADS=8
export HF_ENDPOINT=https://hf-mirror.com
step() { echo "[qw] $(date '+%F %T') $1"; }

# 0) 等夜链完成（GPU 释放）
for _ in $(seq 1 60); do
  grep -q "汇总完成" /tmp/ddet_sv_chain.log 2>/dev/null && break
  sleep 30
done
step "夜链结束，Qwen 流程开始"

# 1) 重分词（与 CodeT5 版行对齐）
$PY scripts/semeval_prepare_qwen.py > /tmp/ddet_qw_prep.log 2>&1 \
  || { step "重分词失败（见 ddet_qw_prep.log）"; exit 1; }
step "1/4 重分词完成"

# 2) 接入冒烟（fail-closed）
$PY scripts/check_qwen.py > /tmp/ddet_qw_check.log 2>&1 \
  || { step "冒烟失败（见 ddet_qw_check.log）"; exit 1; }
grep -q "QWEN_SMOKE_PASS" /tmp/ddet_qw_check.log || { step "冒烟未通过"; exit 1; }
step "2/4 冒烟通过"

# 3) 小样本训练冒烟
$PY scripts/semeval_finetune.py --task b --limit 300 --epochs 1 --batch 2 \
    --config configs/ddet_qwen.yaml --data-dir data/processed/semeval_qwen \
    --init none --lr-encoder 2e-4 --out runs/semeval_qwen_smoke \
    > /tmp/ddet_qw_smoke.log 2>&1 || { step "训练冒烟失败"; exit 1; }
grep -q "\[out\]" /tmp/ddet_qw_smoke.log || { step "训练冒烟未收尾"; exit 1; }
rm -f runs/semeval_qwen_smoke/*.pt && step "3/4 训练冒烟通过（冒烟检查点已清理省盘）"

# 4) 正式 B 臂（2ep；LoRA lr=2e-4、头 3e-4、有效批 8）
$PY scripts/semeval_finetune.py --task b --weights sqrt --loss ce --aug-ws 0.3 --batch 4 \
    --config configs/ddet_qwen.yaml --data-dir data/processed/semeval_qwen \
    --init none --lr-encoder 2e-4 --out runs/semeval_qwen_b \
    > /tmp/ddet_qw_b.log 2>&1 || step "B 主臂失败"
step "4/4 B 主臂完成"

$PY - <<'PYEOF' > /tmp/ddet_qw_summary.txt 2>&1
import json
from pathlib import Path
p = Path("runs/semeval_qwen_b/eval.json")
if p.exists():
    d = json.load(open(p))
    print("Qwen-B: val =", d["val"]["macro_f1"], " test =", d["test"]["macro_f1"])
    print("对照 CodeT5-B: val 0.5021 / test 0.3040")
else:
    print("Qwen-B: [缺]")
PYEOF
step "汇总完成 -> /tmp/ddet_qw_summary.txt"
