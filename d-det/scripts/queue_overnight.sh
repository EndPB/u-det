#!/usr/bin/env bash
# 夜间自动链（2026-09-25）：手工特征 → probs 导出 → s2 能力展示 → 三任务堆叠 → B 瓶颈重试 → 汇总
set -u
cd "$(dirname "$0")/.."
PY=/root/miniconda3/envs/udet/bin/python
export OMP_NUM_THREADS=8
export HF_ENDPOINT=https://hf-mirror.com
step() { echo "[sv] $(date '+%F %T') $1"; }

step "overnight 链启动"
$PY scripts/semeval_stats.py > /tmp/ddet_sv_stats.log 2>&1 || step "stats 失败（fail-soft 继续）"
step "stats 完成"
$PY scripts/semeval_probs_dump.py > /tmp/ddet_sv_probs.log 2>&1 || step "probs 失败（继续）"
step "probs 完成"
$PY scripts/semeval_s2_showcase.py > /tmp/ddet_sv_showcase.log 2>&1 || step "showcase 失败（继续）"
step "showcase 完成"
$PY scripts/semeval_stack.py > /tmp/ddet_sv_stack.log 2>&1 || step "stack 失败（继续）"
step "stack 完成"
$PY scripts/semeval_finetune.py --task b --readout-rank 8 --readout-init 0.02 \
    --weights sqrt --loss ce --aug-ws 0.3 --out runs/semeval_b_r8init \
    > /tmp/ddet_sv_retry.log 2>&1 || step "retry 失败（继续）"
step "retry 完成"
$PY - <<'PYEOF' > /tmp/ddet_sv_summary.txt 2>&1
import json
from pathlib import Path
print("========== 夜间链汇总 ==========")
for f in ("runs/semeval_r/sweep.json", "runs/semeval_r/showcase.json",
          "runs/semeval_stack/results.json"):
    p = Path(f)
    print("=" * 20, f)
    print(p.read_text() if p.exists() else "[缺]")
p = Path("runs/semeval_b_r8init/eval.json")
if p.exists():
    d = json.load(open(p))
    print("[B retry r8 init0.02] val=", d["val"]["macro_f1"], " test=", d["test"]["macro_f1"])
else:
    print("[B retry] [缺]")
PYEOF
step "汇总完成 -> /tmp/ddet_sv_summary.txt"
