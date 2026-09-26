#!/usr/bin/env bash
# B 上的读出秩端到端确认：r ∈ {4, 8, 11}（与 B 主臂同超参：sqrt 权重 + CE + 空白扰动，2ep）
# 对照 = 既有直连头 runs/semeval_b（val 0.5021 / test 0.3040）
set -u
cd "$(dirname "$0")/.."
PY=/root/miniconda3/envs/udet/bin/python
export OMP_NUM_THREADS=8
export HF_ENDPOINT=https://hf-mirror.com
step() { echo "[rb] $(date '+%F %T') $1"; }

step "B 秩确认队列启动（r=4/8/11，各 2ep）"
for r in 4 8 11; do
  $PY scripts/semeval_finetune.py --task b --readout-rank "$r" --weights sqrt --loss ce \
      --aug-ws 0.3 --out "runs/semeval_b_r$r" > "/tmp/ddet_sev_b_r$r.log" 2>&1 \
      || step "r=$r 失败（见 /tmp/ddet_sev_b_r$r.log）"
  step "r=$r 完成"
done

step "===== 汇总 ====="
$PY - <<'PY'
import json
from pathlib import Path
print("\n== B 读出秩确认（macro-F1）==")
d = json.load(open("runs/semeval_b/eval.json"))
print(f"  plain头（既有基线）: val={d['val']['macro_f1']:.4f}  test={d['test']['macro_f1']:.4f}")
for r in (1, 2, 4, 8, 11, 16):
    p = Path(f"runs/semeval_b_r{r}/eval.json")
    if not p.exists():
        continue
    d = json.load(open(p))
    print(f"  s2-r={r}: val={d['val']['macro_f1']:.4f}  test={d['test']['macro_f1']:.4f}")
print("\n（冻结特征扫描参考：val 峰值 r=11 0.4154；r=8 0.4025；plain768 0.3979）")
PY
step "B 秩确认完成"
