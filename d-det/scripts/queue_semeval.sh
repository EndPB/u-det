#!/usr/bin/env bash
# SemEval-2026 Task 13 适配队列（冻结骨干 v1.0；每臂 ≤2ep）
#   0) 等数据准备完成   1) 冒烟(A)   2) A 主臂（平衡子集 + 阈值校准）
#   3) C 主臂（sqrt 类权重）   4) C 臂2（focal+sqrt）   5) B 主臂（sqrt）   6) B 臂2（inv）
#   7) 汇总（各 eval.json + 多数类基线 + 榜单参照）
set -u
cd "$(dirname "$0")/.."
PY=/root/miniconda3/envs/udet/bin/python
export OMP_NUM_THREADS=8
export HF_ENDPOINT=https://hf-mirror.com
step() { echo "[sev] $(date '+%F %T') $1"; }

# ---- 0) 等待数据准备
step "0/7 等待数据准备（meta.json）…"
ok=0
for _ in $(seq 1 90); do
  if [ -f data/processed/semeval/meta.json ] && grep -q "\[done\]" /tmp/ddet_semeval_prep.log 2>/dev/null; then
    ok=1; break
  fi
  sleep 30
done
[ "$ok" = "1" ] || { step "数据准备超时/失败（见 /tmp/ddet_semeval_prep.log）"; exit 1; }
step "数据就绪"

# ---- 1) 冒烟（A，300 条 1ep；fail-closed）
$PY scripts/semeval_finetune.py --task a --limit 300 --epochs 1 \
    --out runs/semeval_smoke > /tmp/ddet_sev_smoke.log 2>&1 \
    || { step "冒烟失败（见 /tmp/ddet_sev_smoke.log）"; exit 1; }
grep -q "\[out\]" /tmp/ddet_sev_smoke.log || { step "冒烟未正常收尾"; exit 1; }
if grep -qi "traceback" /tmp/ddet_sev_smoke.log; then step "冒烟含 Traceback，中止"; exit 1; fi
step "1/7 冒烟通过"

# ---- 2) A 主臂（平衡子集；空白扰动增强；阈值校准）
$PY scripts/semeval_finetune.py --task a --weights none --loss ce --aug-ws 0.3 \
    > /tmp/ddet_sev_a.log 2>&1 || step "A 主臂失败"
step "2/7 A 完成"

# ---- 3) C 主臂（sqrt 类权重 + CE）
$PY scripts/semeval_finetune.py --task c --weights sqrt --loss ce --aug-ws 0.3 \
    > /tmp/ddet_sev_c_sqrt.log 2>&1 || step "C-sqrt 失败"
step "3/7 C-sqrt 完成"

# ---- 4) C 臂2（focal γ=2 + sqrt 权重）
$PY scripts/semeval_finetune.py --task c --weights sqrt --loss focal --gamma 2.0 \
    --aug-ws 0.3 --out runs/semeval_c_focal > /tmp/ddet_sev_c_focal.log 2>&1 || step "C-focal 失败"
step "4/7 C-focal 完成"

# ---- 5) B 主臂（sqrt 类权重）
$PY scripts/semeval_finetune.py --task b --weights sqrt --loss ce --aug-ws 0.3 \
    > /tmp/ddet_sev_b_sqrt.log 2>&1 || step "B-sqrt 失败"
step "5/7 B-sqrt 完成"

# ---- 6) B 臂2（inv 全逆频率权重）
$PY scripts/semeval_finetune.py --task b --weights inv --loss ce --aug-ws 0.3 \
    --out runs/semeval_b_inv > /tmp/ddet_sev_b_inv.log 2>&1 || step "B-inv 失败"
step "6/7 B-inv 完成"

# ---- 7) 汇总：我们的结果 + 多数类基线 + 榜单参照
step "===== 汇总 ====="
$PY - <<'PY'
import json
from pathlib import Path
import pyarrow.parquet as pq
import numpy as np
from sklearn.metrics import f1_score

print("\n== 我们的结果（macro-F1；test 为官方镜像 1K 样本）==")
for tag in ("semeval_a", "semeval_c", "semeval_c_focal",
            "semeval_b", "semeval_b_inv"):
    p = Path("runs") / tag / "eval.json"
    if not p.exists():
        print(f"  [缺] {tag}"); continue
    d = json.load(open(p))
    v, t = d["val"], d["test"]
    extra = (f" @thr={t.get('threshold'):.2f}" if t.get("threshold") else "") + \
            (f" | test@0.5={t.get('macro_f1@0.5')}" if "macro_f1@0.5" in t else "")
    print(f"  {tag}: val={v['macro_f1']:.4f} test={t['macro_f1']:.4f}{extra}")

print("\n== 多数类基线（同一评测子集）==")
for task, va_f, te_f in (("a", "a_val", "a_test"), ("b", "b_val", "b_test"),
                         ("c", "c_val", "c_test")):
    for name, f in (("val", va_f), ("test", te_f)):
        t = pq.read_table(f"data/processed/semeval/{f}.parquet", columns=["label"])
        y = np.array(t.column("label").to_pylist())
        maj = np.bincount(y).argmax()
        f1 = f1_score(y, np.full_like(y, maj), average="macro")
        print(f"  {task}-{name}: n={len(y)} 多数类={maj} macro-F1={f1:.4f}")

print("\n== 榜单参照（官方私有测试；不可直接比较，仅方向参考）==")
print("  A 冠军 0.9971 (TeleAI) / 2nd 0.8241 / 3rd 0.8019")
print("  B 冠军 0.5082 (TeleAI) / 2nd 0.4560 / 3rd 0.4488")
print("  C 冠军 0.7855 (YoungDSMLKZ) / 2nd 0.7333 / 3rd 0.7140")
PY
step "SemEval 适配队列完成"
