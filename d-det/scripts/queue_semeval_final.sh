#!/usr/bin/env bash
# SemEval 适配队列（精简版：3 臂 = A + C-sqrt + B-sqrt）
#   用户决策（2026-09-24）：B/C 各只保留一个方案（C-focal、B-inv 取消，节省租金）。
#   A、C-sqrt 已完成；本脚本负责：B 冒烟 → B-sqrt 主臂 → 汇总。
set -u
cd "$(dirname "$0")/.."
PY=/root/miniconda3/envs/udet/bin/python
export OMP_NUM_THREADS=8
export HF_ENDPOINT=https://hf-mirror.com
step() { echo "[sev] $(date '+%F %T') $1"; }

step "final 队列启动（B 单臂 + 汇总；C-focal/B-inv 已按用户决策取消）"

# ---- 0) 前置断言：A / C 结果已在
[ -f runs/semeval_a/eval.json ] || step "警告：缺少 runs/semeval_a/eval.json"
[ -f runs/semeval_c/eval.json ] || step "警告：缺少 runs/semeval_c/eval.json"

# ---- 1) B 冒烟（11 分类路径快速验证，fail-closed）
$PY scripts/semeval_finetune.py --task b --weights sqrt --loss ce --limit 300 \
    --epochs 1 --out runs/semeval_smoke_b > /tmp/ddet_sev_smoke_b.log 2>&1 \
    || { step "B 冒烟失败（见 /tmp/ddet_sev_smoke_b.log）"; exit 1; }
grep -q "\[out\]" /tmp/ddet_sev_smoke_b.log || { step "B 冒烟未正常收尾"; exit 1; }
if grep -qi "traceback" /tmp/ddet_sev_smoke_b.log; then step "B 冒烟含 Traceback，中止"; exit 1; fi
step "1/2 B 冒烟通过"

# ---- 2) B 主臂（sqrt 类权重 + CE）
$PY scripts/semeval_finetune.py --task b --weights sqrt --loss ce --aug-ws 0.3 \
    > /tmp/ddet_sev_b_sqrt.log 2>&1 || step "B 主臂失败"
step "2/2 B 完成"

# ---- 3) 汇总：3 臂结果 + 多数类基线 + 榜单参照
step "===== 汇总 ====="
$PY - <<'PY'
import json
from pathlib import Path
import pyarrow.parquet as pq
import numpy as np
from sklearn.metrics import f1_score

print("\n== 我们的结果（3 臂；test 为官方镜像 1K 样本）==")
for tag in ("semeval_a", "semeval_c", "semeval_b"):
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
print("\n注：C-focal、B-inv 两臂按预算决策取消（B/C 各保留一个方案）。")
PY
step "SemEval 精简队列完成"
