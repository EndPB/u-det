#!/usr/bin/env bash
# A×Qwen 验证（用户令"开始"，2026-09-25）：①冻结探针（原始底座）②A 微调 2ep ③汇总。
set -u
cd "$(dirname "$0")/.."
PY=/root/miniconda3/envs/udet/bin/python
export OMP_NUM_THREADS=8
export HF_ENDPOINT=https://hf-mirror.com
export PATH=/root/miniconda3/envs/udet/bin:$PATH

step(){ echo "[aq $(date '+%F %T')] $*"; }

step "A×Qwen 验证启动（磁盘 $(df -BG /root/autodl-tmp | awk 'NR==2{print $4}')）"

# ---------- ① 冻结探针（原始底座，零训练） ----------
if [ ! -f runs/semeval_qwen_probe/a_raw_test.npz ]; then
  $PY scripts/semeval_qwen_probe.py --task a --weights none --tag raw \
    > /tmp/ddet_aq_probe.log 2>&1 || step "探针失败（继续）"
fi
step "① 冻结探针完成"

# ---------- ② A 微调 2ep（Qwen 底座） ----------
if [ ! -f runs/semeval_qwen_a/eval.json ]; then
  $PY scripts/semeval_finetune.py --task a --weights sqrt --loss ce --aug-ws 0.3 \
    --epochs 2 --batch 4 --no-last --config configs/ddet_qwen.yaml \
    --data-dir data/processed/semeval_qwen --init none --lr-encoder 2e-4 \
    --out runs/semeval_qwen_a > /tmp/ddet_aq_ft.log 2>&1 \
    || step "A 微调失败（继续）"
fi
step "② A 微调完成：val=$(python3 -c "
import json; from pathlib import Path
p = Path('runs/semeval_qwen_a/eval.json')
print(round(json.load(p.open())['val']['macro_f1'], 4) if p.exists() else '-')" 2>/dev/null || echo -)"

# ---------- ③ 汇总 ----------
$PY - > /tmp/ddet_aq_summary.txt 2>&1 <<'PY'
import json
from pathlib import Path
print("========== A×Qwen 验证汇总 ==========")
p = Path("runs/semeval_qwen_probe/probe.json")
print("-- 冻结探针（原始底座 + LR）：")
print(p.read_text() if p.exists() else "[缺]")
print("-- A 微调（runs/semeval_qwen_a/eval.json）：")
q = Path("runs/semeval_qwen_a/eval.json")
if q.exists():
    d = json.load(q.open())
    print("  val :", json.dumps(d["val"], ensure_ascii=False)[:500])
    print("  test:", json.dumps(d["test"], ensure_ascii=False)[:500])
else:
    print("  [缺]")
print("-- 对照（CodeT5 时代）：冻结 s1 .6614（AUC .8213）｜微调崩 .3955｜"
      "rvoid 单特征 .6688｜rvoid+s1 融合 .7492（prior22）")
PY
step "③ 汇总 -> /tmp/ddet_aq_summary.txt"
step "=====AQ_DONE====="
