#!/usr/bin/env bash
# 补跑队列（2026-09-25 上午）：收 qwen-C → DS-B → 7B（冒烟门）→ 终版集成 → 汇总。
# 说明：qwen-C 已在跑（本脚本只等）；DS 用新分词的 semeval_ds；7B 已在 config 注入 auto_trainable。
set -u
cd "$(dirname "$0")/.."
PY=/root/miniconda3/envs/udet/bin/python
export OMP_NUM_THREADS=8
export HF_ENDPOINT=https://hf-mirror.com
export PATH=/root/miniconda3/envs/udet/bin:$PATH

step(){ echo "[fin $(date '+%F %T')] $*"; }
free_gb(){ df -BG /root/autodl-tmp 2>/dev/null | awk 'NR==2{print $4}' | tr -d 'G'; }
run_val(){ $PY - "$1" <<'PY' 2>/dev/null || echo "-"
import json,sys
from pathlib import Path
p=Path(sys.argv[1])/"eval.json"
print(json.load(open(p))["val"]["macro_f1"] if p.exists() else "-")
PY
}
have_probs(){ [ -f "$1/probs_val.npz" ] && [ -f "$1/probs_test.npz" ]; }

step "补跑队列启动（磁盘 $(free_gb)GB）"

# ---------- S1：等 qwen-C（进程消失或 eval.json 出现即继续；上限 5h） ----------
for i in $(seq 1 300); do
  [ -f runs/semeval_qwen_c/eval.json ] && break
  pgrep -f "out runs/semeval_qwen_c" >/dev/null 2>&1 || break
  sleep 60
done
step "S1 qwen-C 结束：val=$(run_val runs/semeval_qwen_c)（磁盘 $(free_gb)GB）"

# ---------- S2：qwen-C probs 兜底 + 清盘 ----------
if [ -f runs/semeval_qwen_c/best.pt ] && ! have_probs runs/semeval_qwen_c; then
  $PY scripts/semeval_dump_run.py --run runs/semeval_qwen_c --task c \
    --config configs/ddet_qwen.yaml --data-dir data/processed/semeval_qwen \
    --init none --tag qwen_c > /tmp/ddet_fin_dumpqwc.log 2>&1 \
    || step "S2 dump qwen_c 失败（保留检查点）"
fi
if have_probs runs/semeval_qwen_c; then
  find runs/semeval_qwen_c -name "*.pt" -delete 2>/dev/null
  step "S2 清盘：qwen_c 检查点已删（probs 保留，磁盘 $(free_gb)GB）"
fi

# ---------- S3：DS-1.3B B 臂（semeval_ds 分词；支持断电断点续传） ----------
if [ -f runs/semeval_ds_b/best.pt ] && [ ! -f runs/semeval_ds_b/eval.json ]; then
  step "S3 恢复模式：DS 训练已完成但评测被中断，从 best.pt 补评测"
  $PY scripts/semeval_dump_run.py --run runs/semeval_ds_b --task b \
    --config configs/ddet_ds.yaml --data-dir data/processed/semeval_ds \
    --init none --tag ds_b > /tmp/ddet_fin_dumpds.log 2>&1 \
    || step "S3 恢复导出失败（继续）"
  $PY - <<'PY' || step "S3 eval.json 合成失败"
import json
import numpy as np
from pathlib import Path
run = Path("runs/semeval_ds_b")
out = {}
for split in ("val", "test"):
    d = np.load(run / f"probs_{split}.npz")
    p, y = d["probs"], d["y"]
    pred = p.argmax(1)
    f1s = []
    for c in sorted(set(y.tolist())):
        tp = ((pred == c) & (y == c)).sum()
        fp = ((pred == c) & (y != c)).sum()
        fn = ((pred != c) & (y == c)).sum()
        f1s.append(2 * tp / max(2 * tp + fp + fn, 1))
    out[split] = {"macro_f1": float(np.mean(f1s)),
                  "acc": float((pred == y).mean()), "n": int(len(y))}
json.dump({"task": "b", "recovered": True, "val": out["val"], "test": out["test"]},
          open(run / "eval.json", "w", encoding="utf-8"), ensure_ascii=False, indent=2)
print("DS 恢复评测 OK", out)
PY
elif [ -f data/processed/semeval_ds/b_train.parquet ] && [ ! -f runs/semeval_ds_b/eval.json ]; then
  $PY scripts/semeval_finetune.py --task b --weights sqrt --loss ce --aug-ws 0.3 \
    --epochs 2 --batch 4 --no-last --config configs/ddet_ds.yaml \
    --data-dir data/processed/semeval_ds --init none --lr-encoder 2e-4 \
    --out runs/semeval_ds_b > /tmp/ddet_fin_ds.log 2>&1 || step "S3 DS 训练失败（继续）"
fi
if have_probs runs/semeval_ds_b; then
  find runs/semeval_ds_b -name "*.pt" -delete 2>/dev/null
fi
step "S3 DS 结束：val=$(run_val runs/semeval_ds_b)（磁盘 $(free_gb)GB）"

# ---------- S4：7B（默认跳过；用户令 2026-09-25"7B 暂时不跑"。要跑: RUN_7B=1 bash queue_finish.sh） ----------
if [ "${RUN_7B:-0}" = "1" ] && [ -f checkpoints/qwen2.5-coder-7b-instruct-gptq-int4/config.json ]; then
  $PY scripts/check_qwen.py configs/ddet_qwen7b.yaml > /tmp/ddet_fin_chk7b.log 2>&1 || true
  if grep -q "QWEN_SMOKE_PASS" /tmp/ddet_fin_chk7b.log; then
    if [ ! -f runs/semeval_qwen7b_b/eval.json ]; then
      $PY scripts/semeval_finetune.py --task b --weights sqrt --loss ce --aug-ws 0.3 \
        --epochs 2 --batch 2 --no-last --config configs/ddet_qwen7b.yaml \
        --data-dir data/processed/semeval_qwen --init none --lr-encoder 1e-4 \
        --out runs/semeval_qwen7b_b > /tmp/ddet_fin_7b.log 2>&1 \
        || step "S4 7B 训练失败（继续）"
    fi
  else
    step "S4 7B 已跳过（RUN_7B!=1；用户令暂时不跑）"
  fi
fi
step "S4 7B 结束：val=$(run_val runs/semeval_qwen7b_b)"

# ---------- S5：7B probs 兜底 + 清盘 ----------
if [ -f runs/semeval_qwen7b_b/best.pt ] && ! have_probs runs/semeval_qwen7b_b; then
  $PY scripts/semeval_dump_run.py --run runs/semeval_qwen7b_b --task b \
    --config configs/ddet_qwen7b.yaml --data-dir data/processed/semeval_qwen \
    --init none --tag qwen7b > /tmp/ddet_fin_dump7b.log 2>&1 \
    || step "S5 dump 7B 失败（保留检查点）"
fi
if have_probs runs/semeval_qwen7b_b; then
  find runs/semeval_qwen7b_b runs/semeval_ds_b -name "*.pt" -delete 2>/dev/null
  step "S5 清盘完成（磁盘 $(free_gb)GB）"
fi

# ---------- S6：终版集成（吸收 qwen_c / ds_b / qwen7b 成员） ----------
$PY scripts/semeval_ensemble_v2.py > /tmp/ddet_fin_ens.log 2>&1 || step "S6 集成失败"
[ -f runs/semeval_ensemble/results.json ] && \
  cp runs/semeval_ensemble/results.json runs/semeval_ensemble/results_final.json
step "S6 集成完成"

# ---------- S7：汇总 ----------
$PY - > /tmp/ddet_fin_summary.txt 2>&1 <<'PY'
import json
from pathlib import Path
print("========== 补跑队列汇总 ==========")
for f in ("runs/semeval_ensemble/results_final.json",):
    p = Path(f)
    print("=" * 20, f)
    print(p.read_text() if p.exists() else "[缺]")
print("=" * 20, "各臂 val/test")
for tag in ("semeval_b", "semeval_b_big", "semeval_qwen_b", "semeval_qwen_b_big",
            "semeval_qwen_c", "semeval_qwen7b_b", "semeval_ds_b"):
    p = Path(f"runs/{tag}/eval.json")
    if p.exists():
        d = json.load(open(p))
        print(f"[{tag}] val={d['val']['macro_f1']:.4f} test={d['test']['macro_f1']:.4f}")
    else:
        print(f"[{tag}] [无]")
PY
step "S7 汇总 -> /tmp/ddet_fin_summary.txt"
step "=====FIN_DONE===== 磁盘 $(free_gb)GB"
