#!/usr/bin/env bash
# 总排程（用户令"全部排上"，2026-09-25 深夜）：
#   S0 等 Qwen-B 链 → S1 旧臂 probs 导出+清盘 → S2 大数据/特征/ngram/子空间v2
#   S3 CodeT5 大数据臂 → S4 Qwen 大数据臂 → S5 集成(中途) → S6 Qwen-C(条件)
#   S7 DS-1.3B 尾臂(条件) → S8 清盘(消费完毕的检查点) → S9 7B QLoRA(磁盘条件)
#   S10 终版集成 + 汇总。全部 fail-soft：单步失败不阻断后续。
set -u
cd "$(dirname "$0")/.."
PY=/root/miniconda3/envs/udet/bin/python
export OMP_NUM_THREADS=8
export HF_ENDPOINT=https://hf-mirror.com
export HF_HUB_ENABLE_HF_TRANSFER=0
LOG=/tmp/ddet_all_chain.log

step(){ echo "[all $(date '+%F %T')] $*"; }
free_gb(){ df -BG /root/autodl-tmp 2>/dev/null | awk 'NR==2{print $4}' | tr -d 'G'; }
run_val(){ $PY - "$1" <<'PY' 2>/dev/null || echo "-"
import json,sys
from pathlib import Path
p=Path(sys.argv[1])/"eval.json"
print(json.load(open(p))["val"]["macro_f1"] if p.exists() else "-")
PY
}
have_probs(){ [ -f "$1/probs_val.npz" ] && [ -f "$1/probs_test.npz" ]; }

step "===== 总链启动，磁盘可用 $(free_gb)GB ====="

# ---------- S0：等前序（Qwen 子链 230960 / 夜间重试 230526；均可能已结束） ----------
for i in $(seq 1 200); do
  [ -f runs/semeval_qwen_b/eval.json ] && break
  kill -0 230960 2>/dev/null || break
  sleep 90
done
for i in $(seq 1 60); do
  kill -0 230526 2>/dev/null || break
  sleep 60
done
# 双保险：只要机器上还有 semeval_finetune 在跑（任何来源），继续等（上限 2h）
for i in $(seq 1 120); do
  pgrep -f "semeval_finetune.py" >/dev/null 2>&1 || break
  sleep 60
done
QWB=$(run_val runs/semeval_qwen_b)
step "S0 完成：前序清空，Qwen-B val=${QWB}（磁盘 $(free_gb)GB）"

# ---------- S1：旧臂 probs 导出 + 清 .pt ----------
$PY scripts/semeval_dump_old.py > /tmp/ddet_all_dumpold.log 2>&1 \
  || step "S1a dump_old 失败（继续）"
for tag in b_r4 b_r8 b_r11 b_r8init; do
  if [ -f "runs/semeval_r/probs_${tag}_val.npz" ]; then
    find "runs/semeval_${tag}" -name "*.pt" -delete 2>/dev/null
  fi
done
find runs/semeval_qwen_smoke -name "*.pt" -delete 2>/dev/null
step "S1 完成：旧臂 probs 已导出、.pt 已清（磁盘 $(free_gb)GB）"

# ---------- S2：CPU 阶段后台先行（大数据 + stats2 + ngram），与 S0b 重叠 ----------
(
  $PY scripts/semeval_prepare_big.py > /tmp/ddet_all_big.log 2>&1 || echo "[cpu] prepare_big 失败"
  $PY scripts/semeval_stats2.py > /tmp/ddet_all_stats2.log 2>&1 || echo "[cpu] stats2 失败"
  $PY scripts/semeval_ngram.py > /tmp/ddet_all_ngram.log 2>&1 || echo "[cpu] ngram 失败"
) > /tmp/ddet_all_cpu.log 2>&1 &
CPU_PID=$!
step "S2-CPU 已后台启动（pid=$CPU_PID）"

# ---------- S0b：Qwen 基建（数据重分词 → 冒烟 → Qwen-B 2ep） ----------
if [ ! -f data/processed/semeval_qwen/b_train.parquet ]; then
  $PY scripts/semeval_prepare_qwen.py > /tmp/ddet_all_prepqw.log 2>&1 \
    || step "S0b-1 prepare_qwen 失败（跳过 Qwen 系）"
fi
if [ ! -f runs/semeval_qwen_b/eval.json ]; then
  $PY scripts/check_qwen.py configs/ddet_qwen.yaml > /tmp/ddet_all_chkqw.log 2>&1 \
    || step "S0b-2 check_qwen 失败（尝试继续）"
  QW_OK=0
  grep -q "QWEN_SMOKE_PASS" /tmp/ddet_all_chkqw.log && QW_OK=1
else
  QW_OK=0
  step "S0b 前置跳过：Qwen-B 已由前序队列产出"
fi
if [ "$QW_OK" = "1" ] && [ -f data/processed/semeval_qwen/b_train.parquet ]; then
  if [ ! -f runs/semeval_qwen_b/eval.json ]; then
    $PY scripts/semeval_finetune.py --task b --limit 64 --epochs 1 --batch 2 --no-last \
      --config configs/ddet_qwen.yaml --data-dir data/processed/semeval_qwen \
      --init none --lr-encoder 2e-4 --out runs/semeval_qwen_smoke \
      > /tmp/ddet_all_smokeqw.log 2>&1 || true
    if grep -q "\[final\]" /tmp/ddet_all_smokeqw.log; then
      $PY scripts/semeval_finetune.py --task b --weights sqrt --loss ce --aug-ws 0.3 \
        --epochs 2 --batch 4 --no-last --config configs/ddet_qwen.yaml \
        --data-dir data/processed/semeval_qwen --init none --lr-encoder 2e-4 \
        --out runs/semeval_qwen_b > /tmp/ddet_all_qwb.log 2>&1 \
        || step "S0b-3 Qwen-B 训练失败（继续）"
    else
      step "S0b-3 Qwen 训练冒烟未过（跳过 Qwen-B）"
    fi
  fi
fi
step "S0b 完成：Qwen-B val=$(run_val runs/semeval_qwen_b)"

# ---------- S2 收尾：等 CPU 后台 + 子空间 v2（GPU） ----------
wait $CPU_PID 2>/dev/null
$PY scripts/semeval_subspace_v2.py > /tmp/ddet_all_sub2.log 2>&1 \
  || step "S2d subspace_v2 失败（继续）"
step "S2 完成：数据/特征/ngram/子空间v2（磁盘 $(free_gb)GB）"

# ---------- S3：CodeT5 大数据臂（60K 训练 1ep ≈ 原 B 的 1.65 倍暴露） ----------
if [ -f data/processed/semeval_big/b_train.parquet ] && [ ! -f runs/semeval_b_big/eval.json ]; then
  $PY scripts/semeval_finetune.py --task b --weights sqrt --loss ce --aug-ws 0.3 \
    --epochs 1 --no-last --data-dir data/processed/semeval_big \
    --out runs/semeval_b_big > /tmp/ddet_all_ctxbig.log 2>&1 \
    || step "S3 ctx-big 失败（继续）"
fi
step "S3 完成：ctx-big val=$(run_val runs/semeval_b_big)（磁盘 $(free_gb)GB）"

# ---------- S4：Qwen 大数据臂（1ep，LoRA） ----------
if [ -f data/processed/semeval_qwen_big/b_train.parquet ] && [ ! -f runs/semeval_qwen_b_big/eval.json ]; then
  $PY scripts/semeval_finetune.py --task b --weights sqrt --loss ce --aug-ws 0.3 \
    --epochs 1 --batch 4 --no-last --config configs/ddet_qwen.yaml \
    --data-dir data/processed/semeval_qwen_big --init none --lr-encoder 2e-4 \
    --out runs/semeval_qwen_b_big > /tmp/ddet_all_qwbig.log 2>&1 \
    || step "S4 qwen-big 失败（继续）"
fi
step "S4 完成：qwen-big val=$(run_val runs/semeval_qwen_b_big)（磁盘 $(free_gb)GB）"

# ---------- S5：集成（中途，先落袋） ----------
$PY scripts/semeval_ensemble_v2.py > /tmp/ddet_all_ens1.log 2>&1 \
  || step "S5 集成失败（继续）"
[ -f runs/semeval_ensemble/results.json ] && \
  cp runs/semeval_ensemble/results.json runs/semeval_ensemble/results_pass1.json
step "S5 完成：集成(中途)"

# ---------- S6：Qwen-C（条件：任一 Qwen-B val≥0.50） ----------
QMAX=$($PY - <<'PY' 2>/dev/null || echo 0
import json
from pathlib import Path
vals=[]
for p in ("runs/semeval_qwen_b/eval.json","runs/semeval_qwen_b_big/eval.json"):
    if Path(p).exists(): vals.append(json.load(open(p))["val"]["macro_f1"])
print(max(vals) if vals else 0)
PY
)
if $PY -c "import sys; sys.exit(0 if float('$QMAX')>=0.50 else 1)" 2>/dev/null; then
  $PY scripts/semeval_finetune.py --task c --weights sqrt --loss ce --aug-ws 0.3 \
    --epochs 2 --batch 4 --no-last --config configs/ddet_qwen.yaml \
    --data-dir data/processed/semeval_qwen --init none --lr-encoder 2e-4 \
    --out runs/semeval_qwen_c > /tmp/ddet_all_qwc.log 2>&1 \
    || step "S6 qwen-C 失败（继续）"
  step "S6 完成：qwen-C val=$(run_val runs/semeval_qwen_c)"
else
  step "S6 跳过：Qwen 最高 val=$QMAX < 0.50"
fi

# ---------- S7：DeepSeek-Coder-1.3B 尾臂 ----------
if [ -d checkpoints/deepseek-coder-1.3b-instruct ] && [ ! -f runs/semeval_ds_b/eval.json ]; then
  $PY scripts/semeval_finetune.py --task b --weights sqrt --loss ce --aug-ws 0.3 \
    --epochs 2 --batch 4 --no-last --config configs/ddet_ds.yaml \
    --data-dir data/processed/semeval_qwen --init none --lr-encoder 2e-4 \
    --out runs/semeval_ds_b > /tmp/ddet_all_ds.log 2>&1 \
    || step "S7 DS 臂失败（继续）"
fi
step "S7 完成：DS 臂 val=$(run_val runs/semeval_ds_b)（磁盘 $(free_gb)GB）"

# ---------- S8：清盘（仅清"probs 已落地"的检查点） ----------
# S8a Qwen-B 兜底导出（若跑在补丁生效前，无 probs 时补导）
if [ -f runs/semeval_qwen_b/best.pt ] && ! have_probs runs/semeval_qwen_b; then
  $PY scripts/semeval_dump_run.py --run runs/semeval_qwen_b --task b \
    --config configs/ddet_qwen.yaml --data-dir data/processed/semeval_qwen \
    --init none --tag qwen_b > /tmp/ddet_all_dumpqwb.log 2>&1 \
    || step "S8a qwen_b 补导失败（保留检查点）"
fi
# S8b 删除已消费的大检查点（probs 已入集成；best 重训可复原）
for d in semeval_qwen_b semeval_qwen_b_big; do
  if have_probs "runs/$d"; then
    find "runs/$d" -name "*.pt" -delete 2>/dev/null
    step "S8 清盘：runs/$d 检查点已删（probs 保留）"
  fi
done
find runs/semeval_b_big runs/semeval_qwen_c runs/semeval_ds_b runs/semeval_qwen7b_b \
     -name "last.pt" -delete 2>/dev/null
step "S8 完成（磁盘 $(free_gb)GB）"

# ---------- S9：Qwen2.5-Coder-7B QLoRA（条件：磁盘 ≥9GB） ----------
FREE=$(free_gb)
if [ "$FREE" -ge 9 ] 2>/dev/null; then
  HFCLI=/root/miniconda3/envs/udet/bin/huggingface-cli
  if [ ! -f checkpoints/qwen2.5-coder-7b-instruct-gptq-int4/config.json ] && [ -x "$HFCLI" ]; then
    $HFCLI download Qwen/Qwen2.5-Coder-7B-Instruct-GPTQ-Int4 \
      --local-dir checkpoints/qwen2.5-coder-7b-instruct-gptq-int4 \
      > /tmp/ddet_all_dl7b.log 2>&1 || step "S9a 7B 下载失败（跳过）"
  fi
  if [ -f checkpoints/qwen2.5-coder-7b-instruct-gptq-int4/config.json ]; then
    $PY -c "import gptqmodel" 2>/dev/null \
      || $PY -m pip install -q gptqmodel > /tmp/ddet_all_gptqpip.log 2>&1 \
      || step "S9b gptqmodel 安装失败（试跑）"
    $PY scripts/check_qwen.py configs/ddet_qwen7b.yaml > /tmp/ddet_all_chk7b.log 2>&1 \
      || step "S9c 7B 冒烟失败（尝试继续）"
    if grep -q "QWEN_SMOKE_PASS" /tmp/ddet_all_chk7b.log; then
      $PY scripts/semeval_finetune.py --task b --weights sqrt --loss ce --aug-ws 0.3 \
        --epochs 2 --batch 2 --no-last --config configs/ddet_qwen7b.yaml \
        --data-dir data/processed/semeval_qwen --init none --lr-encoder 1e-4 \
        --out runs/semeval_qwen7b_b > /tmp/ddet_all_qw7b.log 2>&1 \
        || step "S9d 7B 主臂失败（继续）"
      step "S9 完成：qwen7b val=$(run_val runs/semeval_qwen7b_b)"
    else
      step "S9 跳过：7B 冒烟未过"
    fi
  else
    step "S9 跳过：7B 权重不可用"
  fi
else
  step "S9 跳过：磁盘 ${FREE}GB < 9GB"
fi

# ---------- S10：终版集成 + 汇总 ----------
$PY scripts/semeval_ensemble_v2.py > /tmp/ddet_all_ens2.log 2>&1 \
  || step "S10a 终版集成失败"
find runs/semeval_qwen_smoke -name "*.pt" -delete 2>/dev/null
$PY - > /tmp/ddet_all_summary.txt 2>&1 <<'PY'
import json
from pathlib import Path
print("========== 「全部排上」总链汇总（%s） ==========" % Path('/tmp/ddet_all_chain.log').stat().st_mtime)
for f in ("runs/semeval_ensemble/results.json", "runs/semeval_r/subspace_v2.json",
          "runs/semeval_r/showcase.json", "runs/semeval_stack/results.json"):
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
step "S10 完成：汇总 -> /tmp/ddet_all_summary.txt"
step "=====ALL_DONE===== 磁盘可用 $(free_gb)GB"
