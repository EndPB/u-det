#!/usr/bin/env bash
# v0.4 几何损失衔接队列（用户指令：4ep 完成后自动衔接）
#   0) 等 v0.3.0 续跑（4ep）完成   1) margin 标定（读 4ep dump）
#   2) GPU 冒烟（--limit 100，fail-closed）   3) v0.4.0 主实验（2ep，基座=4ep last.pt）
#   4) v0.4.0 评测+探针   5) v0.4.1 对照臂（+covreg）   6) v0.4.1 评测+探针   7) 汇总
set -u
cd "$(dirname "$0")/.."
PY=/root/miniconda3/envs/udet/bin/python
export OMP_NUM_THREADS=8
export HF_ENDPOINT=https://hf-mirror.com
step() { echo "[v040q] $(date '+%F %T') $1"; }

# ---- 0) 等待 4ep 续跑完成（PID 死亡 + 收尾标记/日志收尾；最多 6h，软超时）
EXT_PID=${EXT_PID:-68627}
[ -f /tmp/ddet_v030_ext.pid ] && EXT_PID=$(cat /tmp/ddet_v030_ext.pid)
step "0/7 等待 v0.3.0 续跑（4ep）完成（PID=$EXT_PID）…"
done_=0
for _ in $(seq 1 720); do
  if ! kill -0 "$EXT_PID" 2>/dev/null; then
    if grep -q "续跑完成" /tmp/ddet_v030_ext.log 2>/dev/null || grep -q '"epoch": 3' runs/v0.3.0_fullft/eval.json 2>/dev/null; then
      done_=1; break
    fi
  fi
  sleep 30
done
if [ "$done_" != "1" ]; then
  if [ -f runs/v0.3.0_fullft/last.pt ]; then
    step "软超时：未见收尾标记，但 last.pt 存在，继续衔接（请复核 ext 日志）"
  else
    step "超时且无 last.pt，放弃衔接"; exit 1
  fi
fi
step "4ep 完成，衔接开始"

# ---- 1) margin 标定（固定 margin 斥力；失败用配置默认值）
MARGIN=$($PY scripts/calibrate_margin.py 2>/dev/null | grep -oP 'MARGIN=\K[0-9.]+' | tail -1)
if [ -z "$MARGIN" ]; then MARGIN=1.5; step "标定失败，用默认 margin=$MARGIN"; else step "1/7 margin=$MARGIN"; fi

# ---- 2) GPU 冒烟（fail-closed：必须正常收尾且无 NaN）
$PY train.py --config configs/ddet_v040.yaml --tag v040_smoke --limit 100 --epochs 1 \
    --set loss.rep_margin=$MARGIN > /tmp/ddet_v040_smoke.log 2>&1 \
    || { step "冒烟失败（见 /tmp/ddet_v040_smoke.log）"; exit 1; }
grep -q "完成，产物目录" /tmp/ddet_v040_smoke.log || { step "冒烟异常而未正常收尾"; exit 1; }
if grep -qi "traceback" /tmp/ddet_v040_smoke.log; then step "冒烟含 Traceback，中止"; exit 1; fi
if [ -f runs/v040_smoke/metrics.csv ] && grep -qi "nan" runs/v040_smoke/metrics.csv; then
  step "冒烟日志出现 NaN，中止"; exit 1
fi
step "2/7 冒烟通过"

# ---- 3) v0.4.0 主实验（rep + fisher + xfam；2ep）
$PY train.py --config configs/ddet_v040.yaml --tag v0.4.0_geom \
    --resume runs/v0.3.0_fullft/last.pt --epochs 2 --set loss.rep_margin=$MARGIN \
    > /tmp/ddet_v040_r0.log 2>&1 || { step "v0.4.0 训练失败（见 /tmp/ddet_v040_r0.log）"; exit 1; }
step "3/7 v0.4.0 训练完成"

# ---- 4) v0.4.0 评测 + 探针（ckpt 保底：best 不存在时回退 last）
CKPT0=runs/v0.4.0_geom/best.pt; [ -f "$CKPT0" ] || CKPT0=runs/v0.4.0_geom/last.pt
$PY train.py --config configs/ddet_v040.yaml --eval --ckpt $CKPT0 --dump-scores \
    > /tmp/ddet_v040_eval0.log 2>&1 || step "v0.4.0 评测失败"
$PY scripts/probe_fragments.py --config configs/ddet_v040.yaml \
    --ckpt $CKPT0 --n 500 --dump-samples \
    > /tmp/ddet_v040_frag0.log 2>&1 || step "v0.4.0 片段探针失败"
$PY scripts/probe_scale_verify.py --config configs/ddet_v040.yaml \
    --ckpt $CKPT0 --tag v0.4.0_geom \
    --pair-file data/processed/pairs_qwen15.parquet --npz runs/v0.4.0_geom/scale_scores.npz \
    > /tmp/ddet_v040_scale0.log 2>&1 || step "v0.4.0 跨规模探针失败"
if [ -f data/processed/pairs_ds13.parquet ]; then
  $PY scripts/probe_scale_verify.py --config configs/ddet_v040.yaml \
      --ckpt $CKPT0 --tag v0.4.0_geom_ds13 \
      --pair-file data/processed/pairs_ds13.parquet --npz runs/v0.4.0_geom/scale_scores_ds13.npz \
      > /tmp/ddet_v040_scale_ds13_0.log 2>&1 || step "v0.4.0 ds13 探针失败"
fi
step "4/7 v0.4.0 评测+探针完成"

# ---- 5) v0.4.1 对照臂（v0.4.0 + covreg；独立从同一 4ep 基座）
$PY train.py --config configs/ddet_v041.yaml --tag v0.4.1_covreg \
    --resume runs/v0.3.0_fullft/last.pt --epochs 2 --set loss.rep_margin=$MARGIN \
    > /tmp/ddet_v040_r1.log 2>&1 || { step "v0.4.1 训练失败（见 /tmp/ddet_v040_r1.log）"; exit 1; }
step "5/7 v0.4.1 训练完成"

# ---- 6) v0.4.1 评测 + 探针（ckpt 保底：best 不存在时回退 last）
CKPT1=runs/v0.4.1_covreg/best.pt; [ -f "$CKPT1" ] || CKPT1=runs/v0.4.1_covreg/last.pt
$PY train.py --config configs/ddet_v041.yaml --eval --ckpt $CKPT1 --dump-scores \
    > /tmp/ddet_v040_eval1.log 2>&1 || step "v0.4.1 评测失败"
$PY scripts/probe_fragments.py --config configs/ddet_v041.yaml \
    --ckpt $CKPT1 --n 500 --dump-samples \
    > /tmp/ddet_v040_frag1.log 2>&1 || step "v0.4.1 片段探针失败"
$PY scripts/probe_scale_verify.py --config configs/ddet_v041.yaml \
    --ckpt $CKPT1 --tag v0.4.1_covreg \
    --pair-file data/processed/pairs_qwen15.parquet --npz runs/v0.4.1_covreg/scale_scores.npz \
    > /tmp/ddet_v040_scale1.log 2>&1 || step "v0.4.1 跨规模探针失败"
if [ -f data/processed/pairs_ds13.parquet ]; then
  $PY scripts/probe_scale_verify.py --config configs/ddet_v041.yaml \
      --ckpt $CKPT1 --tag v0.4.1_covreg_ds13 \
      --pair-file data/processed/pairs_ds13.parquet --npz runs/v0.4.1_covreg/scale_scores_ds13.npz \
      > /tmp/ddet_v040_scale_ds13_1.log 2>&1 || step "v0.4.1 ds13 探针失败"
fi
step "6/7 v0.4.1 评测+探针完成"

# ---- 7) 汇总（4ep 基座 / v0.4.0 / v0.4.1）
step "===== 汇总（v0.3.0_fullft 4ep / v0.4.0_geom / v0.4.1_covreg）====="
$PY - <<'PY'
import json
from pathlib import Path
for tag in ("v0.3.0_fullft", "v0.4.0_geom", "v0.4.1_covreg"):
    p = Path("runs") / tag / "eval.json"
    if not p.exists():
        print("[summary]", tag, "缺 eval.json")
        continue
    m = json.load(open(p))["metrics"]
    for k, v in m.items():
        row = {kk: (round(vv, 4) if isinstance(vv, float) else vv) for kk, vv in v.items()}
        print("[summary]", tag, k, row)
PY
step "v0.4 衔接队列完成"
