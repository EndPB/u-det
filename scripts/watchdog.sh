#!/usr/bin/env bash
# 夜间看门狗：保证下面两份结果都能产出，中途崩了自动接着跑。
#
#   runs/base_codet5_tok/eval.json   CodeT5 滑动窗口基线（4 epoch）的 test 指标
#   runs/v0.4.2/eval.json            v0.4.2（分块编码器）**4 epoch** 的 test 指标
#                                    （先跑 2 轮 → 再追加 2 轮，与 v0.4.1 @4ep 对齐预算）
#
# ★ 安全前提（宁可不动手，也不乱动手）：
#   1) 只在**确认没有任何任务在跑**时才行动 —— 先查一次，隔 60s 再复核一次。
#      判据是"没有 queue_*.sh 进程"且"没有 python train.py 进程"。
#   2) flock 独占锁（若系统没有 flock 就跳过加锁），保证同时只有一个看门狗。
#   3) 重试次数有上限，避免崩溃死循环。
#   4) 每次行动都写进日志，早上可以完整回溯。
#   5) 它**从不杀进程**，只会在空闲时补跑。
#
# 降级策略：v0.4.2 第一次失败后，重试时自动加 `--block-batch 8`
# （分块编码器每次前向并的块数减半 ⇒ 激活显存减半）。block_batch 不影响数学结果
# （块之间本就独立），只影响显存与吞吐，所以降级后训出来的模型仍然可比。
#
# 启动：setsid nohup bash scripts/watchdog.sh > /tmp/watchdog.log 2>&1 &
set -u
cd /root/autodl-tmp/u-det

PY=/root/miniconda3/envs/udet/bin/python
LOG=/tmp/watchdog.log
PIDFILE=/tmp/watchdog.pid
MAX_TRIES=4
BASE_TAG=base_codet5_tok
BASE_CFG=configs/baseline_codet5_tok.yaml
V042_TAG=v0.4.2
V042_CFG=configs/udet_v042.yaml
V042_EPOCHS=4                     # 目标轮数：先 2 轮，再追加 2 轮
tries=0
beat=0

say() { echo "[wd $(date '+%m-%d %H:%M:%S')] $*" >> "$LOG"; }

active() {
  pgrep -f 'bash scripts/queue_' >/dev/null 2>&1 && return 0
  pgrep -f 'python train\.py' >/dev/null 2>&1 && return 0
  return 1
}

epochs_done() {   # $1 = tag；输出该 run 的 metrics.csv 里出现过的 epoch 数
  awk -F, 'NR>1{print $1}' "runs/$1/metrics.csv" 2>/dev/null | sort -un | wc -l
}

eval_tag() {      # $1 = tag, $2 = config
  say "评估 runs/$1/best.pt"
  OMP_NUM_THREADS=8 "$PY" train.py --config "$2" --tag "$1" \
    --eval --ckpt "runs/$1/best.pt" > "/tmp/$1.eval.log" 2>&1
  if [ -f "runs/$1/eval.json" ]; then
    say "✓ runs/$1/eval.json 已生成"
  else
    say "✗ runs/$1 评估失败（详见 /tmp/$1.eval.log）"
  fi
}

# ------------------------------------------------------------------ #
# 单实例保护：**PID 文件 + `kill -0` 判活**。
#
# 为什么不用另外两种写法（都是 2026-09-20 实测踩到的坑）：
#   1) `flock`：fd 会被子进程（`sleep`）继承。一旦看门狗被 kill，残留的 sleep
#      仍占着锁文件，新看门狗永远起不来（“已有看门狗在运行”）。
#   2) `pgrep -f 'bash scripts/watchdog.sh'`：会被**命令行里提到脚本名**的
#      外层 shell 误命中（启动命令本身就是一串含该脚本名的文本）。
# PID 文件 + `kill -0` 没有这两个问题，且能自动识别/接管陈旧 PID 文件。
if [ -f "$PIDFILE" ]; then
  old=$(cat "$PIDFILE" 2>/dev/null)
  if [ -n "$old" ] && kill -0 "$old" 2>/dev/null; then
    echo "已有看门狗在运行（PID $old），本次退出"
    exit 0
  fi
  echo "发现陈旧 PID 文件（$old 已不存在），接管"
fi
echo $$ > "$PIDFILE"

say "=== 看门狗启动（PID $$）目标：$BASE_TAG 的 eval.json + $V042_TAG 训满 $V042_EPOCHS 轮 ==="

while true; do
  sleep 300
  beat=$((beat + 1))
  if active; then
    [ $((beat % 12)) -eq 0 ] && say "巡检：有任务在跑，继续等待"
    continue
  fi
  sleep 60
  if active; then continue; fi          # 复核：排除两段任务之间的空窗

  base_ok=0; [ -f "runs/$BASE_TAG/eval.json" ] && base_ok=1
  v042_ep=$(epochs_done "$V042_TAG")
  v042_ok=0
  [ -f "runs/$V042_TAG/eval.json" ] && [ "$v042_ep" -ge "$V042_EPOCHS" ] && v042_ok=1
  if [ "$base_ok" = 1 ] && [ "$v042_ok" = 1 ]; then
    say "=== 两份结果都已产出（v0.4.2 $v042_ep 轮），看门狗正常退出 ==="
    exit 0
  fi
  say "空闲且结果不全（基线=$base_ok v0.4.2=$v042_ok，已完成 $v042_ep 轮）→ 开始补救"

  # ① 基线：先把 epoch 补到 4，再补评估
  if [ "$base_ok" = 0 ]; then
    ep=$(epochs_done "$BASE_TAG")
    if [ "$ep" -lt 4 ] && [ -f "runs/$BASE_TAG/last.pt" ]; then
      say "基线只完成 $ep 轮，续跑到 4 轮"
      OMP_NUM_THREADS=8 "$PY" train.py --config "$BASE_CFG" --tag "$BASE_TAG" \
        --resume "runs/$BASE_TAG/last.pt" --epochs $((4 - ep)) \
        > "/tmp/${BASE_TAG}.retry.log" 2>&1
    fi
    [ -f "runs/$BASE_TAG/best.pt" ] && eval_tag "$BASE_TAG" "$BASE_CFG"
  fi

  # ② v0.4.2：训练到 $V042_EPOCHS 轮（必要时降级）→ 评估
  if [ "$v042_ok" = 0 ]; then
    ep=$v042_ep
    if [ "$ep" -ge "$V042_EPOCHS" ]; then
      say "v0.4.2 已训满 $ep 轮，只需评估"
    else
      left=$((V042_EPOCHS - ep)); [ "$left" -lt 1 ] && left=1
      extra=""
      if [ "$tries" -ge 1 ]; then
        extra="--block-batch 8"
        say "第 $((tries + 1)) 次尝试：加 --block-batch 8（激活显存减半）"
      fi
      if [ "$ep" -ge 1 ] && [ -f "runs/$V042_TAG/last.pt" ]; then
        say "v0.4.2 续跑 $left 轮（已完成 $ep）"
        OMP_NUM_THREADS=8 "$PY" train.py --config "$V042_CFG" --tag "$V042_TAG" \
          --resume "runs/$V042_TAG/last.pt" --epochs "$left" $extra \
          > "/tmp/${V042_TAG}.retry${tries}.log" 2>&1
      else
        say "v0.4.2 从头跑 $V042_EPOCHS 轮"
        OMP_NUM_THREADS=8 "$PY" train.py --config "$V042_CFG" --tag "$V042_TAG" \
          --epochs "$V042_EPOCHS" $extra > "/tmp/${V042_TAG}.retry${tries}.log" 2>&1
      fi
    fi
    [ -f "runs/$V042_TAG/best.pt" ] && eval_tag "$V042_TAG" "$V042_CFG"
    if [ ! -f "runs/$V042_TAG/eval.json" ] || [ "$(epochs_done "$V042_TAG")" -lt "$V042_EPOCHS" ]; then
      tries=$((tries + 1))
      if [ "$tries" -ge "$MAX_TRIES" ]; then
        say "=== 已达最大重试次数 $MAX_TRIES，看门狗退出（请人工检查 /tmp/${V042_TAG}.retry*.log）==="
        exit 1
      fi
    fi
  fi

  sleep 60
done
