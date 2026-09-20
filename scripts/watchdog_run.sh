#!/usr/bin/env bash
# 通用夜间看门狗：盯住**一个** run，保证它训满 N 轮、并且**重新**产出一份
# 与该 ckpt 对应的 eval.json（可选同时 `--dump-raw` 落盘原始概率供离线分析）。
#
# 用法：
#   bash scripts/watchdog_run.sh <TAG> <CONFIG> <EPOCHS> [--dump-raw]
#
# 例（夜间启动，与训练队列互不干扰）：
#   setsid nohup bash scripts/watchdog_run.sh v0.4.5 configs/udet_v045.yaml 4 --dump-raw \
#     > /tmp/watchdog_v045.log 2>&1 &
#
# 与旧版 `scripts/watchdog.sh` 的区别：旧版的目标（base_codet5_tok + v0.4.2）是
# 写死的、产出即退出；本脚本目标由参数给出，可复用。旧脚本保留作为那次任务的存档。
#
# ★ 安全前提（与旧版一致，来历见 docx/lessons.md D 类）：
#   1) **只在确认没有任何任务在跑时才动手** —— 先查一次，隔 60s 再复核一次，
#      排除两段任务之间的空窗。判据：没有 `bash scripts/queue_*` 且没有 `python train.py`。
#   2) **PID 文件 + `kill -0`** 做单例保护。不用 `flock`（fd 会被子进程继承，
#      看门狗被 kill 后残留的 sleep 会永久占锁）、不用 `pgrep -f watchdog`（会被
#      命令行里提到脚本名的外层 shell 自匹配）。两者都是 2026-09-20 实测踩到的坑。
#   3) 重试次数有上限，避免崩溃死循环。
#   4) 每次行动都写进日志，早上可完整回溯。
#   5) **它从不杀进程**，只在空闲时补跑。
#
# ★ 为什么续跑用 `--resume last.pt --epochs <left>`：
#   train.py 里 `start_epoch = resume_epoch + 1`，见 docx/lessons.md D3。
#   所以补 `left` 轮后累计恰好等于目标轮数，不会多跑也不会少跑。
set -u
cd /root/autodl-tmp/u-det

TAG=${1:?用法: watchdog_run.sh <TAG> <CONFIG> <EPOCHS> [--dump-raw]}
CFG=${2:?缺少 CONFIG}
EPOCHS=${3:?缺少 EPOCHS}
DUMP=${4:-}

PY=/root/miniconda3/envs/udet/bin/python
PIDFILE="/tmp/watchdog_${TAG}.pid"
MAX_TRIES=4
tries=0
beat=0

say() { echo "[wd:$TAG $(date '+%m-%d %H:%M:%S')] $*"; }

active() {
  pgrep -f 'bash scripts/queue_' >/dev/null 2>&1 && return 0
  pgrep -f 'python train\.py' >/dev/null 2>&1 && return 0
  return 1
}

# 已完成轮数。
#
# ⚠️ **不能数 metrics.csv 里的 distinct epoch**：csv 是每个 log_every 写一行的，
# 进行中的 epoch 也会被写进去，于是"跑到 epoch 2 的 20%"会被数成 3 轮，
# 看门狗就会少补一轮（要等下一次重试才自愈）。
# 权威来源是 last.pt 里的 `epoch` 字段 = 最后一次**完整跑完**的 epoch 下标，
# 所以已完成轮数 = epoch + 1。ckpt 读不出来时才退回数 csv。
epochs_done() {
  local e
  e=$(OMP_NUM_THREADS=1 "$PY" -c "
import torch
c = torch.load('runs/$1/last.pt', map_location='cpu', weights_only=False)
print(int(c.get('epoch', -1)) + 1)
" 2>/dev/null | tail -1)
  case "$e" in ''|*[!0-9]*) e=-1 ;; esac
  if [ "$e" -lt 0 ]; then
    awk -F, 'NR>1{print $1}' "runs/$1/metrics.csv" 2>/dev/null | sort -un | wc -l
  else
    echo "$e"
  fi
}

# eval.json 是否**比 best.pt 新**。
# 光看 `[ -f eval.json ]` 不够：上一轮留下的旧 eval.json 会让看门狗误判"已完成"，
# 从而放走一个其实没评估的 run（这正是把 _2ep 快照与正式结果混淆的同一类错误）。
eval_fresh() {
  [ -f "runs/$1/eval.json" ] || return 1
  [ -f "runs/$1/best.pt" ]  || return 1
  [ "runs/$1/eval.json" -nt "runs/$1/best.pt" ]
}

# ------------------------------------------------------------------ #
# 单实例保护：PID 文件 + `kill -0` 判活（理由见文件头 ★ 安全前提 2）
if [ -f "$PIDFILE" ]; then
  old=$(cat "$PIDFILE" 2>/dev/null)
  if [ -n "$old" ] && kill -0 "$old" 2>/dev/null; then
    echo "已有 $TAG 的看门狗在运行（PID $old），本次退出"
    exit 0
  fi
  echo "发现陈旧 PID 文件（$old 已不存在），接管"
fi
echo $$ > "$PIDFILE"

say "=== 启动（PID $$）目标：$TAG 训满 $EPOCHS 轮，且 eval.json 比 best.pt 新${DUMP:+（+dump-raw）} ==="

while true; do
  sleep 300
  beat=$((beat + 1))
  if active; then
    [ $((beat % 12)) -eq 0 ] && say "巡检：有任务在跑，继续等待"
    continue
  fi
  sleep 60
  if active; then continue; fi          # 复核：排除两段任务之间的空窗

  ep=$(epochs_done "$TAG")
  if [ "$ep" -ge "$EPOCHS" ] && eval_fresh "$TAG"; then
    say "=== 已达标（$ep 轮，eval 比 ckpt 新），看门狗正常退出 ==="
    exit 0
  fi

  if [ "$tries" -ge "$MAX_TRIES" ]; then
    say "✗ 已补救 $tries 次仍未达标（$ep 轮），放弃空转（详见 /tmp/$TAG.*.log）"
    exit 1
  fi
  tries=$((tries + 1))

  # ① 先把轮数补到 $EPOCHS
  if [ "$ep" -lt "$EPOCHS" ]; then
    if [ -f "runs/$TAG/last.pt" ]; then
      left=$((EPOCHS - ep)); [ "$left" -lt 1 ] && left=1
      say "第 $tries 次补救：已有 $ep 轮，续跑 $left 轮 → 累计 $EPOCHS"
      OMP_NUM_THREADS=8 "$PY" train.py --config "$CFG" --tag "$TAG" \
        --resume "runs/$TAG/last.pt" --epochs "$left" > "/tmp/$TAG.retry.log" 2>&1
    else
      say "第 $tries 次补救：没有 last.pt，从头跑 $EPOCHS 轮"
      OMP_NUM_THREADS=8 "$PY" train.py --config "$CFG" --tag "$TAG" \
        --epochs "$EPOCHS" > "/tmp/$TAG.retry.log" 2>&1
    fi
  fi

  # ② 再（重新）评估，保证 eval.json 与当前 best.pt 对应
  say "评估 runs/$TAG/best.pt"
  OMP_NUM_THREADS=8 "$PY" train.py --config "$CFG" --tag "$TAG" \
    --eval --ckpt "runs/$TAG/best.pt" $DUMP > "/tmp/$TAG.eval.log" 2>&1
  if eval_fresh "$TAG"; then
    say "✓ runs/$TAG/eval.json 已生成（$ep 轮）"
  else
    say "✗ runs/$TAG 评估失败（详见 /tmp/$TAG.eval.log）"
  fi
done
