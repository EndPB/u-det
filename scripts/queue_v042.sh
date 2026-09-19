#!/usr/bin/env bash
# v0.4.2（分块编码器 codet5blk，2 epoch）接力队列。
#
# 用户指令：「等现在的跑完再跑」—— 所以要等 CodeT5 滑动窗口基线（含它尾部的 eval）
# 彻底结束之后才启动，避免两个任务抢同一张 12GB 卡。
#
# 等待条件用 **PID 存活**（唯一不会"早已为真"的信号），并在放行前校验基线产物；
# 历史上用过"日志行数 >= N"做等待条件，那个条件在启动瞬间就为真，
# 于是并发起了第二个训练，差点把产物目录写坏。这里不再重蹈覆辙。
set -u
cd /root/autodl-tmp/u-det

PID="${1:?用法: queue_v042.sh <pid>（要等待的进程 PID）}"
PY=/root/miniconda3/envs/udet/bin/python

echo "[queue] $(date +%H:%M:%S) 等待 PID $PID（CodeT5 基线队列）结束…"
while kill -0 "$PID" 2>/dev/null; do sleep 30; done

echo "[queue] $(date +%H:%M:%S) PID 已退出；校验基线产物"
if [ -f runs/base_codet5_tok/eval.json ]; then
  echo "[queue] 基线 eval.json 存在 ✓"
else
  echo "[queue] ⚠ 未找到 runs/base_codet5_tok/eval.json —— 基线可能中途失败，"
  echo "[queue]   但 v0.4.2 与它无关，继续启动；事后请检查 /tmp/base_codet5_tok*.log"
fi

echo "[queue] $(date +%H:%M:%S) 启动 v0.4.2（codet5blk 分块编码器，block=128，2 epoch）"
OMP_NUM_THREADS=8 nohup "$PY" train.py --config configs/udet_v042.yaml \
  --tag v0.4.2 --epochs 2 > /tmp/v042.log 2>&1 &
wait $!

echo "[queue] $(date +%H:%M:%S) v0.4.2 训练结束，评估 best.pt"
OMP_NUM_THREADS=8 "$PY" train.py --config configs/udet_v042.yaml \
  --tag v0.4.2 --eval --ckpt runs/v0.4.2/best.pt 2>&1 | tee /tmp/v042_eval.log
echo "[queue] $(date +%H:%M:%S) 全部完成"
