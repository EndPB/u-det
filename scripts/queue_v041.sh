#!/bin/bash
# v0.4.1（最终版）：只到 **4 epoch**，且**只等、不再起训练**。
#
# 背景：周期 1（e0/e1）已完成；周期 2（e2/e3）由更早那版队列启动、现为孤儿进程仍在跑。
# 本脚本只负责「等周期 2 落盘 -> 评测」，绝不 launch 任何训练
# （上一版按「>=2 行」等待，而周期 1 早已写完 2 行，会在 30 秒后重复起一个训练 —— 已修）。
export OMP_NUM_THREADS=8
PY=/root/miniconda3/envs/udet/bin/python
CFG=configs/udet_v04_cons.yaml
cd /root/autodl-tmp/u-det

echo "[queue] $(date +%H:%M:%S) 等周期 2（e2/e3）落盘 —— 需要 metrics.jsonl 满 4 行"
while [ "$(grep -c . runs/v0.4.1/metrics.jsonl 2>/dev/null || echo 0)" -lt 4 ]; do sleep 60; done
sleep 30                                   # last.pt 写在 metrics.jsonl 之后
echo "[queue] $(date +%H:%M:%S) 4 epoch 到齐，开始评测"
$PY train.py --config $CFG --tag v0.4.1 --eval --ckpt runs/v0.4.1/best.pt >> /tmp/v041.log 2>&1
echo "[queue] $(date +%H:%M:%S) 完成（v0.4.1 停在 4 epoch）"
