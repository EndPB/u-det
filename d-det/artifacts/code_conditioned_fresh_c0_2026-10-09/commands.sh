#!/usr/bin/env bash
# fresh C0 重建（2026-10-09；§9 唯一动作）实际执行命令
set -x
cd /root/autodl-tmp/u-det
# manifest 先写后拟合；随后 dev+inner 全折拟合、确定性复拟合、诊断对照
OMP_NUM_THREADS=8 /root/miniconda3/envs/udet/bin/python scripts/cc_fresh_c0.py
