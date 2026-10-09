#!/usr/bin/env bash
# 规范成员序 fresh C0（2026-10-10；跨侧复现裁定的 member-order 修复，指导 §12）实际执行命令
set -x
cd /root/autodl-tmp/u-det
# (1) 重建特征 bundle：member_idx 改为 admission 顺序；断言与哈希先行落盘（失败即停止）
OMP_NUM_THREADS=8 /root/miniconda3/envs/udet/bin/python scripts/cc_build_features.py
# (2) 规范序 fresh C0（词法仍为折内拟合，仅 fit_rows；eval 行只 transform）
OMP_NUM_THREADS=8 /root/miniconda3/envs/udet/bin/python scripts/cc_fresh_c0_canonical.py
