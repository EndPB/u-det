#!/usr/bin/env bash
# 修正版 fresh C0（2026-10-10；spec_violation_global_lexical_fit 修复）实际执行命令
set -x
cd /root/autodl-tmp/u-det
# 折内拟合 char/word vectorizer（仅 fit_rows 文本）→ transform eval；pre/post manifest 双哈希
OMP_NUM_THREADS=8 /root/miniconda3/envs/udet/bin/python scripts/cc_fresh_c0_corrected.py
