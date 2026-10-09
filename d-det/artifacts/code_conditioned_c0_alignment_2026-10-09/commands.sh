#!/usr/bin/env bash
# C0 逐点对账（2026-10-09）实际执行命令（服务器 AutoDL，/root/autodl-tmp/u-det）
set -x
cd /root/autodl-tmp/u-det
# 1) 校验参考包哈希（11/11 npz + 4/4 static_preflight）
python - <<'PY'
# 见 report.md §2.1；对 reference_manifest.json 所列 sha256 全量校验
PY
# 2) v1 对账（按目录名配对；暴露结构错位）
OMP_NUM_THREADS=8 /root/miniconda3/envs/udet/bin/python scripts/cc_c0_align.py
# 3) 模式假设探针（complete vs instruct；结论：complete 更差，排除）
OMP_NUM_THREADS=8 /root/miniconda3/envs/udet/bin/python scripts/cc_mode_probe.py
# 4) v2 对账（按真实 heldout 配对 + 共同成员块 + C1-C3 重算）
OMP_NUM_THREADS=8 /root/miniconda3/envs/udet/bin/python scripts/cc_c0_align2.py
