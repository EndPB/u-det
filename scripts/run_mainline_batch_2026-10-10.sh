#!/usr/bin/env bash
# Mainline C0-C3 one-shot batch (2026-10-10): C0 reused (frozen canonical run),
# C1/C2/C3 rerun on canonical inputs. One batch, one summary dir.
set -e
cd /root/autodl-tmp/u-det
B=d-det/artifacts/mainline_c0c3_batch_2026-10-10
mkdir -p "$B/logs"
PY=/root/miniconda3/envs/udet/bin/python
echo "=== batch start $(date -u +%FT%TZ) HEAD=$(git rev-parse HEAD) ==="
OMP_NUM_THREADS=8 "$PY" scripts/cc_c1_prompt_conditioned_mainline_2026-10-10.py 2>&1 | tee "$B/logs/c1.log"
echo "=== c1 done $(date -u +%FT%TZ) ==="
OMP_NUM_THREADS=8 "$PY" scripts/cc_c2_static_proxy_mainline_2026-10-10.py 2>&1 | tee "$B/logs/c2.log"
echo "=== c2 done $(date -u +%FT%TZ) ==="
OMP_NUM_THREADS=8 "$PY" scripts/cc_c3_invariance_mainline_2026-10-10.py 2>&1 | tee "$B/logs/c3.log"
echo "=== BATCH_DONE $(date -u +%FT%TZ) ==="
