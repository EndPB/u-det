#!/usr/bin/env bash
set -euo pipefail
echo '=== AutoDL preflight ==='
date -Is || true
hostname || true
uname -a || true
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
else
  echo 'nvidia-smi unavailable'
fi
python - <<'PY'
import importlib.metadata as m
import platform
print('python', platform.python_version())
for name in ('torch','transformers','numpy','scikit-learn','tokenizers','huggingface-hub','pandas','pyarrow'):
    try:
        print(name, m.version(name))
    except Exception:
        print(name, 'missing')
try:
    import torch
    print('torch.cuda.is_available', torch.cuda.is_available())
    print('torch.version.cuda', torch.version.cuda)
    if torch.cuda.is_available():
        print('gpu', torch.cuda.get_device_name(0), 'count', torch.cuda.device_count())
except Exception as exc:
    print('torch check failed', repr(exc))
PY
df -h . || true
free -h || true
