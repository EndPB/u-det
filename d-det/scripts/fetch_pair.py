"""按名字下载一对 base/instruct 模型（hf-mirror）。

用法：python scripts/fetch_pair.py deepseek-coder-1.3b [org]
默认 org=deepseek-ai，落到 checkpoints/<name>-base / -instruct。
"""
import os
import sys
from pathlib import Path

os.environ.setdefault('HF_ENDPOINT', 'https://hf-mirror.com')
from huggingface_hub import snapshot_download  # noqa: E402

root = Path(__file__).resolve().parents[1]
name = sys.argv[1]
org = sys.argv[2] if len(sys.argv) > 2 else 'deepseek-ai'
pats = ['*.json', '*.txt', '*.safetensors', '*.bin', '*.model']
for role in ('base', 'instruct'):
    repo = '%s/%s-%s' % (org, name, role)
    dst = str(root / 'checkpoints' / ('%s-%s' % (name, role)))
    print('[fetch] %s -> %s' % (repo, dst), flush=True)
    snapshot_download(repo, local_dir=dst, allow_patterns=pats)
print('[fetch] done')
