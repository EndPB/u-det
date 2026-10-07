# d-det/checkpoints（2026-10-07 服务器清理后）

当前仅保留：

- `codet5-base/`（ACL 主线与全部现有报告使用的编码器，保留）

## 已删除（2026-10-07；配对线 v0.1–v0.4 已闭环，全部相关特征/产物已落盘，可按需重下）

| 目录 | 大小 | 恢复方式 |
|---|---|---|
| `deepseek-coder-1.3b-base` | 2.6G | `python scripts/fetch_pair.py deepseek-coder-1.3b`（默认 org=deepseek-ai） |
| `deepseek-coder-1.3b-instruct` | 5.1G | 同上 |
| `qwen2.5-coder-0.5b-base` | 954M | `python scripts/prepare.py pairmdl`（默认即 0.5B 家族） |
| `qwen2.5-coder-0.5b-instruct` | 954M | 同上 |
| `qwen2.5-coder-1.5b-base` | 2.9G | `python scripts/prepare.py pairmdl --pair-name qwen2.5-coder-1.5b` |
| `qwen2.5-coder-1.5b-instruct` | 2.9G | 同上 |

共约 **15.4 GB**。下载脚本已内建 hf-mirror（`HF_ENDPOINT=https://hf-mirror.com`，国内直连可用）。
历史 revision/哈希线索：`scripts/hf_provenance.py`、`docx/d-det-v0.1.md` 调研记录。

⚠️ 恢复提示：`.gitignore` 已忽略 `checkpoints/`（不入库）；本文件仅存于服务器。
