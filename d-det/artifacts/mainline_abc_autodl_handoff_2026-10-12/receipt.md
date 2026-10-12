# A/B/C 大数据包 AutoDL 接收回执

日期：2026-10-12  
远端：`codex-gpu:/root/autodl-tmp/u-det`  
状态：文件上传完成；服务器当前 `nvidia-smi` 不可用，本轮未启动训练。

## 远端独立核验

服务器使用 `/root/miniconda3/bin/python scripts/verify_mainline_abc_bundle.py` 返回：

```json
{"status": "BUNDLE_OK", "files": 18, "tasks": ["A", "B", "C"], "test_opened": false}
```

远端 SHA-256：

| 文件 | SHA-256 |
|---|---|
| `d-det/data/mainline_abc_200k_2026-10-12/manifest.json` | `a5328bf3a082d262fa65e300f6f77dcd5e86405827b7d89296d65a7ee361c7bc` |
| `scripts/mainline_abc_gpu_batch.py` | `c128f262e7ed7621d966c04a3b2142091d8079e0f7a99d1fe9f3e0a8a1a59a55` |
| `scripts/verify_mainline_abc_bundle.py` | `16b5be0b9e07aa1ec20aff9aa07fd83423767d29fb4412a436a4fa68131bfb8d` |
| `d-det/docx/d-det_AutoDL_ABC三任务快速主线批次指导_2026-10-12.md` | `abc4d405c4eb9a7331292e1f7e47f350e638d0026812f799acb7b06f8f0685fe` |

数据目录远端包含 A/B/C 三项各 200,000 train、12,000 validation、格式特征和 6 个 `jsonl.gz` 标准库读取副本；没有 test 文件。远端不依赖 `pyarrow`。

## GPU 恢复后的单条命令

```bash
cd /root/autodl-tmp/u-det
/root/miniconda3/bin/python scripts/mainline_abc_gpu_batch.py --epochs 2 --batch 32
```

该命令一次运行 A、B、C 的 `codet5_frozen` 与 `codet5_layout` 两个快速配置，产物写入 `d-det/artifacts/mainline_abc_gpu_batch_2026-10-12/`。
