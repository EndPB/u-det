# A/B/C 大数据包 AutoDL 接收回执

日期：2026-10-12  
远端：`codex-gpu:/root/autodl-tmp/u-det`  
状态：文件上传完成；该包已纠偏为 AICD numeric external control，**禁止启动训练**，无论服务器 GPU 是否可见。

## 远端独立核验

服务器使用 `/root/miniconda3/bin/python scripts/verify_mainline_abc_bundle.py` 返回：

```json
{"status": "BUNDLE_OK", "files": 19, "tasks": ["A", "B", "C"], "test_opened": false}
```

远端 SHA-256：

| 文件 | SHA-256 |
|---|---|
| `d-det/data/mainline_abc_200k_2026-10-12/manifest.json` | `d99f74fb4b650d358e58f736150d7888ed48ed34ad429aa61f06114dc2eaa77c` |
| `scripts/mainline_abc_gpu_batch.py` | `2227d9768693f76f095954548183727371ac37033f736f16440cb3addb4b69cd` |
| `scripts/verify_mainline_abc_bundle.py` | `16b5be0b9e07aa1ec20aff9aa07fd83423767d29fb4412a436a4fa68131bfb8d` |
| `d-det/docx/d-det_AutoDL_ABC三任务快速主线批次指导_2026-10-12.md` | `c09f43c363b2f66f9c9f18eb30515ee1676f7d77f30bb7cdaf5d50a9b30d7aac` |

数据目录远端包含 A/B/C 三项各 200,000 train、12,000 validation、格式特征和 6 个 `jsonl.gz` 标准库读取副本；没有 test 文件。远端不依赖 `pyarrow`。

## 纠偏后的服务器动作

只运行：

```bash
/root/miniconda3/bin/python scripts/verify_mainline_abc_bundle.py
```

不要执行 `mainline_abc_gpu_batch.py`。真正 d-det ABC 批次必须另行冻结任务中心化、语义/来源分离、AMMC train-only 视图和 Human-inclusive 标签包。

## 纠偏复核

RTX 3080 Ti 已可见。服务器实测旧训练命令立即输出 `WITHDRAWN_EXTERNAL_CONTROL: no training started` 并退出。数据包 19/19 通过，新增角色标记明确禁止把 AICD T1/T2/T3 当项目 ABC 主线；未启动本批训练。
