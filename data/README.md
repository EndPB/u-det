# data/

本目录存放数据集，**不入 git**（体积大），通过脚本随时重建。

```
data/
├── raw/                      # 原始下载数据（download_data.py）
│   └── CoDET-M4/
│       ├── dataset_without_comments.parquet   # 全量数据，500,552 行 / 约 437 MB
│       └── README.md
├── processed/                # 划分 / 清洗 / 分词后的数据（prepare_data.py）
│   ├── codet_m4/             # datasets arrow 格式（train / validation / test）
│   └── summary.json          # 数据统计摘要
└── cache/                    # datasets 缓存（可选）
```

## 下载与准备

```bash
# 1) 下载全量数据（默认走 hf-mirror 镜像）
python scripts/download_data.py

# 2) 统计 + 划分 train/val/test 并保存
python scripts/prepare_data.py

# 3) （可选）抽样调试
python scripts/prepare_data.py --sample 20000
```

## 字段说明（CoDET-M4）

| 字段 | 类型 | 说明 |
| --- | --- | --- |
| `code` | string | 原始代码片段 |
| `cleaned_code` | string | **去注释后的代码**（推荐作为模型输入） |
| `language` | string | 编程语言 |
| `model` | string | 生成该样本的模型标识 |
| `target` | string | 标签列 |
| `source` | string | 样本来源 |
| `split` | string | 官方划分标记 |
| `features` | struct | 8 个手工统计特征（平均函数长度、可维护性指数等） |

数据来源：<https://huggingface.co/datasets/DaniilOr/CoDET-M4>（MIT License）
