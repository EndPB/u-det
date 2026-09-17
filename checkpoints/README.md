# checkpoints/

预训练编码器权重目录，**不入 git**（体积大）。

```bash
# 默认下载 CodeT5-base（约 0.9 GB）
python scripts/download_encoder.py

# 其它编码器示例
python scripts/download_encoder.py --model Salesforce/codet5-large
python scripts/download_encoder.py --model microsoft/graphcodebert-base --name graphcodebert-base
```

下载完成后目录结构：

```
checkpoints/
├── codet5-base/                 # Salesforce/codet5-base
│   ├── config.json
│   ├── pytorch_model.bin        # ~892 MB
│   ├── spiece.model             # 分词器
│   ├── tokenizer_config.json
│   └── special_tokens_map.json
└── codet5-base.json             # 下载元信息（源仓库 / 端点）
```

代码中这样加载：

```python
from udet.encoders import build_encoder, build_tokenizer

encoder = build_encoder("codet5", model_name_or_path="checkpoints/codet5-base")
tokenizer = build_tokenizer("checkpoints/codet5-base")
```
