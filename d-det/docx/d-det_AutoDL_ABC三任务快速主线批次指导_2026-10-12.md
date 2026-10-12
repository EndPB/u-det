# AutoDL 指导：A/B/C 大数据快速主线批次

日期：2026-10-12  
状态：已在本机冻结并上传前校验；服务器无 GPU 时只接收文件和核验，不启动训练。

## 目标

一次批次同时推进三个任务：

| 任务 | 数据 | 首轮读数 |
|---|---|---|
| A | AICD T1，二分类 | Macro-F1、AUROC、balanced accuracy |
| B | AICD T2，12 个数字类，包含 Human | Macro-F1、balanced accuracy、每类 recall |
| C | AICD T3，4 个数字类 | Macro-F1、balanced accuracy、混淆矩阵 |

本机已经把每项训练集冻结为 200,000 条、验证集 12,000 条；测试文件不上传、不读取。T2/T3 的数字标签保留原始 ID，标签语义映射另行登记，不把数字顺序擅自改成 family 名称。

## 服务器接收后先做

```powershell
/root/miniconda3/bin/python scripts/verify_mainline_abc_bundle.py
/root/miniconda3/bin/python scripts/mainline_abc_gpu_batch.py --epochs 2 --batch 32
```

如果 `nvidia-smi` 不可用，只执行第一条核验和文件清单，不执行第二条。GPU 恢复后从第二条开始，整批运行 A、B、C，不拆成零散实验。数据同时提供 `jsonl.gz` 标准库读取副本，服务器不需要安装 `pyarrow`。

## 首轮配置

每项包含两个快速配置：

1. `codet5_frozen`：冻结 CodeT5-small，训练统一 MLP 读出；作为快速 CodeT5 基线。
2. `codet5_layout`：同一 CodeT5 表征拼接原始布局统计，包括 tab、空格、行尾空白、空行率、行长均值/方差和缩进统计。

该批次用于快速判断三个任务是否存在可用增量。若布局配置在 A/B/C 的原始验证集提升，再追加一次完整 CodeT5 微调和 AMMC 训练视图批次；不通过时不靠增加轮数或损失项挽救。

## 产物

服务器回传：`metrics.json`、每项两种配置的验证分数、混淆矩阵、GPU 监控、命令、环境、数据 manifest 和 SHA256SUMS。所有结果注明 `test_opened=false`。

## 解释边界

该批次验证 A/B/C 与 CodeT5、格式通道的效果，不能单独证明纯后训练因果。主线按“旗舰模型 base 差异在工作范围内可忽略”推进；base/post 谱系校验作为后补工作。

