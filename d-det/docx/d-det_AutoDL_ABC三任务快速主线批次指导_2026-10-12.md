# AutoDL 指导：AICD 数值外部控制批次（已冻结，禁止作为 d-det ABC 主线）

日期：2026-10-12  
状态：**外部控制包，禁止启动**。它不是 d-det 的 A/B/C 主线；真正主线必须使用任务中心化、语义分离、AMMC 训练视图和正确的 Human-inclusive 标签包。纠偏记录见 `d-det_ABC路线纠偏与主线恢复记录_2026-10-12.md`。

## 角色

该包只用于评估 AICD 数值标签上的大规模读取、格式通道和 CodeT5 工程链路：

| 外部配置 | 数据 | 首轮读数 |
|---|---|---|
| T1 | AICD 2 个数字类 | Macro-F1、AUROC、balanced accuracy |
| T2 | AICD 12 个数字类 | Macro-F1、balanced accuracy、每类 recall |
| T3 | AICD 4 个数字类 | Macro-F1、balanced accuracy、混淆矩阵 |

本机已经把每项训练集冻结为 200,000 条、验证集 12,000 条；测试文件不上传、不读取。T2/T3 的数字标签保留原始 ID，不能把它们写成 d-det 的 B/C 标签，也不能把这个包当作任务中心化或语义配对数据。

## 服务器接收后只做

```powershell
/root/miniconda3/bin/python scripts/verify_mainline_abc_bundle.py
```

无论 GPU 是否可见，都**不执行训练命令**。数据同时提供 `jsonl.gz` 标准库读取副本，服务器不需要安装 `pyarrow`。该包只保留接收、哈希和边界核验记录。

## 历史快筛配置（不进入主表）

每项包含两个快速配置：

1. `codet5_frozen`：冻结 CodeT5-small，训练统一 MLP 读出；作为快速 CodeT5 基线。
2. `codet5_layout`：同一 CodeT5 表征拼接原始布局统计，包括 tab、空格、行尾空白、空行率、行长均值/方差和缩进统计。

上述历史配置只检查了冻结编码器加布局统计的外部数值预测，未实现 d-det 的语义分离、任务中心化或 AMMC，不能回答核心方法是否有效。其结果继续保留，不决定 d-det 主线是否启动或停止。

## 产物

保留已有本机快筛分数、CPU 结果、数据 manifest 和接收回执；本批不新增服务器训练产物。

## 解释边界

ABC 在项目中仅表示二分类、含 Human 的家族归因、四分类三个问题。AICD 外部配置、其固定类数和官方协议不能替代项目目标；该快筛也不能作为 d-det 或 CodeT5 全参微调的结论。

