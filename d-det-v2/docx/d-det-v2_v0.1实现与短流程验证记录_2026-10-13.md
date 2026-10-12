# d-det-v2 v0.1 实现与短流程验证记录

日期：2026-10-13。状态：**已实现并通过短流程验证；正式 GPU 批次未运行；test 未读取**。
执行者：AutoDL 端（本仓库工作区）。代码：`d-det-v2/src/v0.1/`（独立新写；`d-det/` 原代码零改动）。

## 1. 交付内容

| 文件 | 说明 |
|---|---|
| `src/v0.1/data.py` | 记录加载 / 规范化去重 / task 划分（calib、5 折）/ 内容代理 Q 与 AMMC 变换 / 分块分词 |
| `src/v0.1/center.py` | 任务中心 μ、标准化 ridge（alpha 内层选择）、5 折 OOF 参考、全局均值 |
| `src/v0.1/model.py` | 分块编码器池化、阶段一模型、F/D 校正头（低秩 8 + 零初始化末层；D 含 signed/mag 读出） |
| `src/v0.1/train.py` | CLI：prep / tokens / s1 / feat / s2 / smoke |
| `src/v0.1/p0.py` | C-P0 强控制（word/char TF-IDF、布局元数据、B1 概率、校准子集融合） |
| `src/v0.1/eval.py` | 主表 / 每类与混淆 / 长度分桶 / 长度匹配 / AMMC 压力 / 配对 bootstrap CI / Ours-F 派生检测 |
| `src/v0.1/configs.yaml` | 首批默认值（《设计指导》§10.2） |

## 2. 数据准备结果（prep 全量）

- 记录 25,440（train/dev/test = 16,800 / 5,280 / 3,360）；exact 与规范化重复均为 0（保留裁定已登记）；
- 训练内校准子集 210 task（10%，语言分层）；主训练 1,890 task；5 折各 378 task；
- 数据源 sha256 已登记：`artifacts/v0.1/prep/manifest.json`；
- 变换接受率（train，请求→生效）：
  - comment：c 1103/2400、cpp 1341/2400、cs 701/2400、go 1082/2400、java 890/2400、php 1463/2400、py 1088/2400
    （未生效 = 文件无注释、或注释标记位于字符串内被正确保留、或解析失败）；
  - ws：各语言 2110–2380/2400；rename：py 2,067/2400（其余语言按设计回退并计数）；
- 长度分桶（tokenizer token 数）：≤512 = 11,314；513–1,024 = 10,913；1,025–2,048 = 3,201；>2,048 = 12；
- 验证器抽查：C 系扫描器正确剥离 `/* */` 块注释；字符串内 `//` 正确保留；Python rename 返回值可复解析。

## 3. 实现级选择（首批固定；与《设计指导》§5/§10.2 对齐）

- 校准子集参与范围：仅 checkpoint/阈值选择，不参与训练、中心与参考器；
- 参考器协议：按 task 5 折 OOF；推理保留 5 个参考器、分别构造校正 logits 后取均值（单次 sigmoid/softmax）；
- `E_sem` = 冻结预训练 CodeT5-base（Q(x) 均值池化）；不下载新权重；
- AMMC 仅用于阶段一 B1 训练曝光（原文 1/2；comment .2 / ws .3 / rename .4 独立触发；全未生效回退并计数）；
  阶段二头 = 干净文本冻结特征 + 同一曝光序列/采样器；
- C-P0 融合权重在训练内校准子集拟合（组件对该子集均为 OOF）；
- 长代码 512×≤4 块（首 2+尾 2）、有效 token 加权池化；标记 2048 token 覆盖上限；
- 机制小头（预声明）：F = r-only / ref-only；D = signed-only / mag-only（随主批完成）。

## 4. 短流程验证（smoke）

`python train.py smoke`（小样本：每 split 8 task）：prep → tokens → s1(F/D, B1) → feat(F/D) → s2(F/D)
→ `p0.py`（F/D）→ `eval.py`（F/D，含压力与 CI）全链路通过；日志见 `artifacts/v0.1/prep/smoke_log.txt`。
Smoke 数值不具统计意义（仅验证管道、形状、缓存、协议连通性）。

## 5. 未完成 / 边界

- **正式 GPU 批次未运行**：B0/B1 微调（2 epoch、3 seed）、Ours/B2/B3/机制、C-P0、统一评估等待指令；
- test split 未读取（开关 `test_read=false`；读取需显式授权）；
- `rename` 首批仅 Python；其他语言在 AMMC/Q 中回退计数（按设计报告）；
- 运行时间预估（单卡 3080 Ti）：F 每 (arm,seed) 约 25–40 min、D 约 8–15 min；特征与评估为分钟级；
  整批 3 seed × 2 task 预计 6–10 小时（可分批续跑，缓存与 checkpoint 均已落盘）。
