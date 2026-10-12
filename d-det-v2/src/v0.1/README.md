# v0.1（首批实现；已实现并短流程验证，**正式批次未运行**）

范围：按《d-det_家族归因与二分类检测_设计及执行指导_2026-10-13.md》实现首批批次。
**独立且极简**：本版代码独立新写，不 import/不复制/不修改 `d-det/` 旧代码（**原代码保持原样**）；
旧线仅作数据/权重的只读来源。

## 文件

| 文件 | 内容 |
|---|---|
| `data.py` | 记录加载/规范化去重/task 划分（calib、5 折）/内容代理 Q 与 AMMC 变换/分块分词 |
| `center.py` | 任务中心 μ（F: Human+族各 1/8；D: Human 1/2、族 1/14）、标准化 ridge（alpha 内层按 task 组选）、5 折 OOF、全局均值 |
| `model.py` | 分块编码器池化（≤4×512，首 2+尾 2，有效 token 加权）、阶段一模型、F 校正头（低秩 8+GELU+零初始化末层）、D 校正头（[z;z⊙z]/signed/mag） |
| `train.py` | CLI：`prep` / `tokens` / `s1` / `feat` / `s2` / `smoke`（两阶段、六臂、机制小头） |
| `p0.py` | C-P0 强控制（word/char TF-IDF、布局元数据、B1 概率；融合权重在校准子集拟合） |
| `eval.py` | 主表/每类/混淆/长度分桶/长度匹配/AMMC 压力/配对 CI/Ours-F 派生检测 |
| `configs.yaml` | 首批默认值（《设计指导》§10.2） |

## 运行（当前服务器 = AutoDL 有卡机；一次性批次顺序）

```bash
cd d-det-v2/src/v0.1
python train.py prep && python train.py tokens          # CPU；写 manifest 到 artifacts/v0.1/prep/
for s in 0 1 2; do
  for t in F D; do
    python train.py s1 --task $t --arm B0 --seed $s     # 阶段一：直接微调（原文）
    python train.py s1 --task $t --arm B1 --seed $s     # 阶段一：AMMC 预算匹配（Ours 的冻结来源）
    python train.py feat --task $t --seed $s            # s(x)、冻结 h/基线 logits、5 折 OOF 参考
    python train.py s2 --task $t --seed $s              # ours / b2 / b3 + 机制小头（冻结编码器）
    python p0.py  --task $t --seed $s                   # C-P0（CPU）
    python eval.py --task $t --seed $s                  # dev 主表 + 分桶 + 压力 + CI
  done
done
# test 需显式授权：python eval.py --task F --seed 0 --splits dev,test --test
```

## 实现约定（首批固定选择；已写入 manifest/报告）

- 校准子集 = train 的 10%（按 task、语言分层；210 任务），仅用于 checkpoint/阈值选择；不参与训练与中心/参考器；
- 5 折 OOF 按 task 分组；**推理保留 5 个参考器**，分别构造校正 logits 后取均值再 sigmoid/softmax（不先平均参考）；
- `E_sem` = **冻结预训练 CodeT5-base** 在 Q(x) 上的均值池化（不下载新权重）；
- AMMC 仅作用于阶段一（B0 = 原文；B1 = 每曝光原文 1/2、增强 1/2；comment .2 / ws .3 / rename .4 独立触发，未生效回退计数）；
  阶段二头使用干净文本的冻结特征（与阶段一同一曝光序列/采样器）；
- rename（作用域内私有标识符）首批仅支持 Python（其他语言回退计数；接受率见 `artifacts/v0.1/prep/manifest.json`）；
- C-P0 融合在校准子集拟合（四个组件对该子集均为 OOF；词法/布局组件在 train 主训练集拟合，B1 概率来自未见过校准子集的 B1）；
- 长代码 = 512-token 块、最多 4 块（首 2+尾 2）、有效 token 加权池化；标记 2048 token 覆盖上限；
- 产物：重型文件在 `d-det-v2/work/v0.1/`（records/缓存/checkpoint/逐样本预测，不进 git）；
  记录在 `d-det-v2/artifacts/v0.1/`（prep/manifest、runs、reports）。

## 当前状态（2026-10-13）

- `prep` 已跑全量：25,440 条（train/dev/test = 16,800/5,280/3,360；去重 0 例；校准 210 任务、主训练 1,890、5 折各 378）；
- `tokens` 已缓存：≤512 = 11,314 / 513-1024 = 10,913 / 1025-2048 = 3,201 / >2048 = 12；
- `smoke` 全链路（prep→tokens→s1→feat→s2→p0→eval）通过；日志 `artifacts/v0.1/prep/smoke_log.txt`；
- **正式 GPU 批次未运行**（等待"运行批次"指令）；`test_read=false`。
