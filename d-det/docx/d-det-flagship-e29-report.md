# d-det E29-A 执行报告（服务端 → 指导 AI）

> 依据：`docx/d-det_E29核心方向指导_2026-09-30.md`（基线 70f4760）。
> 产物：`artifacts/flagship_e29_a/{config.json, manifest.json, fold_metrics.json,
> density_metrics.json, solver.log, report.md}`（本报告副本为 report.md）。
> 纪律声明：**只读 b_train/b_val**（特征复用 E28 train/val-only 缓存并做 md5 清单逐一校验）；
> 不加载 test_seen/unseen 或 E25 四划分特征；未继续 top-k 变体；`test_accessed=false`；
> 随机切分仅作参照、不触发晋级（预注册操作化规则见 config.json `exit_rule`）。

---

## 0. 摘要

- **协议按规范执行**：AI 按 generator 5 折 GroupKFold（留出性自检：验证折∩训练折 generator=0），
  Human 随机分折仅入检测；R0=m_raw、R1=μ；每折训练折标准化 + inner 80/20 选 C、验证折只评一次；
  密度 = λ=20 收缩对角高斯 (8)(9)；随机切分对照同协议跑完。求解 24.4 min（总 1470s）。
- **主结果**：generator-held-out 家族 bal：**R0 .1789±.0641、R1 .1873±.0645**；随机切分：
  R0 .4277±.0093、R1 .4202±.0065 ⇒ **跨生成器保留的信号仅 ≈ 随机切分的 43%（R0）/45%（R1）**，
  约为随机猜测（1/8）的 1.4–1.5 倍且折间波动大（.11–.31）。
  **检测几乎不塌**：gen-held-out det .954/.958 vs random .957/.961（−0.3pt）——AI–Human 判别可跨生成器泛化，
  家族归因不可。
- **密度对照**：random 切分下密度**从不**胜线性（0/5，均值差约 −22pt）；gen 切分下 R1 密度 3/5 折胜
  （均值 .1929 vs .1873）——按出口规则 density_ok=True，但 **signal_ok=False**（min 双臂 gen 均值
  .1789 < 预注册线 .18，R1 .1873 仅勉强越线）⇒ **E29-B 未触发**。
- **出口判定（规范 §3.3）**：当前 family 信号**主要是 generator/实验室记忆与分布混杂**——随机切分
  远高于跨生成器（≥23pt 差），且跨生成器残余信号弱而不稳 ⇒ **停止流形/对比损失路线（B 不启动），
  转向数据元数据审计**。此结论只依据 generator-held-out，不使用随机切分触发。
- 边界声明：不将本次结果写成"旗舰模型后训练特征已被分离"；不写成"归因空间不需要额外维度"；
  密度诊断只是对角收缩高斯，不是完整 DMHM。

---

## 1. 数据与折结构（只读 b_train/b_val；合并 n=22972：AI 13472 + Human 9500）

- **generator 数目事实（关键）**：AI 共 **24 个 generator**；每家族 generator 数：
  **OpenAI 1**、DeepSeek 3、Qwen 5、Meta 5、**Mistral 2**、IBM 3、**01-ai 2**、Google 3
  ⇒ **5/8 家族 <5 个 generator**。按规范"generator 少则记录实际折数、不复制样本"：
  这些家族在多数折的证据折中整体缺席（留出单/双 generator≈留出整个实验室），
  验证折族覆盖为每折 3–5 个家族（逐折明细见 fold_metrics.json）。
- 折样本量：每折 AI 验证 2673–2712（各家族 ~1700 样本成块落入某折；Google 1572）；
  Human 按 RandomState(1) 均分（两种划分共用）。
- 每折记录：family/generator/language/长度计数、训练/验证 generator 数（19–20 / 4–5）。

## 2. 主指标：generator-held-out vs 随机切分（5 折均值±std）

| 表示 | 家族 bal（gen-held-out） | 家族 bal（random） | 检测 AUROC（gen） | 检测 AUROC（random） |
|---|---:|---:|---:|---:|
| R0 m_raw | **.1789 ± .0641** | .4277 ± .0093 | .9538 | .9571 |
| R1 μ | **.1873 ± .0645** | .4202 ± .0065 | .9581 | .9607 |

逐折（gen；R1）：.3073 / .1680 / .1116 / .1721 / .1774；逐折（random；R1）：
.4165 / .4225 / .4143 / .4159 / .4320。检测逐折均在 .94–.965（gen）内。
- 与随机切分的差距：R0 **−24.9pt**、R1 **−23.3pt**；相对随机猜测（.125）：1.4×/1.5×。
- 折0 与折1 的水平（.31/.17）与折2（.11）的落差与"哪些家族落在验证折、其 generator 是否为大块"
  直接相关（例如折0 验证含 OpenAI 整族 n=1700，而训练折完全无 OpenAI ⇒ 该族召回恒 0 仍计入 balanced）。
- 每折 C* 均由训练折 inner 选择（家族 .01–.3、检测 .03–1.0 区间内，明细见 fold_metrics 的 inner_grid）。

## 3. 收缩对角密度 vs 线性判别（λ=20）

| 集合 | R0 密度 bal 均 | R0 胜线性折数 | R1 密度 bal 均 | R1 胜线性折数 |
|---|---:|---:|---:|---:|
| gen-held-out | .1688 | 2/5 | .1929 | **3/5** |
| random | .2036 | 0/5 | .2380 | 0/5 |

- random 切分下对角线高斯远弱于线性（−19~−22pt）——与内核 E21"交互呈分布式交叉"的旧观察一致：
  该数据上家族信号大量在**跨维协方差**中，对角密度天然读不到（此处仅作观察，不作新结论）。
- gen 切分下密度相对线性回升（R1 3/5 折小幅胜）——但这是"弱信号区间内的此消彼长"，
  不足以改变主结论（signal_ok=false）。

## 4. 出口判定（规范化操作，预注册于 config.json）

| 条件 | 结果 |
|---|---|
| signal_ok：min(双臂 gen 家族 bal 均值) ≥ 0.18 | **False**（R0 .1789 / R1 .1873——R1 仅越线 0.7pt，且未通过 min 规则） |
| density_ok：任一臂密度多数折（≥3/5）胜线性 | True（仅 R1，且只在 gen 切分下） |
| **E29-B 触发** | **False** |

- 阈值敏感度如实说明：R1 均值 .1873 恰在 .18 上方、R0 .1789 恰在下方——规则输出对阈值敏感；
  但**定性结论不依赖阈值**：跨生成器信号 ≈ 随机切分的 43–45%、折间波动大、且密度优势只在
  弱信号区间出现（random 下 0/5）。
- 规范出口措辞（§3.3）："若 R0/R1 在 generator-held-out 上都接近随机或远低于随机切分结果"——
  本数据符合"远低于随机切分结果"分支 ⇒ **当前 family 信号主要是 generator 记忆或分布混杂；
  停止增加流形和对比损失，转向数据元数据审计**。
- 不触发即不执行 E29-B（未创建 e29_b 产物；未做任何 SupCon/A 矩阵试验）。

## 5. 发现与边界

1. **检测任务跨生成器泛化良好**（.954–.958），与 E24/E25 的"检测稳、归因难"格局一致；
   家族任务在"新 generator（新家族版本）"上仅剩弱信号。
2. **无 task_id**：跨生成器折仍可能含同题不同 generator 的题目簇混杂，"同题不泄漏"依旧不能声称。
3. 本审计只回答"来源可分性是否跨生成器稳定"；**不识别** R_f/q_f/Z_f 或纯后训练因果效应（§2 边界沿用）。
4. 不把 R0/R1 的家族分类结果写成"旗舰模型后训练特征已被分离"；也不写"归因空间不需要额外维度"。
5. 数据元数据审计方向（按规范建议）：generator 粒度太粗（5/8 家族 ≤3 个 generator，OpenAI 只 1 个）
   ——后续若要继续家族线，优先补 generator 多样性与 task_id/共同题目配对，而非再加损失项。

## 6. 复现与文件

```bash
cd /root/autodl-tmp/u-det/d-det
# 冒烟（artifacts/flagship_e29_a_smoke；~2 min；跑完已清理）
OMP_NUM_THREADS=8 python scripts/flagship_e29a.py --smoke
# 全量（~24.5 min：inner 网格 + 密度对照 + 双划分；默认拒绝覆盖）
OMP_NUM_THREADS=8 python scripts/flagship_e29a.py
```

- 产物：`artifacts/flagship_e29_a/`（6 件；manifest 含 commit、E28 特征 sha256=d83f63f6bb83c591…、
  输入清单 sha256、每折 generator 清单、计数、C 网格、λ=20、tau=null、seed、test_accessed=false）；
  运行日志 `/tmp/ddet_e29a_full.log`。
- 只读依赖：`runs/flagship_e28/features_train_val.npz`（train/val-only；md5 逐一校验通过）、
  b_train/b_val parquet；**未触碰** test/unseen 与 E25 features.npz。

---

## 7. 勘误与结论边界（2026-09-30，依据 `d-det_ACL核心假设与E30协议修正_2026-09-30.md` §1）

1. **撤回**本报告 §0/§4 中"当前 family 信号主要是 generator/实验室记忆与分布混杂"的表述——
   该归因超出证据。E29-A 协议存在两处缺陷：
   a) OpenAI 只有一个 generator：留出它时训练集不含任何 OpenAI 样本，而该家族仍被计入闭集
   family balanced accuracy（召回恒 0 拖低指标）；
   b) 密度对照实现缺陷：缺失家族被赋予伪造的零均值/单位先验——密度结果不作为结论依据。
2. 正确表述：E29-A 说明"随机切分下的家族信号中包含相当多 generator（含整实验室）级别的成分，
   跨 generator 残留信号弱且不稳"；**E29-B 暂停是正确的实验纪律，但不是对流形或对比假设的证伪**。
3. 修正协议（family-conditional 两折、七家族、固定 C=0.1、同本随机控制）的 H1 重检结果见
   `docx/d-det-flagship-e30-report.md`：出口未通过——"当前协议下未观察到稳定跨 generator 家族信号"。
