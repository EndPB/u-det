# d-det E30 执行报告（服务端 → 指导 AI）

> 依据：`docx/d-det_ACL核心假设与E30协议修正_2026-09-30.md`（基线 bd12725）。
> 产物：`artifacts/flagship_e30/{config.json, manifest.json, metrics.json, density_metrics.json,
> predictions.npz, solver.log, report.md}`（本报告副本 = report.md）。
> 纪律声明：只读 b_train/b_val（特征复用 E28 train/val-only 缓存 + md5 清单逐一校验）；
> 分类器固定 **C=0.1**（无 inner tuning）；标准化仅训练折；`test_accessed=false`；
> 出口规则运行前写入脚本/配置，未事后调整。修正 E29-A 的结论边界（见 §5 与 E29 报告 §7 勘误）。

---

## 0. 摘要

- **H1 最小实验按修正协议完成**：排除 OpenAI（单 generator）后的**七家族 / 23 generators**；
  family-conditional **两折** split（每家族两侧均有 generator，断言两侧不相交且非空）；
  每家族固定种子打乱 generator（2→1/1、3→1/2、5→2/3）；随机控制 = 同一七家族、每家族与
  gen 折**相同样本数**、同一分类器（C=0.1）。求解 0.5 min（总 37.5s）。
- **主结果（gen-held-out vs 同本随机控制，BA_F / BA_G）**：

| 表示 | 模式 | fold0 BA_F / BA_G | fold1 BA_F / BA_G | 检测 AUROC |
|---|---|---:|---:|---:|
| R0 raw（主） | gen | **.3316 / .3276** | **.2402 / .2234** | .9481 / .9419 |
| R0 raw | random | .3869 / .3354 | .3803 / .3377 | .9527 / .9482 |
| R1 μ（次要） | gen | **.3388 / .3324** | **.2430 / .2268** | .9550 / .9473 |
| R1 μ | random | .3902 / .3439 | .3772 / .3502 | .9554 / .9514 |

- **出口判定：未通过（passed=false）**——BA_F 与 BA_G 在 R0、R1 上、两折中**全部**低于随机控制
  （差值 −0.4 ～ −14.0pt；仅 R0 BA_G fold0 −0.78pt 与 R1 BA_G fold0 −1.15pt 接近持平）。
  ⇒ 按规范措辞：**"当前协议下未观察到稳定跨 generator 家族信号"**（不写"信号由记忆造成"）。
- **背景读数（仅陈述）**：gen 折 BA_F .24–.34 高于机会水平 1/7≈.143（约 1.7–2.4×），但两折、
  两表示下均低于同组成随机控制——跨与不跨 generator 之间的差距是**强度差**而非"有无信号"的二分；
  规范要求该差距不得上升为"记忆造成"的因果结论。
- **检测几乎不受影响**：gen − random 差值 −0.04～−0.63pt（R1 仅 −0.04/−0.41pt）；
  但按规范，**检测稳定不能替代 H1 的家族证据**。
- H2/H3 不启动（H1 未通过）；密度诊断（修正缺失类 −∞、方差下限 1e-8）全部低于线性读出（仅诊断）。

---

## 1. 协议与实现（规范 §4 逐项）

1. **闭集**：有 ≥2 generator 的七家族（DeepSeek 3、Qwen 5、Meta 5、Mistral 2、IBM 3、
   01-ai 2、Google 3；共 23 generators；AI 样本 11772 + Human 9500）。"七家族"=当前数据的
   实验室集合，不称"七个旗舰家族"。
2. **family-conditional 两折**（每家族两侧均有 generator）：
   - fold0：val=A（如 DeepSeek-R1；Qwen Coder-1.5B-Instruct+QwQ-32B；…）、train=B；
     fold1 互补。逐族 sample 数：fold0 val {DeepSeek 173, Qwen 818, Meta 863, Mistral 1038,
     IBM 985, 01-ai 1525, Google 469}，fold1 互补（AI train/val = 5901/5871 与 5871/5901）。
   - 断言通过：每家族两侧非空、generator 集合不相交、每折两侧家族齐全。
   - 注：数据集存在 generator 名笔误"Qwen2.5-Codder-14B-Instruct"（原样保留）。
3. **表示**：R0=raw（主）；R1=μ（次要——**使用 E24 监督头得到，不能宣称完整 pipeline 的新
   generator 泛化**）。标准化仅训练折（std 下限 1e-2）。
4. **分类器**：单个 L2 多项逻辑回归，**固定 C=0.1**（预注册；不在单 generator 子集上 inner tuning）；
   检测同为固定 C=0.1 的二项逻辑回归（Human 二折 Twofold 随机划分，RandomState(1)，两模式共用）。
5. **随机控制**：每家族从样本池随机二堆，样本数与 gen 折逐族相同、同一分类器、同一 C
   （不跨折池化、不改变类平衡）。
6. **密度诊断**：λ=20 对角收缩高斯；缺失类 score=−∞（本实验两侧家族齐全，未触发）；方差下限 1e-8；
   只作诊断、不参与出口。
7. **预测留存**：`predictions.npz` 含每样本 row_id（source split+row）、family、generator、
   language、length、split、classes_（7 族），以及 2 臂×2 模式×2 折的预测数组。

## 2. 出口判定（规范 §4；规则运行前固定）

| 检查 | fold0 diff | fold1 diff | 通过 |
|---|---:|---:|---|
| R0 · BA_F | −0.0553 | −0.1401 | ✗ |
| R0 · BA_G | −0.0078 | −0.1143 | ✗ |
| R1 · BA_F | −0.0514 | −0.1342 | ✗ |
| R1 · BA_G | −0.0115 | −0.1234 | ✗ |
| 检测（附报） | R0 −0.0046 / R1 −0.0004 | R0 −0.0063 / R1 −0.0041 | — |

（diff = gen − random。规则："BA_F 与 BA_G 在 R0、R1 上都 > 随机控制且两折方向一致"。）
**结论（按规范措辞）**：当前协议下**未观察到稳定跨 generator 家族信号**。

## 3. 逐家族与逐 generator 明细（gen 折，R1；R0 见 metrics.json）

逐家族召回（R1）：fold0 {DeepSeek .445, Qwen .379, Meta .269, Mistral .089, IBM .474,
01-ai .004, Google .712}；fold1 {DeepSeek .033, Qwen .122, Meta .249, Mistral .314, IBM .478,
01-ai .206, Google .298}。
逐 generator 样例（R1）：fold0 Google：gemma-3n-e4b-it **.712**；fold1 Google：gemma-3-27b-it
.341 / codegemma-2b **.020**；fold0 DeepSeek：R1 .445；fold1 DeepSeek：V3-0324 .031 /
coder-1.3b-base .051；fold1 Mistral：Devstral .314（fold0 Mistral-7B-Instruct .089）。
⇒ 同一实验室内部 generator 间差异极大，且方向随折翻转（DeepSeek R1↔V3 两折几乎互换）；
这是"跨 generator 稳定性"未达出口的直接表象（陈述性，不作因果归因）。

## 4. 密度诊断（仅诊断）

BA_F：gen R0 .207/.216、R1 .253/.205；random R0 .230/.218、R1 .261/.252——全部低于线性读出，
与 E28/E29 的观察一致（对角线高斯读不到跨维信息）。按规范不参与 H1 出口。

## 5. 结论与边界

1. **H1 结论（规范措辞）**：当前协议下未观察到稳定跨 generator 家族信号；两表示、两折、
   两指标对随机控制的差值全为负（仅两个 BA_G fold0 接近持平）。不得写成"信号由记忆造成"，
   也不得写成"存在稳定迁移"。
2. **检测**：gen 与 random 差距最小（−0.04～−0.63pt）——检测读出的来源信号跨 generator 更稳
   （与 E29-A 一致）；但按规范不能以检测稳定替代 H1 的家族证据。
3. **H2/H3**：不启动。只有 H1 通过才做同一折上的 matched ablation（先几何 XOR 任务条件，
   不得同时引入；低秩归因空间不作先验）。
4. **E29-A 边界修正（同步勘误至 E29 报告 §7）**：撤回"家族信号主要是 generator 记忆"表述——
   E29-A 协议有两处缺陷（OpenAI 单 generator 时留出它→训练无该类却计入闭集 bal；密度对照的
   缺失类处理被伪造零均值/单位先验）；E29-B 暂停是纪律而非对流形/对比假设的证伪。
5. **论文口径**：后训练相对分布公式（指导 §2）→ H1 跨 generator 可迁移性（本轮未通过）→
   H2 几何读出 → H3 任务分离；每步只改它直接检验的假设。

## 6. 复现与文件

```bash
cd /root/autodl-tmp/u-det/d-det
OMP_NUM_THREADS=8 python scripts/flagship_e30.py --smoke   # ~0.4 min（已清理）
OMP_NUM_THREADS=8 python scripts/flagship_e30.py           # ~0.6 min；默认拒绝覆盖
```

- 产物：`artifacts/flagship_e30/`（7 件）；manifest 含 commit、E28 特征 sha256=d83f63f6bb83c591…、
  输入清单 sha256、逐折 generator 分配、断言说明、C=0.1、λ=20、seed、test_accessed=false；
  预测文件含每预测 row_id/family/generator/language/length/split/classes_。
- 只读依赖：b_train/b_val parquet 与 `runs/flagship_e28/features_train_val.npz`（md5 校验通过）；
  **未触碰** test/unseen 与 E25 features.npz。
