# d-det 旗舰域 E35 报告：论文级 H2 直接检验（全 generator 关系集）

> 规范来源：`docx/d-det_E35论文级H2直接检验指导_2026-09-30.md`（基线提交 `25824ca`）。
> 产物：`artifacts/flagship_e35/{config,manifest,metrics,predictions.npz,solver.log}`。

## 一句话结论

在 E30 七家族/23 generators 全关系集上，固定 B0 主轴 + 跨 generator 正对 SupCon（F1）相对纯残差 CE（F0）：**两折的 BA_F 与 BA_G 四个差值全部为正（方向一致），且有效 anchor 守卫通过（fold0 .714 / fold1 .286，检验充分）——但 pooled 增量仅 +0.28pt（BA_F）/+0.30pt（BA_G），远小于 1pt 门槛 ⇒ 按预注册出口 2：H2 在当前 768 维表示与跨 generator 正对定义下不支持；停止几何损失路线**。论文主线按规范 §5 收窄为"检测 + 描述性闭集归因 + 迁移损失与数据构型分析"。

## 摘要表（BA_F / BA_G；机会水平 1/7≈.1429）

| 模型 | fold0（val 5871） | fold1（val 5901） | pooled（11772） |
|---|---|---|---|
| 主轴 ℓ₀（加权 B0，冻结） | .3023 / .2934 | .2466 / .2230 | .2545 / .2442 |
| F0：主轴 + 残差 CE | .2998 / .2904 | .2441 / .2207 | .2510 / .2413 |
| F1：F0 + 0.1·L_cross（跨 gen 正对） | .3015 / .2912 | **.2480 / .2253** | **.2538 / .2443** |
| **Δ(F1−F0)** | **+0.17pt / +0.08pt** | **+0.39pt / +0.46pt** | **+0.28pt / +0.30pt** |
| 随机切分参照（主轴，仅参照） | .3701 / .3391 | .3795 / .3682 | — |

- 预注册出口 1 条件：两折 BA_F/BA_G 均正（✓ 4/4）**且 pooled ≥1pt（✗，+0.28/+0.30pt）** → 出口 1 不达成；按"增量小于 1 个百分点"适用**出口 2**。
- 出口 3（接近机会水平）：不适用——pooled F0/F1 BA_F ≈1.76/1.78×机会，且明显低于随机参照（迁移损失 ≈6.8–13.3pt，BA_F）。
- 出口 4（异质性）：增量量级 ≤0.6pt/族、主要集中在 Meta（+1.29pt）；instruct 类几乎持平、Google/DeepSeek 略负 → **只报异质性，不写成总体 H2 通过**。

## 1 协议与预注册

- **数据与折**：E30 七家族（排除单 generator 的 OpenAI）/23 generators；每家族 `RandomState(1000+fi)` 打乱 generator：2→1/1、3→1/2、5→2/3；fold0 val=A/train=B、fold1 互补。**折 generator 清单与样本数与 E30 `manifest/metrics` 逐项比对一致**（AI train/val：5901/5871、5871/5901）。
- **主模型**：ℓ₀(z)=W₀z+b₀ 为**加权 B0 主轴**（w=N/(K·|G_f|·N_g)，C=.1；逐折拟合后**全程冻结**，不参与梯度）；残差 r=V·GELU(Uz+b₁)+b₂；ℓ=ℓ₀+γ·W_R·r；**阶段 B 修正门控初始化**（W_R=0、η₀=0.5、γ=0.1·tanh(η)；U Xavier、V std=1e-3）。逐折**初始化审计**：W_R=0 ⇒ 训练前 ℓ≡ℓ₀（logits max|Δ|=0.00e+00，两折）。
- **两臂**：F0=L_F；F1=L_F+0.1·L_cross（τ=.1 只除一次；正对=同家族不同 generator；无正对 anchor 跳过；分母含全批；同 generator 正对不入损失）。F0/F1 共享初始化（deepcopy）、批序（同 RNG，重抽 0）、标准化（仅训练折，std≥1e-2）、epoch（2）与读出头。
- **训练协议**：批 7 族×18=126（族→generator→样本均匀）；47 step/epoch×2；AdamW lr 1e-3 / wd 1e-4 / clip 1；只训练残差支路。**不读 test/unseen；不调门控/τ/λ_C/lr/轮数**。
- **出口（config.json 预注册）**：见 §3。

## 2 结果

**2.1 管线一致性（E30 对接校验）**
- 未加权主轴参照**精确复现 E30 R0**：fold0 .3316/.3276、fold1 .2402/.2234（与 E30 完全一致）→ 折构造/标准化/评估管线同源无误。
- 检测 AUROC 与 E30 完全一致：fold0 .9481、fold1 .9419（E30 协议：全训练侧标准化 + C=.1）。
- 加权主轴（本轮主模型 ℓ₀）：fold0 .3023/.2934、fold1 .2466/.2230（加权在 fold0 上比未加权低 2.9pt；fold1 略高 0.6pt）。

**2.2 有效 anchor（跨 generator 正对覆盖率）**
- 整体：fold0 **.714**（8460/11844 批内样本位）、fold1 **.286**（3384/11844）；组合 .50。两折均远高于守卫阈值 .15。
- 逐族（比例）：fold0 — DeepSeek/Qwen/Meta/IBM/Google = 1.00（训练侧 ≥2 gens），Mistral/01-ai = 0（训练侧 1 gen）；fold1 — Qwen/Meta = 1.00，其余 = 0。即覆盖率结构完全由"该折训练侧 generator 数"决定，与设计预期一致。
- **注意（无剂量效应）**：anchor 较少的 fold1（.286）增量反而更大（+0.39/+0.46pt）于 anchor 丰富的 fold0（.714，+0.17/+0.08pt）。只作记录，不作机理解释。

**2.3 F0/F1 训练健康度**
- L_cross 首→末：fold0 6.37→4.82；fold1 6.55→4.71（对比项生效）。γ 稳定（0.041–0.046，两臂漂移极小）；残差参与度 2.3–5.1%。
- 参数漂移（Frobenius）：F1 的 ‖ΔU‖≈28.2/28.8、‖ΔV‖≈4.8/5.2、‖W_R‖≈0.28/0.49（零初始化生长）；F0↔F1 参数最大元素差 U .069/.073、V .054/.057、W_R .024/.043。
- 首末批 CE（F0/F1 同步记录）：fold0 ep1 末 0.8216/0.8216；fold1 ep1 末 0.6799/0.6608（F1 略低）。

**2.4 逐族与 role（pooled，F1−F0）**
- 逐族 ΔBA_F：Meta +1.29pt、Mistral +0.64pt、Qwen +0.35pt、IBM +0.06pt、01-ai +0.06pt、DeepSeek −0.12pt、Google −0.32pt。
- role（pooled 逐 generator 召回均值）：base .2242（F0 .2173，+0.7pt）、instruct-agent .2870（.2764，+1.1pt）、instruct .2948（.2934，+0.1pt）、instruct-reasoning .2207（.2207，±0）、chat .0413（.0413，±0）。
- 逐 generator 跨度极大（如 F1：gemma-3n-e4b .533 / gemma-3-27b .453 vs codegemma-2b .047；Qwen-1.5B-Instruct .434 vs Qwen-72B .125）——单 generator 子集统计噪声大，不逐点解释。

**2.5 与主轴/随机参照的关系**
- F0 在主轴基础上轻微退化（−0.25pt，两折一致，同阶段 B E0g 的形态）；F1 恢复到主轴水平（fold0 −0.08pt、fold1 +0.14pt）。
- 随机切分参照（主轴，仅参照不作门槛）：fold0 .3701/.3391、fold1 .3795/.3682 → **迁移损失（gen − random）fold0 −6.8/−4.6pt、fold1 −13.3/−14.5pt**，复现 E30"明显迁移损失"的读数。

## 3 判读（预注册出口逐条应用）

1. **出口 1（初步支持）**：两折方向一致性 ✓（4/4 为正），但 pooled ΔBA_F=+0.28pt、ΔBA_G=+0.30pt，均 <1pt → **不达成**。
2. **出口 2（不支持）**："方向不一致**或**增量小于 1 个百分点"——方向一致但增量 <1pt → **成立：H2 在当前 768 维表示与跨 generator 正对定义下不支持；停止几何损失路线**。
3. **出口 3（数据不足）**：不适用（F0/F1 均 ≈1.76/1.78×机会，且显著低于同本随机参照）。
4. **出口 4（异质性限制）**：增量集中在少数族/role（Meta、agent 类），instruct 持平、个别族略负 → 按规范只报异质性，**不写成总体通过**。
- **Anchor 守卫**：通过（.714/.286 ≥ .15）→ 本轮结论**不是"未充分检验"**，而是正式阴性（在跨 generator 正对定义下）。
- 单种子仅作候选证据；1pt 门槛不是显著性检验；不做事后调参解释。

## 4 边界与诚实性声明

- **混合 generator 来源**：数据含 base/instruct/reasoning/不同规模（base 8 / instruct 10 / reasoning 2 / chat 1 / agent 1，OpenAI 被排除），结果只能称"混合 generator 来源关系证据"，**不能称纯后训练因果证据**。
- 跨 generator 正对在各折的「可用家族」不对称（fold0 五族 / fold1 两族；Mistral、01-ai 训练侧永远单 generator）——策略固定，无事后调整；两折合并 pooled 是唯一覆盖全部样本的 generator-held-out 口径（每 generator 恰被留出一次）。
- pooled 低于两折均值不是错误：pooled 按逐族样本量混合（如 01-ai 在 fold0 侧 91 样本 recall .02、fold1 侧 9 样本 .44 → 合并 .05）——已在报告中按口径说明。
- F1 相对 F0 的 +0.28pt（BA_F）与 E31（+0.11pt）、阶段 B（+0.08/+0.29pt）量级一致；方向一致但幅度常数级偏小是全部来源增量证据的共同特征。
- 未使用 test/unseen；未调超参；随机切分仅作迁移损失参照。

## 5 终止条件与主线收窄（规范 §5）

F1 未达到出口 ⇒ 论文主线收窄为：

$$\text{后训练分布重加权动机} \rightarrow \text{768 维表示中的来源可读性} \rightarrow \text{跨 generator 迁移损失与数据构型分析}.$$

- **不再声称**"学习到纯粹归因空间"；不再继续调 SupCon 或添加 DCAN/DMHM 变体。
- H3（检测/归因任务分离）不进入（仅在出口 1 达成时才需要一次 DMHM 式局部几何消融；本轮未触发）。
- 阶段 B 的阴性结果保留为链条一环：控制集（阶段 B）与全关系集（E35）双阴性 ⇒ "受控来源增量"与"跨 generator 关系约束"在当前 768 维表示与预算下均无实用增益。

## 6 复现

```
cd /root/autodl-tmp/u-det/d-det
OMP_NUM_THREADS=8 /root/miniconda3/envs/udet/bin/python scripts/flagship_e35.py           # 全量（24.0s；本机 RTX 3080 Ti）
OMP_NUM_THREADS=8 /root/miniconda3/envs/udet/bin/python scripts/flagship_e35.py --smoke   # 冒烟（独立目录）
```

产物：`artifacts/flagship_e35/`（config.json、manifest.json、metrics.json、predictions.npz〔每折逐样本预测 + family/generator/role/size/language/length/classes〕、solver.log、本报告）。
