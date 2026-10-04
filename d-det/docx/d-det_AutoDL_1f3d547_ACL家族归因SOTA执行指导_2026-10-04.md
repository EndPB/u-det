# d-det AutoDL 执行指导：ACL 家族归因 SOTA 主线

## 0. 最高优先级目标

项目最终目标是：**在可审计、可复现的家族归因协议上，达到 ACL 论文级别的 SOTA 证据**。

H2 词法基线、捷径审计和 round2 编码器实验都是为这个目标服务的诊断环节。它们不能替代家族归因主实验，也不能因为某一轮未超过 TF-IDF 就结束方法探索。

当前代码锚点：

```text
1f3d547b0dbcfc9be67b23351f38076f47cbaf8a
```

## 1. 论文主指标和赛道

所有新实验必须把 **family attribution macro-F1** 作为主指标，同时报告：

- balanced accuracy；
- generator-held-out / family-internal unseen-generator；
- task 或 repository held-out；
- 每个 family、generator、language 的召回率；
- ECE 和 unknown-family 拒识 AUROC。

闭集随机切分、任务留出、仓库留出和未见 generator 必须分开成表，不能合并成一个总分。

主赛道分为：

1. 公开数据闭集归因：与最强公开基线同协议比较；
2. task/repository-held-out：检查是否依赖题目和仓库；
3. family 内 unseen-generator：作为核心鲁棒性目标；
4. open-set：未知 family 不应被强行分配到已知 family。

完整路线参考：`docx/d-det_SOTA家族归因路线图_2026-10-02.md`。

## 2. 已完成工作如何服务 SOTA 目标

### H2 pair 首轮

- `rows=4692`；包内审计 27/27 全通过；`invalid_records=0`。
- TF-IDF 五视图已完成第 5 次逐字节一致复跑。
- 结论是表面 token、标识符、字符串和注释信号必须纳入控制变量，不是停止家族归因研究。

### round2 编码器诊断

- 冻结表示和轻量头低于 TF-IDF，LoRA 有改善但证据不足以称为 SOTA。
- 单变量 SupCon、GRL、协方差项没有稳定贡献，说明下一步不能继续盲目叠加损失。
- 下一轮应转向更强的**多视图来源指纹 + 任务条件化配对 + 跨 generator 评测**。

## 3. 下一阶段实验顺序

### P0：复现强基线

在同一数据和切分上固定复现：

1. 字符/词法 stylometry；
2. CodeT5/CodeT5-Authorship 类监督基线；
3. AST、控制流、命名、注释和格式统计；
4. 成对变换特征与轻量集成。

没有 P0 的完整表格，不把新模型写成 SOTA。

### P1：多视图 late fusion

建立统一特征接口，保留以下分支：

- CodeT5/代码编码器语义表示；
- AST 与控制流统计；
- token、命名、注释、格式 stylometry；
- 同任务 code-to-code 差异；
- 长度、语言、任务、仓库匹配控制。

先做冻结特征和线性/树模型 late fusion，确认增益来自 family 信息，再进入端到端训练。

### P2：任务条件化训练

仅在有明确 `task_id` 的数据上使用配对约束：

- semantic branch：同任务跨 generator 对齐；
- fingerprint branch：同 family 聚合、跨 family 分离；
- nuisance heads：抑制 task、language、repository、length 泄漏；
- family classifier：主损失；
- generator head 和 prototype/retrieval head：辅助损失。

正对必须是同任务、同 source、同 split、同 family、不同 generator。无 `task_id` 的数据只能做外部压力测试，不承担核心 SupCon 训练。

### P3：跨 generator 主实验

预注册三类测试：

1. 闭集同 generator；
2. task/repository held-out；
3. family 内 unseen-generator。

只有两个独立 source、至少两个 seed 或折叠方向一致，且清洗视图仍保留增益，才扩大训练预算。

## 4. AutoDL 开机后的执行守则

```bash
cd /root/autodl-tmp/u-det/d-det
git rev-parse HEAD
git status --short --branch
```

必须确认当前提交是 `1f3d547b0dbcfc9be67b23351f38076f47cbaf8a`。然后：

```bash
source /root/miniconda3/etc/profile.d/conda.sh
conda activate udet
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
export TOKENIZERS_PARALLELISM=false
```

资源规则：

- 不下载或生成 35GB 原始数据；
- 不覆盖 H2 round1 或 ACL round1–round4 产物；
- 每个新方法使用独立目录；
- 记录 commit、split manifest、seed、环境变量和 SHA-256；
- test 只在协议冻结后评估一次。

## 5. 成功标准

家族归因达到论文级 SOTA，至少需要同时满足：

1. 在公开数据同协议上超过最强公开基线；
2. task/repository-held-out 仍有正增益；
3. family 内 unseen-generator 不低于强 stylometry 和 CodeT5 基线；
4. 去掉任务、语言、长度和表面视图后，增益仍可解释；
5. 多 seed 或多折方向一致；
6. unknown-family 有可校准的拒识，而不是封闭集强制分类。

单一随机切分上的高分、单个 source 的提升或 raw-only 增益，都不构成 SOTA 证据。

## 6. 当前决策

当前不重复 H2 基线，也不继续无目标地堆叠 round2 损失。下一次 AutoDL 工作应进入 P0/P1：复现强基线、统一多视图特征和评测，再以 P2 任务条件化训练作为主方法路线。

论文叙事应聚焦：**任务语义、代码结构和来源指纹的分离建模，能否在严格 unseen-generator 家族归因上超过表面基线。**
