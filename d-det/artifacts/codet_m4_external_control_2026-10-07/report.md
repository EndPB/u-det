# CoDET-M4 external source-fingerprint control

日期：2026-10-07
状态：完成；这是外部控制与捷径审计，不是同任务 H2 配对证据。

## 数据边界

使用本地 `codet_m4_balanced_control_v1` 固定包（CoDET-M4 revision `4d4e665037cb797cb5381c0b95c1d33e30420b8e`），共 18,870 行，拆分为 train/val/test。每个 split × model × language 单元按固定 hash 选择，跨 split 代码 hash 为 0。包中 `model` 是 CoDET-M4 自己的来源标识，不能映射成 Droid、AuthorBench、LLM-CodeGen、STACAD 或 AICD 的 family。

该包没有 task/prompt id，源文件已去除 comments。因此它只回答两个外部控制问题：

1. 代码文本和结构特征能否区分 `target=ai/human`？
2. 在五个具名 AI model（`codellama/gpt/llama3.1/nxcode/qwen1.5`）内部，来源模型是否仍有可读信号？

`human` 和 `__none__` 不被伪装成五个 AI model 中的类别；它们只进入 detection 任务。

## 固定协议

- 文本特征：字符 TF-IDF，3–5 grams，`min_df=2`，最多 60,000 特征，sublinear TF；只在 train 拟合词表。
- 结构特征：上游八个数值字段 `avgFunctionLength`、`avgIdentifierLength`、`avgLineLength`、`emptyLinesDensity`、`functionDefinitionDensity`、`maxDecisionTokens`、`maintainabilityIndex`、`whiteSpaceRatio`；train 拟合标准化器。
- 读出：`LogisticRegression(class_weight="balanced", max_iter=1000)`，固定参数，不用 test 选参。
- 评估：固定 val/test；报告 macro-F1、balanced accuracy 以及按语言的 test 分解。
- 重跑入口：[`codet_m4_external_control_baseline.py`](../../scripts/codet_m4_external_control_baseline.py)；原始结果：[`metrics.json`](metrics.json)。

## Test results

| task | feature | n | BA | macro-F1 |
|---|---|---:|---:|---:|
| human/AI detection | char TF-IDF | 4,191 | **.8916** | **.8806** |
| human/AI detection | structural features | 4,191 | .6777 | .6506 |
| five-model attribution | char TF-IDF | 2,992 | **.5636** | **.5532** |
| five-model attribution | structural features | 2,992 | .3002 | .2788 |

字符模型在三个语言上的 detection BA 为 C++ `.9103`、Java `.8798`、Python `.8847`；五模型 attribution BA 为 C++ `.6080`、Java `.6078`、Python `.4738`。结构特征的五模型 attribution BA 为 C++ `.2830`、Java `.2873`、Python `.3303`，接近但高于五类机会水平 `.2000`，显示结构统计存在有限来源捷径，但不能替代代码语义。

## 结论边界

1. CoDET-M4 的 detection 信号很强，且纯结构特征也有可见信号；因此任何 ACL family attribution 改进都必须报告结构/语言控制，避免把检测或仓库特征误写成家族几何。
2. 五模型 attribution 的字符 BA `.5636` 明显高于结构 BA `.3002`，但这是同一 CoDET-M4 source label space 内的闭集外部控制，不能与 Droid family BA 或 H2 generator-heldout BA 直接排序。
3. 包没有同题跨 generator 配对，不支持 H2 正对、任务中心化因果解释或 generator-invariant 结论。
4. `target`、`model`、`language`、`source` 的角色必须分别登记；不能把 `target=ai` 当作 family，不能把缺失 model `__none__` 变成新 generator。
