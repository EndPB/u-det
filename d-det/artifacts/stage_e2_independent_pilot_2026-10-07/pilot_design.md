# pilot_design：为什么该 unit（不）能检验 H2（2026-10-07）

## Path A：Meta / h2_llm_codegen_v2（最接近 ready）

- 三 generator 齐备（llama2/llama3/codellama；非空 code task 覆盖 134/152/164 of 168），D0 已有 3 个 admitted heldout 折；
- 能检验：same_task/same_family/different_generator 正对在同族内跨 generator 的判别/迁移（H2 的工程形态）与 H1 族内可读性；干跑池 144 task（保守排除原 test 24 后）。
- 不能检验 / 风险：该命名空间的 task 已用于 pair 轮 train/dev 开发——若指导端不接受此口径，则不能称为“新任务集”；输出为已有生成产物（复用需授权）；无 base/SFT/DPO 元数据 ⇒ 不能检验后训练因果。

## Path B：google/mistral + 第三 generator

- 理论上补齐后同样能检验 H2；但现在**无法给出任何真实可用的第三 generator**（本机无权重/无 API/版本未定）→ 不可检验。

## Path C：OpenAI / AB

- AB 命名空间已暴露，新任务必须全新；需要闭源 API 与新任务集 → 当前不可检验。

## 三个路径共同不能检验的内容

- 跨 dataset 拼接的“普遍 family 几何”；后训练因果；H3 detection/private-adapter（本轮不涉及）。

