# R1：lineage 候选 train/dev 评估（partial）

> 所有边 `partial`（HF cardData.base_model 文档；version 级证据缺）；observational，非随机化；
> task-heldout（fit=798 train → eval=171 dev）；CI=task-cluster (300, seed 20261008)。

| edge | child | base | Δ判对 | single | [S;Δ] | P0 |
|---|---|---|---|---|---|---|
| E1_Athene70B_Llama3 | Nexusflow--Athene-70B | meta-llama--Meta-Llama-3-70B-Instruct | 0.982 | 0.781 | 1.000 | 0.813 |
| E2_AtheneV2Agent_Qwen72B | Nexusflow--Athene-V2-Agent | Qwen--Qwen2.5-72B-Instruct | 1.000 | 1.000 | 1.000 | 1.000 |
| E3_AtheneV2Chat_Qwen72B | Nexusflow--Athene-V2-Chat | Qwen--Qwen2.5-72B-Instruct | 1.000 | 1.000 | 1.000 | 1.000 |
| E4_SkyT1Flash_Qwen32B | NovaSky-AI--Sky-T1-32B-Flash | Qwen--Qwen2.5-32B-Instruct | 0.988 | 0.994 | 1.000 | 0.795 |
| E5_SkyT1Flash_SkyT1Preview | NovaSky-AI--Sky-T1-32B-Flash | NovaSky-AI--Sky-T1-32B-Preview | 0.749 | 0.553 | 1.000 | 0.789 |
| E6_SkyT1Preview_Qwen32B | NovaSky-AI--Sky-T1-32B-Preview | Qwen--Qwen2.5-32B-Instruct | 0.988 | 0.988 | 1.000 | 0.760 |
| E7_QwQ32B_Qwen32B | Qwen--QwQ-32B-Preview | Qwen--Qwen2.5-32B-Instruct | 0.994 | 0.988 | 1.000 | 0.778 |

## edge 识别（7-way）
- dev acc=0.416（chance 0.143），macro-F1=0.389
- per-edge acc: E1_Athene70B_Llama3=0.71, E2_AtheneV2Agent_Qwen72B=0.13, E3_AtheneV2Chat_Qwen72B=0.75, E4_SkyT1Flash_Qwen32B=0.15, E5_SkyT1Flash_SkyT1Preview=0.60, E6_SkyT1Preview_Qwen32B=0.34, E7_QwQ32B_Qwen32B=0.23

## 合法表述（§4.3）
- 全部 edge 为 partial → 只能写“模型卡记录的 base relation 候选”；不得进入论文标题/摘要的 post-training effect。
- 八条硬条件中 documented relation ✓；same task/protocol ✓；independent provenance ✓；version 独立证据缺 → partial。
