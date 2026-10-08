# R1 后训练比较（corrective；train/dev）

| edge | delta_only(I) | raw_joint_lr(I) | mlp_swap(I) | side(I) | full_P0(I) |
|---|---|---|---|---|---|
| E1_Athene70B_Llama3 | 0.983 | 0.965 | 0.924 | 0.765 | 0.826 |
| E2_AtheneV2Agent_Qwen72B | 1.000 | 0.994 | 1.000 | 1.000 | 1.000 |
| E3_AtheneV2Chat_Qwen72B | 1.000 | 0.994 | 0.988 | 1.000 | 1.000 |
| E4_SkyT1Flash_Qwen32B | 0.987 | 0.994 | 0.988 | 0.994 | 0.850 |
| E5_SkyT1Flash_SkyT1Preview | 0.751 | 0.373 | 0.389 | 0.367 | 0.313 |
| E6_SkyT1Preview_Qwen32B | 0.987 | 0.994 | 0.988 | 0.988 | 0.809 |
| E7_QwQ32B_Qwen32B | 0.993 | 0.993 | 1.000 | 0.988 | 0.812 |

（complete 侧见 metrics；逐项 CI95 同文件）

## 数学边界
- 线性 swap 分数差 f([S,D])−f([S,−D])=2·w_D·D（S 项与截距消去）——线性 swap 读数不是 S-A 非线性交互证据；小 MLP 版为非线性对照。

## checkpoint/component 隔离支持表

| edge | 共享端点 | 移除 incident 后训练边 | isolated_rawLR(comp/inst) | 状态 |
|---|---|---|---|---|
| E1_Athene70B_Llama3 | — | 6 条 | 0.550/0.526 | evaluable_descriptive |
| E2_AtheneV2Agent_Qwen72B | Qwen--Qwen2.5-72B-Instruct | 5 条 | 1.000/0.988 | evaluable_descriptive |
| E3_AtheneV2Chat_Qwen72B | Qwen--Qwen2.5-72B-Instruct | 5 条 | 0.994/0.982 | evaluable_descriptive |
| E4_SkyT1Flash_Qwen32B | NovaSky-AI--Sky-T1-32B-Flash,Qwen--Qwen2.5-32B-Instruct | 3 条 | 0.930/0.947 | evaluable_descriptive |
| E5_SkyT1Flash_SkyT1Preview | NovaSky-AI--Sky-T1-32B-Flash,NovaSky-AI--Sky-T1-32B-Preview | 4 条 | 0.339/0.374 | evaluable_descriptive |
| E6_SkyT1Preview_Qwen32B | NovaSky-AI--Sky-T1-32B-Preview,Qwen--Qwen2.5-32B-Instruct | 3 条 | 0.912/0.930 | evaluable_descriptive |
| E7_QwQ32B_Qwen32B | Qwen--Qwen2.5-32B-Instruct | 4 条 | 0.860/0.959 | evaluable_descriptive |

> 全部边仍 partial；不得写 verified post-training effect 或 checkpoint-heldout 泛化。
