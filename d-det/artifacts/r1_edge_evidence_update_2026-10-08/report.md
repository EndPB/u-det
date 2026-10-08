# R1 证据更新：可识别性与表面控制（不升级因果）

- provenance 核对：两模式行齐备、任务覆盖完整（7/7 边断言通过）
- 表面回归对 Δ 的 R²=0.3620（fit=11172 行）
- 方向读数口径：torch 线性对称判别器（旧目录的 sklearn LR 口径在正交化残差上因每维均值为零而在零驻点失效；本目录统一 torch 口径）

## 逐边并列读数（instruct 侧）

| edge | delta | delta⊥surface | p0 | side | 证据等级 |
|---|---|---|---|---|---|
| E1_Athene70B_Llama3 | 0.708 | 0.591 | 0.421 | 0.719 | partial |
| E2_AtheneV2Agent_Qwen72B | 0.965 | 0.649 | 0.287 | 0.953 | partial |
| E3_AtheneV2Chat_Qwen72B | 0.959 | 0.620 | 0.275 | 0.942 | partial |
| E4_SkyT1Flash_Qwen32B | 0.971 | 0.573 | 0.450 | 0.977 | partial |
| E5_SkyT1Flash_SkyT1Preview | 0.854 | 0.877 | 0.251 | 0.415 | partial |
| E6_SkyT1Preview_Qwen32B | 0.977 | 0.579 | 0.409 | 0.971 | partial |
| E7_QwQ32B_Qwen32B | 0.965 | 0.474 | 0.515 | 0.965 | partial |

## edge-heldout（7 折 LOO；描述性）

- 留出边 delta 均值=0.889

| edge | heldout delta |
|---|---|
| E1_Athene70B_Llama3 | 0.573 |
| E2_AtheneV2Agent_Qwen72B | 0.942 |
| E3_AtheneV2Chat_Qwen72B | 0.947 |
| E4_SkyT1Flash_Qwen32B | 0.971 |
| E5_SkyT1Flash_SkyT1Preview | 0.854 |
| E6_SkyT1Preview_Qwen32B | 0.971 |
| E7_QwQ32B_Qwen32B | 0.965 |

## child-heldout（6 组 LOO；描述性）

| child | held edges | delta 均值 |
|---|---|---|
| Nexusflow--Athene-70B | E1_Athene70B_Llama3 | 0.573 |
| Nexusflow--Athene-V2-Agent | E2_AtheneV2Agent_Qwen72B | 0.942 |
| Nexusflow--Athene-V2-Chat | E3_AtheneV2Chat_Qwen72B | 0.947 |
| NovaSky-AI--Sky-T1-32B-Flash | E4_SkyT1Flash_Qwen32B,E5_SkyT1Flash_SkyT1Preview | 0.906 |
| NovaSky-AI--Sky-T1-32B-Preview | E5_SkyT1Flash_SkyT1Preview,E6_SkyT1Preview_Qwen32B | 0.909 |
| Qwen--QwQ-32B-Preview | E7_QwQ32B_Qwen32B | 0.965 |

> 全部边为 `partial observational`（documented base relation candidate）；不得写 verified post-training effect。
