# R1 交换接口修正（corrective rerun；不覆盖旧结果）

正确接口：pos=[S;D] vs neg=[S;−D]（交换只反 D）。旧 [S;D] vs −[S;D] 列为 sign-construction diagnostic。

| edge | swap_direction | old_diagnostic | side | p0 |
|---|---|---|---|---|
| E1_Athene70B_Llama3 | 0.965 | 1.000 | 0.778 | 0.819 |
| E2_AtheneV2Agent_Qwen72B | 0.994 | 1.000 | 1.000 | 1.000 |
| E3_AtheneV2Chat_Qwen72B | 0.994 | 1.000 | 1.000 | 1.000 |
| E4_SkyT1Flash_Qwen32B | 0.994 | 1.000 | 0.994 | 0.865 |
| E5_SkyT1Flash_SkyT1Preview | 0.380 | 1.000 | 0.374 | 0.316 |
| E6_SkyT1Preview_Qwen32B | 0.994 | 1.000 | 0.988 | 0.819 |
| E7_QwQ32B_Qwen32B | 0.994 | 1.000 | 0.988 | 0.830 |

- edge-heldout（LOO，instruct）均值: 0.835

## 版本/类型表（partial）
- E1_Athene70B_Llama3: level=partial; post-training or further training on documented base; specifics unverified
- E2_AtheneV2Agent_Qwen72B: level=partial; post-training or further training on documented base; specifics unverified
- E3_AtheneV2Chat_Qwen72B: level=partial; post-training or further training on documented base; specifics unverified
- E4_SkyT1Flash_Qwen32B: level=partial; post-training or further training on documented base; specifics unverified
- E5_SkyT1Flash_SkyT1Preview: level=partial; same-repo iteration (Sky-T1 Preview -> Flash)
- E6_SkyT1Preview_Qwen32B: level=partial; post-training or further training on documented base; specifics unverified
- E7_QwQ32B_Qwen32B: level=partial; post-training or further training on documented base; specifics unverified
