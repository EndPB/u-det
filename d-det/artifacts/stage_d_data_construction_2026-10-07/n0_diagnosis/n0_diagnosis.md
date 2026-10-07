# N0：D1 失败来源只读复盘（2026-10-07）

依据《d-det_AutoDL_D1未过闸门_数据构造下一步指导_2026-10-07.md》§2。**只读**：不训练、不写权重；输入为 D1 `metrics.json`/`predictions.npz` 与 `h2_authorbench_dcan/core.jsonl`。

## 0 已知现象（D1，作为分层诊断引用，不作因果结论）

- test n=1457：tfidf_word `.8200` / p0_fusion_lr `.8388` / codet5_centered `.6487`；task-size `6+` 桶 tfidf_word F1 `.7644`；generator：deepseek `.601`、qwen `.646` vs gemini `.974`、gpt-4.1 `.940`。

## 1 数据结构（train/dev/test；family×generator）

| split | rows | tasks | family rows |
|---|---|---|---|
| train | 6557 | 1900 | claude:821 deepseek:864 gemini:803 llama:893 openai:2296 qwen:880 |
| dev | 1484 | 407 | claude:194 deepseek:196 gemini:183 llama:194 openai:518 qwen:199 |
| test | 1457 | 408 | claude:177 deepseek:188 gemini:193 llama:201 openai:506 qwen:192 |

任务组成：family coverage 分布 {"2": 1413, "3": 564, "4": 252, "5": 201, "6": 285}；task-size 分布 {"2": 1178, "3": 647, "4": 285, "5": 154, "6": 105, "7": 107, "8": 239}

| family | generator | rows | tasks | rows tr/dv/te | tasks tr/dv/te |
|---|---|---|---|---|---|
| claude | claude-3.5-haiku | 1192 | 1192 | train:821/dev:194/test:177 | train:821/dev:194/test:177 |
| deepseek | deepseek-chat | 1248 | 1248 | train:864/dev:196/test:188 | train:864/dev:196/test:188 |
| gemini | gemini-2.5-flash-preview-05-20 | 1179 | 1179 | train:803/dev:183/test:193 | train:803/dev:183/test:193 |
| llama | llama-3.3-70b-instruct | 1288 | 1288 | train:893/dev:194/test:201 | train:893/dev:194/test:201 |
| openai | gpt-4.1 | 1131 | 1131 | train:784/dev:165/test:182 | train:784/dev:165/test:182 |
| openai | gpt-4o | 1062 | 1062 | train:728/dev:172/test:162 | train:728/dev:172/test:162 |
| openai | gpt-4o-mini | 1127 | 1127 | train:784/dev:181/test:162 | train:784/dev:181/test:162 |
| qwen | qwen-2.5-72b-instruct | 1271 | 1271 | train:880/dev:199/test:192 | train:880/dev:199/test:192 |

## 2 分层诊断（test；macro-F1 / BA / CI95 为 500× task-cluster bootstrap）

### 2.1 family×generator

| family/generator | view | n | macro-F1 | BA | CI95 |
|---|---|---|---|---|---|
| claude / claude-3.5-haiku | tfidf_word | 177 | 0.1918 | 0.9209 | [0.1867,0.2847] |
| claude / claude-3.5-haiku | p0_fusion_lr | 177 | 0.2457 | 0.9661 | [0.2427,0.4972] |
| claude / claude-3.5-haiku | codet5_centered | 177 | 0.1535 | 0.8531 | [0.1484,0.2331] |
| deepseek / deepseek-chat | tfidf_word | 188 | 0.1251 | 0.6011 | [0.1165,0.1605] |
| deepseek / deepseek-chat | p0_fusion_lr | 188 | 0.1305 | 0.6436 | [0.1230,0.1668] |
| deepseek / deepseek-chat | codet5_centered | 188 | 0.1063 | 0.4681 | [0.0969,0.1366] |
| gemini / gemini-2.5-flash-preview-05-20 | tfidf_word | 193 | 0.3290 | 0.9741 | [0.3245,0.4987] |
| gemini / gemini-2.5-flash-preview-05-20 | p0_fusion_lr | 193 | 0.4961 | 0.9845 | [0.4908,1.0000] |
| gemini / gemini-2.5-flash-preview-05-20 | codet5_centered | 193 | 0.1604 | 0.9275 | [0.1576,0.2447] |
| llama / llama-3.3-70b-instruct | tfidf_word | 201 | 0.1845 | 0.8557 | [0.1788,0.2366] |
| llama / llama-3.3-70b-instruct | p0_fusion_lr | 201 | 0.1797 | 0.8159 | [0.1725,0.2284] |
| llama / llama-3.3-70b-instruct | codet5_centered | 201 | 0.1374 | 0.7015 | [0.1297,0.1696] |
| openai / gpt-4.1 | tfidf_word | 182 | 0.2422 | 0.9396 | [0.2378,0.4902] |
| openai / gpt-4.1 | p0_fusion_lr | 182 | 0.4986 | 0.9945 | [0.4958,1.0000] |
| openai / gpt-4.1 | codet5_centered | 182 | 0.1420 | 0.7418 | [0.1344,0.1725] |
| openai / gpt-4o | tfidf_word | 162 | 0.1781 | 0.8025 | [0.1718,0.2309] |
| openai / gpt-4o | p0_fusion_lr | 162 | 0.1742 | 0.7716 | [0.1661,0.2254] |
| openai / gpt-4o | codet5_centered | 162 | 0.1045 | 0.4568 | [0.0933,0.1377] |
| openai / gpt-4o-mini | tfidf_word | 162 | 0.1563 | 0.8827 | [0.1521,0.2383] |
| openai / gpt-4o-mini | p0_fusion_lr | 162 | 0.1916 | 0.9198 | [0.1868,0.3227] |
| openai / gpt-4o-mini | codet5_centered | 162 | 0.1129 | 0.5123 | [0.1010,0.1232] |
| qwen / qwen-2.5-72b-instruct | tfidf_word | 192 | 0.1570 | 0.6458 | [0.1474,0.2042] |
| qwen / qwen-2.5-72b-instruct | p0_fusion_lr | 192 | 0.1622 | 0.6823 | [0.1534,0.2117] |
| qwen / qwen-2.5-72b-instruct | codet5_centered | 192 | 0.1277 | 0.4688 | [0.1145,0.1405] |

### 2.2 task-size 桶（1 / 2–5 / 6+）

| bucket | view | n | macro-F1 | BA | CI95 |
|---|---|---|---|---|---|
| size 2-5 | tfidf_word | 1457 | 0.8200 | 0.8125 | [0.7966,0.8400] |
| size 2-5 | p0_fusion_lr | 1457 | 0.8388 | 0.8319 | [0.8189,0.8588] |
| size 2-5 | codet5_centered | 1457 | 0.6487 | 0.6660 | [0.6227,0.6715] |

### 2.3 family×task-size（无 CI，n<24 单元格从 JSON 读全量）

| family/size | view | n | macro-F1 | BA | CI95 |
|---|---|---|---|---|---|
| claude / 2-5 | tfidf_word | 177 | 0.1918 | 0.9209 | skipped_small_n |
| claude / 2-5 | p0_fusion_lr | 177 | 0.2457 | 0.9661 | skipped_small_n |
| claude / 2-5 | codet5_centered | 177 | 0.1535 | 0.8531 | skipped_small_n |
| deepseek / 2-5 | tfidf_word | 188 | 0.1251 | 0.6011 | skipped_small_n |
| deepseek / 2-5 | p0_fusion_lr | 188 | 0.1305 | 0.6436 | skipped_small_n |
| deepseek / 2-5 | codet5_centered | 188 | 0.1063 | 0.4681 | skipped_small_n |
| gemini / 2-5 | tfidf_word | 193 | 0.3290 | 0.9741 | skipped_small_n |
| gemini / 2-5 | p0_fusion_lr | 193 | 0.4961 | 0.9845 | skipped_small_n |
| gemini / 2-5 | codet5_centered | 193 | 0.1604 | 0.9275 | skipped_small_n |
| llama / 2-5 | tfidf_word | 201 | 0.1845 | 0.8557 | skipped_small_n |
| llama / 2-5 | p0_fusion_lr | 201 | 0.1797 | 0.8159 | skipped_small_n |
| llama / 2-5 | codet5_centered | 201 | 0.1374 | 0.7015 | skipped_small_n |
| openai / 2-5 | tfidf_word | 506 | 0.1558 | 0.8775 | skipped_small_n |
| openai / 2-5 | p0_fusion_lr | 506 | 0.1894 | 0.8992 | skipped_small_n |
| openai / 2-5 | codet5_centered | 506 | 0.1220 | 0.5771 | skipped_small_n |
| qwen / 2-5 | tfidf_word | 192 | 0.1570 | 0.6458 | skipped_small_n |
| qwen / 2-5 | p0_fusion_lr | 192 | 0.1622 | 0.6823 | skipped_small_n |
| qwen / 2-5 | codet5_centered | 192 | 0.1277 | 0.4688 | skipped_small_n |

## 3 错误审计（exact / normalized / 跨 split / generator 偏置）

- 全库 exact 重复组：0（跨 split 0）；norm_ws：0（跨 split 0）；norm_lex：46（跨 split 25）；同 task 跨 split：0

| view | n_err | err_rate | exact dup(te) | norm dup(te) | exact code in train | norm code in train |
|---|---|---|---|---|---|---|
| tfidf_word | 253 | 0.1736 | 0 | 0 | 0 | 0 |
| p0_fusion_lr | 225 | 0.1544 | 0 | 0 | 0 | 0 |
| codet5_centered | 516 | 0.3542 | 0 | 0 | 0 | 0 |

`tfidf_word` 每 generator 错误与主要误判目标：

| generator | n_err | top wrong targets |
|---|---|---|
| claude-3.5-haiku | 14 | openai×7; deepseek×3; llama×2 |
| deepseek-chat | 75 | openai×43; qwen×18; llama×9 |
| gemini-2.5-flash-preview-05-20 | 5 | openai×4; claude×1 |
| gpt-4.1 | 11 | deepseek×9; llama×1; qwen×1 |
| gpt-4o | 32 | qwen×20; llama×6; deepseek×5 |
| gpt-4o-mini | 19 | llama×6; deepseek×6; qwen×5 |
| llama-3.3-70b-instruct | 29 | openai×17; qwen×7; deepseek×4 |
| qwen-2.5-72b-instruct | 68 | openai×37; llama×18; deepseek×11 |

top confusions：deepseek->openai×43; qwen->openai×37; openai->qwen×26; openai->deepseek×20; qwen->llama×18; deepseek->qwen×18; llama->openai×17; openai->llama×13

`p0_fusion_lr` 每 generator 错误与主要误判目标：

| generator | n_err | top wrong targets |
|---|---|---|
| claude-3.5-haiku | 6 | openai×3; deepseek×2; qwen×1 |
| deepseek-chat | 67 | openai×33; qwen×18; llama×12 |
| gemini-2.5-flash-preview-05-20 | 3 | openai×3 |
| gpt-4.1 | 1 | deepseek×1 |
| gpt-4o | 37 | qwen×19; deepseek×13; llama×4 |
| gpt-4o-mini | 13 | deepseek×6; qwen×4; llama×2 |
| llama-3.3-70b-instruct | 37 | openai×17; qwen×14; claude×3 |
| qwen-2.5-72b-instruct | 61 | openai×36; llama×15; deepseek×9 |

top confusions：qwen->openai×36; deepseek->openai×33; openai->qwen×23; openai->deepseek×20; deepseek->qwen×18; llama->openai×17; qwen->llama×15; llama->qwen×14

## 4 失败来源判断（只读结论）

1. **组成/支持不均衡证据**：仅 285/2715 个 task 含全部 6 族（占 10.5%）；family coverage 2~6 分布见 §1；task-size 与 family×task-size 分层显示误差随组成变化——D1 的 test 评价混合了不同竞争集，不能直接读作纯内容可读性。
2. **generator 异质证据**：同一 family 内只有 openai 有 3 个 generator；其余 5 个 family 各 1 个——family 级数字实际绑定单个 generator 的指纹，跨 generator 归因信号无法与 family 语义分离。
3. **哈希跨 split 审计**：exact 跨 split 0 组、norm_ws 0 组、**norm_lex（去掉注释/字符串后）25 组**（涉及 119 行）——存在少量跨 split 骨架重复，是真实的数据卫生问题（已作为 N1 剔除依据）；同 task 跨 split = 0。D1 低于 P0 不是 exact/ws 级重复或同 task 泄漏伪影。
4. **需 N1 才能回答的部分**：现有数据能否提供 ≥3 个可靠 generator 的正式 family；以及组成均匀化（六族齐全子集）后内容信号是否仍低于 P0（N2 预注册后执行）。

