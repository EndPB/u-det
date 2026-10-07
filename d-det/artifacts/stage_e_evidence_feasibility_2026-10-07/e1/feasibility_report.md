# E1：独立数据可行性（2026-10-07，仅元数据扫描；未下载/未生成/未训练）

依据指导 §3；输入为服务器已有 summary/manifest/审计 JSON（hash 见 `dataset_feasibility.json.inputs`）。

## 1 论文目标与 family 语义

- 主任务 = 模型来源/家族归因；detection 与拒识另表。family 语义必须在协议中明确为 vendor group / 共享谱系 lineage / generator 集合；同 vendor 不同版本（gpt-4.x）不等于已知共享权重谱系；Llama 系列也不能凭名字当作唯一同基座对。
- 缺 base/SFT/DPO 配对元数据（LCv2 README 明示）；本轮不把 task 中心化当作后训练因果证据。

## 2 现有来源可识别性（要点；全量见 JSON）

| dataset | rows | tasks | family/generator | dup audit | test 暴露 | 可独立留出 |
|---|---|---|---|---|---|---|
| h2_authorbench_dcan | 9498 | 2715 | 6 vendor 族 / 8 gen | exact/ws/lex done | 已评估（3 次 test 读取） | no |
| h2_llm_codegen_v2 | 1512 | 168 | 5 族 / 9 gen（meta=3，google/mistral=2） | not run | partial（pair 读 22 task） | partial |
| h2_stacad_* | 144958 | 22053 | 7 models（非 vendor 分组） | not run | yes | no |
| h2_droid_full_selected | 146718 | gen-heldout | 32 machine gen | exact 有 | yes | no |
| aicd_t2（numeric） | 1111199 | missing | 12 numeric（映射缺失） | exact 有（跨 split 1,848） | yes | no |
| codet_m4/balanced | 500552 | missing | model 级（含 null） | not run | yes | no |

## 3 覆盖率与折支持（回传 §6.4）

- AB 每族 generator 数：openai=3；claude/deepseek/gemini/llama/qwen=1（`coverage_matrix.csv`）。
- LCv2：meta=3（三折 admitted）；google=2、mistral=2（4 折 diagnostic-only）；ibm/microsoft=1。
- 10 折 train/dev 正支持与 test task 数：见 `dataset_feasibility.json.fold_support`。
- 同任务覆盖：AB 2,715 task 中六族齐全 285（剔除后 280）；LCv2 每 task 9 输出（5 族）。

## 4 pilot 蓝图（只规划；training_allowed=false）

- status = **partial**：现成 triple 仅 openai(AB) 与 meta(LCv2)；google/mistral 缺第 3 个 generator；任务池无“同一新任务集”的现成组合。
- 估计规模：≈300 task × 3 族 × 3 gen ≈ 2,700 输出 ≈ 4MB 原文 + ~10MB 索引/缓存（增量磁盘 ≈15MB 量级）。
- 功效：power_unknown（粗规则：~1,200 task 量级才能分辨 1pt；需预注册模拟）。
- 缺口：第 3 个 google/mistral generator（生成授权）、新任务提示集、闭源 API/预算、license 确认。

## 5 出口

- E1 = **partial**：交付缺口清单与预算；训练/生成/外部调用**未请求、未执行**；不自行删除历史资产。

