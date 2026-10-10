# H3 本机 hard-negative 修订 v4 与 AutoDL 接收指导

版本：2026-10-11  
状态：`revise_data`；v4 已通过 hard-negative 视图的长度阈值，但 untouched clean-dev 仍未通过，尚未授权 GPU。

## 1. v4 做了什么

本机 CPU 使用精确配额约束的最小成本匹配：每个 task 固定 1 个 human + 3 个不同 generator 的 AI；每个 generator 在整个 split 获得精确全局配额；目标函数为 `0.70 × log-length 差异 + 0.30 × (1 − masked-token cosine)`。这避免 v3 的轮换配额把长度差残留留给某些 generator。

脚本：`scripts/h3_hardneg_revision_v4.py`。它复用 v3 的掩码、probe 和读取函数，包内同时保留 v3 脚本。  
数据：`d-det/data/h3_stacad_revision_v4/`（gitignore）。  
证据：`d-det/artifacts/h3_hardneg_revision_v4_2026-10-11/`。

## 2. 读数

| 视图 | raw task-macro | masked task-macro | length task-macro |
|---|---:|---:|---:|
| untouched clean-dev | .8813 | .6547 | .7248 |
| hard-negative v4 | .8427 | .6343 | .6907 |

hard-negative v4 的长度读数已低于 `.70`，掩码词法也继续下降；但 untouched clean-dev 的 `.7248` 仍超过预注册阈值。因此本轮裁定仍为 `formal_gpu_ready=false`、`gate=revise_data`。

## 3. 来源边界

已逐个检查 STACAD corpus/crossprompt 的 JSONL 字段：corpus 只有 `file_name/human_src/llm_src/paraphrased_by/label/split_v1`，crossprompt 只有 prompt/model 字段。没有 repository、project 或 source URL。v4 不从文件名、代码注释或 task_id 猜 project；`provenance_audit.json` 固定记录 `project_provenance=unavailable`。正式结果只能声称 task-heldout，除非另有可审计来源包。

## 4. AutoDL 处理

v4 包上传 AutoDL 只做接收、哈希、编译、行数/配额/指标复核，不启动训练。只有下列条件全部满足，才生成新的 `batch_manifest.json` 并一次性执行完整 GPU 矩阵：

1. untouched clean-dev 的长度捷径也通过，或预注册协议明确把 v4 作为正式评估视图并继续报告 clean-dev 失败；
2. project provenance 补齐，或正式文字严格限制为 task-heldout；
3. lexical control、detection-only、source-only、joint、joint-invariance（可选 adversary）至少两折 × 三 seed 一次性冻结；
4. test、生成、权重下载和代码执行开关保持 false，统一回传指标、逐行 digest、日志、环境和 SHA256SUMS。

