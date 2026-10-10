# H3 本机 hard-negative 修订 v3 与 AutoDL 下一批指导

版本：2026-10-11  
状态：`revise_data`；本轮 CPU 数据构造与诊断已完成，尚未授权新的 GPU 主批次。

## 1. 本轮完成

本机直接完成了 H3 v3 数据侧修订，没有读取 test，也没有重跑已关闭的 C0–C3 或 H3 GPU 诊断：

- 从 v1 train/dev 的同一 task 中，为每个 task 保留 1 个 human 与 3 个不同 generator 的 AI；候选按 `0.70 × log1p(bytes)` 差异加 `0.30 × masked-token cosine` 排序；
- generator 采用轮换配额，训练 2,096 task / 8,384 行，dev 659 task / 2,636 行，七个 generator 近似等量；
- hard-negative 只是同 task 配对视图，不是项目留出。原始 alignment 只有 `source_ref=corpus 文件/行`、`file_name`、`task_id`，没有 repository/project/source URL 字段，已由 `provenance_audit.json` 固定记录为 `unavailable`；
- 本机完成 raw、masked-token、length 的 CPU probe，并生成 JSONL、manifest、假设、环境、命令和 SHA256SUMS。

脚本：`scripts/h3_hardneg_revision_v3.py`。  
数据：`d-det/data/h3_stacad_revision_v3/`（gitignore）。  
证据：`d-det/artifacts/h3_hardneg_revision_v3_2026-10-11/`。

## 2. 读数与裁定

| 视图 | raw task-macro | masked task-macro | length task-macro |
|---|---:|---:|---:|
| untouched clean-dev | .8813 | .6547 | .7248 |
| hard-negative v3 | .8778 | .6578 | .7109 |

掩码后词法捷径已显著下降，但长度在 clean-dev 和 hard-negative v3 都仍超过 `.70` 预注册阈值。该结果说明 v3 是有用的挑战集和诊断包，却还不能支撑“正式主模型已摆脱表面信号”的结论。

正式裁定：`formal_gpu_ready=false`、`gate=revise_data`。没有新的 `batch_manifest.json`，不触发 GPU 训练。

## 3. AutoDL 接收原则

本 v3 包可以上传 AutoDL 做完整性、数据角色和 CPU smoke 接收；接收不等于 GPU 授权。服务器侧不得在当前包上自行扩展模型、温度、轮数或损失项，也不得重跑已关闭批次。

只有本机完成下一轮长度控制并产生以下冻结件后，才一次性安排 GPU：

1. untouched clean-dev 与 hard-negative dev 的长度 task-macro 均降到 `.70` 以下，且 lexical/masked 读数、变体、重复和 parser 审计均有完整证据；
2. repository/project provenance 有可审计来源；如果来源无法补齐，所有正式文字必须明确限制为 task-heldout，不得写 project-heldout；
3. 主模型、`lexical_control`、`detection_only`、`source_only`、`joint`、`joint_invariance`（必要时 `joint_adversary`）写入一个冻结 `batch_manifest.json`，至少 2 个 generator-heldout 折 × 3 seed；
4. 脚本、环境、数据、manifest、开关和 SHA256SUMS 一起上传，AutoDL 一次执行整批并统一回传 `metrics.json`、逐行分数 digest、GPU 监控、日志和复核包。

GPU 批次若相对同折 `lexical_control` 没有预注册增量，立即写为 `increment_not_observed` 并停止；不得通过增加容量、温度、轮数或损失项补救。

## 4. 本机与 AutoDL 分工

本机直接完成数据构造、长度/词法/AST/parser/重复审计、轻量 probe、manifest 冻结、打包和哈希。AutoDL 只承担本机无法经济完成的高显存编码或训练；收到包后应一次性做全量核验和完整批次，不拆成需要多次人工传递的小实验。

