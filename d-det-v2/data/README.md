# data（指针目录，不存放数据）

本目录不存放任何数据文件。d-det-v2 的全部数据**复用原 d-det**，路径规则如下：

## 数据根

- 本机绝对路径：`/root/autodl-tmp/u-det/d-det/data/`
- 相对 `d-det-v2/` 根：`../d-det/data/`
- 脚本内（`d-det-v2/scripts/`）：`../../d-det/data/`

## 常用包（按 AB 角色）

- **A 主数据**：`h2_stacad_alignment_v1/`（同题 Human/AI，7 generator × 7 语言）；修订视图 `h3_stacad_revision_v1/`、`h3_stacad_revision_v2/`、`h3_stacad_revision_v3/`、`h3_stacad_revision_v4/`（按需）
- **B 家族归因**：公开系列包与同题语料（`public_same_task_full_2026-10-07/`、series 包等）、`h2_authorbench*/`、`h2_llm_codegen_v2/`、`h2_droid_full_selected/`（外部压力）
- **外部控制（禁止训练）**：`mainline_abc_200k_2026-10-12/`（AICD numeric external control）、`codet_m4*/`、`acl_attribution_collection_v1/` 中的控制集

## 模型与权重

- `../../d-det/models/`（如 `codet5-small/`）；`../../d-det/checkpoints/` 同理

## 规则

1. 不复制、不移动、不改写原数据；只读引用；
2. 冻结包（manifest 所需的 sha256/字节数）在 d-det-v2 自己的 `artifacts/` 中登记；
3. 如需本机快捷访问，可自行软链（`ln -s ../d-det/data data`），但仓库内保持本 README 指针形态（避免 Windows 端 symlink 问题）。
