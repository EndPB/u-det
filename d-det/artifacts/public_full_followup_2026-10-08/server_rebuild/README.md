# server_rebuild/（服务器侧重构材料 · 临时复核）

上游指导（`d-det/docx/d-det_AutoDL_变体迁移执行前闸门指导_2026-10-08.md` §1）要求：交接包身份未解决前，
必须同时保留 `server_rebuild/`（服务器重构版）与 `provided_original/`（原件到达后放置）。

## 内容（本目录为归集快照）

| 文件 | sha256 | 与 expected 清单关系 |
|---|---|---|
| `family_series_admission.server_rebuild.json` | `c079abea65807ce2228822abdb82955f2c122bcdc299fb09093b3a0c52898ee9` | 重构版；**不是**原件；原件期望 sha `cc4a77851702e0bbf57e50636744ad25b572da9a665148b2b76179236b0bdfae` |
| `official_docs_server_fetch/CodeLlama-MODEL_CARD.md` | `4f5feeaee49f128eba13ea5daf36db224143580947bb6e2618ab3b7a3d6ec3b3` | 按官方 URL + commit pin `e81b597e` 拉取（与期望 `official_docs/CodeLlama-Instruct.txt` sha 相同） |
| `official_docs_server_fetch/Qwen2.5-Coder-README.md` | `600d946e07e4cb74cd329cba828f026b2eff08ac5b639d1f4b3898c7ec1700e6` | 按官方 URL + commit pin `5948e971` 拉取（与期望 `official_docs/Qwen2.5-Coder-Instruct.txt` sha 相同） |
| `official_docs_server_fetch/DeepSeek-Coder-README.md` | `aa0a95ca037f1d2c641421417abcaabaca230696cfc79b84cfea235153b8f6f2` | 按官方 URL + commit pin `2f9fd859` 拉取（与期望 `official_docs/DeepSeek-Coder-v1-Instruct.txt` sha 相同） |
| `official_docs_server_fetch/contest_problem.proto` | `c44121c7a16b7b8f0356a8941cee42bf50fa3eb76d4729006f1b25b9f5ec9903` | 按官方 URL + commit pin `fa7a4f81` 拉取（与期望 `official_docs/CodeContests-schema.txt` sha 相同） |
| `official_docs_server_fetch/sources.json` | `1bcfbeedf98ad48554f0d26d525108e7af5c72e94463978034201674fe8b9d0b` | URL/commit/字节/sha 完整来源记录 |

> 注：同内容副本亦保留在原发布位置（`../family_series_admission.server_rebuild.json`、`../official_docs_server_fetch/`），
> 以维持 `811ecfe` 已推送回传中的路径引用；两处 sha256 相同。

## 政策（冻结）

1. 在 `provided_original/` 达到 **16/16** 逐文件核对通过前：**不得删除或覆盖**本目录。
2. **不得**将 `c079abea…` 写成原件哈希；所有引用本目录的结论必须标注 `server_reconstruction_only`。
3. 若原件长期无法到达：重构版定性为 `server_reconstruction_only`，后续结果不得写成"复现本机交接包"，
   也不得使用原件 commit/hash 作为复现证据。
4. 核对结果记录于 `../provenance_resolution.json`（每项含 expected/observed/source/replacement_time/decision）。
