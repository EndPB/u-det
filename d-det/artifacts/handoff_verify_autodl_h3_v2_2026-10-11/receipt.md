# AutoDL H3 v2 接收回执（2026-10-11）

- bundle: autodl_h3_v2_handoff_2026-10-11_upload.zip
- zip_sha256: d045bcd11ddbfe371f15a061ca61b2c53efd9c552da5717ca8b48bc754817826
- received_dir: d-det/data/autodl_h3_v2_handoff_2026-10-11_received_v2
- file_manifest: 22/22 SHA256 OK
- scripts: py_compile OK（服务器 udet env Python 3.12.14；base Python 亦通过）
- gate: revise_data; formal_gpu_ready=false
- test_deserialized: false
- GPU: 服务器会话可见 NVIDIA GeForce RTX 3080 Ti（12,288 MiB）；未运行任何 GPU 任务（本包不触发训练）
- action: BUNDLE_RECEIVED; wait for a new formal GPU manifest

## 服务器侧核验补充（2026-10-11 03:34 +08 / 19:34Z；HEAD `7824621`）

- **解包一致性**：`_received_v2` 与 zip 逐位一致（根 `SHA256SUMS.txt` sha `4bbe531f…`）；`_received` 与 zip 仅行尾差异（CR 剥离后一致），以 `_received_v2` 为准。
- **校验链**：zip sha256 `d045bcd1…` ✓；`unzip -t` 无错；外层 `SHA256SUMS` **22/22 OK**；内层两套 SUMS **11/11 与 3/3**（CR 剥离）OK；`controlled_dev_v2.jsonl` **7,174,917 B** ✓。
- **环境**：host `autodl-container-hmz3teucd6-659d76d1`；Python（udet）3.12.14；磁盘剩余 19 GiB。
- **开关**：未读取 test、未下载权重、未执行样本代码（`test_deserialized=false`；gate=`revise_data`、`formal_gpu_ready=false`、`old_gpu_batch=closed`）。
- **入库（按既有约定）**：文档→`d-det/docx/`；脚本→`scripts/`；产物→`d-det/artifacts/h3_local_revision_v2_2026-10-11/` 与 `.../h3_lexical_diagnostic_v2_2026-10-11/`；`controlled_dev_v2.jsonl`→`d-det/data/h3_stacad_revision_v2/`（gitignore）；zip 与两级解包保留于 `d-det/data/`（gitignore）。
