# AutoDL H3 v2 接收核验指导（2026-10-11）

本包是本机 H3 v2 数据修订结果的接收包，不是 GPU 训练授权包。当前正式闸门仍为 `revise_data`，`formal_gpu_ready=false`；不得在旧冻结包上重跑，也不得在无卡状态把接收核验写成 GPU 实验。

## 远端接收

1. 在 `/root/autodl-tmp/u-det/d-det/data/` 下创建独立目录 `autodl_h3_v2_handoff_2026-10-11_received/`，解压本包，不覆盖历史目录。
2. 对 `SHA256SUMS.txt` 做逐文件核验；再核对 `controlled_dev_v2.jsonl` 的 sha256 和字节数。
3. 运行 `python -m py_compile scripts/h3_local_revision_v2.py scripts/h3_lexical_diagnostic_v2.py`。
4. 读取两个 artifact 的 `gate.json`、`metrics.json`、`report.md`，确认 `formal_gpu_ready=false`、`test_deserialized=false`。
5. 只记录机器、Python、磁盘、CUDA 可见性和接收日志；不要读取原始 test，不下载权重，不执行样本代码。

## GPU 触发条件

本包不能触发 GPU。只有本机下一轮补齐 project/repository provenance、重建 hard-negative 配对、通过 clean-dev 与 controlled-dev 的 lexical/length/view/parser/duplicate 闸门，并另行冻结新的 `batch_manifest.json` 后，才上传新的 GPU 包。届时一次性执行至少两个 generator-heldout 折、三个 seed 和完整 job 矩阵，统一回传。

## 当前关键读数

- clean-dev / controlled-dev 长度 task-macro AUROC：`.7248 / .6679`
- clean-dev / controlled-dev 词法 task-macro AUROC：`.9694 / .9585`
- view probe：`.5569`
- 掩码 token 词法诊断仍为 `.8471 / .7997`

接收完成后只回传 `BUNDLE_RECEIVED`、逐文件校验结果、环境摘要和 GPU 可见性；不要开始训练。
