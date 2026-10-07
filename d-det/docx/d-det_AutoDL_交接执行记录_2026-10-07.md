# d-det：2026-10-07 最小交接包应用与校准执行记录（服务器侧第一回传）

## 0. 回传必含项（按 `AUTODL_AI_HANDOFF_GUIDE_2026-10-07.md` §2）

```text
git HEAD（执行时）: 14b74ee（本轮提交后见文末）
git status: 干净基线之上新增：overlay（7 脚本 + 7 产物目录 + 1 复盘文档）、
            REMOTE_REQUIRED_STATE.json、交接 zip、本轮 *_server 产物与核验档案
preflight: artifacts/autodl_preflight_20261007_155546.log
  GPU: RTX 3080 Ti 12288MiB (driver 595.71.05) | CUDA 12.8 | torch 2.9.1
  transformers 5.17.0 | numpy 2.2.6 | sklearn 1.9.1 | tokenizers 0.23.2 | hf-hub 1.32.0
  python 3.12.14 (udet) | 数据盘 23G 可用 | 内存 90G 可用
```

**必需资产核验（REMOTE_REQUIRED_STATE，must_exist 9/9 OK）**：

| 资产 | 状态 | sha256(16) |
|---|---|---|
| h2_pair_benchmark_v1/pairs.jsonl | OK 83,142,931 B | 962d59ccd102dda6 |
| h2_pair_benchmark_v1/generator_fold_index.jsonl | OK 1,822,812 B | 2d6cffd2e3c0d786 |
| h2_pair_benchmark_v1/manifest.json | OK | 3a42bbe95aefeb93 |
| h2_authorbench/core.jsonl | OK 6,230,760 B | 5b15025a8eea7768 |
| h2_authorbench/fold_plan.json | OK | d351f60abbfea798 |
| models/codet5-small/{config,vocab,merges,pytorch_model.bin} | OK（新上传）；**与随包 authorbench 产物 model_sha256 全 True** | c067a637… / 43bb485f… / 5d346f84… / 968fb0f4… |

optional_later 4 件：全部就绪（其中 llm_codegen_v2、stacad_alignment 为本轮从保留 bundle 提取并校验）。
P0 产物：`acl_sota_p0/summary_p0_v1.json` 在位（fusion .8388 / STACAD .7330 / Droid .1645，只读核对通过）。

## 1. 执行结果（stage A/B/C 全部通过）

| 步骤 | 结果 | 与随包参考对比 |
|---|---|---|
| A. 数据角色审计 | pairs 4692、invalid 0、task 冲突 0、**admitted=6 折**；8/8 包矩阵 | **逐字段一致**（含折计数、hash、跨 split 统计） |
| B1. H2 pair（CodeT5-small，5 视图） | raw BA .6735 / F1 .6750 / AUC .7546 | Δ≤0.0004（五视图全部 |Δ|≤0.0031） |
| B2. AuthorBench task-heldout | test BA **.597222** / F1 **.576036**；best_C=0.03 | **Δ=0.000000（完全一致）** |
| C. P0 只读核对 | 三参照值在位，未重复训练 | 按手册 §5-C 口径完成 |

细节与逐项对比：`artifacts/handoff_verify_2026-10-07/`（README + 三个 JSON + logs）。

## 2. 产出目录（不覆盖随包产物，全部独立 `_server` 后缀）

- `d-det/artifacts/acl_local_stage_a_2026-10-07_server/`（data_role_matrix.json + report.md）
- `d-det/artifacts/h2_generator_heldout_codet5_small_2026-10-07_server/metrics.json`
- `d-det/artifacts/authorbench_codet5_small_family_2026-10-07_server/metrics.json`
- `d-det/artifacts/handoff_verify_2026-10-07/`（资产/对比/日志）
- `u-det/artifacts/autodl_preflight_20261007_155546.log`

## 3. 过程记录与偏差

1. 交接 zip 33/33 成员 SHA-256 全过；**中文名 docx 用 Python zipfile 解压**（unzip 会乱码，沿用既有教训）。
2. overlay 同名文件仅 4 项：3 份 docx 与仓库现有**逐字节一致**、`authorbench_cpu_baseline.py` 亦一致
   （已跟踪），无覆盖损失；其余均为新增。
3. 4 个控制包从保留的 `autodl_h2_bundle_3715711_upload.zip` 提取（venv 内嵌套 zip），
   按各包上传清单校验：stacad 8/8、llm_codegen_v2 10/10、aicd 8/8、m4 7/7 全过 —— **无需补传**。
4. 两个校准脚本默认 `--out` 指向随包产物目录；为遵守"不覆盖旧产物"改用 `_server` 后缀输出，
   数值用于与参考对比（结论：命中）。
5. 环境：`OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 TOKENIZERS_PARALLELISM=false`，GPU fp16 autocast，
   脚本内置 seed=20261007；preflight 显示 GPU 空闲后启动，总耗时 h2 ≈ 4 min、authorbench ≈ 12 s。

## 4. 缺口与下一步

- **无阻塞缺口**（must_exist 9/9；原 optional 两件已随本轮补全）。
- 唯一保留事项：`h2_alignment_v3.zip`、`agent_attribution_v1.zip` 等其余嵌套包仍在 bundle zip 内
  （未提取，需要时按同样流程提取校验即可）。
- 建议下一步（待指导端裁定）：进入总结 §7 阶段 B 的**任务条件化多视图强基线**
  （AuthorBench task-aware / LLM-CodeGen v2；先冻结特征 + 线性/树模型，先不引入 adversarial）。
