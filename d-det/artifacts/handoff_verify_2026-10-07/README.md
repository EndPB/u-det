# 交接包应用与校准复现记录（2026-10-07 · 服务器侧）

对应交接包：`d-det/data/autodl_minimal_handoff_2026-10-07.zip`（sha256 `838432b4…f740c2`，
33/33 成员核验通过；中文名 docx 用 Python zipfile 解压）。

## 本目录内容

| 文件 | 说明 |
|---|---|
| `codet5_small_verify.json` | 上传资产 7 文件 sha256；tokenizer（`RobertaTokenizer(vocab=,merges=)`，32100 词表）与权重结构检查；**与随包 authorbench 产物 `model_sha256` 交叉对照 = 全 True** |
| `stage_a_compare.json` | 数据角色审计与随包参考的对比（矩阵 8/8 包逐字段一致、pair 折逐项一致） |
| `stage_b_compare.json` | H2 pair 五视图 + AuthorBench 的校准对比（逐项 Δ） |
| `logs/` | preflight 日志、两次校准运行日志 |

## 关键结果（全部命中手册预期）

1. **Preflight**：RTX 3080 Ti 12G、torch 2.9.1+cu128、transformers 5.17.0、sklearn 1.9.1、
   tokenizers 0.23.2；数据盘 23G 可用。
2. **Stage A（数据角色审计）**：pair 审计 4692 pairs、0 invalid、task 冲突 0、**admitted=6 折**；
   数据矩阵 8/8 包（rows/tasks/hash/跨 split 计数）与本地参考**逐字段一致**。
   其中 4 个控制包（llm_codegen_v2、stacad_alignment、codet_m4_balanced_control、aicd_t2_numeric_balanced）
   为本轮**从保留的 bundle zip 内提取**，并按其上传清单 SHA-256 全数校验通过（无需补传）。
3. **Stage B（冻结 CodeT5-small 校准）**：
   - H2 pair（6 折平均）：raw BA **.6735**（参考 .6739，Δ−0.0004）；五视图 |Δ|≤0.0031；
   - AuthorBench task-heldout：test BA **.597222** / macro-F1 **.576036**（**与参考 Δ=0.000000**），
     best_C=0.03，dev/boot CI/counts 全部一致。
4. **Stage C（P0 只读核对）**：`acl_sota_p0/summary_p0_v1.json` 三参照（AuthorBench fusion .8388、
   STACAD 官方协议 .7330、Droid ~.1645）在位；未重复训练。

## 运行命令

```bash
# 应用 overlay（应用前逐成员 SHA-256 校验、同名文件对比备份）
python - <<'P'  # Python zipfile 解压（中文名安全）
P
cp -a /tmp/autodl_handoff/overlay/. /root/autodl-tmp/u-det/
cp /tmp/autodl_handoff/REMOTE_REQUIRED_STATE.json /root/autodl-tmp/u-det/

# Stage A / B（udet 环境；输出一律写 *_server 目录，不覆盖随包产物）
python scripts/audit_acl_local_readiness.py --out d-det/artifacts/acl_local_stage_a_2026-10-07_server
python scripts/h2_generator_heldout_codet5_small.py --views raw ids_only strings_only comments_only all \
    --batch-size 16 --out d-det/artifacts/h2_generator_heldout_codet5_small_2026-10-07_server
python scripts/authorbench_codet5_small_family_baseline.py --batch-size 16 \
    --out d-det/artifacts/authorbench_codet5_small_family_2026-10-07_server
```

环境记录：`OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 TOKENIZERS_PARALLELISM=false`；GPU fp16 autocast；
脚本内置 seed=20261007。
