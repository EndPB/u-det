# AutoDL 服务器侧 AI 工作交接指南（最小覆盖包）

这是对已有 AutoDL `u-det` 工作目录的增量覆盖包。包内**没有数据集原文、模型权重、历史预测数组或凭据**；这些资产应复用服务器已有版本。服务器侧 AI 先核验缺口，只有明确缺失并且确实阻塞当前阶段时，才请求针对性补传。

## 1. 应用覆盖包

假设服务器已有项目目录为 `/root/autodl-tmp/u-det`：

```bash
unzip autodl_minimal_handoff_2026-10-07.zip -d /tmp/autodl_handoff
cd /root/autodl-tmp/u-det
cp -a /tmp/autodl_handoff/overlay/. .
```

覆盖前先备份同名文档、脚本和轻量报告；不要覆盖服务器原有数据、权重和大型预测文件。包内 `BUNDLE_MANIFEST.json` 可用于逐文件 SHA256 校验。

## 2. 第一条回传必须包含

```bash
git rev-parse HEAD
git status --short --branch
bash scripts/autodl_preflight.sh | tee artifacts/autodl_preflight_$(date +%Y%m%d_%H%M%S).log
```

回传当前 HEAD、工作树、GPU/显存/CUDA、Python/PyTorch/Transformers/scikit-learn、剩余磁盘/内存、`REMOTE_REQUIRED_STATE.json` 中的资产存在性与哈希、已有 P0 产物，以及准备执行的唯一阶段。完成这些检查前不要启动长任务。

## 3. 必须复用的远端资产

当前阶段默认要求服务器已有：

```text
d-det/data/h2_pair_benchmark_v1/pairs.jsonl
d-det/data/h2_pair_benchmark_v1/generator_fold_index.jsonl
d-det/data/h2_pair_benchmark_v1/manifest.json
d-det/data/h2_authorbench/core.jsonl
d-det/data/h2_authorbench/fold_plan.json
d-det/models/codet5-small/config.json
d-det/models/codet5-small/vocab.json
d-det/models/codet5-small/merges.txt
d-det/models/codet5-small/pytorch_model.bin
```

`h2_authorbench_dcan`、`h2_llm_codegen_v2`、Droid 和 STACAD 属于后续或外部压力测试，不要因为包内有相关报告就自动重新上传或解压。CodeT5-small 不存在时先回报缺失，不要自动联网下载大权重。

## 4. 环境与 tokenizer 约束

优先复用服务器已有 `udet` 环境。实际版本必须记录，不要盲目重装。新增脚本使用：

```python
RobertaTokenizer(vocab=".../vocab.json", merges=".../merges.txt")
```

注意实际文件是 `merges.txt`；这里的要点是使用 `vocab=` 和 `merges=` 显式参数，不要把 `vocab.json` 传给旧版 `vocab_file`，否则可能生成错误的五词表 tokenizer。使用本地权重时保持 `local_files_only=True`。

## 5. 运行顺序

### A. 数据角色审计

```bash
python scripts/audit_acl_local_readiness.py
```

验收：6 个 admitted generator-heldout pair folds；Google/Mistral 因正对不足的 fold 必须继续 diagnostic-only，不能制造伪正对。

### B. 冻结 CodeT5-small 复核

```bash
python scripts/h2_generator_heldout_codet5_small.py \
  --views raw ids_only strings_only comments_only all --batch-size 16
python scripts/authorbench_codet5_small_family_baseline.py --batch-size 16
```

这是环境校准，不是新主方法。预期方向：H2 pair raw 六折 BA 约 `.6739`；AuthorBench task-heldout test BA 约 `.5972`、macro-F1 约 `.5760`。如果偏离，先查数据 hash、split、tokenizer、截断、pooling、dtype 和 test 是否被提前读取。

### C. P0 只读核对

先阅读 `d-det/artifacts/acl_sota_p0/README.md` 和总研究总结。已有 P0 产物时优先做指标/哈希复核，不要重复完整训练。P0 参照是 AuthorBench fusion macro-F1 `.8388`、STACAD 官方协议 `.7330`、Droid generator-heldout 约 `.1645`。

## 6. ACL 主线

论文命题是：后训练/来源差异可能在代码表示中留下可迁移的来源几何；归因空间必须把任务效应、检测轴和家族来源轴分开，并在 unseen-generator 上验证。

- **H1 来源可读性**：比较 raw、joint、centered 与词法/结构控制，报告 macro-F1、BA_F、逐类 recall、task-cluster CI 和 language/length/metadata 控制。逐题中心化属于 transductive 诊断，不能冒充单样本部署。
- **H2 跨 generator 关系保真**：正对只能是同 family、不同 generator；训练/验证/测试 generator 必须隔离。无 train/dev 正对的 fold 不进入主聚合，随机切分只能作迁移损失参照。
- **后训练差异**：使用双侧 `u=[h_plus;h_minus]`、`S=(h_plus+h_minus)/2`、`A=(h_plus-h_minus)/2`。联合双侧表示是主视图，`A` 是消融；没有共同 base/model role 控制时，只能写来源可读性/迁移诊断。
- **H3 检测与归因分离**：只有 H1/H2 通过后，才比较 detection-only、family-only、joint、shared/private adapter 或梯度分离。DCAN、DMHM、SupCon、低秩、梯度分离不得同时加入。

## 7. 禁止事项

- 不把检测分数当 family attribution。
- 不把 CoDET-M4 五模型标签映射到其他数据集 family。
- 不把 AICD 数字标签 0–11 命名为 family，直到官方 mapping/preprocessing 被核实。
- 不把随机切分分数写成 unseen-generator 结论。
- 不把 transductive task-centered 数字和 ordinary inference 数字放进同一主表。
- 不覆盖旧产物；每次实验使用新目录并记录 commit、数据 hash、环境、seed、参数、预测键和 test 读取时间。

## 8. 每次实验交付

```text
config.json  metrics.json  predictions.npz  report.md
SHA256SUMS.txt  logs/
```

如果提升小于 1 个百分点、折间方向不一致、检测下降或只在单 generator 生效，默认进入诊断/附录，不升格为主方法。每次结束后回传运行命令、输出目录、哈希、环境、seed、耗时、完整指标、失败项和下一步建议。\n