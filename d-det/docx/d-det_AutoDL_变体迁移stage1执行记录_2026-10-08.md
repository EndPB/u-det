# d-det AutoDL：变体迁移 stage-1（train/dev）执行记录（2026-10-08）

- 授权依据：《AutoDL 服务器重构版最小授权与补传豁免》2026-10-08（§1 允许 train/dev；§4 执行边界；test 需另行授权）。
- 上游提交：`dccd1d2`（执行前闸门 C1-C8 全过）。
- 本轮性质：**train/dev 拟合/评估**。不读 test、不生成新样本、不下载权重。
- 产出目录：`d-det/artifacts/variant_transfer_stage1_2026-10-08/`
- 脚本：`scripts/variant_transfer_stage1_execute.py`（执行）、`scripts/variant_transfer_stage1_report.py`（报告）。

## 1 授权、标注与开关

- `source_status.json`：**source_status=server_reconstruction_only**；original_bundle_verified=false；claims_of_byte_identity=forbidden；重构哈希 `c079abea…` 登记，**不得**写作原件哈希 `cc4a7785…`（原件仍未上传，16/16 awaiting）。
- `execution_switches.json`（v2）：**training_allowed=true**（scope=train/dev）；generation_allowed=false；test_read_allowed=false。

## 2 执行设置（全部来自冻结配置）

- 数据：BCC full/instruct 11 成员 × {train,dev} 行 = 10,659 行（**test 行显式排除 1,881**，不进入内存）。
- 折：3 系列 × heldout（7b/13b/34b/70b、1.5B/7B/14B/32B、1.3b/6.7b/33b）× 负集 2 版本（size_mix / size_matched）= **22 子实验**。
- 读出（10）：tfidf_char、tfidf_word、sem_base、sem_small、style_lr、style_lgb、metadata_only、size_length_only、P0_fusion、P0_equal。
- 选择协议（冻结）：C 网格按 dev AUROC 最大、平局取更小 C；SGD 5ep best-dev × seeds 0/1/2 概率均值；LGBM(800) 固定超参；bootstrap 500（task-cluster，seed 20261008）。
- 环境：udet（sklearn 1.9.1 / scipy / lightgbm 4.7 / torch 2.9.1+cu128 / RTX 3080 Ti）；特征编码 97.3s；全量 22 折 **3,118.8s**。
- 特征哈希：style/meta `b518d23c…`；emb small `9b2aedfc…`（512d）；emb base `c4dc6881…`（768d）（见 feature_manifest.json）。

## 3 结果概览（dev；选择内估计，非迁移证据）

- **P0_fusion dev AUROC**：22 折范围 **0.917 – 0.990**；按系列：CodeLlama 0.955–0.970；Qwen 0.971–0.990；DeepSeek 0.917–0.938。
- **best single 成员**：几乎全部为 `style_lgb`（0.890–0.986）。
- **控制读出**：metadata_only 0.811–0.967；size_length_only 0.576–0.885（**显著低于内容特征** → 系列信号不是单靠规模/长度）。
- **负集版本敏感性（P0_fusion Δ matched−mix）**：CL +0.007~+0.009；Qwen −0.002~+0.009；DS −0.011~+0.011（版本间差异小，方向不稳定；两版并行报告，遵从预注册）。
- **R2 表示迁移（系列中心距离）**：base euclid 0.536–0.65、cosine ~0.50–0.55（弱）；小模型类似；train 距离-logsize 秩相关 |ρ|<0.2（无强尺寸混杂）。
- 明细见 `train_dev_report.md`（22 折 × 10 读出 + CI + R2）与 `train_dev_metrics.json`。

## 4 §5 产物要求对照

| 要求 | 状态 |
|---|---|
| execution config 与 amendment hash | ✓（`6aba9c4d…` / `4d0d1631…`，记录于 metrics JSON） |
| train/dev 行数、task 数、heldout 排除证明 | ✓（每折 train=7,980/4,788/3,192；dev=1,710/1,026/684；test 显式排除；fold 矩阵在闸门产物） |
| 预测 | ✓ predictions/ 22 文件（10 读出 × dev 行，含 unit/task/y/score） |
| 指标 + task-cluster bootstrap 500 CI | ✓（AUROC/AP/task-macro，每读出一组 CI） |
| size/length-only baseline | ✓（每折）+ metadata_only |
| P0 fusion / equal | ✓（每折） |
| 命令、日志、代码版本、SHA256SUMS | ✓（commands.txt、logs/、git_head、34 件 SHA256SUMS） |
| source_status 标注 | ✓（所有顶层产物 + 报告头部） |

## 5 解释边界与下一步

- dev 为选择集（C/SGD 选择在 dev 上），数字不能当作 heldout 迁移证据；迁移信号只能在 **test 单读**（独立授权：时间/数据 hash/输出字段）后评估。
- 不得表述：复现本机原件；unseen independent family；后训练因果；无污染确认性 benchmark。
- 产物冻结（本 commit 后）；**下一步：向指导端申请 test 单读授权**。原件到达后按其独立任务补核对（不覆盖本轮产物）。
