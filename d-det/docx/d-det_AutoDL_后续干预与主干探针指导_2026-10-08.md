# AutoDL 后续指导：来源信号干预与主干探针

日期：2026-10-08  
基线提交：`958eeb4`  
目标：解释已观察到的系列迁移信号来自哪里，并评估它是否依赖 CodeT5 表示。  
当前权限：只允许新的 train/dev 探索；禁止重新读取已经使用过的 test；禁止生成新模型输出。

## 1. 先固定研究问题

现有结果已经证明：在固定 BigCodeBench `full/instruct`、task-heldout 和系列内尺寸留出设置下，来源信号具有较高可读性。下一步不再追求更高 AUROC，而是区分三个假设：

### H-format：表面格式假设

模型主要利用换行、缩进、空白、注释、字面量、命名和模板残留。安全的格式/词法干预后，读出应显著下降。

### H-content：内容条件化假设

去掉可疑表面信息后，任务语义和代码结构仍保留部分系列信号。干预后重新训练的读出仍高于有限控制，才支持这个假设。

### H-backbone：表示依赖假设

迁移结果不是 CodeT5 encoder 表示的偶然特性。保持数据、切分、读出、预算完全相同，更换一个 encoder-only、一个 encoder-decoder 和一个 decoder-only 冻结表示后，结论方向仍应大体一致。

这三个问题必须分开；“原分类器在干预数据上掉分”和“干预后重新训练仍可读”不是同一个实验。

## 2. 严格边界

- 不读取旧 test，不用保存的 `test_scores.npz` 选择干预、主干、阈值或报告折。
- 不覆盖 `variant_transfer_stage1b_2026-10-08/`、`variant_transfer_stage2_2026-10-08/` 或原始分数。
- 所有新产物写入 `d-det/artifacts/post_stage2_intervention_2026-10-08/`。
- 新结果全部标记 `exploratory_train_dev_only=true`；不能作为新的确认性 test 或 H2 主表。
- 不下载大模型权重，除非某个主干被单独授权；优先使用服务器已有本地权重并记录字节 hash。
- 现有 `source_status=server_reconstruction_only`、`original_bundle_verified=false`、`claims_of_byte_identity=forbidden` 继续保留。

## 3. Phase A：安全干预矩阵

所有干预都使用相同的已冻结 train/dev task split、11 个成员、两个负集版本和三个 feature seed。先在不训练的情况下生成审计摘要，再训练新的 train/dev 读出。

### A0 原始对照

从保存的 train/dev 原始输入和 stage-1b 对象重放一份只读对照。它只用于确认新 pipeline 与旧 dev 分数相符，不产生新 test 数字。

### A1 格式安全规范化

只处理不会改变 token 语义的部分：统一 CRLF/LF、删除行尾空白、统一文件末尾换行，并保留原始和规范化 hash。不要自动重排缩进、括号或语句。

### A2 注释屏蔽

用状态机或已有语言 tokenizer 屏蔽注释，必须区分字符串、字符常量、模板字符串和注释状态。屏蔽失败的行单独记录，不静默删除。输出只保存规范化后的 hash 和统计，不把代码正文写入 Git。

### A3 字面量屏蔽（敏感性分析）

将字符串、数字和路径字面量替换为类型占位符。需要记录替换计数、失败率和代码长度变化；因为这可能改变任务语义，只能作为敏感性分析，不能命名为“去风格后的纯代码”。

### A4 标识符屏蔽（暂不默认执行）

只有在语言 tokenizer 能稳定区分局部变量、函数名、API 名和关键字时才允许执行。不能用简单正则把库 API、类型名和变量名混为一类。若多语言覆盖率不足，状态写为 `not_executed`，不要为了完整矩阵强行生成。

每个版本至少保存：`transform_id`、版本、规则、成功/失败行数、原始 code hash 与变换后 hash 的映射 hash、token 数变化、截断比例和语义风险说明。

## 4. Phase A 的两个读出

### A-readout-1：冻结原对象变换评分

将 stage-1b 已保存对象直接用于变换后的 train/dev，只做 transform/predict。这个结果回答：**原分类器对表面干预有多脆弱**。由于词表和表示空间没有为变换重训，不能解释为干预后最佳可读性。

### A-readout-2：干预后重新训练

每个干预版本在 train 上重新拟合同一组固定读出，在 dev 上按原规则选择：

- TF-IDF char/word；
- CodeT5-small/base 冻结表示；
- style LR/LGBM；
- code-layout control；
- oracle size/length control；
- P0-fusion 和 P0-equal。

不新增模型、损失或超参网格。报告原始 dev、变换后 dev、相对原始 dev 的 paired task-cluster delta。不要把 dev 结果写成迁移或 test 证据。

## 5. 必须保存的干预结果

每个 `transform_id × arm × fold × readout` 保存：

- AUROC、AP、task-macro AUROC；
- 500 次 task-cluster bootstrap（seed `20261008`）；
- 正类率、有效 task 数、失败 task 数；
- 原始→变换后的分数差、字符/token/行数变化；
- size/length-only 与 code-layout 控制；
- config hash、代码 commit、环境版本、输入/特征 hash。

主要比较使用同一个 train/dev task 索引和同一 bootstrap task 抽样序列。不能按哪个干预掉分最大来事后挑主结果。

建议判读：

- 原对象大幅掉分、重新训练恢复：主要是表示/词表不匹配，不能说信号消失；
- 原对象和重新训练都下降，且控制下降更小：支持表面信号依赖；
- 两者都保持高分：支持干预后仍存在稳定可读成分，但仍不能称因果后训练信号；
- 只剩 code-layout 或 size/length 高分：把结果归为混杂诊断，不推进主干矩阵。

## 6. Phase B：最小主干探针，而不是完整矩阵

在 Phase A 未完成前不展开主干规模扫描。若本地已有权重且无需下载，最多执行以下最小比较：

| 表示 | 目的 |
|---|---|
| CodeT5-small/base encoder 表示 | 现有基线，保持不变 |
| 一个 encoder-only 代码模型 | 检查纯 encoder 表示是否复现方向 |
| 一个 decoder-only 代码模型的冻结 hidden state | 检查 decoder 表示是否复现方向 |

选择规则必须先写入配置：同一 tokenizer 截断预算、同一 pooling、同一 train/dev split、同一线性 probe、同一 C 网格、同一 seed 和相同特征缓存格式。不要把 decoder 的生成 log-prob 与 encoder embedding 混作同一表示实验。

每个主干至少记录：模型/版本、参数量、tokenizer、最大长度、截断覆盖率、pooling 层、hidden size、权重 SHA-256、冻结状态和显存峰值。

### 最小 2×3 设计

- 架构：encoder-only / encoder-decoder / decoder-only；
- 尺寸：只在同一主干已有 small/base 时比较，否则不补下载；
- 读出：固定 Logistic Regression 和一个 style_lgb 控制；
- 数据：只跑原始和一个安全 `comments_masked` 版本；
- 统计：同一 500 次 task-cluster bootstrap；
- 结论：只比较方向和迁移损失，不按最高 AUROC 选“最佳主干”。

如果三类架构方向一致，论文可把结果写成“表示选择不改变主要现象”；如果方向不同，必须把它作为结果，不能隐藏不利主干。

## 7. 资源和停止规则

- 先只做 A1/A2；A3 作为敏感性，A4 默认不做。
- 先跑 1 个 series、1 个 heldout fold 的 smoke，确认变换失败率、token 覆盖和预测格式，再展开 11 折。
- 任何 test 文件访问、旧 test 预测重算、下载新权重或新生成都立即停止并回传。
- 变换导致语义破坏、失败率超过预声明阈值或跨语言规则不一致时，停止该变换，不修改数据纳入规则。
- 如果 CodeT5-small/base 和一个已有主干方向一致，停止扩展；不要为了寻找显著差异继续添加主干。
- 如果结果只显示 style/metadata 仍然很强，下一步转向数据处理与模板审计，不做更多 backbone sweep。

## 8. 回传清单

AutoDL 只回传：

1. `intervention_manifest.json`；
2. 每个变换的规则、失败率、hash 和 token/长度统计；
3. 原对象评分与重新训练评分的 train/dev 对照；
4. 两个负集版本的 paired delta；
5. Phase B 主干可用性和权重 hash（未执行则明确 `not_executed`）；
6. `commands.txt`、日志、环境版本、Git head/status 和 SHA256SUMS；
7. 明确声明 `test_read=false`、`generation=false`、`exploratory_train_dev_only=true`。

只有这份回传完成后，指导端才决定是否值得做一个额外主干；当前不允许再开新的 test 或完整主干矩阵。
