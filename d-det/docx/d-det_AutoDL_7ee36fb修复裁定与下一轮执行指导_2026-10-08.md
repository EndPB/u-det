# d-det AutoDL 7ee36fb 修复裁定与下一轮执行指导

日期：2026-10-08；服务器当前关机，本文用于下次开机后的执行入口。

## 1. 结果裁定

端点平衡 48 折构造审计通过，anchor-only 和 partner-only 权重均为 0.5000。relation residual=0.8501，independent-source composition=0.8455，共享 task-cluster bootstrap 差值 +0.00465，95% CI [+0.00176,+0.00749]。这只能写成当前三个已知公开系列、train/dev 开发协议中的小幅关系 residual 增量；P0_pair=0.7391 仍是 partial surface 控制，不能称完整 H2 主结果。

role-blind 结果暂缓使用：core_role_blind.py 先对标准化后的 A 块取负，而真正交换原始输入应重新计算 standardize([S,-A])，两者相差 2 μ_A/σ_A。interface_audit.json 没有验证标准化后的双视图；raw S+A=0.9990、role-blind/oracle 比较和“残差增量消失”统一标为 pending corrective rerun。train_twin() 也未设置 torch.manual_seed(seed)。

weighted_auc 与 sklearn sample-weight AUROC 的合成 tie 审计一致，最大误差约 1.1e-15；但 weighted_ap 全 tie 返回约 0.8333，预期为 0.5，本轮 AP 不进入论文。

R1 正确 [S;D] 对 [S;-D] 接口下 E5 约 0.380，edge-heldout E1 约 0.526、E5 约 0.368；旧接近 1.0 数字降级为 sign-construction diagnostic。当前 BCC 全 machine；DroidCollection 无 task_id/prompt_id，CoDET-M4 不是同题语义配对，H3 仍为 H3_original_unavailable_current_set；STACAD-v2 仅是候选。

## 2. 下次开机必做

1. 用同一 fit mu/sd 分别标准化 raw [S,A] 与 raw [S,-A]，禁止对标准化 A 块直接取负；补原始和标准化交换断言。
2. 设置 Python/NumPy/Torch/CUDA seed，保存每 seed 初始化、loss 和分数哈希；重跑 task-dev 和 model 5-fold，不读旧/new test。
3. 按预声明 tie 规则修复 weighted AP，并用 tie、全正、全负、单正例合成数据与 sklearn 核对。
4. 用相同 48 折补齐 lexical/style/meta 双侧 composition 与完整 symmetric pair P0，按既有 fusion/equal 方法在每折的 train 上重新拟合来源分类器/关系头；旧 stage-2 二分类对象的标签/切分并不兼容三系列 composition，不能直接重放其分数充当新协议 P0。主比较固定为 residual−composition 与 residual−full_P0。拟合及融合选择只在内层 train tasks 完成，外层 dev只作开发评测。

### 接口修复的确定方案

优先在 fit 原始两个方向的完整集合上估计标准化：A块均值严格设为零，A尺度为 sqrt(mean(A²))；S/R块按其 fit均值/标准差。这样真实交换后，A的标准化块也严格反号。若保留 canonical fit的非零 mu_A，则必须分别从原始 A 和 -A变换，不能反号标准化块。任选方案先冻结，勿根据 dev选方案。回传实际 mu_A/sigma_A、交换前后最大误差、随机输入顺序后分数严格变号审计。

oracle 的 residual 若仅在原始已知 instruct侧计算并复用于两方向，明确为受辅助信息的上界；不要求它满足可部署未知角色接口，不把它的高分当机制证据。role-blind的内层 cross-fit按task分组（同task所有model同行），不以行号mod5代替任务分组。用同一任务抽样序列报告各臂差值CI，指标名统一为 model-balanced pair-direction accuracy，本轮 .9742等不是AUROC。

先仅跑一个冻结的 task fold smoke及完整变换断言，再完成三种子 task/model折；复用缓存，不重新编码/采集。任何代表折由元数据预定，不能根据旧效果挑选。

### 关系统计和强基线的解释

关系端点平衡构造本机48/48复核通过，AUROC函数复核通过；没有理由重跑现有48折模型。AP仅从保存分数重算，逐tie组累计TP/FP，用组末precision乘该组recall增量，与sklearn average_precision_score(sample_weight=...)对照；不能对tie组内部顺序逐行积分。全负输入单独约定警告/0，避免把无效AUROC写作数值。

composition在本轮未经fit-only特征标准化或等权系列采样；补强baseline时，使用对称fit scaler、系列等权训练、fit内部cross-fit概率和固定dev内层选择，与当前canonical baseline并列。保留当前 +.46pt 数字，若新基线抹去增量照实归档。三种子逐个分数/损失需要保存，再报告ensemble，不能把三seed平均称三seed稳定。

当前null生成是对每个unordered pair采独立Bernoulli标签，不是保持原正负计数的置换。可以称随机标签sanity，不报精确randomization-test p。residual null保留了真标签训练的composition，是条件于baseline的sanity，不能作为整个pipeline的零信号证明。无需额外扩张置换预算，本轮主证据仍是paired增量。

原总路线1pt是正式方法的实用阈值；+.46pt只满足小幅开发增量口径，不改旧闸门，不以此自动启动H3。

## 3. R1 后训练差异主线与 H3 可行性（可与修复同时推进）

R1 不依赖F2的gate。七边只作观察性谱系候选，先按模式分别比较正确方向的 A/Delta-only、raw联合、小MLP、单侧来源组合和完整P0。线性f([S,D])-f([S,-D])=2w_D·D，S项和截距完全消失，因此本轮线性swap读数不是S-A非线性交互证据。逐边paired CI、fit-only选择、固定三seeds；不做巨大模型矩阵。

E1 instruct seen-edge实际 .9649，不是六条边都 .994；E1 edge-heldout .5263，E5 seen .3801/edge-heldout .3684，全保留；不得取max(p,1-p)或依dev翻方向。E2/E3共享parent，E4/E5/E6共享checkpoint，留一条edge不是checkpoint-heldout。另列去除所有incident edges的endpoint/component隔离支持表：不足的fold记insufficient，不借共享端点恢复“新checkpoint泛化”。

本机负责补来源/版本/父子训练证据及新同题输出素材；AutoDL用已存在train/dev缓存继续方法对比。新任务确认应在方法冻结后另有未参与开发的任务/输出，不能把当前171 dev或旧test重切分。STACAD的66780对是AI-AI same-task跨generator对，不是66780个human-AI对；3180 human×7 AI名义支持为22260对，实际有效性按内容/许可/重复/语言对齐另审计。每family仅1 generator不能检验族内unseen-member H2，但可支撑Human/AI与闭集来源读出的H3机制pilot。先支持审计和预声明；本轮不启动正式H3，也不读其test。

不新增 encoder/decoder/参数量对比，不下载权重、不生成新输出、不覆盖 7ee36fb。开关保持 train/dev training=true、test_read=false、generation=false、weights_downloaded=false。

## 4. 交接与执行停止条件

只需本文、既有远端代码和train/dev缓存；不补传57文件本机复核快照，不传权重/完整语料/历史test。缺缓存则回传具体路径/形状/哈希缺口，不能用汇总表虚构逐样本重算。plan规格重建允许，标server-reconstructed；无需等待本机原件。

输出独立目录role_blind_residual_corrective、relation_endpoint_balanced_strong_p0、r1_posttraining_comparison；保留config/开关/data-role/逐seed预测索引和hash/metrics/报告/日志/命令/git/SHA。回传四点：真实标准化swap断言、role-blind相对matched baseline的paired差值、relation相对完整P0的差值、R1逐边/endpoint隔离边界。先修复测试再训练；若断言失败停止对应branch，其余branch可继续。修复后负结果照实闭合，不增加一轮换名称或调gate。

本机复核入口：scripts/audit_core_round5_interfaces.py；审计JSON：d-det/artifacts/core_round5_local_review_2026-10-08/interface_and_metric_audit.json。该审计只使用保存源码/摘要及合成数组，无训练、语料、test或权重读取。

