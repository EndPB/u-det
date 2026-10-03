# AutoDL：701d0df 审计修正与下一步执行指导

日期：2026-10-03  
审阅提交：`701d0df4e444a3f1a1af4ed8cd2f35f231a3c8d4`  
服务端项目：`/root/autodl-tmp/u-det/d-det`

本文件接替 `d-det_AutoDL_3420857之后服务器AI继续指导_2026-10-03.md`。第二轮已执行，下一步先修评测和结论，再决定方法实验。保留 round1、round2 原始产物，在新目录记录修正版；不要覆盖历史数字。

## 1. 可以保留的进展与暂不能下的结论

第二轮在相同 AuthorBench 任务划分上报告：TF-IDF .7639；metadata-only .4753；filtered TF-IDF .4572；任务内中心化并训练侧标准化的冻结表示 .6376（转导）；LoRA 三 seed .7088/.7372/.7399，均值 .7286。LoRA 训练脚本在 evaluate 中确实调用了 model.eval()，因此下述 MLP 评测模式问题不直接否定这三个 LoRA 读数。

但不能直接断言：

- “长度捷径被排除”：仅按长度分桶、与 metadata-only 比较不足以排除混淆；
- “内容 token 是主要原因”：filtered 同时移除注释、字符串、标识符并折叠空白，混合了词法和格式变化；
- “所有机制无贡献、瓶颈一定在表示容量”：旧 MLP 指标存在评测问题，消融规模也不足以给出因果排除；
- “LoRA 稳定超过 late-fusion”：late-fusion 参照需要修复；两者头结构、目标和训练预算不同；
- “LoRA 已改善跨 generator 归因”：本轮仅在 AuthorBench task-held-out 训练、测试，没有新增 generator-held-out 或 file-held-out 方法实验。

## 2. 首要修复：MLP 的 train/eval 模式

源码证据：

- `scripts/dcan_four_models.py` 的 `MLP` 包含 BatchNorm1d 和 Dropout；`run_seed` 的 dev、train 特征提取、test 特征提取和 `evaluate` 未切换这些 MLP 至 eval。
- `scripts/dcan_round2_ablations.py` 的 `run_variant` 同样未调用相关模块的 eval。
- `torch.no_grad()` 只关闭梯度，不关闭 Dropout，也不阻止 BatchNorm 在训练模式下使用当前批统计、更新 running buffers。

影响：dev/test 的预测具有随机性、依赖批组成；BatchNorm buffers 会被评估数据更新。这里不能指控使用了 test 标签训练，但旧结果不符合预期的独立单样本评测，也不能用于严格消融比较。影响大小须重跑确定。

修复要求：

1. 每个训练 epoch 开始，所有可训练分支、头及 adversary 调用 train()。
2. dev、test 和用于拟合融合 LR 的 train 表示提取前，调用 eval()，并使用 no_grad/inference_mode。
3. 保存 best state 时包括 BatchNorm buffers；恢复后再次 eval。评估结束若继续训练，恢复 train()。
4. 从头训练修正版 MLP：旧 checkpoint 已经按随机 dev 评估选 epoch，不足以只重算一次 test。
5. 原流程复现与比较选择规则改动分开标记：第一步只修模式，保持原 dev 规则。若下一阶段统一按融合 dev Macro-F1 选 epoch，所有比较臂同时执行并明确是新协议。融合 LR 只在 train 表示上拟合，不用 test 拟合或选参数。
6. 重跑同 seed 的修正版 late-fusion/full-disentangle 基线及四个消融，每臂 3 seeds；加入一个“全部开关同原模型”的 no-op 检查，先确认模块初始化、数据批序、损失、dev 规则和预测可复现。

必须检查：

- 两次对相同输入评估输出相同；
- eval 前后 BatchNorm running_mean/running_var/num_batches_tracked 不变；
- 同样本换评估 batch 大小/组成后，输出仅有数值容差内差异；
- train/dev/test task_id 不交叉；test 不参与 epoch/融合器选择；
- 每项损失非零梯度、有效正对覆盖率、每 family 覆盖均有记录。

注意目标本身：full-disentangle 同时用 CE(head_s) 鼓励 semantic 表示保留 family 信息，又用 family-GRL 鼓励移除该信息。先如实记录这种目标冲突，修复评测后再用单变量对照研究，不把 .707 vs .703 写成机制无效的证明。实际关闭的机制为 SupCon、semantic-GRL、fingerprint-GRL、交叉协方差四项；两种 fingerprint nuisance target 可单独列明，不能混用“四/五机制”计数。

## 3. LoRA：保留读数，补全公平对照与交付

`scripts/dcan_round2_trainable.py` 的 evaluate 已使用 model.eval()。先在服务器加载保存的 adapter+head，核对 seed 和 best_dev，复算其 test 概率与已保存预测是否一致；保存基座权重 SHA256、LoRA 配置、tokenizer、family 顺序、截断规则与软件版本。

补充同配置 head-only：同 encoder、tokenize 512=384+128、同线性分类头、相同 CE、采样、学习率中对应 head 的部分、相同 epoch/early-stop 规则与 3 seeds，只把 adapter 开关关闭。不能把 round1 的 MLP+SupCon/40ep 当作唯一冻结对照。last_block 只跑 3ep 的 dev 结果可以作为资源优先级判断，不能称该方法已被充分淘汰。

没有训练逐 epoch 日志，就不能写“3 seeds 均未平台化”：best_epoch 分别为 12/11/9，只能确认 seed0 最优落在 12ep 上限。先保存 dev 曲线，再判断是否增加预算；当前不授权凭 test 结果反复加 epoch。

代码还要修：

- `--smoke` 当前写入与正式运行相同的 metrics/config/predictions 路径，给所有脚本增加独立 --out 和防覆盖检查；后续冒烟仅评估 dev，不读取 test 进行方案筛选。
- 若梯度累积最后一个窗口不足 accum，仍提交该窗口并按实际 microbatch 数缩放。当前 6557 行、bs16 有 410 个 microbatch，accum2 恰整除，但通用实现仍应修正。
- collate 按 tokenizer 的 pad_id 填充；mask 按实际长度建立；evaluate 不硬编码 pad_id=0。
- 预测包必须包含 task_id、样本 ID/代码 hash、y_true、family 顺序、split hash 和 seed；现有 trainable/ablations 预测包只有概率矩阵，不能独立验证顺序或直接进行配对任务 bootstrap。
- 保留最多一个 adapter+head checkpoint 是资源限制；不要根据 test 挑选 seed，报告仍保留所有 seed。

## 4. TF-IDF 诊断结论修正

先利用已有结果改写报告，无须重跑四个来源的大基线：

“联合匿名化、去注释/字符串和空白归一化后，Macro-F1 从 .7639 降至 .4572；说明当前性能依赖被这些处理改变的信号，尚不能区分标识符、注释、字面量、格式或任务混淆各自的贡献。”

如继续诊断，固定相同训练/验证选择协议，每次仅改变一种处理（标识符、字符串、注释、空白各一项），不把 regex 联合变换视为纯内容移除。词法处理须考虑字符串中的 //、/*，避免先删注释破坏字符串。长度桶边界由 train 拟合后应用于 dev/test；当前 shortcuts.metrics 用传入的 test 长度计算分位数，与“train 四分位”注释不一致。所有 LR 记录 convergence；filtered SGD 当前只记录 dev、不恢复 best epoch，应明确固定5ep或实施同 raw 的 dev 选择规则。

统计比较使用同 task 的配对 bootstrap，报告每 seed 差值和任务级置信区间。三 seed 均值差不能单独证明“噪声级”“显著提升”或“没有贡献”。报告表中 fd .707 相对 .703 是约 +0.4 个百分点，不是 ±0.0pt。

## 5. 执行范围与产物

本轮顺序：模式修复与 no-op 冒烟 → 修正版 MLP/消融 3 seeds → LoRA checkpoint 复算与匹配 head-only → 修正报告。暂不新增大型模型、数据或大规模 STACAD/AICD 训练，不进入新的 GRL/温度/门控网格。

输出新目录：`artifacts/acl_dcan_round3_audit/`。至少保存 mode_assertions、config、metrics、dev 曲线、带样本身份的压缩预测、数据及代码 hash、环境、checkpoint 复算报告和正式日志。新报告写入 `docx/d-det-acl-dcan-round3-audit-report.md`。

保留旧 round1/round2，标注哪些数字待修正。CPU 预处理仍 OMP=MKL=2；先 df 核对空间，至少保留1GB可用磁盘；复用现有 encoder 和 29MB 冻结特征缓存，不下载新权重。LoRA 上轮约10GB显存，仅在服务器运行；本机3050 4GB只做文档/代码审计。

完成后回答：修复评测后 late-fusion/消融差值是否仍成立？匹配条件下 adapter 是否改善任务留出？联合清洗结论是否收窄？下一步是否已有必要的证据和资源进入 generator-held-out 方法实验？不能用没有运行的外部评测宣称 SOTA。
