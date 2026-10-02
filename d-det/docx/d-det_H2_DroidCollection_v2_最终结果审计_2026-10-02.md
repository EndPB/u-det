# DroidCollection v2 H2 正式结果审计与收尾建议

日期：2026-10-02  
服务端提交：3ab2e46  
实验脚本基线：de88cad

## 结论

146,718 条正式主样本的实验已经完成并复现通过。F1 是 F0 加入跨 generator 正对约束的唯一方法臂。两折结果如下：

| 结果 | fold 0 | fold 1 | pooled |
|---|---:|---:|---:|
| F0 BA_F | .1493 | .1680 | .1580 |
| F1 BA_F | .1420 | .1601 | .1517 |
| F1−F0 BA_F | −0.73pt | −0.79pt | −0.63pt |
| F1−F0 BA_G | −0.68pt | −0.60pt | −0.60pt |

四个折级差值全部为负，pooled 的两个差值也为负。按照预注册出口 3，当前 DroidCollection v2、768 维冻结表示和该跨 generator 正对定义下，不支持 H2。

这次结果同时说明，扩大数据量没有解决核心问题。B0 的 generator-held-out BA_F 只有 .1470/.1624，pooled 为 .1543，机会水平为 1/7=.1429；随机切分主轴约为 .327–.329。也就是说，主要瓶颈是 generator shift 下家族信息几乎没有被当前表示稳定保留，而不是 SupCon 权重或训练轮数不足。

检测 AUROC 为 .9511/.9553，只能说明人类/机器检测信号较强，不能推出家族归因空间有效。

## 数据和复现

- 主包：146,718 行，其中 125,718 条机器生成、21,000 条人类对照；
- 覆盖 7 个 family、32 个 machine generator；
- source_row_sha1 无重复；
- 两折 generator 集合不相交，held-out generator 均有测试样本；
- test 只用于最终评估，diagnostic 文件未参与训练或评分；
- 特征缓存 146,718/146,718 行，无跳过；
- 两次运行的 metrics 逐项一致，predictions 逐位一致，输入与缓存 hash 一致。

服务端正式产物：

    d-det/artifacts/h2_droid_v2/

其中应保存 audit.json、config.json、manifest.json、metrics.json、predictions.npz、solver.log、repro_compare.json 和 report.md。

## 必须补充的覆盖限制

当前报告的总体 anchor 比例为 fold 0=.714、fold 1=.857，足以通过原先的“整体比例不低于 .15”守卫，但这个指标不能代表每个 family-fold 都被 H2 损失检验：

- fold 0 的 CodeLlama 训练侧只有一个 generator，因此该 family 的跨 generator anchor 为 0；
- fold 0 和 fold 1 的 IBM Granite 训练侧都只有一个 generator，因此该 family 两折的跨 generator anchor 都为 0；
- fold 1 的 CodeLlama 有两个训练 generator，因此该折有有效 anchor。

因此最终表述应为：

> 在大多数 family-fold 单元有有效跨 generator 正对的正式协议下，F1 未显示增益；CodeLlama fold 0 和 IBM Granite 两折没有被跨 generator 对比损失直接训练，故对这些单元只能报告未充分检验。

不要把 .714/.857 写成“所有 family-fold 的 anchor 覆盖充分”。这是一项报告边界修正，不需要重新训练。

## 研究决策

1. 不再继续调 SupCon 温度、损失权重、门控、学习率或 epoch。
2. 不再添加 DMHM、DCAN、局部密度、协方差、正交或低秩分支。
3. 不再扩大 DroidCollection 原始数据，也不再把更多数据量当作解决 H2 的主要路径。
4. 论文主张收窄为：后训练差异在检测任务中较稳定；在闭集或随机切分中存在一定来源可读性；但 generator-held-out 的跨 family 归因在当前冻结表示中接近机会水平，当前来源残差和跨 generator 对比约束没有带来稳定增益。
5. AuthorBench 继续作为 task-aware 机制审计；STACAD 只作为外部多语言模型归因验证，不替代 H2 的 family-generator 层级。

后续工作应转入最终表格、图、限制和方法讨论。若未来仍要重新检验 H2，必须先使用每个 family 至少三个 generator 的 task-aware 数据构造新的折叠协议；不能通过继续调当前损失来补救。
