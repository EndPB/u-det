# v3 canonical C0 本机补交包对账

本轮已覆盖并核验服务器补交的三个 train/dev feature 文件：bundle、features_manifest、feature_member_order_check。成员序、行序、fold lexical fit 和词表均保持 11/11 一致。

在本机 Python 3.12.3 / NumPy 2.2.6 / sklearn 1.9.1、OMP=8 下重跑 canonical C0。dev 指标与服务器值在 1e-3 内，task-macro 逐折一致；但服务器只发布 score digest，没有发布逐行分数数组。本机与服务器 digest 字段仍为 0/99 相等，因此不能计算或声称 `row_score_max_abs <= 1e-3`。

裁定：`cross_side_score_pending`。C0 尚未标记 `aligned`；C1–C3 继续停止，test、生成和权重开关保持关闭。若要闭合严格闸门，只需服务器提供 11 折 train/dev 的压缩逐行分数数组，或在完全匹配的运行时重跑；不需要 test 数据、权重或原始语料。
