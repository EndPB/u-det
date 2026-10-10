# H3 v2 本机数据修订假设

若将长度控制冻结为训练拟合的语言内分位桶，并在每个 dev task 选择跨 generator 的最近长度 AI 对照，同时按 label×language×generator×length_bin 检查变体接受率，则可区分长度残余、变体标记和 parser 失败造成的表面信号。只有 clean-dev 与 controlled-dev 的控制读数都下降、view probe 接近机会、且 project provenance 补齐，才允许生成新的 GPU batch_manifest。

本轮只做 CPU 数据审计与轻量 probe；test 未反序列化，旧 GPU 诊断不重跑。
