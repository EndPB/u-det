# AutoDL H3 执行入口

本文件已收敛为入口。当前完整指南见 [AutoDL 当前交接指南](AUTODL_AI_HANDOFF_GUIDE_2026-10-10.md)，H3 通过闸门后的执行矩阵见 [H3 修订后新主线批次指导](d-det_AutoDL_H3修订后新主线批次指导_2026-10-10.md)，研究总账见 [ACL 研究总账与可迭代路线](d-det_ACL总研究总结与可迭代路线_2026-10-10.md)。

当前裁定：小显存和 CPU 友好的数据任务直接在本机完成，不单独写 AutoDL 指导或打扰用户；只有需要较大显存或长时间 GPU 的阶段才交给 AutoDL，并按完整批次一次执行。当前 H3 数据闸门为 `revise_data`：length-only task-macro AUROC `.8459`、lexical-only `.9261`，因此先由本机完成长度分桶平衡、七语言变体、parser 补齐和两组跨任务骨架重复裁定；不启动 H3。闸门通过后，AutoDL 批次应同时覆盖同题 Human/AI detection、AI source head、task/project/generator 隔离、UIT-AMMC 式 train-only 变体、至少两个 generator-heldout 折和三个 seed；旧 C0–C3 不重跑，test、生成、权重和代码执行保持关闭。

每次迭代请在总账版本日志追加日期、commit、数据包 SHA、闸门、结果和最小下一步，并把新产物写入独立目录。
