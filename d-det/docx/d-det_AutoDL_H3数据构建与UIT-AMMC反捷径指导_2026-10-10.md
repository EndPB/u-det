# AutoDL H3 执行入口

本文件已收敛为入口。当前完整指南见 [AutoDL 当前交接指南](../../AUTODL_AI_HANDOFF_GUIDE_2026-10-10.md)，研究总账见 [ACL 研究总账与可迭代路线](d-det_ACL总研究总结与可迭代路线_2026-10-10.md)。

当前裁定：服务器无卡时只接收文件、做 CPU 数据审计和 shortcut probes；数据构建、切分、哈希、AST 安全变体、TF-IDF 和低显存工作优先在本地完成。真实 H3 需要同题 Human/AI detection、AI source head、task/project/generator 隔离和 UIT-AMMC 式 train-only 变体；旧 C0–C3 不重跑，test、生成、权重和代码执行保持关闭。

每次迭代请在总账版本日志追加日期、commit、数据包 SHA、闸门、结果和最小下一步，并把新产物写入独立目录。
