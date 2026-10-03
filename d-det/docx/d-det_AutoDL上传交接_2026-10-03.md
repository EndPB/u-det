# AutoDL 数据上传交接（2026-10-03）

项目根目录：`/root/autodl-tmp/u-det`。本次仅上传数据、准备脚本与研究指导，不启动训练，也不覆盖已有训练结果。

## 上传内容与入口

- `d-det/data/acl_attribution_collection_v1/`：集合说明、来源清单、统一元数据索引、AICD T1/T2/T3 原始 parquet 与基础审计。
- `d-det/data/h2_authorbench/`：八模型覆盖的任务子集及原始公开压缩包。
- `d-det/data/h2_authorbench_dcan/`：按相同 prompt 筛选的跨来源对齐候选核心。
- `d-det/data/h2_droid_full_selected/`：generator/family 显式字段的本地选定数据；原有 `h2_droid_full_selected_upload/` 仍保留。
- `d-det/data/stacad_v2/`：官方压缩包、说明与已抽取的配对数据/辅助测试数据。
- `d-det/data/codet_m4/`：公开 parquet、下载记录与元数据计数审计。
- `d-det/data/llm_codegen_raw/` 和 `h2_llm_codegen/`：公开 CSV 原始数据与初步整理的对齐候选。
- `scripts/`：相关下载、审计、构建脚本。
- `d-det/docx/`：SOTA 路线图、数据集合执行指导、本交接说明和上传 manifest。

所有传输文件应按 `autodl_upload_manifest_2026-10-03.json` 的 SHA256 校验。上传校验仅证明传输完整，不表示训练数据或实验协议已经审计完毕。

## 开始实验前必须处理的事项

1. AICD 的数字标签映射仍待官方验证；公开 parquet 不包含 generator/task/language，不能凭猜测分层报告。T1/T2/T3 是任务视图，其总行数不是独立样本总数。
2. AuthorBench 两个子集源自同一压缩包且彼此重叠，分别生成的 task split 不能合并训练/测试。选择一个主协议，统一 group split 后再作比较。
3. STACAD 的现成 group 是 `file_name`，目前保证的是文件级留出；未恢复 repository_id 前不能声称仓库级留出。
4. LLM-CodeGen 当前 core 是待审计候选，尚不能直接作为正式训练集：检查原始 CSV 编码、response 解析失败、空代码、代码围栏/解释文本、C/C++语言、相同 CWE 的不同 prompt。正式任务泛化评估应以 CWE 统一分组，避免 simple/secure 的同一问题跨 split。
5. LLM-CodeGen 当前 family 是组织来源的初步归并（例如 Google/Meta），不同架构谱系不能据此宣称同一家族；保留 generator 名称并重新审核谱系映射。同任务不等于经过功能等价验证。
6. CoDET-M4 本地审计显示 5 个具名 AI model、human 和缺失 model 值；只有两值 target=ai/human，缺失 model 不可当作新的生成器。
7. 跨来源 exact/near duplicate 检查、任务泄漏审计、功能正确性筛选尚未完成。优先产出这些审计结果，再建立 baseline。不要把待审计候选直接写成完整 ACL 数据贡献。
8. 部分旧准备脚本含 Windows 本地绝对路径；服务器复用时先改为项目相对路径并保留 provenance，不要盲目运行。

## 建议服务端下一步

先审计上述数据结构与切分，修复 LLM-CodeGen 解析/分组，再输出可训练清单和 group-safe split。随后在选定协议上训练 baseline 与语义/来源分离方法；保留正式测试集，不在其上调参。启动 GPU 实验属于后续工作，本次上传不启动任何实验。

服务器数据盘空间较紧张：不要再解压完整 STACAD 压缩包，不要把所有 AICD 配置转换为重复 JSONL；按 parquet batch 与 JSONL 流式处理。先确认缓存/模型权重的空间预算，再开始训练。
