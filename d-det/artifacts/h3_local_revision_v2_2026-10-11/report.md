# H3 v2 本机数据修订报告

## 结果

- controlled-dev：2636 行、659 个 task；按 train-only、language 内 log1p(bytes) 四分位冻结，task 内保留 human 与最多 3 个不同 generator 的最近长度 AI。
- clean-dev lexical task-macro AUROC：0.9694；controlled-dev：0.9585。
- clean-dev length task-macro AUROC：0.7248；controlled-dev：0.6679。
- view-probe task-macro AUROC：0.5569；结果已按 label、language、generator、length_bin 展开。
- parser 失败按语言、标签、generator 分层；当前文件来自 v1 invalid_examples 清单，未把解析失败行删除。

## 裁定

`revise_data`，`formal_gpu_ready=false`。长度/词法控制仍高于准入线，project/repository provenance 仍不可用；本轮没有生成新的 GPU 批次，也没有读取 test。旧 H3 GPU 诊断已关闭，不在旧冻结包上重跑。

下一步是补齐项目来源并重新构造 label×language×generator×length_bin 平衡包，随后重新跑全量 train/dev 闸门；只有全部通过才制作一次性 GPU manifest。
