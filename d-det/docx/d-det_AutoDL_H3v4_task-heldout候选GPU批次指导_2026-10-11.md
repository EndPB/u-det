# AutoDL H3 v4 task-heldout 候选 GPU 批次指导

版本：2026-10-11  
状态：**已被本机 AMMC 类增强诊断取代，禁止启动**。候选包仅保留作审计历史。

## 批次边界

这是一个新的 task-heldout 候选主线：v4 hard-negative train/dev、2 个 generator-heldout fold、3 个 seed、4 个 job，共 24 个 job。clean-dev 的 `.7248` 长度读数作为负控制继续报告；project provenance 缺失，因此结果只能写 task-heldout，不能写 project-heldout。

固定开关：`test_read=false`、`generation=false`、`weights_downloaded=false`、`code_execution=false`。模型必须使用服务器已有的 `d-det/models/codet5-small`，不得下载权重。

## 执行状态

本机已在 RTX 3050 上完成同协议的增强训练与 raw/masked lexical 对照（18 个检测运行 + 3 个独立 source-only 运行）。增强后 raw lexical 捷径仍强，且相对 raw lexical 的增量未在两折、两个评估视图和三 seed 上同时通过预注册 CI；裁定为 `increment_not_observed_close_v4_frozen_candidate`。

因此下面的旧命令**不再执行**，即使 CUDA 恢复也不启动该候选包。下一轮必须先完成 AMMC 组合视图、transform-only/view-only probe 和新的本机数据闸门，再生成新的 manifest 与完整 GPU 批次指导。

## 历史命令（仅供审计，不得执行）

在仓库根目录执行：

```bash
/root/miniconda3/bin/python scripts/h3_formal_v4_gpu_batch_2026_10_11.py \
  --out d-det/artifacts/h3_formal_v4_gpu_batch_2026-10-11
```

脚本固定运行 `detection_only`、`source_only`、`joint`、`joint_invariance`，每个 2 折 × 3 seed；不接受临时增加的配置。完成后统一保留 `metrics.json`、`run_manifest.json`、日志、GPU 监控和 SHA256SUMS。

## 失败规则

主检测任务相对同折 v4 masked lexical control 没有预注册增量时，结论为 `increment_not_observed`，停止，不加容量、温度、轮数或损失项。任何 CUDA 不可用的会话只做接收核验，不把 CPU 运行冒充 GPU 实验。

