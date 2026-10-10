# AutoDL H3 GPU 诊断批次指导（一次性执行版）

版本：2026-10-11  
对应代码：提交 `98a9f79` 的 `scripts/h3_gpu_diagnostic_batch_2026_10_11.py`  
批次性质：**GPU 诊断批次**。H3 v1 正式数据闸门仍记录为 `revise_data`；本批次的结果用于判断模型容量、联合读出和安全变体一致性，不直接改写正式 H3 主表。

## 1. 这次服务器必须做什么

一次性跑完 24 个固定运行：

- `detection_only`：6 次（2 个 generator-heldout 折 × 3 seed）；
- `source_only`：6 次（任务留出读出 × 3 seed；generator-heldout 的未知类别不伪造分数）；
- `joint`：6 次（共享 MLP + detection/source 两个 head）；
- `joint_invariance`：6 次（joint + train-only parent/variant 一致性）。

CodeT5-small 只使用服务器已有的本地权重，先在 CUDA 上做一次 mean-pooling 编码，再在 CUDA 上完成 24 个 head 训练。固定参数：最大长度 512、编码 batch 16、head batch 256、3 epoch、seed=`20261011,20261012,20261013`。不读取 test 正文，不生成新数据，不下载权重，不临时改参数。

## 2. 上传清单

将以下文件和目录放到服务器同一个仓库路径（默认 `/root/autodl-tmp/u-det`）：

```
scripts/h3_gpu_diagnostic_batch_2026_10_11.py
d-det/artifacts/h3_gpu_diagnostic_batch_2026-10-11/batch_manifest.json
d-det/data/h3_stacad_revision_v1/train_balanced.jsonl
d-det/data/h3_stacad_revision_v1/dev_clean.jsonl
d-det/data/h3_stacad_revision_v1/train_variants.jsonl
```

服务器上必须已有：

```
d-det/models/codet5-small/{config.json,pytorch_model.bin,vocab.json,merges.txt}
```

数据 SHA256 固定为：

- `train_balanced.jsonl` = `152602c39f6af8268e774d235196a0ee7a4b6485d21e8cf10cd76faba96c6d58`
- `dev_clean.jsonl` = `fe7955a5cfb55093274e7da5721d485e213e6e3521d0b31b61ec7a2c8388b9cf`
- `train_variants.jsonl` = `e4d38e932015a4933825d5ef09ae05b759822455c2fd0762f166827a7cc33a98c`

## 3. 服务器执行

必须先确认有 CUDA；无 CUDA 时只记录 `CUDA_UNAVAILABLE` 并停止，不能把 CPU 冒烟冒充完成：

```bash
cd /root/autodl-tmp/u-det
source /root/miniconda3/etc/profile.d/conda.sh
conda activate udet
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4

python - <<'PY'
import torch
assert torch.cuda.is_available(), "CUDA_UNAVAILABLE"
print(torch.cuda.get_device_name(0), torch.cuda.get_device_properties(0).total_memory)
PY

mkdir -p d-det/artifacts/h3_gpu_diagnostic_batch_2026-10-11
nvidia-smi --query-gpu=timestamp,name,memory.used,memory.total,utilization.gpu,temperature.gpu,power.draw --format=csv -l 10 > d-det/artifacts/h3_gpu_diagnostic_batch_2026-10-11/gpu_utilization.csv &
GPU_MONITOR_PID=$!

python scripts/h3_gpu_diagnostic_batch_2026_10_11.py \
  --out d-det/artifacts/h3_gpu_diagnostic_batch_2026-10-11 \
  --jobs detection_only source_only joint joint_invariance \
  --epochs 3 --batch-size 256 --encode-batch-size 16 --max-length 512 \
  2>&1 | tee d-det/artifacts/h3_gpu_diagnostic_batch_2026-10-11/run.log.txt
STATUS=${PIPESTATUS[0]}
kill "$GPU_MONITOR_PID" 2>/dev/null || true
printf 'BATCH_STATUS=%s\n' "$STATUS"
exit "$STATUS"
```

预期产物：

- `metrics.json`：24 次运行、每折/每 seed 指标、聚合读数、CUDA/权重/输入哈希；
- `run_manifest.json`：实际批次指针；
- `run.log.txt`：完整日志（使用 `.txt`，避免仓库 `*.log` 忽略）；
- `gpu_utilization.csv`：显卡利用率、显存和温度时间序列。

## 4. 回传与判读

整目录一次性打包回传，不接收零散 job：

```bash
cd /root/autodl-tmp/u-det
tar -czf /tmp/h3_gpu_diagnostic_batch_2026-10-11.tar.gz \
  d-det/artifacts/h3_gpu_diagnostic_batch_2026-10-11
sha256sum /tmp/h3_gpu_diagnostic_batch_2026-10-11.tar.gz
```

回传后本机只做汇总和 CI 复核：

- detection：看两折、三 seed 的 AUROC/BA 方向和波动；
- source：只报告已见 generator 的 task-heldout accuracy；未知 generator 明确标记 unsupported；
- joint：比较 detection-only 的增量；
- joint-invariance：检查 parent/variant 一致性及检测增量；
- GPU 利用率和日志用于确认确实完成了 CUDA 训练，而不是仅做文件核验。

任一 job 失败，原样保留失败日志；不加 epoch、温度、容量或损失项补跑。正式 H3 闸门仍按本机修订后的 length/lexical/AST/transform 规则单独裁定。
