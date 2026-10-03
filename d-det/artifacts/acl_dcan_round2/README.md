# acl_dcan_round2 —— 3420857 之后三个问题的诊断产物（2026-10-03）

> 规范：`docx/d-det_AutoDL_3420857之后服务器AI继续指导_2026-10-03.md`。不覆盖 round1 产物。

## 布局

- `split_manifest.json` —— 切分说明（DCAN 沿用任务级 train/dev/test；test 不参与任何选择）。
- `env.json` —— 运行环境（Py3.12 / torch 2.9.1+cu128 / transformers 5.17 / peft 0.21 / 3080Ti）。
- `shortcuts/` —— **Q1 捷径诊断**（`scripts/dcan_round2_shortcuts.py`）：metrics.json + predictions.npz。
- `ablations/` —— **Q2 单变量消融**（`scripts/dcan_round2_ablations.py`）：4 变体 × 3 seeds + round1 参照。
- `trainable/` —— **Q3 可训练表示**（`scripts/dcan_round2_trainable.py`）：`smoke_metrics.json`（last_block .4867 vs lora .6067 @3ep）+
  正式 LoRA 3 seeds（≤12ep，bf16，bs16×accum2）；`metrics.json` + `predictions.npz`。
- 唯一 checkpoint：`runs/acl_dcan_round2/best_lora.pt`（1.2MB，LoRA+head；不入库）。

## 关键数字（task-held-out，DCAN；机会 1/6≈.167）

| 方法 | macro-F1 | 备注 |
|---|---:|---|
| char TF-IDF（round1） | **.7639** | 同切分强基线 |
| **LoRA(r8,q/v) 3 seeds** | **.7286 ± .0140** | 可训练表示最优；balAcc .7467/ECE .066；299,526 参数；~10GB VRAM |
| late-fusion（round1） | .7134 ± .0060 | 参照基线 |
| lf − SupCon / fd − semGRL / fd − fpGRL / fd − noorth | .7153 / .7073 / .7074 / .7071 | **无单变量超过 .713（实质）** |
| frozen head-only（round1 sem-only） | .6881 ± .0131 | |
| codeT5_center_std → LR | .6376 | **转导**（用 test 任务兄弟；非部署方法） |
| last_block（3ep 冒烟，未正式） | .4867 | 慢热，被 LoRA 淘汰 |
| metadata-only / filtered TF-IDF / raw LR | .4753 / .4572 / .4844 | 捷径诊断 |

## 复现命令

```bash
cd /root/autodl-tmp/u-det/d-det
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 TOKENIZERS_PARALLELISM=false
P=/root/miniconda3/envs/udet/bin/python
$P scripts/dcan_round2_shortcuts.py
$P scripts/dcan_round2_ablations.py --seeds 3 --epochs 40
$P scripts/dcan_round2_trainable.py --smoke --scheme both          # 方案筛选
$P scripts/dcan_round2_trainable.py --scheme lora --seeds 3 --epochs 12
```
