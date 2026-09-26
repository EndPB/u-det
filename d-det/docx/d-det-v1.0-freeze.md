# d-det v1.0：冻结方案（单一最优配置；汇报与 SemEval 基准）

> 2026-09-24 冻结 ｜ 状态：**生效** ｜ 后续所有对外汇报与 SemEval 相关工作以本文档为准
> 权重指纹：`runs/v0.4.1_covreg/last.pt` **sha256 = e7e6565c…95a22e4**；`config.yaml` sha256 = ef00282a…0bde90a22

---

## 1. 冻结内容（一句话）

**d-det v1.0 = `v0.4.1_covreg` 权重**（CodeT5-base 全参 + 四路监督 + 四项几何损失）**+ 双轴融合决策协议**
（默认 `fused = LR([s1, s2])`；单头 s1 仅限分布内使用）。

- 权重：`runs/v0.4.1_covreg/last.pt`（epoch 5；因 monitor 0.9232 ≤ resume 0.9233 未生成 best.pt，`last` 即最终权重；
  训练脚本已加保底——此后 `best.pt=last.pt` 不会再缺）。
- 配置：`configs/ddet_v041.yaml`（训练时另加 `--set loss.rep_margin=4.528`）。
- 结构：`DualScoreModel`（mean 池化；`w1→s1` 人机轴 / `w2(rank1)→s2` 偏好位移轴 / `tok_head` token 级）。
- 配方：全参微调（109.6M）＋四流 `[m4, hybrid, pair, pairxf]`；损失 `s1=1, token=1, s2=1(m=1), rep=1(m=4.528),
  fisher=1, xfam=0.5, cov=10, mlm=0.2, orth=0.05`；2ep（基于 v0.3.0-4ep 续训）；lr 3e-4 / enc 2e-5；accum 8；bf16。
- 决策协议：**融合（s1+s2）为默认**；报数口径 = 5×4 重复分层 CV + 逻辑回归（与 `probe_scale_verify.py` 一致）；
  部署时用目标域少量标注拟合 LR；**纯 s1 只用于分布内**（跨风格/OOD 见 §4）。

**备用参照（不默认使用，仅在特定场景）**
- `runs/v0.3.0_fullft/last.pt`（4ep）：**s1 的 OOD 最强**（instruct 池 0.99）、ds13-base fused 最佳（0.9846）；
- `runs/v0.3.0_fullft/best.pt`（若被 epoch3 覆盖则以 last 为准）：m4 F1 峰值 0.9776（2ep 权重另存价值）。

---

## 2. 为什么是它（证据，F1 优先）

**m4/test（n=1940）**

| 组 | **F1** | acc | AUC |
|---|---|---|---|
| **v1.0 = v0.4.1** | **0.9775** | 0.9773 | 0.9953 |
| v0.4.0 | 0.9766 | 0.9763 | 0.9955 |
| v0.3.0 2ep（峰值） | 0.9776 | 0.9773 | 0.9966 |
| v0.3.0 4ep | 0.9756 | 0.9753 | 0.9969 |
| u-det v0.4.4 | 0.9775 | 0.9773 | — |
| base_codet5_ft | 0.9756 | 0.9751 | — |
| base_codet5_tok | 0.9736 | 0.9732 | — |

**hybrid/test（F1）**：v1.0 = 0.7877 / 0.5008 / 0.8167（line 微超 u-det 的 0.7878；token 距 u-det 0.2pt）
**片段（acc@0.5）**：v1.0 = full 0.9820 / full_frag **0.9861** / head 0.9537 / tail 0.9282 / mid 0.8886（full 两项全系最佳）
**跨族（qwen15 / ds13）**：v1.0 的 s2\* = **0.9999 / 1.0000**、fused = 0.9987~0.9992（instruct）/ 0.9878~0.9812（base）

**为什么不是另外两个**（三者差异全部在噪声内：m4 σ≈0.34pt；片段 σ≈1.3pt@n=500）：
- vs v0.4.0：v0.4.0 的 mid 0.9026 更好（差 7/500 样本），但 v1.0 的 m4 F1、full/full_frag、ds13-instruct fused 更高，
  且包含完整配方（去冗余项）；mid 的差距不显著。
- vs v0.3.0-2ep：F1 峰值同为噪声级；v1.0 在 hybrid/fragments/s2\* 上全面更强，只在 s1-OOD 上弱（用融合或备用 s1 解决）。

**交叉融合矩阵（新证据，`scripts/cross_fusion.py`）**

| 池 | 最佳组合 | AUC |
|---|---|---|
| qwen15 instruct | v0.3.0-s1 + 任意 v0.3.0/v0.4.x-s2 | **1.0000** |
| qwen15 instruct | **abA-s1 + v1.0-s2** | **0.9997**（s2 能救最弱的 s1） |
| ds13 instruct | v0.3.0-s1 + v0.4.x-s2 | **1.0000** |
| qwen15 base | v0.3.0-s1 + v1.0-s2 | **0.9900** |
| ds13 base | v0.3.0-s1 + v0.3.0-s2 | **0.9846** |

结论：**s2 轴是独立且强力的第二判别轴**（v1.0 的 s2 可把任意 s1 拉到 0.9997+）；**s1 轴跨风格最强的是 v0.3.0**。
（如更看重片段 mid，可把主权重改冻 v0.4.0——请以此文档更新为准。）

---

## 3. 冻结流程（复现）

```bash
cd /root/autodl-tmp/u-det/d-det
# 训练（已完成于 2026-09-24）：
python train.py --config configs/ddet_v041.yaml --tag v0.4.1_covreg \
    --resume runs/v0.3.0_fullft/last.pt --epochs 2 --set loss.rep_margin=4.528
# 评测与探针（已完成；ckpt 用 last.pt）：
python train.py --config configs/ddet_v041.yaml --eval --ckpt runs/v0.4.1_covreg/last.pt --dump-scores
python scripts/probe_fragments.py    --config configs/ddet_v041.yaml --ckpt runs/v0.4.1_covreg/last.pt --n 500
python scripts/probe_scale_verify.py --config configs/ddet_v041.yaml --ckpt runs/v0.4.1_covreg/last.pt \
    --tag v0.4.1_covreg --pair-file data/processed/pairs_qwen15.parquet --npz runs/v0.4.1_covreg/scale_scores.npz
python scripts/cross_fusion.py --title qwen15 \
    --npz v0.3.0=runs/v0.3.0_fullft/scale_scores.npz \
    --npz v0.4.0=runs/v0.4.0_geom/scale_scores.npz \
    --npz v0.4.1=runs/v0.4.1_covreg/scale_scores.npz
```

---

## 4. 已知缺陷与使用禁忌

1. **instruct/chat 风格 OOD 上 s1 弱**（0.86~0.88；v0.3.0 为 0.99）→ 必须用融合；或换用 v0.3.0-4ep 的 s1；
   去冗余（covreg）**不能**修复此点（v0.4.1 实测）。
2. **片段 mid = 0.8886**（仍低于 abA 0.9165；v0.4.0 的 0.9026 为本谱系最佳）。
3. **"显著超越"禁语**：m4 主口径在噪声带内（对外措辞 = "打平全量微调基线"）；
   "显著"只可用于跨族/零标注第二轴（且对外引用前需补真基线跨族读数——待办）。
4. AUC 略降（0.9953 vs 2ep 0.9966）：排序质量未同步提升，报告时如实标注。

---

## 5. SemEval-2026 Task 13 应用计划（适配流程）

（官方评测期 2026-01 已结束 → 本流程按"外部基准 + 可复用适配"定义；后续同类赛道可直接套用。）

- **映射**：Task A（人机二分类）= s1（+fused）；Task C（Human/Machine/Hybrid/**Adversarial**）= s2 + `tok_head`
  （Adversarial=RLHF 模仿人类，正是 s2 的定义域；Hybrid 对应 token 级）；Task B（家族归因）= 需 P2 探针/低秩子空间（暂缓）。
- **适配方案（遵守预算纪律：每任务 = 一个新方法 ≤2ep）**：
  1. 官方 train 分层子采样（A ~30-50K / C ~50-100K；含未见语种样本），从 v1.0 权重 `--resume` 续训 2ep；
  2. 任务头：A 直接复用 s1（+s2 融合）；C/B 在共享 encoder 上接新分类头（多类交叉熵，macro F1 早停）。
- **零 GPU 备选（推荐先做）**：冻结 v1.0 抽特征（s1 / s2 / tok 统计 / 池化 h 投影）+ 线性分类头（CPU 训练）——
  快速检验可行性，成本 <1h。
- **合规**：仅用官方 train；不得用第三方 AI 检测器；通用/代码预训练模型允许（v1.0 合规）。
- **交付**：官方 `scorer.py` 口径（macro F1）；提交 CSV（id,label）。

---

## 6. 材料索引（冻结包）

| 件 | 路径 |
|---|---|
| 权重（v1.0） | `runs/v0.4.1_covreg/last.pt`（sha256 e7e6565c…22e4） |
| 配置 | `configs/ddet_v041.yaml`（+ `loss.rep_margin=4.528`） |
| 评测/探针产物 | `runs/v0.4.1_covreg/{eval.json, fragments.json, scale_scores*.npz, scores_*.pt}` |
| 备用权重 | `runs/v0.3.0_fullft/last.pt`（s1-OOD 备用）、`runs/v0.4.0_geom/last.pt`（mid 备用） |
| 关键脚本 | `cross_fusion.py` / `probe_scale_verify.py` / `probe_fragments.py` / `calibrate_margin.py` / `check_v040_grads.py` |
| 完整报告 | `docx/d-det-v0.4.md`（两臂）；`docx/d-det-v0.3.md`（前代）；README §9（命令） |
```
