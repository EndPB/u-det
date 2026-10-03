# acl_dcan_round3_audit · 审计修正轮产物

对应报告：`docx/d-det-acl-dcan-round3-audit-report.md`；执行规范：`docx/d-det_AutoDL_701d0df审计修正与下一步指导_2026-10-03.md`。

## 内容导航

| 文件 | 说明 |
|---|---|
| `config.json` | 全部修正版配置（MLP 8 臂、FT 匹配 head-only、统计口径、边界） |
| `env.json` | 软件/硬件环境（OMP=MKL=2） |
| `split_manifest.json` | 切分未变声明 + 不变量复核证据（跨 split task_id=0） |
| `mode_assertions.json` | 评测模式断言（双评一致/BN 冻结/批组成容差/split/test 隔离） |
| `mlp/` | 修正版 MLP 8 臂 ×3 seeds：`metrics.json`（含 noop_repro）、`dev_curves.json`、`predictions.npz`（含 task_id/source_sha256/model_name/y/family_order/split_hash + 各臂概率，键名 `{arm}_s{seed}_fuse` 等） |
| `mlp_pre_fix_archive/` | **无效归档**：初版漏 `opt.zero_grad`，数字全部作废（见 NOTE_INVALID.txt） |
| `stats_paired_bootstrap.json` | 任务级配对 bootstrap（B=2000，408 任务）：4 组对比逐 seed 差值与 CI95 |
| `shortcuts_fix.json` | Q1 口径修正：train 拟合长度桶 / LR 收敛记录 / filtered 5ep 协议 |
| `bn_diag.json` | 评测模式差异归因（A/D/B/C 四口径，late_fusion s0） |
| `checkpoint_recompute.json` | round2 LoRA ckpt 复算：fp16 逐位一致、基座 sha256、配置与截断记录 |
| `dev_curves_round2_lora.json` | round2 LoRA 逐 epoch dev 曲线存档（best .7285@12 / .7283@11 / .7271@9） |
| `head_only/` | 匹配 head-only（3 seeds）：`metrics.json`、`predictions.npz`、`predictions_meta.json`（身份侧车：split_hash 与 MLP 包一致） |
| `head_vs_lora.json` | head-only vs round2 LoRA 逐 seed 配对 bootstrap（同 encoder/tokenize/头/lr/预算） |
| `hashes.json` | 数据/基座/ckpt/脚本/关键产物的 sha256 |
| `logs/` | 正式运行日志（mlp.log、head_only.log） |

## 关键数字（速查）

- 修正版 MLP（fuse，3 seeds）：late_fusion **.7381±.0043**、full_disentangle **.7356±.0021**、lf_nosupcon .7274、fd_nosemgrl .7280、fd_nofpgrl .7387、fd_noorth .7364、semantic_only .6987、fingerprint_only .5881。
- SupCon 差值 +1.1pt（逐 seed CI 均跨零）；其余开关无一致正贡献（orth s2 显著负 −2.0pt）。
- no-op：dev 曲线逐位一致、概率 maxdiff 0.0。
- 匹配 head-only vs LoRA：**head-only .4163±.0112 vs LoRA .7286±.0140**（配对差 −31.2pt，逐 seed CI 全离零；`head_vs_lora.json`——同 encoder/tokenize/头/CE/采样/head-lr/预算，仅关 adapter）。

## 重跑命令（防覆盖已内置）

```bash
P=/root/miniconda3/envs/udet/bin/python
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 $P scripts/dcan_round3_mlp.py --seeds 3 --epochs 40 --noop-check --out artifacts/acl_dcan_round3_audit/mlp
$P scripts/dcan_round3_ft.py --verify-ckpt runs/acl_dcan_round2/best_lora.pt
$P scripts/dcan_round3_ft.py --scheme head_only --seeds 3 --epochs 12 --out artifacts/acl_dcan_round3_audit/head_only
$P scripts/dcan_round3_stats.py && $P scripts/dcan_round3_head_vs_lora.py
$P scripts/dcan_round3_shortcut_fix.py && $P scripts/dcan_round3_bn_diag.py
```
