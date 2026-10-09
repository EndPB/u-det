# code_conditioned_c0_strong_p0_2026-10-09（C0 强 P0 重建）

性质：复现/审计轮。四 fit-only 组件（CodeT5-small semantic linear / char TF-IDF / word TF-IDF / style-meta linear）等权 z-score 融合；11 member-heldout 折。

- dev row-level mean = **0.9329**（CI 0.9228–0.9424）；task-macro = 0.9442
- inner train-only row-level = 0.9301
- 与本机 approx 参考（.9344/.9324）差：dev −0.0015 / inner −0.0023（>1e-3；本机逐点参考表未传输，见 metrics.reference_check）
- 规格探针史见 `metrics.json.reference_check.spec_probe_history`；探针脚本在 `probe_scripts/`
