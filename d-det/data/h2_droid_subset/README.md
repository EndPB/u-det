# H2 DroidCollection subset

This directory is a small, uploadable candidate dataset for the cross-generator H2 experiment. It was selected from `project-droid/DroidCollection` at revision `9a42843be99456a3af70d2b9c9a53a4d7624598f` through `https://hf-mirror.com`.

## Contents

- `core.jsonl` — 18,000 primary rows: 14,220 `MACHINE_GENERATED` rows and 3,780 `HUMAN_GENERATED` controls. It covers 7 model families and 32 machine generators. The original source split is retained in `split_source`.
- `diagnostic_hybrid_adversarial.jsonl` — 2,000 diagnostic-only rows: 1,000 `MACHINE_REFINED` and 1,000 `MACHINE_GENERATED_ADVERSARIAL`. Do not mix these into the primary H2 fit or score.
- `summary.json` — source revision, selected families/generators, source label counts, output counts, and schema.
- `fold_plan.json` — deterministic two-fold generator-held-out assignment. For each family, one generator subset is held out in each fold; the other generators provide training/validation rows.

The JSONL files are intentionally small enough for transfer and streaming reads. The builder downloaded one Parquet shard at a time, iterated row groups, and deleted each raw shard immediately after processing; no full dataset or embedding was materialized locally.

## Server-side use

1. Upload this directory to AutoDL.
2. Read `core.jsonl` in streaming mode.
3. Follow `fold_plan.json`: machine `train` rows use `split_source=train` and training generators; machine validation rows use `split_source=dev` and the same generators; held-out evaluation uses `split_source=test` and held-out generators. Keep human controls in the matching source split.
4. Report family attribution on held-out generators (BA_F and per-family results). A generator classifier cannot claim unseen-generator identification when the held-out generator label was absent from training; treat generator-level scores as seen-generator diagnostics only.
5. Run the diagnostic file as a separate robustness analysis for refined/adversarial code.

## Limitations

DroidCollection provides `Code`, `Generator`, `Model_Family`, `Label`, `Language`, `Source`, generation/rewrite metadata, and split information, but no public `task_id` or `prompt_id`. Therefore this subset supports a controlled generator-heldout attribution test, not a proof of same-task paired human/AI comparisons. Avoid claiming task-level causal or prompt-invariant generalization without adding a task-aware dataset or metadata.

The selected rows are a bounded quota sample, not a re-export of the full corpus. Re-run the builder only if a larger quota is needed.
