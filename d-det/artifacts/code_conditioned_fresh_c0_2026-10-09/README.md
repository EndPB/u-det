# code_conditioned_fresh_c0_2026-10-09

Fresh train/dev C0 reconstruction (per §9 adjudication / fresh_c0_protocol.json).

- `manifests/` + `manifests_index.json`: per-fold manifests **written & hashed before fitting**
  (effective series map sha, heldout member/series, fit/eval member ids, positive/negative/train
  row hashes, eval task hash, component specs, code commit).
- `metrics.json`: dev (row .9329 / tm .9442) + inner (.9301/.9515), bootstrap CIs, determinism
  self-check (refit max|Δ|=0.0), bit-comparison vs the 2968ff9 C0 (max|Δfused|=0.0),
  diagnostic vs the historical package under the official reading (not the gate).
- `score_digests.json`: sha256 over canonical float64 arrays (eval: taskpos order; train: fit
  order) for cross-side alignment.
- `spec_sheet.json`: complete replication spec (components, fusion, metrics, environment).
- `report.md`: main report (Chinese). `logs/`, `commands.sh`, `git_*`, `SHA256SUMS.txt`.
- score npz live in `local/` (not committed; rebuild with `scripts/cc_fresh_c0.py`).

Switches: train/dev only; test_read=false; generation=false; weights_downloaded=false;
code_execution=false. C1-C3 not rerun (requires aligned fresh C0).
