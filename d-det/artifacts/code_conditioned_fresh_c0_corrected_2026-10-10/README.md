# code_conditioned_fresh_c0_corrected_2026-10-10

Corrected fresh train/dev C0 (fold-fit-only lexical TF-IDF). Supersedes
`code_conditioned_fresh_c0_2026-10-09` (blocked: spec_violation_global_lexical_fit).

- `manifests/`: per-fold v2 manifests — `lexical_fit_spec` written before fitting
  (vectorizer params, fit-row key hash, fit-texts hash), `lexical_fit_attestation`
  appended after fitting (vocab size / vocab sha256 / IDF sha256); dual manifest
  hashes (`manifest_sha256_pre_fit` / `manifest_sha256_post_fit`).
- `metrics.json`: dev .9328/.9451, inner .9296/.9520; determinism refit max|Δ|=0.0
  (vocab hash match); impact vs blocked global-fit run (r .997-.998); historical
  diagnostic (official reading; not the gate).
- `score_digests.json` / `spec_sheet.json`: cross-side 1e-3 gate instruments.
- `report.md`: main report (Chinese). Score npz in `local/` (gitignored).

Switches: train/dev only; test_read=false; generation=false; weights_downloaded=false;
code_execution=false. C1-C3 remain stopped until the corrected C0 passes protocol
audit and the 1e-3 cross-side gate.
