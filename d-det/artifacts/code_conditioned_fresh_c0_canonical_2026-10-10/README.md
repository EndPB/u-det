# code_conditioned_fresh_c0_canonical_2026-10-10

Canonical-order fresh train/dev C0 (member-order fix per guidance §12 + retained
fold-fit-only lexical fix). Server-side, train/dev only.

Supersession chain (C0):
1. `code_conditioned_fresh_c0_2026-10-09` — BLOCKED (global lexical fit).
2. `code_conditioned_fresh_c0_corrected_2026-10-10` — lexical protocol fix done,
   but bundle member order was still permuted (CL, DS, Q vs admission CL, Q, DS).
3. **this directory** — canonical member order + fold-fit-only lexical; the run
   used for the pending cross-side score-digest comparison.

## What was fixed (§12)

- `scripts/cc_build_features.py`: `member_idx` rebuilt in admission order; row
  assertions (`member_idx[i] == admission.members_order.index(rows[i].model_id)`)
  plus `members_order_sha256` / `row_mapping_sha256` were persisted to
  `…/features/feature_member_order_check.json` BEFORE the bundle write.
- Old/new bundle compared bitwise: only `member_idx` changed (by the recorded
  permutation `[0,1,2,3,8,9,10,4,5,6,7]`) and `member_ids` was added; every other
  array and `emb_task_small.npz` are bitwise identical.
- Per-fold manifests (v3): `feature_mapping`, `script_sha256`, `runtime_attestation`.
- Controlled re-run evidence: CodeLlama folds are invariant (max|Δ| = 0, r = 1);
  the seven Qwen/DeepSeek folds were semantically redefined (previous run's fold
  labels mapped cyclically to wrong members).

## Key hashes

- members_order_sha256 = `1afd29b6…` (= admission map / local audit value)
- row_mapping_sha256 = `b38a41e4…` (`model_id|task_id|split|solution_sha256`, \n-joined, no trailing NL)
- bundle.npz = `d05f8c89…` (was `be5cc9d6…`), emb_task_small.npz = `f1e75299…` (unchanged)
- scripts: canonical `a3911260…`, cc_common `851b62e9…`, build_features `15304671…`

## Results (dev)

row-level mean **.9442**, task-macro **.9585**, pooled .9427 (CI row [.9346,.9531], tm [.9491,.9672]);
inner row .9491 / tm .9666.

Gate status: `prepared (canonical member order); cross-side score comparison pending`.
Test reading, generation, weight downloads, code execution: all disabled.

## Files

- `manifests/` 11 pre/post-fit fold manifests (v3) + `manifests_index.json`
- `metrics.json`, `score_digests.json` (with fused stats), `spec_sheet.json` (v3),
  `runtime_attestation.json`, `feature_order_fix_summary.json`
- `scripts/` frozen copies used for the run; `commands.sh`; `logs/`
- score npz live in git-ignored `local/` (intentionally not committed)
- `SHA256SUMS.txt` (excludes `local/`)
