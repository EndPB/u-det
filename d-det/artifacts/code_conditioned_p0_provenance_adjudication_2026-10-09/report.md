# C0 P0 provenance adjudication (2026-10-09, corrected)

## Decision

`edf3813` correctly establishes that the continuous C0 scores are not point-aligned with the server C0. Its inferred 7-cycle, however, is obtained by matching character-score blocks to server blocks. It is not, by itself, proof that the historical local training actually used a different series map.

## What the local record proves

- The registered command passes the current `family_series_admission.json` to `attribution_family_member_ho.py`.
- `attribution_family_metric.py` derives each heldout series from that supplied map and excludes same-series seen members.
- `audit_public_full_p0_local.py` reconstructs the evaluation rows with exactly that map.
- Direct reconstruction from the current admission reproduces all 11 stored `p0_scores.npz` `y` vectors and task arrays exactly, including the transition positions and row counts.

Thus the current map is supported by the command, source code, labels, and task order. The old output still lacks a per-fold effective-map hash and a complete component specification, so exact numerical alignment is unresolved. The 7-cycle should be recorded as a **score-block alignment hypothesis**, not as settled historical provenance.

## Consequence

C0 remains `not_aligned`. The historical local P0 is map-consistent but not an exact reference because semantic/lexical fitting specifications and per-fold manifests are incomplete. Server C0 and C1–C3 remain server-internal development diagnostics; they must not be presented as a confirmed comparison to the historical local P0.

Do not rename directories, rewrite labels, or infer the historical map again from continuous scores. The finite repair is a fresh train/dev C0 whose per-fold manifest records the effective map, fit/eval member IDs, row/task hashes, component specs, and code commit before fitting. Only an aligned fresh C0 permits C1–C3 reruns. Test, generation, weights, and code execution remain disabled.
