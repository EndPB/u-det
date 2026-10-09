# Fresh C0 specification adjudication (2026-10-10)

## Verdict

`spec_violation_global_lexical_fit`; C0 is blocked before the alignment gate. The reported fresh C0 is reproducible relative to the previous server C0, but it is not a valid execution of the declared fold-fit-only lexical protocol.

## Evidence

`spec_sheet.json` declares `fold-fit rows texts only (fold-train-only vocabulary & IDF)`. In `cc_fresh_c0.py`, the vectorizers are fit before the fold loop at lines [225, 228]. The operation is `fit_transform(texts)` over all 10,659 train/dev rows. Later folds only slice the already-fitted matrices, so dev and heldout-member text influence the vocabulary and IDF. The same applies to the word vectorizer.

The semantic and style/meta scalers and classifiers are fit inside each fold; this finding is limited to lexical TF-IDF construction. The 11 manifests are structurally complete, and the self-check against the previous C0 is useful reproducibility evidence, but neither repairs the leakage.

## Required action

Run a corrected fresh C0 with both vectorizers fitted inside each fold using only `fit_rows`, then transform `ev_rows`. Add the fit-row hash and vocabulary/IDF specification to every manifest. Recompute C0 metrics and digests. Keep C1–C3 stopped until the corrected C0 passes both protocol audit and the `1e-3` cross-side score/metric gate.
