# H3 v3 hard-negative hypothesis

STACAD alignment exposes task/file identity but no repository or project identity.  This CPU pass therefore uses a conservative same-task hard-negative view: one human and three distinct-generator AI rows selected by a masked-token similarity plus log-length score.  It does not call task identity project provenance.

The pre-registered readout is the masked and raw lexical probe on the hard-negative train/dev pair view, together with the untouched clean-dev reference.  A future GPU manifest is allowed only if project provenance is supplied or the claim is explicitly limited to task-heldout detection, the clean and hard-negative views are frozen, and the main model has a pre-registered gain over the corresponding lexical control.

This pass is CPU-only; test rows remain unopened and the closed C0-C3 and H3 diagnostic batches are not rerun.
