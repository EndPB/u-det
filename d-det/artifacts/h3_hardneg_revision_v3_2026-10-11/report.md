# H3 v3 hard-negative CPU revision

## Decision

This is a data construction and diagnostic package, not GPU authorization.  `project_provenance=unavailable`; task-heldout is the strongest supported claim until repository/project evidence is added.

## Readout

| view | raw task-macro | masked task-macro | length task-macro |
|---|---:|---:|---:|
| clean-dev reference | 0.8813 | 0.6547 | 0.7248 |
| hard-negative v3 | 0.8778 | 0.6578 | 0.7109 |

The hard-negative view is selected per task and generator from the existing aligned rows; it does not create synthetic labels.  The clean-dev reference remains the primary untouched check.

