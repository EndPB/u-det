# code_conditioned_c1_prompt_conditioned_2026-10-09（C1 题面条件化代码归因）

性质：机制诊断（5 视图 × linear/MLP，对照 C0；train/dev only）。
最佳视图 code_only-MLP：row 0.8921 / task-macro 0.9042；
full-MLP：row 0.8847 / task-macro 0.8998。
全部视图 ΔTM(vs C0) 为负（−4.0 ~ −36.5pt），0–1/11 折为正 ⇒ 闸门失败（非接近）。
full−code_only（task-macro，MLP）=-0.0044 ⇒ 题面条件未改善任务内排序。
