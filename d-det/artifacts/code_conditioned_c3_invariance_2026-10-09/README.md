# code_conditioned_c3_invariance_2026-10-09（C3 增强一致性训练）

性质：语义安全增强一致性（comment .2 / whitespace .3 / rename .4；λinv=0.1；train/dev only）。
C3 row 0.8748 / task-macro 0.8972；ΔTM vs C0 = -0.0473（0/11 为正）⇒ 闸门失败。
安全审计：接受样本 AST 破坏 0 / 接口变化 0；拒绝=安全跳过或空操作（见 transform_acceptance.jsonl）。
变换后归因保持：rename 符号一致 95.4%、comment 91.6%、whitespace 86.4%（详见 metrics.transformed_eval）。
