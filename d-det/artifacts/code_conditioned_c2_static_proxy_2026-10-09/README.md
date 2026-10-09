# code_conditioned_c2_static_proxy_2026-10-09（C2 静态代理条件）

性质：static_proxy_conditioning（full 视图 + λv=0.1 静态代理辅助 MSE；无测试执行）。
C2 row 0.8846 / task-macro 0.8998；
ΔTM vs C0 = -0.0446（0/11 为正）；vs C1 full-mlp = -0.0001（辅助损失零效应）。
长度/style 残差化后 Δ(row/TM) = -0.0690/-0.0796。
代理审计：n_ast_nodes 与长度相关 0.70/0.71（不得重命名作代码约束）；parse_ok 零方差已排除。
