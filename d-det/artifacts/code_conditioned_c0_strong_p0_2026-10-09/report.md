# C0 强 P0 重建（server reconstruction；2026-10-09）

性质：复现/审计轮（非新方法结果）。dev 行级均值 = 0.9329；inner 行级均值 = 0.9301（本机参考 .9344/.9324）。

## dev

- row-level mean = **0.9329** CI95 [0.922783464398192, 0.9424106185186518]
- task-macro mean = 0.9442 CI95 [0.933830077086656, 0.9530402711323762]
- pooled = 0.9323；member-macro = 0.9329

| component | row-level mean |
|---|---|
| semantic | 0.8694 |
| char_tfidf | 0.8697 |
| word_tfidf | 0.8824 |
| style_meta | 0.8924 |

## inner

- row-level mean = **0.9301** CI95 [0.9198764814789788, 0.9394584039302268]
- task-macro mean = 0.9515 CI95 [0.9436063180770843, 0.9591213474025974]
- pooled = 0.9303；member-macro = 0.9301

| component | row-level mean |
|---|---|
| semantic | 0.8661 |
| char_tfidf | 0.8633 |
| word_tfidf | 0.8804 |
| style_meta | 0.8993 |

## 参考对照

- dev Δ vs .9344 = -0.0015；inner Δ vs .9324 = -0.0023；within_1e-3 = False
- 本机逐点参考表未随本轮传输；此处仅对 ACL §48 的 approx 值做报告性对照；冻结规格下的残差 Δdev=-0.0015/Δinner 待读；严格 1e-3 逐点审计需本机强 P0 参考分数。

## 规格探针（预声明，逐步定位）

| 变体 | dev row-level |
|---|---|
| S1_LR_insample_z | 0.9245 |
| S1_LR_oof_z | 0.9236 |
| semantic_base | 0.8649 |
| semantic_small+base | 0.8553 |
| wide_LR_C4 | 0.9303 |
| wide_LR_balanced | 0.9294 |
| wide_SGD3_frozen | 0.9329 |
- 冻结：wide_SGD3（char 2-5 min_df=1 / word 1-3 min_df=1 / SGD×3 seed 集成）
- 行映射修复：emb row mapping used local member index before; fixed to global 115-model index; texts.jsonl.gz check 10659/10659

> 全部拟合 fit-only；dev 仅开发评测；无 test/生成/权重下载。
