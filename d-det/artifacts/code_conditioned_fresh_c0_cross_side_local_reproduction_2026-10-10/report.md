# Local reproduction and cross-side C0 adjudication

## Verdict

The local reproduction completed on all 10,659 train/dev rows with no test access. It does not close the `1e-3` gate. The blocking reason is structural: AutoDL's feature manifest orders members as CodeLlama, DeepSeek, Qwen, while the admission map and `cc_common.fold_setup()` interpret member indices as CodeLlama, Qwen, DeepSeek. CodeLlama positions match; the seven Qwen/DeepSeek positions are permuted.

This means the server bundle's `member_idx` can be interpreted under a different member order from the fold labels. It explains why the local and server lexical attestations match on the four CodeLlama folds but are cyclically shifted on the seven Qwen/DeepSeek folds. The corrected lexical fit itself is present; the bundle-to-label mapping remains unresolved.

## Local run

The local run rebuilt the 92-dimensional stylometry block, 10-dimensional metadata block, three size/length columns, and 512-dimensional CodeT5-small code representation from existing train/dev caches. It completed in 370 seconds under Python 3.11.5, NumPy 1.26.4, and scikit-learn 1.7.1. Its dev row mean was `.943931` and task-macro mean `.958058`; these numbers are diagnostic only because the server used Python 3.12.14, NumPy 2.2.6, scikit-learn 1.9.1 and, more importantly, a differently ordered feature bundle.

Cross-side checks: fit-text hashes equal on 0/11 folds, lexical vocabulary pairs equal on 4/11 folds (the CodeLlama folds), and fused score digests equal on 0/11 folds.

## Required correction

Before any C1-C3 decision, rebuild the feature bundle with a canonical `members_order` identical to the admission map and record its hash in the feature manifest and C0 manifests. AutoDL then needs one corrected train/dev C0 run. Test reading, generation, weight downloads, and code execution remain disabled.
