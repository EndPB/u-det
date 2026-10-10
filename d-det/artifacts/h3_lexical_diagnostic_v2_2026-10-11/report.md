# H3 v2 词法捷径分解

四种 CPU 词法视图均在相同 train/dev task 口径下拟合；这不是 GPU 主实验。

| view | clean-dev task-macro | controlled-dev task-macro |
|---|---:|---:|
| raw | 0.8969 | 0.8500 |
| no_comments | 0.8853 | 0.8543 |
| normalized_ws | 0.8902 | 0.8374 |
| masked_tokens | 0.8471 | 0.7997 |

若 masked_tokens 仍高，捷径主要来自标识符/字面量/结构组合；下一轮应建立同题、同语言、同长度且 project-heldout 的 hard-negative 配对，而不是继续堆叠模型。
