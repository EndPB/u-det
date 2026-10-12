# d-det-v2（平行新起点）

创建：2026-10-12。与 `u-det/d-det/` **平行**的新主线文件夹；只含基础目录结构，不含数据。

## 结构

- `docx/`：独立总账 **`d-det-v2_总账与主线定义_2026-10-12.md`**（本线唯一入口，先读它）
- `scripts/`、`artifacts/`、`configs/`、`runs/`：实验脚本与产物（自动创建）
- `data/`：**不使用独立数据**；全部复用原 d-det 数据（见 `data/README.md`）

## 线级定位（一句话）

在"旗舰模型 base 一致"的强假设下，用**代码条件的后训练差异**表示，在既有全量数据上快速验证 **A（二分类）与 B（家族归因）** 相对同预算 CodeT5 基线的增量。**只做 A/B**；C 暂缓。

## 路径约定

- 数据：`u-det/d-det/data/`（相对本目录 `../d-det/data/`）
- 模型/权重：`u-det/d-det/models/`（相对 `../d-det/models/`）
- 历史总账与旧实验（只读）：`u-det/d-det/docx/`、`u-det/d-det/artifacts/`
