# H3 本机数据修订 v1：跑前登记

日期：2026-10-10；登记先于本轮 probes。旧报告与旧结果不修改。

## 固定问题和比较

1. 七语言 parser 的实际覆盖率、跨任务/跨 split 的 syntax-tree 与骨架重复是否明确？先移除旧报告两组重复涉及的全部四个 task，再对新增跨 split 重复以整 task 排除 train 侧，保留 dev 侧；同 split 跨任务骨架仅标记，不因骨架同形自动判断语义重复。
2. 长度控制是否缓解旧 length-only .8459？同一训练集分别用完整七 AI 和按九维长度距离选择三 AI 的训练视图拟合；主评估均为同一 untouched clean-dev。额外报告 matched-dev（三 AI）敏感性分析，禁止替代主评估。
3. comment=.2、whitespace=.3、rename=.4 对所有语言的 train 行独立抽样；接受必须无 parser ERROR/MISSING，保留 token/syntax-tree（改名按明确映射对齐），Python 还需 stdlib AST 对齐。通用改名只接受函数内单独局部声明、全局无同名/成员/字符串/动态引用等保守子集；无法确认即拒绝。解析器和语法树一致仅为 safety proxy，不声称运行正确性。

## 固定实现与口径

- seed=20261010，bootstrap=500，CPU-only 本机；不运行数据代码、不下载模型权重、不生成模型输出。
- test 行在 JSON 反序列化之前按容器 split 标记跳过；不提取、解析、去重、增强或建模 test 代码。只保留已有文件 SHA，test 不重切分。
- 长度特征沿用旧九维：字符/行/非空行/空行/最大行/平均行/空白分词/空格/制表。log1p 后尺度仅从 train 拟合，按 task 内到 human 的距离取最邻近三 AI，不调整 dev 阈值。
- length LR(C=1,max_iter=1000)、metadata LR、AST histogram LR、char+word TF-IDF + LinearSVC(C=1)；词表仅 train 拟合。原始及长度匹配训练视图各自独立拟合；全部结果均报告，不以 dev 选择配置。
- primary=clean-dev；matched-dev=sensitivity（基于特征匹配的选择估计量，具有选择偏差）。旧 .9261 仅作历史参照；主模型必须与修订数据上重新拟合且同 eval 行的 lexical control 配对比较。
- task bootstrap；源 metadata 只代表可观测文件/语言字段，未提供 project provenance，不能把 .5 写成项目泄漏已排除。

## 标签支持边界

STACAD 为七个 observed generator/source，每个 source 只有一个 generator，没有同家族多 generator 正对。source head 只能报告 task-heldout 的已见 source 分类；generator-heldout 用于 Human/AI 检测，留出 generator 的 source 标签不进入闭集分类分母。H2 family transfer=unsupported，不以目录名补齐。

## 预定出口

- integrity 或 accepted syntax/tree 不一致：stop。
- 主评估任一固定 length 模型 task-macro >=.70：该模型 shortcut risk high，整轮保留 revise_data；匹配 dev 通过不能替代主评估。
- transform audit 未全覆盖、scope 安全证据不足、project provenance 缺失：明确列出限制，不声称 H3 全闸门通过。
- 数据 revision 完成不等于 H3 方法验证；数据侧全部必要条件通过后才写 executable GPU batch，方法增量只在 GPU 运行后判断，不能把未来增量作为开始训练的先决条件。
