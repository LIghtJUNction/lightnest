# LightNest 实验记录索引

这些文档保留实验时的原始结论，包括**未通过的模型、错误早停、旧能力退步和负面结果**。文档不是当前可选源码版本；唯一最新实现见仓库 `hybrid/`、`baseline/LightNest_RGB_v2/` 和 `tools/template_io/`。

| 方向 | 文档 |
|---|---|
| 递归计算、模型容量与参数设计 | [递归](2026-10-17-recursive.md)、[容量](2026-10-17-capacity.md)、[架构](2026-10-17-architecture.md) |
| 依赖分析、Transformer 对照 | [依赖](2026-10-17-dependency.md)、[Transformer](2026-10-17-transformer.md) |
| 原始机器码、ELF 及泛化试验 | [ELF 效果](2026-10-18-elf-effect.md)、[扩量](2026-10-18-elf-scale.md)、[真实文件](2026-10-18-elf-real.md) |
| 增长、渐进式与突破实验 | [增长](2026-10-18-growth.md)、[渐进式](2026-10-18-progressive.md)、[突破](2026-10-18-breakthrough.md) |
| 困难样本与大规模训练 | [困难训练](2026-10-18-harder.md)、[数据标准化](2026-10-18-dataset.md)、[10k 扩量](2026-10-18-scale-10k.md) |
| 红蓝绿三模块 | [RGB v0](2026-10-18-rgb-v0.md)、[RGB v1](2026-10-18-rgb-v1.md)、[RGB v2](2026-10-19-rgb-v2.md) |
| I/O 复用与自适应推理 | [模板 I/O](2026-10-18-template-io-v1.md)、[Hybrid v1](2026-10-19-hybrid-v1.md) |

| 必要修改的训练反馈 | [反馈分配机制 v1（2026-10-20）](2026-10-20-required-edit-mechanism.md)：同预算对照，效果有限，未晋升 |

**比较原则：** 每份报告的分母、模型起点、程序类别与数据冻结时间可能不同，不能拼接百分比。只有前后在同一批样本及同一评分规则下的对照才可比较。实验用过的保留题今后必须当作公开回归。

有关当下采用决定和下一轮验证门槛，见 [LESSONS](../LESSONS.md) 和 [NEXT_ITERATION](../NEXT_ITERATION.md)。
