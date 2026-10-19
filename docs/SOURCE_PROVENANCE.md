# 源码选择与来源

本次同步时间：2026-10-20。主版本为 2026-10-19 的 Hybrid v1；当前稳定入口仍为 C64。没有混入早期模型权重或原始实验数据。

- 原始 Hybrid ZIP：`234ee0f38c546207195a3fb1ac968d45592fb35b51f84d1777d90c2c4051f079`（SHA256）。包含 `hybrid/*.py` 与必需的 `baseline/LightNest_RGB_v2/` 当前源码和固定 C64 架构定义。本次只筛选源码、必要协议/元数据与原始 Markdown 报告；不重新实现原网络。
- 原始 Template I/O ZIP：`920717a92c7aa0cc8dba5c3b8beb26f59b9a6c2e24078f869e708a782453ccfa`（SHA256）。将 `src/*.py` 映射到 `tools/template_io/*.py`，保留 `PROTOCOL.json`，其余大结果不加入。
- `docs/experiments/` 是原实验 Markdown 的只读记录。README/LESSONS/NEXT_ITERATION 是按报告结果撰写的仓库说明，不代表重新测得实验指标。

原文及交接报告声明来源许可尚未核实；保留 [NOTICE](../NOTICE.md)，不新加无法确认的许可证。请在继续公开分发、合并外部代码前核验相关许可。

不在 Git 中放置 `.pt`、原始 `.jsonl`、二进制文件、旧归档、临时结果或模型缓存。权重/数据所在云盘参见 [ARTIFACTS](ARTIFACTS.md)。
