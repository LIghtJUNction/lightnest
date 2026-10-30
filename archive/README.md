# 历史架构归档

`legacy/` 保留 2026-10-20 及以前的 C64、RGB、Hybrid、结构学习、编译修复、
局部修复记忆、必要修改损失、模板 I/O、测试、Lean 证明和研究文档。
移动时的逐文件 SHA256 见 `legacy/ARCHIVE_MANIFEST.json`；这是移动前工作树的内容，
包含本次会话此前已经完成的 uv 和文档整理。

这些文件用于复现和比较，不再作为当前架构入口。当前 `lightnest_native/` 预测不导入
历史神经网络；历史数据转换和预训练初始化单独记录。

`rpn3_rv32/` 保留本次先开发的 RV32 新架构、测试和说明。用户随后将主线扩大到
多 CPU 架构、优先本机 x86-64，因此它只作为历史 TRAIN 预训练和对照。
已完成一个 2,048 更新终点，公开验证 240/240；未完成两个种子及原计划最终对照，
不能把它宣布为通过完整晋升。原始训练和权重仍在仓库外保留。

从项目根目录运行：

```bash
uv run --locked python -m unittest discover -s archive/legacy/tests -t archive/legacy -v
uv run --locked python -m unittest discover -s archive/legacy/tools/template_io -p 'test_*.py' -v
uv run --locked python archive/legacy/tools/check_lean.py
```

原实验命令和相对路径见 [历史 README](legacy/README.md) 与 [复现文档](legacy/docs/README.md)。
运行原来的 `hybrid`、`experiments` 等模块时先 `cd archive/legacy`，再以 `uv run --locked python ...`
执行；uv 会向上找到根目录项目。源码树检查统一使用根目录 `tools/check_source_tree.py`。
数据与权重仍放在仓库外，不因移动源码而迁移或删除外部档案。

## Current-line research modules

The current model package is `lightnest/`: RGB policy and complete-ELF generation,
with the optional H-Net dynamic-chunking decoder. Standalone studies that are not
part of that architecture live in `research/`: finite behavior planning, DFA
learning, exact x86 execution primitives, and alternative edit policies. Their
tests and experiment drivers remain under the root `tests/` and `tools/` folders
so the recorded studies stay runnable; imports use `archive.research.*`.

## Universal Machine v1

`universal_native/` 保存原主线模型、数据与执行工具。当前 RGB 主线位于
`lightnest/`，多架构编解码接口保留在 `lightnest/isa.py`。
旧实验记录中的文件路径与源码哈希属于原始提交，不重新标记为 RGB 结果。
