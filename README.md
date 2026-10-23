# LightNest

**以多种 CPU 架构的机器码理解、修改和生成作为长期目标，先在本机 Linux x86-64 上训练。**
当前主线是 PyTorch 实现的 Universal Machine v1：处理变长指令、架构标识、请求、
寄存器依赖和可见输入/输出要求，预测整段代码的修改集合。

当前已接入 iced-x86 与 LLVM 多目标编解码、ELF 函数提取、本机编译训练语料和受限
执行器。**解码支持、实际执行支持与经过训练的能力分别记录**；其他平台尚未训练，
不能据此宣称已学会所有 CPU 指令或成为最强模型。实际结果见 [训练记录](docs/RESULTS.md)。

## 使用

```bash
uv sync --locked
uv run --locked python tools/check.py
uv run --locked python -m lightnest_native targets
uv run --locked python -m lightnest_native decode --architecture x86_64 --hex 4889f84801f0c3
uv run --locked python -m lightnest_native inspect-elf /path/to/program.o
```

汇编器从标准输入读取汇编，通过 LLVM 输出机器码；有未解决重定位时拒绝返回
独立代码块。对应入口为 `uv run --locked python -m lightnest_native assemble --architecture aarch64`。

修改代码时，用验证式解码入口指定真实检查点和仓库外的新工作目录：

```bash
uv run --locked python -m lightnest_native repair \
  --checkpoint /path/to/universal-104729.pt --workdir /tmp/lightnest-repair-new < requests.jsonl
```

每行含 `architecture`、`command`、十六进制 `binary` 和 `examples`；可见样例包含
三个 uint64 输入及一个 uint64 输出。`repair` 在受限子进程核对可见样例，按神经
不确定性做有预算的候选搜索；通过仍不是正确性证明，不覆盖文件。
`predict` 保留纯神经、完全不执行候选的诊断入口。当前训练检查点只接受 x86-64，
跨平台请求会明确拒绝。

## 目录

```text
lightnest_native/   当前多架构接口、PyTorch 模型、x86-64 语料和受限执行器
tests/             当前模型、指令后端、ELF 与执行边界测试
tools/             统一检查和源码树检查
docs/              当前架构、复现、能力边界及实测结果
archive/legacy/    原 C64、RGB、Hybrid 和其他历史实验，保留测试与证明
archive/rpn3_rv32/  本次先完成的 RV32 新架构，对照保留
```

训练数据、检查点、优化器状态和逐条结果均保存在仓库外。源代码检查使用 Python 3.13、
PyTorch 2.10 CPU 和锁定依赖；需要 GCC、Clang、LLD、LLVM 工具。本机执行器使用 Linux
seccomp、内存/时间限制和只读可执行代码页。

[架构](docs/ARCHITECTURE.md) · [复现](docs/REPRODUCE.md) · [结果](docs/RESULTS.md)
· [开发约定](CONTRIBUTING.md) · [归档](archive/README.md) · [来源与版权](NOTICE.md)
