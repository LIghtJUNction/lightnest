# LightNest

当前开发架构是 **RPN3**：用 PyTorch 编码受限 RV32 指令、请求和依赖关系，
一次预测所有必要修改位置及替换字节。旧 C64、RGB、Hybrid、编译修复和其他实验
已完整移入 [`archive/legacy/`](archive/README.md)。

这是合成程序的受限修改研究。推断不读取答案、不执行宿主机候选，也不表示已能
修改真实大型软件。新模型是否超过旧模型，以逐条对照结果为准；见 [实验结果](docs/RPN3_RESULTS.md)。

## 使用 uv

```bash
uv sync --locked
uv run --locked python tools/check.py
uv run --locked python -m lightnest_next --help
```

源码检查需要 GCC、Clang 和 LLD。训练和推断使用锁定的 Python 3.13、PyTorch 2.10 CPU
和 NumPy 2.3.5。完整检查包括新架构、归档回归和模板工具测试。

## 训练与推断

原始数据、父模型和训练产物保存在仓库外，具体步骤见 [复现说明](docs/REPRODUCE.md)。
推断必须提供已训练检查点，不自动用随机权重代替。

```bash
uv run --locked python -m lightnest_next predict --checkpoint /path/to/rpn3-104729.pt
```

从标准输入读取 JSONL，每行只有 `command` 和十六进制 `binary`。
返回候选补丁、完整输出字节、STOP 状态和未验证标记；不会覆盖输入文件。
仅接受历史训练所覆盖的 12 种 ADD/SUB 请求句式，最多 1,024 字节和 8 处修改。

## 目录

```text
lightnest_next/   当前 PyTorch 架构、数据准备、训练和评价
tests/           新架构与证据边界测试
tools/           统一检查、源码树检查
docs/            当前架构、复现与实测结果
archive/legacy/  历史源码、测试、证明、协议和研究记录
```

[架构](docs/ARCHITECTURE.md) · [开发约定](CONTRIBUTING.md) · [历史实验](archive/legacy/docs/experiments/README.md)
· [外部档案](archive/legacy/docs/ARTIFACTS.md) · [来源](archive/legacy/docs/SOURCE_PROVENANCE.md) · [版权说明](NOTICE.md)

归档的 Lean 定理只覆盖原实验中明确写出的不变量，不证明新网络的预测正确性。
运行归档证明检查：`uv run --locked python archive/legacy/tools/check_lean.py`。
