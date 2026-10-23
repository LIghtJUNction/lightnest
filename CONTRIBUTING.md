# 开发约定

## 环境与检查

使用 uv 管理 Python 3.13、项目虚拟环境和依赖。`pyproject.toml` 声明直接依赖，
`uv.lock` 固定完整依赖；Torch 使用官方 CPU 索引，与现有实验保持一致。
编译测试还需要 GCC、Clang、LLD 和 LLVM 工具；检查入口会在缺少这些工具时失败，避免静默跳过编译测试。
`archive/legacy/environment.json` 记录原实验环境，不代表当前机器的环境。

```bash
uv sync --locked
uv run --locked python tools/check.py
```

`tools/check.py` 统一执行源码树检查、Python 语法检查、新架构测试、归档回归和
template/cache/delta 测试。不需要外部权重，也不启动训练。
它不生成 Python 字节码；编译测试使用临时目录。

修改归档 Lean 规格或证明工具时，安装 `archive/legacy/formal/lean-toolchain` 指定的工具链，再执行：

```bash
uv run --locked python tools/check.py --lean
```

这会额外执行完整证明检查。普通 Python 测试通过不代表 Lean 证明已通过。
只运行某组测试时，使用对应复现文档里的命令；完整提交检查使用统一入口。
执行旧文档中的 Python 命令时，在命令前加 `uv run --locked`，即可使用项目环境。
修改依赖后执行 `uv lock`，将 `pyproject.toml` 和 `uv.lock` 一起提交。
`requirements.txt` 仅供旧工具兼容，由锁文件导出，不手工维护。依赖变更后同步更新：

```bash
uv export --locked --no-hashes --no-emit-project --emit-index-url --output-file requirements.txt
```

## 目录与修改范围

- `lightnest_native/`：当前多架构编解码、PyTorch 网络、本机数据、训练和受限执行。
- `tests/`：无需外部权重的新架构与证据边界检查。
- `tools/`：统一检查与源码树检查。
- `archive/legacy/`：历史架构、实验、测试、Lean 证明与文档；保留原内部相对结构。
- `archive/rpn3_rv32/`：本次历史任务预训练架构及其测试，用于对照和初始化。
- `docs/`：当前架构、实测结果和复现步骤，入口见 [文档导航](docs/README.md)。

新增 Python 代码使用四空格缩进、UTF-8 和 LF；编辑器设置见 `.editorconfig`。
保留既有导入入口，不为排版批量改写历史实验或来源快照。

## 实验资料与结论

权重、数据集、运行日志、原始结果、编译产物和缓存放在仓库外；不要用
`git add -f` 绕过忽略规则。外部资料说明见 [ARTIFACTS](archive/legacy/docs/ARTIFACTS.md)。
保留协议、源码来源和上游许可；来源未核实前不要自行添加项目许可证。

修改机制时运行对应行为测试。修改研究结论时提供冻结输入、预算、对照和验收证据；
区分单元测试、已有实验重放、新实验和模型晋升。历史报告保留当时的失败和限制，
新增结果另写文档，不覆盖旧记录或改写冻结哈希。
