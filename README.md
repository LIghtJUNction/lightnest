# LightNest

RGB 是当前唯一训练主线：用 PyTorch 学习理解、编辑和生成完整二进制文件，先做本机 Linux x86-64。

参考北邮 [MiTA](https://arxiv.org/html/2602.01219v6) 对注意力与输入相关前馈网络的解释：
绿区压缩全局信息并保留可路由的原始细节；蓝区快速提议；红区共享循环状态，按需增加思考深度。
红区在同一文件上复用证据库，接受编辑后重新观测。

目前真实语料包括 16 个安装在本机的命令行程序、2,048 个完整 ELF 编辑任务和 864 个
完整 ELF 生成样本。故障是人工注入的；生成课程目前限定为指定退出码或输出 8 字节。
这不等于任意程序理解、任意代码生成或已训练所有 CPU 指令。

```bash
uv sync --locked
uv run --locked python tools/check.py
uv run --locked python -m lightnest_rgb --help
uv run --locked python -m lightnest_rgb inspect /usr/bin/sort
uv run --locked python -m lightnest_rgb targets
uv run --locked python -m lightnest_rgb decode --architecture x86_64 --hex 4889f84801f0c3
```

多架构编解码接口沿用 iced-x86 / LLVM；实际训练和执行目前只有 x86-64。
完整 ELF 解析给出入口、段、节、依赖与导入符号；这些确定性解析结果与模型学到的能力分别记录。
Any2Any 主线的总输入容量为 1 MiB，局部注意力窗口为 4,096 token；神经笔记槽负责压缩长文件上下文，
不会把容量提升误报成已经完成任意文件语义理解。
新增的[外部代码与说明对齐实验](docs/SEMANTIC_ALIGNMENT.md)仍低于词汇匹配基线，
尚未通过语义验收。

生成器直接输出完整文件的所有字节；不会拼接文件头、套用模板或调用编译器代替模型生成。
编辑器只接收原文件、请求和可见样例；候选在隔离环境中执行，测试通过也不是正确性证明。
输出使用新路径，原文件保持不变。训练和使用步骤见 [RGB 复现](docs/RGB_REPRODUCE.md)；
单文件自解释 ELF 见 [Self-hosting ELF](docs/SELF_HOSTING.md)。

```text
lightnest_rgb/             RGB 主线、完整 ELF 工具、多架构编解码
archive/universal_native/  原 Universal Machine v1 模型与工具
archive/rpn3_rv32/         早期 RV32 架构
archive/legacy/            其他历史实验与证明
 tests/                   主线和归档行为测试
 tools/                   本地统一检查
 docs/                    设计、实验结果和复现步骤
```

Python 3.13 / PyTorch 2.12.1 CPU 的依赖由 uv 锁定。测试需要 GCC、Clang、LLD、LLVM、
bubblewrap 和 libseccomp；真实语料构建还需要对应命令行程序。权重、原始可执行文件、
缓存、训练日志和逐条结果全部在仓库外。

[RGB 设计](docs/RGB_DESIGN.md) · [历史结果](docs/RESULTS.md)
· [开发约定](CONTRIBUTING.md) · [归档](archive/README.md) · [来源](NOTICE.md)
