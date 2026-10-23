# LightNest

受限 RV32 程序修改研究项目：读取请求及机器码，分析依赖，生成候选修改，并用有界解释器测试结果。**不是通用语言模型，也没有证明能安全修改真实大型软件。**

## 当前版本与验收状态

- **最新机制探索：Repair memory v1（2026-10-20）**。不改神经架构，保留最多8个有可见进展的局部候选。2,448个任务中，相对同样允许双修改的随机搜索净增90个修复，候选尝试减少45.12%；但错误放行从41增至45，不默认启用。见 [机制报告](docs/experiments/2026-10-20-repair-memory-v1.md) 与 [复现](docs/reproduce/repair-memory-v1.md)。
- **Compiled v3** 保留16类编译C程序与真实zlib补充实验、执行路径筛选及反例。见 [报告](docs/experiments/2026-10-20-compiled-v3.md)。
- **当前稳定默认路径：C64（固定64轮）**。本轮无神经训练，不修改C64、Hybrid或原权重，不默认启用候选修复工具。
- **Research v2** 保留结构规则、学习对照及反例；**Hybrid v1 / RGB v2** 保留现有神经源码；**template_io v1** 提供精确缓存、结构缓存与差量工具。

编译修复任务额外接收可见的输入/输出测试要求。它与旧版仅接收请求文本和机器码的结构变换任务不同，不可直接比较准确率。真实函数的错误也是人工注入，不是生产缺陷修复成绩。

## 仓库范围

只提交最新源码、必要协议、测试和研究文档。不提交模型权重、数据集、原始结果、生成的ELF/机器码、共享库或缓存。完整原始档案留在外部存储。

```text
experiments/repair_memory_v1/     有证据的局部进展保留、五种对照与评价
experiments/compiled_v3/          C生成、RV32IM解释器、候选搜索、评价与复算
experiments/compiled_external_v3/ zlib函数提取、上游许可与独立补充实验
experiments/research_v2/          既有结构/学习对照与反例
hybrid/                          状态桥和模式切换研究
baseline/LightNest_RGB_v2/       当前RGB/C64依赖源码
formal/                          Lean规格、证明与公理审计，不含构建产物
tools/template_io/               缓存与差量工具
tools/replay_compiled_v3.py       最新源码重放既有正式决策
tests/                           不依赖外部权重的源码检查
docs/                            实验报告与复现步骤
```

`baseline/.../work/.../arch/breakthrough_model.py` 是当前加载需要的固定架构定义，不是旧模型权重。

## 本地检查

Python 3.13。旧源码测试需要原依赖；编译与记忆机制实验本身只需标准库和GCC、Clang、LLD。自动检查会安装编译工具，避免编译测试因缺工具而跳过。

```bash
uv sync --locked
uv run --locked python tools/check.py
```

统一入口会检查源码树、Python 语法和两组测试；缺少 GCC、Clang 或 LLD 会失败。
环境准备和目录约定见 [开发说明](CONTRIBUTING.md)，全部文档见 [文档导航](docs/README.md)。

所有输出目录必须在仓库外。模型推断和旧模型回归仍需 [外部权重与数据](docs/ARTIFACTS.md)，不能使用随机初始化冒充已训练模型。

## Lean形式化辅助验证

安装官方Elan并使 `lake` 可用后，在根目录执行：

```bash
uv run --locked python tools/check_lean.py
```

固定Lean 4.35.0，使用附带标准库，无需模型权重、GPU或Mathlib。完整检查在临时目录重建证明、审计公理、重新调用内核、对比Python实现，并确认故意错误的证明被拒绝；缺少工具会失败，不伪装为通过。每次push和PR都会运行独立的 `Lean proofs` CI。

证明覆盖补丁不变量、共享循环分段、预算、有限验收记录保持及局部候选提交规则，不证明神经准确率、修复语义或整个Python/C实现正确。用法与未覆盖范围见 [Lean验证说明](docs/FORMAL_VERIFICATION.md)。

## 阅读顺序

[最新机制报告](docs/experiments/2026-10-20-repair-memory-v1.md) · [Compiled v3](docs/experiments/2026-10-20-compiled-v3.md) · [Research v2](docs/experiments/2026-10-20-research-v2.md) · [探索经验](docs/LESSONS.md) · [架构边界](docs/ARCHITECTURE.md) · [实验记录](docs/experiments/) · [来源说明](docs/SOURCE_PROVENANCE.md) · [版权说明](NOTICE.md)

zlib提取代码的独立许可见 [LICENSE](experiments/compiled_external_v3/LICENSE.md)。通过少量样例不等于正确性证明。候选只在有界解释器执行，不覆盖输入文件，不运行宿主机候选。
