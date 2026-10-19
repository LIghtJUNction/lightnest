# LightNest

受限 RV32 程序修改研究项目：读取请求及机器码，分析依赖，生成候选修改，并用有界解释器测试结果。**不是通用语言模型，也没有证明能安全修改真实大型软件。**

## 当前版本与验收状态

- **最新探索：Compiled v3（2026-10-20）**。实际完成 16 类编译 C 程序的 3,264 个正式任务和真实 zlib 函数的 579 个补充任务。执行路径筛选在相同检查条件下减少候选尝试 31.09%，但有限测试仍放行错误候选。见 [完整报告](docs/experiments/2026-10-20-compiled-v3.md)。
- **当前稳定默认路径：C64（固定 64 轮）**。本轮无神经训练，不修改 C64、Hybrid 或原权重，不默认启用候选修复工具。
- **Research v2** 保留结构规则、学习对照及反例；**Hybrid v1 / RGB v2** 保留现有神经源码；**template_io v1** 提供精确缓存、结构缓存与差量工具。

本轮编译修复任务额外接收可见的输入/输出测试要求。它与旧版仅接收请求文本和机器码的结构变换任务不同，不可直接比较准确率。真实函数的错误也是人工注入，不是生产缺陷修复成绩。

## 仓库范围

只提交最新源码、必要协议、测试和研究文档。不提交模型权重、数据集、原始结果、生成的 ELF/机器码、共享库或缓存。完整原始档案留在外部存储。

```text
experiments/compiled_v3/           C 生成、RV32IM 解释器、候选搜索、评价与复算
experiments/compiled_external_v3/  zlib 函数提取、上游许可与独立补充实验
experiments/research_v2/           既有结构/学习对照与反例
hybrid/                           状态桥和模式切换研究
baseline/LightNest_RGB_v2/        当前 RGB/C64 依赖源码
formal/                           Lean规格、证明与公理审计，不含构建产物
tools/template_io/                缓存与差量工具
tools/replay_compiled_v3.py        最新源码重放既有正式决策
tests/                            不依赖外部权重的源码检查
docs/                             实验报告与复现步骤
```

`baseline/.../work/.../arch/breakthrough_model.py` 是当前加载需要的固定架构定义，不是旧模型权重。

## 本地检查

Python 3.13。旧源码测试需要原依赖；新的编译实验本身只需标准库和 GCC、Clang、LLD。自动检查会安装编译工具，避免编译测试因缺工具而跳过。

```bash
python -m pip install -r requirements.txt
python tools/check_source_tree.py
python -m unittest discover -s tests -t . -v
PYTHONPATH=tools/template_io python -m unittest discover -s tools/template_io -p 'test_*.py' -v
```

完整新实验命令见 [Compiled v3 复现](docs/reproduce/compiled-v3.md)。所有输出目录必须在仓库外。模型推断和旧模型回归仍需 [外部权重与数据](docs/ARTIFACTS.md)，不能使用随机初始化冒充已训练模型。

## Lean 形式化辅助验证

安装官方 Elan 并使 `lake` 可用后，在根目录执行：

```bash
python tools/check_lean.py
```

固定 Lean 4.35.0，使用附带标准库，无需模型权重、GPU或Mathlib。完整检查在临时目录重建证明、审计公理、重新调用内核、对比Python实现，并确认故意错误的证明被拒绝；缺少工具会失败，不伪装为通过。每次push和PR都会运行独立的 `Lean proofs` CI。

证明覆盖补丁不变量、共享循环分段、预算和有限验收记录保持，不证明神经模型准确率或整个Python/PyTorch实现正确。用法、证据比较入口及未覆盖范围见 [Lean验证说明](docs/FORMAL_VERIFICATION.md)。

## 阅读顺序

[最新报告与下一步](docs/experiments/2026-10-20-compiled-v3.md) · [Research v2](docs/experiments/2026-10-20-research-v2.md) · [探索经验](docs/LESSONS.md) · [架构边界](docs/ARCHITECTURE.md) · [实验记录](docs/experiments/) · [来源说明](docs/SOURCE_PROVENANCE.md) · [版权说明](NOTICE.md)

zlib 提取代码的独立许可见 [LICENSE](experiments/compiled_external_v3/LICENSE.md)。通过少量样例不等于正确性证明。候选只在有界解释器执行，不覆盖输入文件，不运行宿主机候选。
