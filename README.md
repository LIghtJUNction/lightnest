# LightNest

受限 RV32 程序修改研究项目：读取请求及机器码，分析依赖，生成逐字节修改动作，并用有界解释器测试结果。**不是通用语言模型，也没有证明能安全修改真实大型软件。**

## 当前版本与验收状态

- **最新探索：Research v2（2026-10-20）**。比较四条候选路径，真实训练四个判断头；受限结构任务由反向依赖规则胜出，神经候选未通过旧能力保持验收。它不是神经能力突破，也不默认启用。见 [实验结果](docs/experiments/2026-10-20-research-v2.md)。
- **当前稳定默认路径：C64（固定 64 轮）**。Hybrid、RGB 蓝区路由、扩量候选及本轮学习候选均不替换默认模型。
- **Hybrid v1** 保留 StateLink、监督 THINK/EMIT 切换与策略梯度源码；**RGB v2** 提供基础组件和可选执行后端；**template_io v1** 提供精确缓存、结构缓存和差量工具。
- 不默认加载或运行研究模型；不提交模型权重。

## 仓库范围

**只提交最新源码、必要的协议配置、测试和研究文档。** 模型检查点、训练与评估数据、原始结果、二进制候选、旧版源码压缩包和缓存留在 Google 云盘，不进 Git。

```text
experiments/research_v2/        当前多路径实验、独立复算、计时和反例
hybrid/                         当前 StateLink + 模式策略研究源码
baseline/LightNest_RGB_v2/      Hybrid 依赖的 RGB/C64 源码（不含权重、样本）
formal/                         Lean规格、证明与公理审计，不含构建产物
tools/template_io/              模板、精确缓存、静态关系缓存、差量工具
tests/                          无模型权重的源码检查
docs/                           探索经验、实验报告、运行说明、下一轮计划
```

`baseline/LightNest_RGB_v2/runtime/base/lightnest/work/.../arch/breakthrough_model.py` 是当前 C64 加载必需的固定架构定义，不是旧版候选权重。

## 检查源码（不需要下载模型）

推荐 Python 3.13；测试环境使用 PyTorch 2.10.0+cpu。

```bash
python -m pip install -r requirements.txt
python tools/check_source_tree.py
python -m unittest discover -s tests -t . -v
PYTHONPATH=tools/template_io python -m unittest discover -s tools/template_io -p 'test_*.py' -v
```

模型推断、训练和历史模型回归另需外部权重及数据，不可用随机初始化代替。Research v2 的训练还需要 NumPy 2.3.5，完整步骤见 [复现说明](experiments/research_v2/REPRODUCE.md)。原始档案见 [ARTIFACTS](docs/ARTIFACTS.md)。

## Lean 形式化辅助验证

安装官方 Elan 并使 `lake` 可用后，在根目录执行：

```bash
python tools/check_lean.py
```

固定 Lean 4.35.0，使用附带标准库，无需模型权重、GPU或Mathlib。完整检查在临时目录重建证明、审计公理、重新调用内核、对比Python实现，并确认故意错误的证明被拒绝；缺少工具会失败，不伪装为通过。每次push和PR都会运行独立的 `Lean proofs` CI。

证明覆盖补丁不变量、共享循环分段、预算和有限验收记录保持，不证明神经模型准确率或整个Python/PyTorch实现正确。用法、证据比较入口及未覆盖范围见 [Lean验证说明](docs/FORMAL_VERIFICATION.md)。

## 从哪里阅读

[本轮结果](docs/experiments/2026-10-20-research-v2.md) · [探索经验](docs/LESSONS.md) · [架构边界](docs/ARCHITECTURE.md) · [下一轮方案](docs/NEXT_ITERATION.md) · [实验记录](docs/experiments/) · [来源说明](docs/SOURCE_PROVENANCE.md) · [版权说明](NOTICE.md)

不同数据集、模型和指标不可拼成同一条进步曲线。结构相关不等于实际语义影响。输出只是候选补丁，不在宿主机运行，不覆盖输入文件。
