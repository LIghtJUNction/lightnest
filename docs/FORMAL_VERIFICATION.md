# Lean 辅助验证

这是算法规则的形式化检查层，不是模型准确率证明。当前默认 C64、研究候选、权重、训练数据和评分器均不因接入本工具而改变。

## 运行

安装官方 Elan（Lean 版本管理器），使 `lake` 在 PATH 中。项目固定使用 `leanprover/lean4:v4.35.0`，不跟随 nightly，也不下载 Mathlib。首次使用 Elan 需要联网安装工具链；此后没有第三方 Lean 包依赖。

在仓库根目录执行：

```bash
python tools/check_lean.py
```

完整检查会在新的临时目录中构建，避免旧 `.olean` 掩盖源码问题。依次运行源码检查、`lake build`、传递公理审计、`leanchecker` 内核复查、Python 对照案例和错误证明负例。缺少 Lean、超时、编译失败或任一对照失败都返回非零，不按跳过处理。

可将本次检查凭据写入仓库外的新文件：

```bash
python tools/check_lean.py --report ../lightnest-lean-check.json
```

只运行不依赖 Lean 的工具单元测试：

```bash
python -m unittest discover -s tests -p test_formal_tools.py -v
```

这条命令不检查 Lean 证明，不能替代完整检查。

## 已形式化的范围

| 文件 | 命题 | 前提和边界 |
|---|---|---|
| `formal/LightNest/Patch.lean` | 字节替换保持长度；写入位置可以读回；无关位置保持；不同位置的替换可交换；整批修改保持未触及位置；底稿不匹配时拒绝 | `checkedPatch` 是 `rgb.guard.candidate` 的有限自然数规格。Python 类型检查仍单独测试。底稿指纹相等由外部提供，不证明 SHA-256 无碰撞，也不证明补丁功能正确。 |
| `formal/LightNest/Compute.lean` | 同一状态更新函数的循环可分段；4轮后继续60轮等于64轮；THINK/EMIT消耗不超过预算；零残差保持状态；满足未来输出不变条件时才可提前输出 | 分段要求同一函数、同一输入和完整状态。新增 StateLink、随机性、改权重、改程序、不同浮点运算次序，不能直接套用这个结论。零残差命题使用精确整数，不覆盖 NaN。 |
| `formal/LightNest/Promotion.lean` | 验收通过后，给定记录中的旧成功不会丢失，正确程序不会被误改；同一任务域上的保持关系可逐代传递；拒绝候选时保留父模型 | 只约束已提供的有限、对齐且如实评分的证据。不是所有未来输入的保证，也不证明候选取得了新能力或节省了成本。 |

还保留了一个可直接检查的反例：状态依次为0、1、2时动作都为0，状态到3时动作变成1。因此“连续三次动作相同”不足以证明可以安全提前停止。它针对早退规则的逻辑，不声称真实模型必然出现这个例子。

## 规格怎样连接现有代码

`tools/check_lean.py` 直接加载当前 `baseline/LightNest_RGB_v2/rgb/guard.py`，在固定种子的有效和无效输入上运行实际补丁函数。它把这些实际结果转成临时 Lean 命题，由内核归约检查，与 Lean 规格逐项比较。没有把Python再次计算的结果冒充Lean证明。

检查也读取 `hybrid/model.py` 的实际 `DEPTHS` 常量，比较计算轮数检查规则。候选保持工具的布尔组合和记录重排同样生成Lean对照。

这些有限对照能发现规格与实现偏离，但不是Python程序语义、PyTorch数值内核或RV32解释器的完整形式化等价证明。神经循环本身的数值一致性仍由既有测试负责。

## 证据保持工具

独立调用：

```bash
python tools/formal_contracts.py \
  --parent ../parent-evidence.json \
  --candidate ../candidate-evidence.json
```

文件外层必须提供共同的 `dataset_sha256`、`evaluator_sha256`、`protocol_sha256`，以及 `records` 数组。每行至少有 `id`、`initial_sha256`、`strict`、`false_edit`。前两项标识任务与原程序，后两项必须是真正布尔值。标准 `study.evaluate.score_episode` 已产生这些记录字段；导出时需要额外绑定外层的三个证据指纹。旧结果不能靠编造指纹冒充已登记的证据。

工具拒绝空集、重复任务ID、缺失任务、同ID不同原程序、不同评分/协议/数据指纹，以及“成功且误改”等矛盾字段。返回 `retention_passed` 仅表示通过保持检查，永远不自动更换模型。速度门槛、新任务收益、预留集独立性及原始日志真实性仍须分别核验。

指纹字段的相等检查不证明文件确实对应这份日志。实际运行器必须负责产生和核对这些绑定，不能让候选模型自行提供有利标签。

## 防止虚假通过

- 禁止证明模块使用占位证明、自定义公理、原生求值代替内核证明等逃逸入口。普通 `decide` 仍由内核检查；`native_decide` 不在本项目允许范围。
- 审计已导入的 `LightNest` 全部声明的传递公理依赖，仅允许Lean标准的 `propext`、`Classical.choice`、`Quot.sound`；并在独立进程中用 `leanchecker` 再次调用Lean内核检查构建结果。
- 每次运行都测试错误命题、占位证明、自定义公理、原生证明和不成立的早退命题，确认工具确实拒绝它们。
- 检查证明模块是否全部被根文件导入。防止写了文件却未构建，也防止缺少工具时返回成功。
- 工作流没有写入仓库的权限，不使用模型API或仓库秘密。Lean源码仍可能包含可执行代码；本工具不是对恶意证明程序的安全沙箱。

证明内容、规格、检查工具和CI仍须人工审查。代码审查不能被“CI绿色”取代。

## 加入新算法证明的方式

先写清输入、状态和前提，再定义纯规格和要保持的性质。新增模块放在 `formal/LightNest/` 并加入 `formal/LightNest.lean`。同时增加对应Python实现的有限对照、已知错误版本的负例，以及数值/性能实测。需要Mathlib时另行固定版本和依赖，不在当前轻量构建中自动加入。

优先扩展方向：定义/使用图遍历的健全性、带掩码聚合的维度与归一化条件、量化整数运算的误差界、优化器裁剪的有界性。它们目前都没有完成证明。浮点舍入、溢出、NaN、随机采样和GPU并行次序必须显式建模，不能用理想实数公式代替实际计算。

本目录只提交证明源码、配置和文档。构建物、生成案例、权重和数据不进入Git；检查收据写到仓库外。

## 参考

官方工具链发布：https://github.com/leanprover/lean4/releases/tag/v4.35.0

Lean证明验证和信任边界：https://lean-lang.org/doc/reference/latest/ValidatingProofs/

Lean公理说明：https://lean-lang.org/doc/reference/latest/Axioms/

官方CI动作：https://github.com/leanprover/lean-action
