# WeightLang 效率迭代的下游验收（2026-11-20）

LightNest 已用 WeightLang 本轮新编译的前端与原生服务完成两项真实集成回归，均通过。验收覆盖循环模型的梯度与训练，以及 routed attention 的各模式和参数更新。结构化回执见[验收数据](benchmarks/weightlang-efficiency-2026-11-20.json)。

## 固定源码与依赖关系

- LightNest 被测源码：[`6062f57badac913512ca0ac96a328ef6fbdd8275`](https://github.com/LIghtJUNction/lightnest/commit/6062f57badac913512ca0ac96a328ef6fbdd8275)。
- WeightLang 功能源码：[`072582a03a123b7d95c815d563689383033019d4`](https://github.com/WeightLang/WeightLang/commit/072582a03a123b7d95c815d563689383033019d4)。
- [pyproject.toml](../pyproject.toml) 与 [uv.lock](../uv.lock) 已声明 `weightlang[torch]==0.5.4`，并通过 editable 源指向 `../WeightLang/WeightLang/python`。本轮继续使用该来源，以完整 Git 提交固定验收代码。
- 原生模型入口位于 [lightnest/architecture/native.py](../lightnest/architecture/native.py)，模型结构保存在 `lightnest/architecture/wgt/`。测试显式指定新 `wgt` 和 `wgtime`，服务在独立的 `--test-directory` 中运行并由测试回收。

## 实际结果

| 测试 | 验证范围 | 结果 |
| --- | --- | --- |
| `test_native_and_torch_forward_recurrent_gradients_and_training` | 原生与 Torch 循环模型前向、显式状态梯度及训练更新 | 通过 |
| `test_complete_native_routed_attention_all_modes_and_parameter_update` | 两组小型几何、attention 各模式、参数更新与参考输出 | 通过 |

命令退出码为 0，unittest 报告 `Ran 2 tests in 2.633s`、`OK`。测试源码见 [test_native_model.py](../tests/architecture/test_native_model.py)。回执同时固定编译器和服务的 SHA-256，便于检查配套产物。原始命令、完整输出和本地环境位于 `weights/efficiency-20261120/native-validation.json` 与 `native-regressions.log`。

本轮验证了这些使用路径的行为一致性；没有用测试耗时估计 LightNest 的训练或推理加速，也未将它等同于完整应用、长期训练或 RSI 实验验收。上游性能数据及测量范围见 [WeightLang 效率记录](https://github.com/WeightLang/WeightLang/blob/main/docs/efficiency-2026-11-20.md)。

## 复现

先按上述 WeightLang 提交构建 `target/debug/wgt` 与 `target/debug/wgtime`，再从 LightNest 根目录运行：

```bash
PYTHONPATH=tests/architecture \
WGT_COMPILER="$PWD/../WeightLang/WeightLang/target/debug/wgt" \
WGTIME_TEST_DAEMON="$PWD/../WeightLang/WeightLang/target/debug/wgtime" \
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
PYTHONDONTWRITEBYTECODE=1 TMPDIR=/tmp/weightlang \
.venv/bin/python -m unittest -v \
  test_native_model.NativeModelTests.test_native_and_torch_forward_recurrent_gradients_and_training \
  test_native_model.NativeModelTests.test_complete_native_routed_attention_all_modes_and_parameter_update
```
