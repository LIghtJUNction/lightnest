# Research v2 运行与复现

所有模型实验为显式可选入口，默认 C64 不变。Python 3.13.5、PyTorch 2.10.0+cpu、NumPy 2.3.5。不要在已保存的 `run` 目录重跑写入：写文件用排他创建，防止覆盖冻结结果。

## 1. 源码与依赖

使用 GitHub 的 `experiments/research_v2/`，或本次归档 `source/` 内完全相同的源码。在含 `experiments/` 的目录运行命令。

另下载项目原有的 `LightNest_Hybrid_v1_source_models_results.zip`，解压到工作区之外。SHA256：`234ee0f38c546207195a3fb1ac968d45592fb35b51f84d1777d90c2c4051f079`。它包含未改的 RGB/C64 源码、父模型、训练和公开回归文件。本次增量包不重复包含这些依赖。

原始归档：<https://drive.google.com/file/d/1qRqHhF-NFh08VrAnQNjqbsVhF-xoOV1-/view>

```bash
uv sync --locked
export LIGHTNEST_ARTIFACT_ROOT=/absolute/path/to/LightNest_Hybrid_v1
uv run --locked python -m unittest experiments.research_v2.test_methods -v
```

这 12 项机制检查不需要外部模型权重，可在源代码仓库中不设置环境变量直接运行。训练、模型推断、历史复算必须使用完整原始归档。只有基础源码的 Git 克隆不能代替已训练权重。

## 2. 使用胜出的受限规则工具

这个例子只返回动作，不写入或运行二进制文件；无需加载神经权重。工具接受的固定语法见 `methods.COMMANDS`。示例也是受限结构变换，不是一般代码修复。

```python
from experiments.research_v2.methods import BackwardSlice, common
code = common.assemble('lw x1, 0(x0)\nlw x2, 4(x0)\nadd x10, x1, x2\nebreak')
request = {'command': 'change every add that contributes to the output to sub',
           'binary': code.hex()}
print(BackwardSlice().plan([request])[0])
```

语义限制：`x XOR x` 等无影响运算可能仍被视为结构依赖。超过 8 个修改的任务，正式评估器会按预算判失败；`plan()` 本身只是列出完整计划，不可绕过调用方预算直接执行。

## 3. 从头复现本轮实验

准备原始 `LightNest_scale_v2_10k_dataset.zip`，它用于排除历史字节和结构指纹冲突：<https://drive.google.com/file/d/1D0sRiqpqUEkbsxAGdbACXi9Y7sV5IHPu/view>。取全新输出目录，不能与本归档的 `run/` 相同。

```bash
export RUN=/absolute/path/to/new-research-v2-run
uv run --locked python -m experiments.research_v2.prepare --output "$RUN" --index-archive /path/to/LightNest_scale_v2_10k_dataset.zip
for seed in 104729 130363; do
  uv run --locked python -m experiments.research_v2.run train --root "$RUN" --seed "$seed"
done
for seed in 104729 130363; do
  uv run --locked python -m experiments.research_v2.run develop --root "$RUN" --seed "$seed"
done
```

原实验是在公开校准结果出现后、任何新测试推断之前增加固定阈值 0 和始终 STOP 对照。复现时先保存这一说明，供冻结记录引用，不省略此步骤：

```bash
uv run --locked python - <<'PY'
import os
from pathlib import Path
from datetime import datetime, timezone
from experiments.research_v2.prepare import write
write(Path(os.environ['RUN'])/'driver-amendment.json', {
    'utc': datetime.now(timezone.utc).isoformat(),
    'reason': 'Reproduce the disclosed fixed-zero and always-STOP controls before held-out inference; no new training.',
    'holdout_queries': 0})
PY
for seed in 104729 130363; do
  uv run --locked python -m experiments.research_v2.controls development --root "$RUN" --seed "$seed"
done
uv run --locked python -m experiments.research_v2.controls freeze --root "$RUN"
for seed in 104729 130363; do
  uv run --locked python -m experiments.research_v2.run test --root "$RUN" --seed "$seed"
  uv run --locked python -m experiments.research_v2.controls holdout --root "$RUN" --seed "$seed"
done
uv run --locked python -m experiments.research_v2.audit --root "$RUN"
uv run --locked python -m experiments.research_v2.benchmark --root "$RUN"
uv run --locked python -m experiments.research_v2.boundaries --root "$RUN"
```

必须完成两个种子的公开开发阶段和选择冻结，再执行任何新测试；顺序不可交换。正式报告中的测试题现在已经公开使用，重跑这些命令是复现，不是新的泛化证据。

## 4. 检查已保存结果，不重训

归档 `run/models/` 存四份新判断头及优化器/随机状态；原父模型位于外部 Hybrid v1 归档。`run/holdout-*.json` 与 `development-*.json` 保存每条正式输出。`audit.py` 展示不使用原评分函数的动作重放和第二解释器核算方法。

所有文件的 SHA256 见增量归档的 `SHA256SUMS.json`。原终点冻结在 `run/final-freeze.json`，新增对照与最终选择在 `run/selection-all-paths.json`。移动包后部分来源说明保留测量时的绝对路径；真正运行依赖由 `LIGHTNEST_ARTIFACT_ROOT` 指定。

除非明确另建新实验，不改已冻结源码、权重、测试或阈值，不在宿主机执行候选，不将新权重/数据加入 Git。Git 的测试工作流只跑无需权重的机制测试，不表示自动训练或复现了全部模型成绩。
