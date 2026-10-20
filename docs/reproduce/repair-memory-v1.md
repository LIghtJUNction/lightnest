# Repair memory v1：复现和接续

源码起点：`7a387672105ef68b3117371e19e56819fbe7d6db`，机制源码提交：`0ee97b254aaf3a2a0d90faff5b8e33b6e8b945e4`。默认模型未改动，不需要模型权重。依赖Python 3.13及GCC、Clang、LLD；测量环境使用GCC14.2、Clang17。候选只进入受限解释器。

在已包含本轮源码的仓库根目录运行。禁止使用Python的 `-O` / `-OO` 关闭断言。

## 源码检查

```bash
python -m unittest discover -s tests -p 'test_repair_memory_v1.py' -v
python -m unittest discover -s tests -p 'test_compiled_v3.py' -v
python tools/check_lean.py
```

最后一条需要固定的Lean工具链；不允许把缺失工具当作证明通过。

## 重放交付数据

先解压外部实验包。`source/` 包含本轮运行所需的源码子集，不是完整主仓库；完整Lean检查使用GitHub仓库。令 `EVIDENCE` 指向解压后的 `LightNest_mechanism_v1/`，`RUN` 是此前不存在的仓库外路径：

```bash
export EVIDENCE=/path/to/LightNest_mechanism_v1
export RUN=/path/to/new-memory-replay
mkdir "$RUN"
cp "$EVIDENCE/fresh/dataset.jsonl.gz" "$RUN/dataset.jsonl.gz"
python -m experiments.repair_memory_v1.run evaluate \
  --root "$RUN" --protocol experiments/repair_memory_v1/protocol.json
python -m experiments.repair_memory_v1.run audit --root "$RUN"
python -m experiments.repair_memory_v1.run timing \
  --root "$RUN" --protocol experiments/repair_memory_v1/protocol.json
```

评价会检查冻结源码哈希，旧输出不会被当作新的盲测。运行目录存在 `memory-freeze.json` 时拒绝重跑覆盖。审计和计时文件属于新运行，不能写回旧档案。

## 重新构建原协议的编译程序

```bash
export RUN=/path/to/new-memory-build
python - <<'PYBUILD'
import os
from pathlib import Path
from experiments.compiled_v3.run import build
build(Path(os.environ['RUN']), 'mechanism_evaluation', [6, 7, 8], single=12, double=4)
PYBUILD
```

然后按上述命令评价、复核、计时。重新构建仍是同一实验，不是新的泛化测试。不同编译器可能改变机器码、候选空间和成绩；gzip与构建时间信息也可能改变文件哈希。精确重放用归档数据和固定源码，逐项核对逻辑结果，不要求墙钟时间逐位相同。

原试验分别保存协议、代码哈希、数据构建、最终冻结、全部五种策略决策、失败见证、逐样例得失与计时。下一轮必须另选未使用的构造和测试，不能围绕已知失败调参后继续称其为盲测。
