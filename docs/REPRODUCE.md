# 本机训练与复现

```bash
uv sync --locked
uv run --locked python tools/check.py
```

需要 Linux x86-64、GCC、Clang、LLVM 和 LLD。固定依赖见 `pyproject.toml` 与 `uv.lock`。
历史原始包见 [外部档案](../archive/legacy/docs/ARTIFACTS.md)。数据、运行和权重目录必须在仓库外。

## 历史预训练

归档 RPN3 的准备/训练程序位于 `archive/rpn3_rv32/`。从该目录运行其
`lightnest_next` 模块，按 [归档复现步骤](../archive/rpn3_rv32/docs/REPRODUCE.md)
准备历史 TRAIN、验证和回归。旧路径名称可能反映移动前的位置，应以归档目录为工作目录。
当前本机模型初始化只需要该次 `rpn3-104729.pt` 和对应冻结 TRAIN 清单，不读取其未消费测试。

## 原生语料与两个固定终点

```bash
export OUT=/absolute/path/to/new-universal-run
export HISTORICAL=/absolute/path/to/rpn3-run
export INITIALIZATION="$HISTORICAL/models/rpn3-104729.pt"
uv run --locked python -m lightnest_native prepare --out "$OUT" --historical-run "$HISTORICAL"
uv run --locked python -m lightnest_native train --out "$OUT" --seed 104729 --initialization "$INITIALIZATION"
uv run --locked python -m lightnest_native train --out "$OUT" --seed 130363 --initialization "$INITIALIZATION"
uv run --locked python -m lightnest_native freeze --out "$OUT"
uv run --locked python -m lightnest_native evaluate --out "$OUT" --seed 104729
uv run --locked python -m lightnest_native evaluate --out "$OUT" --seed 130363
```

`prepare` 生成八类 Clang 函数变换：算术、布尔运算、移位、旋转、最值、常量和比较；
每组有需改及已正确版本，均保留可见和独立隐藏样例。历史 TRAIN 再转换 64 组，
不混入旧验证/回归/测试。原生参考同时通过 Python 规格与实际机器码执行检查。

每个种子 1,536 次 AdamW 更新、batch 8，两个起点看到相同采样流。七个原生批次
穿插一个历史批次；完整终点固定，验证只作诊断。源码、依赖锁、数据和初始化
检查点哈希保存在运行目录，变更后需要新运行。两份终点均完成才能冻结，之后
才运行最终测试。重跑公开过的题只是回归，不是新的盲测。

## 输入与输出

```json
{"architecture":"x86_64","command":"replace addition with subtraction","binary":"4889f84801f0c3","examples":[{"inputs":[5,2,0],"output":3}]}
```

运行：`uv run --locked python -m lightnest_native predict --checkpoint /path/to/universal-104729.pt < requests.jsonl`。
这是格式示例，不保证任意未训练请求的预测正确。输出包含全部候选补丁、最终字节、
STOP 和未验证标记。检查点加载严格核对架构和参数；不自动生成随机权重替代。

验证式解码在独立、仓库外的新目录编译执行器：

```bash
uv run --locked python -m lightnest_native repair \
  --checkpoint /path/to/universal-104729.pt --workdir /tmp/lightnest-repair-new < requests.jsonl
```

其候选只按输入中的可见样例核对，不能代替隐藏测试或形式化证明。
验证式解码在纯神经首次终测之后提出，单独冻结一批新的编译程序种子，比较两个入口：

```bash
export VERIFIED=/absolute/path/to/new-verified-evaluation
uv run --locked python tools/evaluate_repair.py prepare --out "$VERIFIED" --previous "$OUT" \
  --models "$OUT/models/universal-104729.pt" "$OUT/models/universal-130363.pt"
uv run --locked python tools/evaluate_repair.py evaluate --out "$VERIFIED"
```

这个阶段不训练新权重。第一次神经终测在后续研究中只作为公开回归，新测试仍
来自已有八类合成程序，不能称为新算法族泛化。

汇编入口读取标准输入。ELF 检查只提取函数，不承诺完成重定位或修复整个可执行文件。
