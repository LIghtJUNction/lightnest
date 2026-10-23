# Compiled v3 复现

## 环境与边界

本轮实测 Python 3.13.5、Clang 17.0.0、GCC 14.2.0、LLD，以及 Python 自带的 zlib 1.3.1。新的编译实验只需 Python 标准库及 GCC/Clang/LLD，不需要模型权重、Torch、GPU 或模型接口。旧研究测试仍需要仓库原依赖。

在 Linux 上安装 `gcc clang lld`。确认 `clang --target=riscv32-unknown-elf` 可用。不同编译器版本可能产生不同指令布局、变异及成绩，不承诺字节一致复现。

所有命令从仓库根目录执行。把 `OUT` 设置到仓库外部的**新目录**，不要复用原始归档。候选只在有界 RV32IM 解释器运行；编译到本机并加载的只有可信参考 C 和解释器本身。

```bash
export OUT=/tmp/lightnest-compiled-v3-new
python -m unittest discover -s tests -p test_compiled_v3.py -v
python -m experiments.compiled_v3.run build --root "$OUT/development" --phase development
python -m experiments.compiled_v3.run evaluate --root "$OUT/development"
python -m experiments.compiled_v3.run audit --root "$OUT/development"
python -m experiments.compiled_v3.run build --root "$OUT/evaluation" --phase evaluation
python -m experiments.compiled_v3.run evaluate --root "$OUT/evaluation" --selection "$OUT/development/selection.json"
python -m experiments.compiled_v3.run audit --root "$OUT/evaluation"
python -m experiments.compiled_external_v3.run --root "$OUT/external" --protocol experiments/compiled_external_v3/protocol.json
```

`build` 自动创建输出目录，若已存在则拒绝，防止覆盖。新执行会产生新的时间戳与 gzip 容器哈希；重现相同程序不等于生成同一归档哈希。JSON 协议在仓库中采用等价排版，原始协议字节保留于实验包。

## 检查已记录实验与最新版修正

原始测量后，路径记录增加了“代码作为数据被读取”的处理。旧冻结文件仍记录测量时源码，不能删掉冻结检查或改写哈希来强行重跑。私有归档同时提供 `measured_source/` 和最新 `source/`。Git 只放最新版本。

要用最新源码检查已有正式决策，不需要原始模型包，也不需要执行旧候选到宿主机：

```bash
python tools/replay_compiled_v3.py /path/to/archive/evaluation /path/to/archive/external
```

该命令读取原数据及逐条决策记录，重跑搜索，比较最终代码 SHA256、决策状态、补丁和尝试次数。写入独立的 `latest-source-replay.json`；有差异即失败。本轮两份收据分别记录 16,320 和 2,316 次，差异列表为空。先复制档案再运行，以免覆盖已有收据。

主实验 `audit` 从完整日志重建统计，并让另一份 Python 解释器执行保存的输出见证。它不等于对全部输入做形式化证明。外部实验在自身驱动结束前完成同样的见证核对。

## 接口与指标

`Request` 仅含 `code`、`entry`、`cases`、`expected`。可见输入输出要求是本轮任务的一部分，原始程序/错误位置/隐藏输入不得传给 `repair()`。最多 128 组可见样例、4,096 个候选，每次 VM 执行最多 30,000 条指令；一次只编辑一条指令。

`candidate` 只表示通过可见样例，不能翻译成“已安全修复”。`unchanged` 只表示没有观察到可见失败，不能翻译成“没有错误”。`abstain` 和 `budget` 都不输出通过结论。正式报告分别记录审计修复、错误候选、漏检、拒绝和原始无错误程序的修改数。

主审计是 256 隐藏输入＋128 固定公开/参考探针，外部审计是 512 隐藏输入＋256 固定公开/参考探针。输入精确去重不等于算法或语义独立。不同阶段不可直接汇总成一个修复准确率。

## 保存规则

提交 Python/C 源码、必要 JSON 协议、许可证、测试和 Markdown。`dataset.jsonl.gz`、`predictions.jsonl.gz`、ELF、RV32 映像、共享库、汇总原始 JSON、缓存等留在外部实验目录。不要把完整归档导入 Git。
