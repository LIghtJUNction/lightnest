# RPN3 复现

从仓库根目录执行，使用 uv 锁定环境。准备项目 [外部档案](../archive/legacy/docs/ARTIFACTS.md)：
Hybrid v1 完整包、10k 扩量数据包、10k 扩量源码/模型/结果包。
`--out` 必须是仓库外的全新目录；后续阶段不覆盖已有终点或记录。

```bash
uv sync --locked
export OUT=/absolute/path/to/new-rpn3-run
uv run --locked python -m lightnest_next prepare \
  --hybrid /path/to/LightNest_Hybrid_v1_source_models_results.zip \
  --dataset /path/to/LightNest_scale_v2_10k_dataset.zip \
  --parents /path/to/LightNest_scale_v2_10k_source_models_results.zip \
  --out "$OUT"
uv run --locked python -m lightnest_next train --out "$OUT" --seed 104729
uv run --locked python -m lightnest_next train --out "$OUT" --seed 130363
uv run --locked python -m lightnest_next freeze --out "$OUT"
uv run --locked python -m lightnest_next evaluate --out "$OUT" --seed 104729
uv run --locked python -m lightnest_next evaluate --out "$OUT" --seed 130363
```

训练只读取规范数据集的 TRAIN 分区，按确定性哈希抽取最多 32,768 条 EDIT/STOP
各半的记录；与旧回归结构重复的记录排除。公开验证分区同样排除旧回归结构，
可能少于请求的 512 条。现有生成器以新种子生成最终任务，并用全部可访问
索引和 Hybrid 记录做字节及中和结构指纹隔离。

默认两个种子、各 2,048 次更新、batch 16、AdamW 学习率 0.0003、梯度裁剪 1。
相同训练流用于两个初始化，固定最终终点，公开验证只作诊断。种子、输入、
源码、模型、依赖环境和流哈希在产物中记录。两份模型都完成后，`freeze` 才能通过。

评价分别运行当前模型、同种子稳定 C64 和扩量父模型，保存逐条输出、旧成功丢失
与新增误改。父模型用原评价器运行，再用当前统一评分路径重新核对。
最后任务已运行后就成为公开回归，不能重命名成新的盲测。

推断读取 JSONL：

```bash
uv run --locked python -m lightnest_next predict --checkpoint "$OUT/models/rpn3-104729.pt" < requests.jsonl
```

每行 `{"command":"change add to sub","binary":"..."}`。不能附带 `target` 或标签。
检查点用 `weights_only=True` 读取并严格核对架构和参数；不回退到随机初始化。
