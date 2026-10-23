# RGB 复现

所有命令通过 `uv run --locked` 使用锁定环境。Linux 原生执行需要 bubblewrap、
libseccomp；构建生成训练标签需要 GNU as / ld。数据必须放在仓库外的新目录。

```bash
OUT=/path/outside/checkout/rgb-run
uv run --locked python -m lightnest_rgb prepare --out "$OUT" --per-program 128
uv run --locked python -m lightnest_rgb train-policy --out "$OUT" --seed 104729
uv run --locked python -m lightnest_rgb train-generator --out "$OUT" --seed 104729
uv run --locked python -m lightnest_rgb freeze --out "$OUT" --name final \
  --policy "$OUT/weights/rgb-104729.pt" --generator "$OUT/weights/generator-104729.pt"
uv run --locked python -m lightnest_rgb evaluate --out "$OUT" --name final --kind policy
uv run --locked python -m lightnest_rgb evaluate --out "$OUT" --name final --kind generation
```

`prepare` 读取当前机器的真实程序，保存 SHA256、包版本、ELF 结构以及实际运行结果。
基础集有 16 个程序；扩展集增加路径、文本和字节工具。基础集有 12 个程序训练，
grep/zstd 验证，sed/jq 测试。每程序 128 个任务，包括无修改任务和
1/2/4/8 处位翻转。只有实际影响可见执行的损坏才被纳入。困难区段尝试失败后允许
回退到文件头，逐条记录 requested_band 与实际 band，不能把回退计作代码区任务。

编辑器先在真实训练程序的 256 字节窗口预训练，再进行完整文件训练。生成器使用
704 个完整链接器生成的 ELF 标签，同时对 12 个真实程序做字节预测；链接器只构建
训练标签，推理不调用它。64 个生成验证样本与 96 个测试样本独立保存。

`freeze` 在模型接触测试请求前固定检查点、源码、语料哈希和任务清单；默认每个未见
程序抽取 32 个分层任务，生成抽取 32 个请求。这是抽样测试，不是跑完全部测试语料。
编辑器只获得四个可见样例，最终输出才用八个隐藏样例检查。原始损坏位置和干净文件
不作为推理输入。执行日志和权重阶段端点保留，训练暂不支持断点续训。

扩大规模时可以使用 `--per-program 256`。当前扩展集包含 23 个真实工具，覆盖文本、压缩、
路径、十六进制和字符串处理；任务数按程序线性增加，训练和测试程序仍按程序隔离。

同一冻结输入可运行 `--mode compressed`、`--mode dense` 或 `--fixed-depth 16` 对照。
这些是同一权重的诊断消融，不是分别调优的独立模型。原始结果包含逐步动作、耗时、
错误 STOP、编码量、循环步数和执行缓存命中。真实端到端耗时包含模型与执行验证。

生成请求文件：`{"exit":37}`，或 `{"stdout":"68656c6c6f21210a"}`（恰好 8 字节）。

```bash
uv run --locked python -m lightnest_rgb generate \
  --checkpoint "$OUT/weights/generator-104729.pt" --request request.json \
  --output /tmp/new-generated.elf --workdir /tmp/new-generation-run
```

生成结果必须由模型停止、通过 ELF 检查并实际满足请求才发布。任何失败都不补头修复。
输出与工作目录均需新路径、位于仓库外。

编辑请求为 `{"command":"...","examples":[{"input":{"args":[],"stdin":""},
"expected":{"status":"ok","returncode":0,"stdout":"","stderr":""}}]}`。
stdin/stdout/stderr 使用十六进制。仅提供合法可见测试，不传损坏标签或隐藏结果。

```bash
uv run --locked python -m lightnest_rgb repair /path/to/damaged.elf \
  --checkpoint "$OUT/weights/rgb-104729.pt" --request repair.json \
  --output /tmp/new-repaired.elf --workdir /tmp/new-repair-run
```

固定长度编辑、最多 8 步、候选与时间有界，失败时返回放弃，不覆盖原文件。
历史架构在 `archive/`，旧结果与训练哈希保留当时路径；归档后的原生入口为
`python -m archive.universal_native`。路径迁移使旧训练协议的源码哈希失配，旧冻结
评估不能冒充未经改动重跑；需要在对应历史提交检出后复现，或者建立新协议。

## 公开验证后的定向训练

初次训练若暴露错误停止或请求相关字节权重不足，可明确运行以下后续课程。它们是
同一模型的继续训练，不是第二个独立随机种子，不可用作多种子稳定性的证据。

```bash
uv run --locked python -m lightnest_rgb refine-policy --out "$OUT" \
  --parent "$OUT/weights/rgb-104729.pt" --seed 104730
uv run --locked python -m lightnest_rgb refine-generator --out "$OUT" \
  --parent "$OUT/training/generator-104729/step-512.pt" --seed 104730
```

编辑课程加强真实训练文件头的下一步修改排序，再训练完整文件。它对文件头有明确偏重，
不能解释为代码区和只读数据区能力都得到提升。生成课程从训练标签内部找出随请求变化
的字节位置，提高这些位置的损失；位置只用于训练，推理仍由模型输出每个字节。
原始运行、父检查点哈希、后续更新和失败验证均保留。最终冻结时明确指定使用的权重。

对照默认使用冻结时每个程序等间隔选出的 8 个任务（共 16 个），主线测试为每个程序
32 个（共 64 个）。与主线结果比较时必须取相同 ID 的子集，不能直接比较两种分母。
`freeze --generation-count 96` 可使用全部 96 个生成测试请求。此前对干净 jq 做过的
无标签计时会写入冻结记录；它没有参与训练，但不宣称模型从未读取过该文件。

强化学习式反馈训练从一个已训练策略继续。每步只执行可见样例，候选的 stderr、退出
信号、超时、入口预检和可见通过数构成奖励；不读取隐藏结果或干净目标字节。

```bash
uv run --locked python -m lightnest_rgb train-feedback --out "$OUT" \
  --parent "$OUT/weights/rgb-104731.pt" --seed 104732 --steps 128
```

这是有界 REINFORCE 微调，不是完整 RL 收敛证明。它保存每步奖励、基线、错误类型、
候选动作和实际执行计数；使用新检查点前重新执行 `freeze`。

如果模型源码增加了红区抽象层或蓝区指导层，旧检查点会被源码哈希保护拒绝。此时用
`train-feedback` 做显式 warm-start 迁移，生成的新检查点才可以加载。

## 统一生成与编辑入口

`act` 是统一 artifact agent。`mode=generate` 从请求生成 ELF，`mode=repair` 从已有 ELF
和可见样例开始编辑。加载 Any2Any checkpoint 后，`mode=understand` 和 `mode=copy_paste`
也进入同一个入口，支持任意字节流的模型观察和原生复制粘贴；ELF 路径经过 ELF 检查和隔离执行器。

```bash
cat >/tmp/act-generate.json <<'EOF'
{"mode":"generate","spec":{"stdout":"5247422e4f4b210a"}}
EOF
uv run --locked python -m lightnest_rgb act \
  --generator "$OUT/training/generator-104730/step-256.pt" \
  --request /tmp/act-generate.json --output /tmp/act-generated.elf \
  --workdir /tmp/act-generate-run
```

红区循环产生高维抽象状态，蓝区把它转换为字节动作、风险和语言 token。预算头从蓝区
状态中学习选择蓝区或一个红区深度；推理时只运行被选中的深度。`async_thoughts` 在每个
思考深度之间把控制权交回事件循环，外部执行反馈可以回流。预算标签来自当前任务中最早
成功的动作，不使用文件长度、重复率或 ELF 特判。
