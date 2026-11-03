# WeightLang 迁移

当前模型架构来源为 `lightnest/architecture/wgt/*.wgt`。正常的 `RGBPolicy`、
`ELFGenerator`、`HNetELFGenerator`、`ArtifactCore`、结构学习器、`LearnedOpenat`
及 `X86World` 入口会读取这些文件；Python 文件保留配置、预处理、损失及宿主接口。
修改 WGT 后需重新构造模型或重启进程。已有模型实例不会热更新。
`archive/` 保留历史对照，不迁移；确定性的 ELF loader、x86/PTX 状态机仍是宿主运行环境。

## 使用的语言和环境

这次迁移同时扩展了本地 WeightLang：`wgt 0.3` 下新增 **`backend torch-dynamic`**。
它是使用 Python 兼容表达式、编译到 PyTorch 的动态模型前端，不是原有静态 `layer/block`
图的无缝扩展，也不是脱离 PyTorch 的独立执行器。它支持多参数、多返回值、变长张量、
显式跨调用状态、动态控制流、共享参数、mask、top-k/gather 和 PyTorch reset-after GRU。
模型构造和完整前向公式都在 WGT 内；不是把原 Python 模型注册成一个黑盒算子。

这个前端尚未发布。需要本次修改后的本地编译器 checkout；版本元数据仍为 0.2.0，
仅凭版本号无法判断是否包含扩展。项目锁定的是可编辑本地依赖：

```text
../WeightLang/WeightLang
```

从项目根执行：

```bash
uv sync --locked
uv run --locked weightc check lightnest/architecture/wgt/generate.wgt --shapes
uv run --locked weightc build lightnest/architecture/wgt/generate.wgt \
  --no-train --no-store --tensor-args '{"width":16,"capacity":64}' \
  -o weights/new-generator
uv run --locked weightc verify weights/new-generator/ELFGenerator --trust-tensor-source
```

已有输出会拒绝覆盖。`check` 只做语法检查，不初始化模型，不执行训练。
`--shapes` 明确报告动态契约：维度和 dtype 由模型守卫与 Torch 在执行时检查，
**没有静态形状证明**。这一形式不支持 Rust 原生模型加载、静态图导出、revision store、
自动复用或分层设备放置；整模型 `.to(device)` 保留 PyTorch 语义。
本次数值对照在 CPU/f32 上执行，未完成 CUDA 等价性或冻结单 ELF 的实际重打包验收。

## 训练、推理与保存

现有训练和推理命令继续使用 WGT 架构，无需切换模型类。
RGB/ELF、Any2Any 和结构训练入口在兼容 `.pt` 外同时保存完整 WGT 模型目录。
模型目录含 `source.wgt`、`weights.safetensors`、`manifest.json`；manifest 包含依赖 WGT
源码快照、构造参数、显式宿主依赖摘要及训练记录。不要只移动 safetensors 文件。

```python
from lightnest.architecture.generate import ELFGenerator, condition_tensor
from lightnest.architecture.weightlang_io import save_model, load_model
import torch

model = ELFGenerator(width=16, capacity=64)
previous = torch.tensor([[256, 1, 2]])
condition = condition_tensor([{"exit": 7}])
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
optimizer.zero_grad()
model(previous, condition)["bits"].square().mean().backward()
optimizer.step()
save_model(model, "weights/my-generator", training={
    "method": "AdamW", "steps": 1, "dataset": "synthetic smoke test"
})
restored, metadata = load_model("weights/my-generator")
output = restored(previous, condition)
```

此示例只验证优化路径，不能证明生成 ELF 的质量。完整训练继续使用现有真实任务损失。
制品保留参数冻结标记，但不包含优化器状态；不能据此声称精确恢复中断的训练。
RGB `load_policy`、ELF `load_generator`、Any2Any 统一代理和自检入口接受完整模型目录或旧 `.pt`。
单 ELF 构建工具仍接收 `.pt`，已补充 WGT/package 数据收集，尚未执行大型重新打包。

动态源码是可信可执行代码，不是沙箱。通用 `load_artifact` 默认拒绝执行动态制品；
调用方必须显式传 `trusted_tensor_source=True`。项目 `load_model` 为本地模型明确选择执行。
宿主代码仍须安装；宿主摘要不符会拒绝加载，WGT 源码快照本身不依赖当前可编辑 WGT 文件。

## 旧检查点转换

旧参数名和布局保持不变，无需转置或重新训练：

```bash
uv run --locked python tools/migrate_weightlang_checkpoint.py old-generator.pt \
  --output weights/imported-generator
```

支持当前 RGB policy、ELF generator、Any2Any 完整检查点。转换记录来源文件 SHA256、
每个张量的显式同名映射，严格检查键/形状/dtype，重载后逐位比较参数。
缺少当前参数的旧检查点会拒绝转换，不用随机参数补齐。
项目保留的宽松旧检查点兼容加载行为与这个严格转换入口不同。
结构模型可通过 `save_model(model, path, kwargs={"width": ...})` 保存。

## 本次验证

基线为本仓库提交 `6210dcaa1387c74616f997990f6c40d5c9598442`，逐文件摘要在
`lightnest/architecture/wgt/baseline.json`。可复跑独立的原代码／WGT 对照：

```bash
uv run --locked python tools/verify_weightlang.py --output weights/new-verification
uv run --locked python tools/check.py
```

21 组覆盖 ELF 深度 0/2/4、流式状态、H-Net dynamic/fixed/dense、RGB 三种注意力、
Any2Any mean/GRU 请求编码、4/8/16 字节语义分块、BPE 语言共享权重和流式状态、
三种结构学习器、LearnedOpenat、X86World。每组检查同名参数、eval/train 前向、
梯度、一次 AdamW 更新及制品保存重载，容差 `atol=1e-6, rtol=1e-5`。
当前样本最大误差为 0。验证用初始化参数和合成输入；没有迁移或宣称存在旧训练权重，
没有完整重训，也不证明所有输入等价或实际任务质量。

逐组指标与测试汇总见 [验证快照](weightlang-verification.json)；完整参数映射和模型保留在
`weights/migration-verified/`。本地验收为 LightNest 332 项（3 项跳过）、WeightLang 相关 118 项。
