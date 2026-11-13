# WeightLang 迁移

模型架构使用 `wgt 0.5`，入口为 `lightnest/architecture/wgt/<组件>/model.wgt`。
`RGBPolicy`、`ELFGenerator`、`HNetELFGenerator`、`ArtifactCore`、结构学习器、
`LearnedOpenat` 和 `X86World` 的公共 Python 入口直接加载这些定义。
Python 文件保留配置、数据预处理、任务损失和应用调度。

## 版本与编译

项目依赖当前本地 WeightLang 0.5.4 SDK，路径为 `../WeightLang/WeightLang/python`。
本文使用当前 WeightLang checkout 的原生模型对象 API；仅版本号 0.5.4 不足以确认具备此接口，需使用包含该接口的源码。
包版本和语言版本分别为 `0.5.4` 与 `wgt 0.5`。

共享 Rust 前端解析 `model ... extends ... {}`、`init(...) {}` 和 `fn ... {}`，
通过 `wgt compile --target torch` 输出编译定义。表达式、方法签名和张量操作采用
Python/PyTorch 语义，由 RustPython 在 Rust 编译阶段验证，SDK 执行生成的普通
PyTorch 模块。`host` 和 `extern` 显式列出宿主依赖。

该执行路径支持当前项目的变长序列、多输入/多返回值、GRU 流式状态、可微掩码、
top-k/gather、共享参数与 H-Net 的直通梯度。形状与 dtype 由模型守卫及 PyTorch
运行时检查。检查器只验证语法和生成代码，动态模型的参数量在实例化后计算。

从项目根目录执行以下命令。

```bash
cargo build --locked --manifest-path ../WeightLang/WeightLang/Cargo.toml \
  -p weightlang-wgt --bin wgt --bin weightc
uv sync --locked
export WGT_COMPILER="$PWD/../WeightLang/WeightLang/target/debug/wgt"
../WeightLang/WeightLang/target/debug/wgt check \
  lightnest/architecture/wgt/generate/model.wgt
../WeightLang/WeightLang/target/debug/wgt compile \
  lightnest/architecture/wgt/generate/model.wgt --target torch
```

本地验证显式使用同一 checkout 下的 `target/debug/wgt`，避免 SDK 默认查找方式变化后使用了其他编译器。
显式的 `WGT_COMPILER` 或 Python `compiler=` 参数具有更高优先级。
使用其他工具链位置时先设置 `WGT_COMPILER`。`wgt check` 与编译都不会构造模型、
执行前向或启动训练。

`target torch` 的模型执行由 PyTorch 提供。Rust/service 模型、原生 AOT、原生设备调度
有各自的接口与支持范围。当前迁移验收使用 CPU/f32；CUDA 和单 ELF 重打包尚待验证。

## 训练、推理和完整模型

现有应用、训练和推理入口保持调用方式，网络计算从新的 WGT 入口加载。
修改 WGT 后重新构造模型或重启进程。公共推理加载器返回 eval 模式的实例，训练时显式调用 `.train()`。

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

保存目录包含 `source.wgt`、`weights.safetensors` 和 `manifest.json`。
manifest 保存编译定义、依赖 WGT 快照、构造参数、冻结状态、宿主摘要及调用方提供的
训练记录。safetensors 同时保存恢复所需的非持久 buffer，重载时恢复参数共享关系。
参数绑定使用 meta 构造布局，避免抽样随机初始权重；模型构造器需要支持 meta device。
保存拒绝覆盖已有目录，失败会清理本次未完成的目标。

SDK 的 `load_module(path, trusted=True)` 显式信任生成代码与宿主依赖，执行前验证
源码、权重及宿主摘要。项目 `load_model` 为本地模型选择此执行方式。
完整快照支持脱离编译器加载，宿主 Python 包仍需安装。
训练记录由调用方提供，优化器状态由应用检查点保存。

## 既有检查点和模型目录

参数名、布局和训练边界保留。原有 `.pt` 仍可从应用入口加载。
上一版 LightNest 的完整 `weightlang.tensor-program` 目录通过项目适配器读取，
核对已知源码摘要、权重摘要、键、形状和 dtype，再绑定到迁移后的架构。
未知旧源码会报错，原目录持续保留。

显式转换为新目录时使用以下命令。

```bash
uv run --locked python tools/migrate_weightlang_checkpoint.py old-generator.pt \
  --output weights/imported-generator
uv run --locked python tools/migrate_weightlang_checkpoint.py \
  weights/migration-verified/generator-depth-2 \
  --output weights/imported-generator-0.5
```

转换记录来源摘要和逐张量同名映射，并在重载后逐位核对参数。

## 验证

```bash
uv run --locked python tools/verify_weightlang.py --output weights/new-verification
uv run --locked python tools/check.py
```

对照使用提交 `6210dcaa1387c74616f997990f6c40d5c9598442` 的原始 Python 实现，
逐文件摘要保存在 `lightnest/architecture/wgt/baseline.json`。
21 组覆盖 ELF 深度与流式状态、H-Net 三种分块、RGB 三种注意力、Any2Any mean/GRU、
语义分块、BPE 共享权重、结构学习器、LearnedOpenat 与 X86World。
每组比较参数、eval/train 前向、梯度、一次 AdamW 更新和完整模型保存重载，
容差为 `atol=1e-6, rtol=1e-5`。输入与参数来自合成验收，完整训练和任务质量另行评估。

本次结果见 [验证快照](weightlang-0.5.4-verification.json)，完整模型和逐张量映射保存在
`weights/migration-wgt-0.5.4-final/`。


## 原生 WGT 对象与后端选择

当前 Python 通用 `weightlang.load/loads/from_ir` 默认返回 `wgt.Model`，执行后端为 Rust/wgtime；不会在原生不支持时自动换成 PyTorch。CPU/f32 输入为列表，参数由服务持有，`train_step` / `train_recurrent` 调用原生优化器。可以继承 `Model` 自定义宿主输入、验证和输出处理。原生导入本身不要求安装 PyTorch。

应用现有十个复杂模型入口全部采用 `<组件>/model.wgt`，并明确 `bind_models(..., backend="torch")`。生成对象同时是 `wgt.Model` 和 `nn.Module`，保留原来的参数名、训练与制品格式；它们仍依赖 mask、top-k、模块继承等 Torch 专用能力，不能称为完整 Rust 模型。历史 archive 是冻结的研究材料，未改写其中的旧源码和入口。

`lightnest.architecture.native.load_recurrent(input_size, hidden_size, *, backend="native", **options)` 直接加载 `wgt/shared_gru/model.wgt`，默认创建真实 Rust 服务实例。这个 GRU 是共享的普通 WGT 定义：输入与 hidden 分别具名，tuple 返回下一 hidden，reset-after-projection 公式使用 std/recurrent。参数按六个门投影和六个 bias 独立声明，当前零初始化、未经训练；不自动加载或假定已有权重。也可以明确 backend="torch" 使用同一源码参与宿主 autograd。

```python
from lightnest.architecture.native import load_recurrent

# 先运行同版本 wgtime，或通过 endpoint= / WGTIME_ENDPOINT 指定已有服务。
with load_recurrent(3, 2, optimizer="adam") as model:
    hidden, = model([0.2, -0.3, 0.1], [0.0, 0.0])
    loss = model.train_step([[0.2, -0.3, 0.1, 0.0, 0.0]], [[0.0, 0.0]], rate=0.001)
```

`tests/architecture/test_native_model.py` 将 PyTorch GRUCell 的 reset/update/new 权重及 bias 显式分拆给共享定义，对照两步 hidden、全部门参数梯度和一次原生 SGD/BPTT 更新。完整 LightNest 应用仍在显式 Torch 后端上验证。当前 Python 原生对象没有统一训练检查点保存接口，不能用参数快照冒充优化器检查点。

入口移动前的源码摘要保存在 `previous-layout-source-hashes.json`，只用来识别已有检查点的精确来源；未知摘要仍拒绝。wheel 包含各目录的 model.wgt，包括共享 GRU。项目继续显式依赖 weightlang[torch]，而不是依赖默认原生 SDK 间接拉入 PyTorch。

本次接口验收对应 WeightLang 源码提交 `0d0bdb9`，需配套构建该版本编译器及 wgtime；已发布包的版本号 0.5.4 本身不保证包含这些源码接口。

## 复杂网络原生入口

`lightnest.architecture.native` 现在提供默认 Rust/wgtime 的 `load_generator`、`load_hnet_generator`、`load_notebook`、`load_attention` 和 `load_hierarchical_recurrent`，完整说明见 [原生复杂模型](native-complex-models.md)。这些入口执行真实原生模型，显式 `backend="torch"` 保留原始 Torch 宿主；原有训练脚本和应用类的兼容入口仍为显式 Torch。不能把新增原生网络入口当作整个 RGBPolicy/Any2Any/X86World 应用已整体迁移完成。
