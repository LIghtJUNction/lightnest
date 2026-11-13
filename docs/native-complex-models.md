# 原生复杂模型

这些接口要求配套的当前 WeightLang 源码编译器和 wgtime，未随 PyPI 0.5.4 重新发布。默认创建 `wgt.Model` 子类，原生执行不导入 Torch，不执行 Python 模型数学或桥接宿主 autograd。Python 负责离散数据准备、位置/调度和显式状态传递。

| 入口 | 原生计算 | 状态/限制 |
| --- | --- | --- |
| `load_rgb_policy(image,command,frames=(),config=None,depth=None,backend="native",**options)` | 完整 RGB 神经图：观察编码、三模式 attention、notebook、共享思考循环、blue/red、风险/语言/预算头 | 默认 width64、depth16；`specialize` 显式切换离散布局与深度，保留权重和 Adam |
| `load_generator(width=128,capacity=2048,backend="native",**options)` | 完整 ELFGenerator：两层 GRU、共享 red 循环、LayerNorm、copy attention、bits/blue/stop 头 | 单序列逐位置；red_depth=0..16；两个显式 hidden 向量。 |
| `load_hnet_generator(width=128,capacity=2048,backend="native",**options)` | 完整 HNetELFGenerator：encoder/main/decoder GRU、动态边界、EMA、confidence STE，以及全部生成头 | chunk_mode=dynamic/fixed/dense；target_ratio 默认4且至少2；五个显式 hidden 向量。 |
| `load_notebook(width=128,slots=32,*,items,backend="native",**options)` | 完整 compress 数学：带 padding 的写入、gate、预算网络、训练时软选择及推理时硬 top-k | 单序列，items 构造时固定，notes 显式；此入口不模拟独立 write/read 服务方法。 |
| `load_attention(width=96,heads=4,landmarks=8,top_k=16,*,items,queries,backend="native",**options)` | 同一图内完成 bank build 和 dense/compressed/routed read，包含共享 query 投影、mask、路由和联合 softmax | items/queries 构造时固定。按原 bank 的 N 个位置保留 detail 选择掩码，不声称紧凑 K 槽的性能；没有跨调用 bank 缓存。 |
| `load_hierarchical_recurrent(input_size,width,backend="native",**options)` | HierarchicalGRU 三种模式和完整显式状态/梯度 | 默认dynamic、target_ratio=4；候选 main 仍在每个位置计算，不声称稀疏计算加速。 |

`backend="torch"` 是显式兼容路径，返回已有原始应用模块，数据/训练继续使用其原 API；不把它冒称为原生。原生参数按照 `.wgt` 声明零初始化（归一化 gamma 为1），属于未经训练的模型。`import_state_dict(existing.state_dict())` 显式完整导入已有模型，按 reset/update/new 拆分门参数，检查所有名称和 shape。成功替换清空旧优化器，随后需要 `model.set_optimizer("sgd")` 或 `"adam"`。不能把这次导入当作优化器检查点恢复。

```python
from lightnest.architecture.native import load_generator

# 需要先启动配套 wgtime；endpoint 也可由 WGTIME_ENDPOINT 指定。
with load_generator(width=128, capacity=2048) as model:
    # 可先显式 model.import_state_dict(...) 绑定已有权重。
    result = model.step(256, [0, 7, 0, 0, 0, 0, 0, 0, 0, 0], red_depth=2)
    next_result = model.step(23, [0, 7, 0, 0, 0, 0, 0, 0, 0, 0],
                             hidden=result["hidden"], position_offset=1)
```

生成器 condition 为 `[mode, exit_byte, eight_stdout_bytes]`，mode=0..2，其余为0..255的整数。`pack_step` 把此数据转换为原生输入端口；请求位、位置、固定 chunk 边界和 pooling 的系数只来自离散输入布局。所有 embedding、投影、归一化、GRU、门控、attention 和可微状态路径均在 WGT 图中。notebook 的 `compress` 默认 query 用带有效掩码的 evidence 均值；hard 必须为 bool，valid 为至少一项 True 的 bool 列表。attention 的 `attend` 接受 observations、valid、condition、queries 和 mode，返回 output/landmarks/summaries/selected。

构造尺寸仍受 WeightLang 的端口宽度16384、维度绑定上限16384、寄存器元素和服务内存预算限制。宽度/容量检查不代替编译器及服务资源检查。没有隐式 batch、GPU、跨服务自动求导或统一原生 checkpoint 保存。一次 `train_recurrent` 才保留跨位置/notes 的梯度；多次 Python `step` 只做推理式状态传递，不假装保留 Torch 计算图。

原生训练输入可由 pack 方法逐端口展平；调用 `train_recurrent` 时：ELFGenerator state 的 input_offset=75、output_offset=17、state_width=2*width；HNetELFGenerator 对应5*width；HierarchicalGRU 为 input_offset=input_size、output_offset=width、state_width=5*width；notebook 的 notes 为 input_offset=items*width、output_offset=0、state_width=slots*width。supervised_offset/width 决定实际监督输出片段，不能把所有诊断输出误当目标。宿主选择 loss、学习率、优化器及序列边界。

验收位于 `tests/architecture/test_native_model.py`：真实 wgtime 中导入与原实现相同的权重，比较完整输出、隐藏状态、多种深度/模式和原生 SGD/BPTT 后的全部参数；notebook 覆盖 padding、显式/默认 query、软/硬预算；attention 覆盖 N<L 的空池窗口、padding 和三种读取模式。新增内核的无效 mask、count、相等值选择、工作预算和独立 Torch VJP 由 WeightLang 测试覆盖。核心验收尺寸为width=8/16、短序列，attention另覆盖N=7、L=3、K=2的非平凡路由，另验证默认width=128生成器的服务准入和执行；这些验证不是性能报告或所有真实模型规模的验收。

Any2Any、独立 GreenObserver 宿主、结构分类器及 X86World 的整个应用宿主仍未全部改成原生图。RGBPolicy 的观察编码已包含在新的完整原生 RGB 图中；原 RGB 流式应用宿主仍保留。它们继续走已有显式 Torch 入口；这些新增原生网络不是对任意 target torch 源码的自动翻译。尤其 X86/ELF/PTX 模拟器和 Any2Any 的随机 dropout/多模态分支需要独立原生实现及验收。

这些新增普通 WGT 定义也可以由同一共享前端导入显式 Torch 后端，例如 `wgt.from_ir(native.definition, backend="torch", device="cpu")`。这只复用定义并按配方创建新权重，不迁移 native 的当前参数；若要相同实例数值，需要显式复制参数。测试对六个定义分别绑定相同随机权重，验证原生和显式 Torch 的全部逻辑输出；原始 target torch 模型另外作为独立语义 oracle 保留。

配套 WeightLang 源码提交见本次验证 JSON 中的 compiler_commit 字段；所有能力均为源码状态，未重新发布 PyPI/crates.io。

RGB 原生宿主只用标准库准备字节块去重、出现位置、种类、地址编码、命令与离散 pooling 布局；所有参数相关计算在一次 WGT forward 中，Python 不串接中间神经网络结果。返回 `(blue, red)` 两个普通数据字典，不提供原 Torch 宿主的第三个流式 state；begin/advance/thoughts 流式兼容尚未接入。`train()`/`eval()` 控制 notebook 软/硬选择；risk 输入显式 stop_gradient。`pack` 返回16个逻辑输入，输出11个端口按 blue 的 logits/bits/risk/language/budget/context_budget、red 的 logits/bits/risk/budget/context_budget 排列。可全部展平后交给原生 train_step；测试比较全部参数的 SGD 与风险专用目标梯度隔离。

新 RGB WGT 使用零/单位参数配方，不等同于原 Torch 随机初始化。构造时 image/command/frames 只决定布局，不训练；模型仍须显式导入已有权重或训练。布局包含唯一块数、出现次数、UTF-8 命令字节数、image 字节数与 image 块数。布局变化前先调用 `model.specialize(image,command,frames,depth=...)`，然后 pack/forward；同一布局可直接替换内容。重绑定使用已固定的原始 WGT 源码和共享 Rust 前端，服务验证参数 schema 与内存准入，保留同一会话的权重、优化器与 work_limit；失败保留原模型。不是任意大小的动态输入：端口16384元素、静态绑定、服务内存与 tape 限制依然适用。

RGB 默认申请256 MiB服务内存，work_limit=20,000,000；可显式覆盖这两个原生 SDK 参数。预算按指令成本计量，不能把它当毫秒或性能保证。默认尺寸验收使用短输入、真实 depth16 前向/Adam；另在同一 Adam 会话切换输入长度与 depth8 后切回，核对第二次更新。`backend="torch"` 仍显式构造原 RGBPolicy；新普通 WGT 源码另外通过 from_ir 在 Torch 执行全部11个逻辑输出。

`NativeRGBConfig` 默认值为 width=64、chunk_size=64、heads=4、landmarks=4、detail_k=8、blue_depth=1、red_depths=(2,4,8,16)、note_slots=16。整数配置必须为正且不接受 bool；width 同时整除8与 heads，chunk_size 为8..256，note_slots 为2..256。red_depths 严格递增、无重复、最大64，blue_depth 小于最小 red_depth；构造/specialize 的 depth 必须大于 blue_depth 且不超过最大 red_depth。config 接受 NativeRGBConfig、同字段字典或原 RGBConfig。显式 Torch 入口返回原 RGBPolicy，depth 须在该对象的 forward 调用传入。
