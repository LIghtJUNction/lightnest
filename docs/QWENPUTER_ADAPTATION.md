# Qwenputer 思路在 LightNest 中的落点

固定计算规则负责精确读写和算术，RGB 负责提出程序、解释反馈和选择编辑。
这是可选 P0 原语，不是完整 CPU、ELF 执行器或训练出来的新能力。
本轮实验、复现命令及未完成检查见 [本地验证报告](QWENPUTER_P0_EXPERIMENT.md)。

## 来源与范围

LightNest 基线：`b457347b3e7b3cd9a6fbcefab34e99927e103025`。
参考 [Qwenputer 作者文章](https://divagr.com/blog/qwenputer) 和
[上游仓库](https://github.com/divagr18/qwenputer)。前一轮记录的上游提交为
`ca13310e2f9f788d1dd3feeea8213c1b0c23913c`；机制参考文件为
`experimental/riscv/docs/rv32.md` 和 `experimental/riscv/chipformer/rv/chat.py`。
本实现独立编写，不复制上游代码，不添加项目许可证，不移植其权重。
本轮不复现上游的性能或模型能力宣传，也不把 RV32 等同于 Linux x86-64。

## 精确状态与思考状态分开

现有 GRU 隐藏状态和 NeuralNotebook 保留。另建显式机器状态，避免让语义摘要冒充
寄存器或 RAM。一个状态可以近似表达“哪里值得改”，另一个必须精确保存某个字节。
本补丁没有修改 `RGBPolicy`、`RGBEditor`、训练入口或原生隔离执行器。

`ExactBitRAM` 使用独热地址 k、二进制值 v、写使能 w 和显式状态 M：

```text
old = k^T M
M_next = M + k [w (v - old)]^T
```

地址只选中第 i 行时，w=1 等价于 `M[i] = v`，w=0 只读取。返回写前值，其他行不变。
遗忘门固定为 1，不用“接近 1”的概率。实现拒绝软地址、多地址、非二进制状态、NaN、
形状和 dtype 不匹配；不原地修改输入，也不在模块中偷偷保存运行状态。

这是稠密正确性基线，不是高效 RAM。64 个 32 位字只有 256 字节逻辑容量；逐位表示在
FP32 下占 8192 字节，BF16 下占 4096 字节。读写和校验仍扫描矩阵。

## 算术写成固定规则，不训练进位

`FixedWordAdder` 把无符号整数拆成位，用固定线性层和 ReLU 逐位传递进位：

```text
s = a + b + carry_in          # s 只能是 0、1、2、3
carry_out = relu(s-1) - relu(s-2)
sum_bit = s - 2*carry_out
```

小整数中间量避开了把整个 32 位整数塞进 BF16/FP32 的精度问题。权重是 buffer，
不是可训练参数。固定 Python 循环展开电路深度，不根据操作数调用整数加法计算答案。
输入编码、输出解码和测试参考仍使用宿主整数；不能把这些边界说成已消除全部宿主工作。

## 接入顺序与验收边界

先统计现有 x86-64 课程真正使用的指令，冻结支持子集、位宽、标志、内存和系统调用语义。
原始指令到语义表示的确定性译码不能修复候选，也不能重新编译另一程序替代候选。
未覆盖语义返回 `unsupported`，预算耗尽返回 `budget`，不能冒充执行成功。

机器状态至少绑定 `artifact_sha256 + input/args_hash + machine_spec_hash + execution_policy_hash`。
接受编辑后默认重建状态，避免用旧结果验证新文件。证据区分
`learned_prediction`、`compiled_core` 和 `native_sandbox`，最终验收仍走现有沙箱。

固定核心后再训练何时执行、怎样使用反馈、改哪里和何时停止。对照原始 RGB、外部执行器、
内部固定电路时，统一程序级拆分、候选数、步数、内存和端到端预算；分别报告覆盖率、
原生任务成功率、旧成功回退和开销。隐藏评测不得进入奖励或自举筛选。

当前没有取指、译码、分支、ELF 加载器、GPU 验证或 RGB 训练。
原语正确不等于模型生成能力提高，更不等于 bootstrap 或 RSI 已实现。
