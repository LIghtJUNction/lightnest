# 二进制语义与自主文本生成

最终目标仍是：内嵌权重的 LightNest 原生 ELF，能正确解释未经训练的自身。下面实验
为它补充可学习的二进制输入和语言输出，不把源码说明匹配当成最终验收。

## 输入与模型

输入来自冻结外部语料 `semantic-v4`。真实外部函数去掉文档字符串和注释后，以固定
文件名 `<artifact>` 编译成完整 Python `.pyc`，包含文件头和完整代码对象。源码和
字节码共享一个字节 BPE、Transformer、Red、神经笔记和语言解码器。推理不解析 AST、
不反编译、不查词汇基线，也不把观察事实交给另一个语言模型改写。

这些 `.pyc` 是外部真实源码的编译产物，不是原地收集的 Linux x86-64 原生程序。
字节码验收不能证明已经理解 ELF 的机器码、加载器或内嵌权重。模块上下文、文档
说明的真实性和完整自解释 ELF 都还需要后续独立验证。

分词器只用训练划分的源码、字节码和说明学习合并规则，初始字母表覆盖全部 256 个
字节。没有未知字节、大小写归一化、字符串清洗或截断；编码后逐次检查能够还原原始
字节。训练集继续排除 LightNest。验证、测试和自身划分只做编译、无损编码和容量
检查，不更新分词或神经权重。

局部注意力分窗处理所有 token，并保留字节位置；各窗不丢弃末尾。全部证据进入有
填充掩码的神经笔记。可学习的门把笔记送回逐 token 状态，保留局部细节。当前 Red
固定运行 4 步；本轮不声称已经训练好自适应思考深度。

候选说明通过学习到的逐 token 匹配评分。文本由 GRU 和对输入的注意力自回归生成，
并学习何时复制输入片段、何时输出词表 token。生成时只传入原始字节，不传入候选
说明或参考答案；EOS、非法特殊 token、UTF-8 有效性和原始输出字节分别记录。

输入上限为 1 MiB，训练语料中的完整 `.pyc` 上限为 4 KiB；长输入的处理能力不等于
长输入语义已经验证。原先的自解释 ELF 未被替换，本轮也没有把它放入训练集。

实现依赖 Hugging Face Tokenizers 的 BPE 训练接口和完整初始字母表参数，使用稳定版
`0.23.3`，不使用 1.0 预发布接口。参见[官方训练参数](https://huggingface.co/docs/tokenizers/main/en/api/trainers)
和[官方发布记录](https://github.com/huggingface/tokenizers/releases/tag/v0.23.3)。

## 冻结与验收

实验目录：`/home/lightjunction/Documents/LightNest-artifacts/2026-10-26/binary-language-v3/`。
`protocol.json` 绑定源码、依赖、数据、字节码版本、分词器和负例表。原始字节码按偏移
保存在单一归档，避免重复写入大量小文件。无法独立编译、超容量和重复的输入单列排除。
不会执行收集来的第三方函数。

训练同时优化双向对比损失和 `0.25 ×` 语言负对数似然。每批一半为随机锚点，一半为
只从训练说明中找出的相似负例。候选只按两个公开验证视图中最弱的词汇基线差值选择。

开启测试和 LightNest 神经盲测前必须同时满足：

1. 源码和字节码验证命中率都高于各自词汇基线，也高于同一分词器配随机神经权重的结果。
2. 32 对未加入训练的同词计数、相反行为程序中，两个视图的成对正确率都不低于 75%。
3. 字节码对应正确说明的语言损失，比错配输入低至少 0.1 nat，排除只学到语言先验的情况。

行为对只包含本地编写的固定算术与比较表达式，其结果通过实际执行编译后的字节码
验证。这个执行器不接收模型生成或第三方代码。规则只构造测试答案，不参与模型推理。

未满足门槛时保持 `not_promoted`，不查询测试/自身划分，保留首次固定验证样本的原始
自由生成文本。即使这些阶段通过，也不自动宣称原生 ELF 自理解目标完成。

## 命令

```bash
uv sync --locked
uv run --locked python archive/research/tools/binary_research.py prepare \
  --out /path/outside/repository \
  --parent /path/to/semantic-v4
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 RAYON_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false \
  uv run --locked python archive/research/tools/binary_research.py train \
  --out /path/outside/repository --steps 4096
uv run --locked python archive/research/tools/binary_research.py evaluate \
  --out /path/outside/repository --checkpoint /path/to/selected-candidate.pt
# `explain-artifact` was an older CLI entry and has been retired;
# the experiment remains under archive/research/.
```

具体候选使用 `selection.json` 的验证集选择。评价入口先复核公开验证和生成条件，
自动阻止未过关候选进入测试集。所有权重、数据、原始输出和日志留在仓库外。

## 停止与云端交接（2026-10-26）

用户要求停止本地工作，训练进程已退出。日志最后记录到 480/4,096 步，磁盘只有
`training-20261027/initial.pt`，它是未训练的初始权重；没有训练后检查点可续跑。
下次需要从头训练，不把初始权重冒充已训练模型。

冻结语料包含 41,922 个训练函数、173 个项目，源码与完整字节码两种视图。词表只从
均衡的 4,096 个训练函数学习；完整语料的哈希、文档去除、固定文件名及训练排除
LightNest 检查见 `data-audit.json`。215 项本地测试通过，没有开启神经测试/自身盲测。

v1 的完整词表统计因换页开销终止，记录保留；v2 完成语料和词表准备；v3 保持全部
输入不变，消除评分中的填充计算后重新冻结。初始神经权重的源码/字节码验证命中率
分别为 5.86%/0.78%，词汇基线为 49.41%/49.22%。本轮没有最终训练结论，也没有
达到原生 ELF 自理解目标。

云端接手时从上述实验目录复制 `protocol.json` 及其 `files` 字段列出的文件到**新的
数据目录**，源码使用本次提交。不要把已有 `training-20261027/` 当成可恢复训练状态。
当前锁定 CPU 版 Torch，训练器仍是 CPU 实现；GPU 依赖、设备适配及新协议冻结留到
下次处理，本次没有启动云端任务。

另有只读容器工具 `archive/research/tools/inspect_packed.py`。实际 7,423,545,524 字节自解释 ELF 中，
读取 1,154,143 字节便提取了 15,000 字节的 Python 主入口，未执行它或载入 GGUF。
输出只给原始 marshal 内容添加标准 `.pyc` 文件头，不改写程序体。提取件明确标记为
`unsanitized`，仍含原始说明文字，不能直接拿来宣称无泄漏的神经自身理解。证据在
`self-entry-inventory.json`，原有自解释 ELF 未替换。
