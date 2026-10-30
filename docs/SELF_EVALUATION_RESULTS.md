# LightNest 自身盲测

训练严格排除了 `/home/lightjunction/Documents/GITHUB/lightnest`。模型使用本机系统文件和
通用 JSON、gzip、ZIP 容器训练，测试阶段才读取 LightNest 源码。

检查点：`any2any-v4b-104738/any2any-104738.pt`，训练 64 步，1,106 个混合样本。

自测读取仓库 251 个文件：

| 类型 | 文件数 | 类型判断 | 字节准确率 |
| --- | ---: | ---: | ---: |
| text | 231 | 100% | 22.83% |
| csv | 18 | 0% | 20.68% |
| json | 2 | 0% | 13.04% |

## 原始模型输出

对 `README.md` 前 256 字节，Any2Any 输出的预测字节解码为：

```text
't  i 7i\\ne�i eeii  \\nii /  di si\\niiiv iiee i vie eeG/ i iifei   i i iti/i iqeisie see'
```

对 `lightnest/architecture/model.py` 前 256 字节，输出为：

```text
'vioq�ifei e o iiiie i v eee e\\ni.ee e7  ii ii  \\n�ie  meei meni  e  veei iie eiUvee'
```

RGB Blue 语言头输出的原始 token 为：

```text
[175, 160, 183, 182, 179, 178, 177, 180, 185, 186, 187, 188, 189, 190, 191, 184,
 176, 174, 173, 172, 171, 170, 169, 168, 167, 166, 165, 164, 163, 162, 161, 181]
```

## 结论

当前模型没有正确理解 LightNest，也没有产生可用的自然语言解释。它能在部分版本中识别
源码是文本，但这不等于语义理解；字节输出是乱码，CSV/JSON 泛化失败，Blue 语言头没有
经过语言训练。该目标尚未完成。后续必须训练通用代码/文本任务并重新盲测，不能把仓库加入
训练集或用规则直接生成 LightNest 描述。

## v5 泛化复测

加入通用 Python 结构摘要目标后，使用未包含 LightNest 的 1,116 个混合样本训练 256 步。
LightNest 盲测结果：252 个文件，text 类型判断 100%，CSV 类型判断 100%，text 字节准确率
75.15%，Python 结构摘要平均绝对误差 0.132。JSON 类型判断仍为 0%。这证明通用结构目标
带来改善，但 JSON 语义和自然语言解释仍未通过，目标继续保持未完成。

## v6 分层盲测

训练 1,116 个通用混合样本 1,024 步后，使用分层观察器编码 LightNest 全部文件，再由 Red
汇总。252 个文件的 text 类型判断为 100%，CSV 为 100%，text 字节准确率为 98.73%，
Python 结构摘要误差为 0.101。JSON 类型判断为 50%。这些数字只证明格式、局部字节和
粗粒度结构预测，仍没有生成可读解释，也没有证明模型理解模块之间的语义关系，因此目标
仍未完成。

## 通用代码解释器盲测

使用系统 Python 示例和库源码训练 64 步，没有读取 LightNest。对 `model.py`、`any2any.py`
和 `evaluate_self.py` 的输出均接近：

```text
pyttooosssssssssssssssssssssssssssssssssssss...
```

这不是源码解释，只是字符先验。解释器实验失败，目标仍未完成。

## 2026-10-26 外部代码与说明对齐

新增通用对比训练，不用字节复制准确率充当语义指标。4,096 步后，在外部项目的 512 项
公开验证中命中 55 项（10.74%），低于纯词汇匹配的 49.41%。因此没有晋升，也没有
查询新冻结的 LightNest 候选说明集合。完整协议与失败证据见
[代码与说明对齐](SEMANTIC_ALIGNMENT.md)；自身理解目标继续保持未完成。
