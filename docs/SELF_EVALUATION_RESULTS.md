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

对 `lightnest_rgb/model.py` 前 256 字节，输出为：

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
