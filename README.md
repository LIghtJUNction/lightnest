# LightNest

受限 RV32 程序修改研究项目：读取请求及机器码，分析依赖，生成逐字节修改动作，并用有界解释器测试结果。**不是通用语言模型，也没有证明能安全修改真实大型软件。**

## 当前版本与验收状态

- **最新研究源码：Hybrid v1（2026-10-19）**，包含 StateLink、监督训练的 THINK/EMIT 切换及策略梯度实验。
- **当前稳定默认路径：C64（固定 64 轮）**。Hybrid、RGB 蓝区路由和扩量候选都未通过“零新增误改＋旧成功逐条保持”的晋升条件。
- **RGB v2** 提供绿区摘要、蓝区候选、红区验证与可选执行后端；**template_io v1** 提供安全精确缓存、结构缓存和文件差量原型。
- 不默认加载或运行研究模型；没有提交模型权重。

## 仓库范围

**只提交最新源码、必要的协议配置、测试和研究文档。** 模型检查点、训练与评估数据、原始结果、二进制候选、旧版源码压缩包和缓存全部留在 Google 云盘，不进 Git 历史。

```text
hybrid/                         当前 StateLink + 模式策略研究源码
baseline/LightNest_RGB_v2/      Hybrid 依赖的当前 RGB/C64 源码（不含权重、样本）
tools/template_io/              模板、精确缓存、静态关系缓存、差量工具
tests/                          无模型权重的源码检查
docs/                           探索经验、实验报告、运行说明、下一轮计划
```

仓库中的 `baseline/LightNest_RGB_v2/runtime/base/lightnest/work/.../arch/breakthrough_model.py` 是当前 C64 加载必需的固定架构定义，不是旧版候选权重。

## 检查源码（不需要下载模型）

推荐 Python 3.13；测试环境使用 PyTorch 2.10.0+cpu。

```bash
python -m pip install -r requirements.txt
python tools/check_source_tree.py
python -m unittest discover -s tests -t . -v
PYTHONPATH=tools/template_io python -m unittest discover -s tools/template_io -p 'test_*.py' -v
```

有权重和数据后才能运行 `python -m hybrid.test_mechanisms` 及 RGB 的全部模型测试，或执行实际推断、训练、回归。缺少外部文件时，测试报错**不等于源码回归失败**。来源和取回方法见 [ARTIFACTS](docs/ARTIFACTS.md)。

## 从哪里阅读

- [探索经验](docs/LESSONS.md) · [架构边界](docs/ARCHITECTURE.md)
- [下一轮实验方案](docs/NEXT_ITERATION.md) · [归档与数据](docs/ARTIFACTS.md)
- [实验报告](docs/experiments/) · [来源说明](docs/SOURCE_PROVENANCE.md) · [版权说明](NOTICE.md)

实验中的不同数据集、不同模型、不同指标不可直接拼成同一条进步曲线。输出只是候选补丁，不在宿主机运行，不覆盖输入文件。
