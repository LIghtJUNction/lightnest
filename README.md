# LightNest

小型神经程序处理研究：读取受限 RV32 机器码，追踪数据来源，提出字节修改，并用独立程序检查结果。

**当前源码：Hybrid v1，基于 RGB v2 和 C64。默认使用原 C64 路径。Hybrid、蓝区路由和扩量权重尚未通过稳定模型晋升，不默认启用。** 这不是通用语言模型，也不能可靠开发或修改大型真实软件。

## 仓库范围

这里只保留最新运行源码、测试、数据格式定义与探索文档。不提交模型权重、训练/测试数据、二进制候选、实验输出、缓存或旧 Git 历史。既有归档留在原存储位置；外部文件说明见 [ARTIFACTS](docs/ARTIFACTS.md)。

`baseline/LightNest_RGB_v2/` 是 Hybrid v1 当前使用的依赖源码，不是另一份可选发布。目录暂时保持原样，以免破坏现有导入和检查点格式；相同组件不再另存 v0/v1 快照。

## 代码导航

| 路径 | 内容 |
|---|---|
| `hybrid/` | 内部状态桥、THINK/EMIT 切换、监督和策略训练、评价与候选接口 |
| `baseline/LightNest_RGB_v2/rgb/` | 绿区摘要、蓝区动作、红区续算、风险判断、缓存与编译循环 |
| `baseline/LightNest_RGB_v2/runtime/` | C64、共享循环、来源关系解析、受限 RV32 执行支持 |
| `baseline/LightNest_RGB_v2/lightnest_data/` | 数据格式、分片、构造组隔离、生成与检查工具 |
| `baseline/LightNest_RGB_v2/study/`、`experiment/` | 当前 RGB 训练、校准、独立评价及成本对照 |
| `tools/template_io/` | 底稿引用、差量输出、结构/精确请求缓存的独立工具 |
| `tests/` | 不依赖外部权重的源码测试 |
| `docs/` | 已测结果、负结果、来源说明、下一轮方案 |

## 本地检查

测量环境为 Python 3.13.5、PyTorch 2.10.0+cpu。先创建虚拟环境，然后：

```bash
python -m pip install --index-url https://download.pytorch.org/whl/cpu torch==2.10.0
python -m pip install -r requirements.txt
python tools/check_source_tree.py
python -m unittest discover -s tests -t . -v
PYTHONPATH=baseline/LightNest_RGB_v2 python -m unittest discover -s baseline/LightNest_RGB_v2/tests -p test_sequence.py -v
PYTHONPATH=tools/template_io python -m unittest discover -s tools/template_io -p 'test_*.py' -v
python -m hybrid.predict --help
```

这些检查只验证机制，不代表模型修复正确率。推断、历史重放和正式训练另外需要外部权重/数据，见 [运行说明](docs/ARTIFACTS.md)。不要用随机初始化结果冒充已训练模型。

## 当前结论与下一步

阅读 [探索经验](docs/LESSONS.md)、[架构边界](docs/ARCHITECTURE.md)、[下一轮方案](docs/NEXT_ITERATION.md)。原始文字报告在 [实验记录](docs/experiments/)。报告中的不同测试集不可拼成同一条增长曲线，原速度成绩也没有在这次源码提交中重新测量。

新实验应先固定起点、预算、数据划分和评价规则，再运行。测试答案不进入预测接口；已公开终测只作为回归集。模型输出只写入新候选，不在宿主机执行，不覆盖原文件。

来源与复制校验见 [SOURCE_PROVENANCE](docs/SOURCE_PROVENANCE.md)。授权边界见 [NOTICE](NOTICE.md)。
