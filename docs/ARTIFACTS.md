# 外部研究档案（Git 不存放这些文件）

| 档案 | 用途 | Google 云盘 |
|---|---|---|
| Hybrid v1 原始包 | 最新源码、训练权重、原始测试、执行轨迹、失败与冻结记录 | [Hybrid v1 ZIP](https://drive.google.com/file/d/1qRqHhF-NFh08VrAnQNjqbsVhF-xoOV1-/view) |
| RGB v2 原始包 | Hybrid 的基础组件/模型与风险对照 | [RGB v2 ZIP](https://drive.google.com/file/d/1VM44PvpLYPZuf5powtSY7Vpy__xGBlB2/view) |
| Template I/O v1 原始包 | 模板、缓存、差量实验输出与工具代码 | [Template I/O ZIP](https://drive.google.com/file/d/1Q9xTyi9XNthRLg819-BWh2kvVZw_he2U/view) |
| 10k 扩量实验 | 训练数据、候选权重、失败案例 | [10k ZIP](https://drive.google.com/file/d/10F7YmrY5VUqo3DduJzyWHkQl5gfrY2Dt/view) |

本仓库源代码从这些包中筛选；完整模型测试需从相应包恢复路径 `baseline/LightNest_RGB_v2/models/`、`models/`、`data/` 及必要清单。**不要把其余 ZIP 内容整体复制进 Git 仓库。**

冻结模型由权重文件而非源码重建；缺权重的 Git 克隆不能直接进行已训练模型推断。若要重新运行已消费数据的实验，只能称复现，不能称新预留测试。运行清单与具体命令见 [Hybrid 复现记录](reproduce/hybrid-v1.md) 和 [RGB 复现记录](reproduce/rgb-v2.md)；路径以原完整归档为准。
