# 论文实验主线说明（中文整理）

这份说明用于后续撰写论文时快速回顾：哪些实验应该保留、各实验承担什么角色、审计模块到底证明了什么，以及联邦学习范式在本文中起到了什么作用。

## 1. 数据集定位

### 1.1 MNIST 是当前论文主线数据集

- 原因：训练稳定、语义评估器准确率较高、联邦与中心化对比已经完整跑通。
- 当前最适合写入正文的主实验，仍然是 `MNIST federated secure baseline` 对比 `MNIST centralized full-train upper bound`。
- 已经保留的核心对比页在：
  - `outputs/mnist_b40x5_fed_vs_central_fulltrain_best/comparison_samples.png`
  - `outputs/mnist_b40x5_fed_vs_central_fulltrain_best/summary.csv`
  - `outputs/mnist_b40x5_fed_vs_central_fulltrain_best/paper_report.md`

### 1.2 STL10 是高分辨率扩展实验，不替代 MNIST 主线

- STL10 的意义在于说明本框架可以扩展到 `96x96` 自然图像。
- 但目前 STL10 的采样质量仍明显弱于 MNIST，更适合作为“高分辨率探索性结果”和“后续工作方向”，不适合取代 MNIST 成为当前论文主结果。
- 因此论文结构上更建议：
  - `MNIST` 放在主实验部分
  - `STL10` 放在扩展实验、附加实验或局限与未来工作部分

## 2. 当前最有代表性的保留实验

### 2.1 联邦安全训练基线

- 目录：`outputs/mnist_b40x5_decoder_x0_semantic100_run`
- 角色：`federated secure baseline`
- 价值：
  - 展示 non-IID 客户端条件下扩散模型仍可训练；
  - 展示 stego 载荷传输与审计验证可以端到端接入联邦训练流程；
  - 作为与 centralized 结果比较时的“分布式训练参考点”。

### 2.2 中心化全参数上界

- 目录：`outputs/mnist_b40x5_decoder_x0_centralized_fulltrain150_v2_run`
- 角色：`centralized upper bound`
- 价值：
  - 在相同主干架构下，给出集中式训练可达到的更高采样质量；
  - 用来量化联邦设定下的质量差距来自哪里；
  - 作为论文中“联邦代价”最清晰的参照。

### 2.3 主比较页面

- 目录：`outputs/mnist_b40x5_fed_vs_central_fulltrain_best`
- 角色：`paper-ready comparison bundle`
- 价值：
  - 一张图配一张表即可支撑论文主对比实验；
  - 适合直接放入正文或稍作修改后放进图表部分。

### 2.4 高分辨率探索实验

- 目录：`outputs/stl10_b32_centralized_fulltrain_run`
- 角色：`high-resolution exploratory baseline`
- 价值：
  - 证明框架可以扩展到更高像素图像；
  - 说明“高分辨率生成质量仍有明显提升空间”。

## 3. 关于审计模块，论文里应该怎么写？

当前审计模块最稳妥、最准确的描述，不应写成“完成了完整安全证明”，而应写成：

- 它验证了客户端训练更新在压缩、异或混淆、隐写嵌入、服务端恢复之后，是否能被无损还原；
- 它提供了逐轮、逐客户端、逐载荷的可追溯记录；
- 它证明的是“传输完整性与审计可追溯性”，不是对强攻击模型下的形式化安全性证明。

## 4. 联邦学习范式在本文中的作用

联邦学习在本文中不是背景设定，而是实验设计本身的重要组成部分。

- 数据不再集中存放，而是分散在客户端；
- 模型不能直接访问原始全量数据；
- 每轮只能通过客户端本地训练后的更新来推进全局模型；
- 因而 non-IID、通信开销、聚合稳定性和安全传输都成为真实问题。

## 5. 当前论文更清晰的叙事顺序

1. `MNIST federated secure baseline`
2. `MNIST centralized upper bound`
3. `MNIST federated vs centralized comparison`
4. `Audit integrity analysis`
5. `Role of federated learning and communication cost`
6. `STL10 high-resolution exploratory extension`

这样主线会更清楚：先讲方法在联邦设定下是否成立，再讲与中心化上界相比差多少，最后再讲高分辨率扩展与当前局限。
