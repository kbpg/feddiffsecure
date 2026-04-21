# 审计层当前真正证明了什么？

当前审计层验证的是：每个客户端可训练更新在安全封装、恢复和重新解析之后，是否保持端到端完整一致。

## 当前原型记录了哪些字段？

- client id
- round id
- 本地样本数
- 平均本地训练 loss
- 原始载荷大小
- 压缩后载荷大小
- carrier PSNR
- 原始载荷 SHA-256
- 恢复载荷 SHA-256
- 验证状态

因此，当前审计回答的问题是：

- “离开客户端的训练更新，在隐写封装和恢复之后有没有被破坏？”

## 它暂时没有证明什么？

当前原型没有直接给出更强的安全结论，例如：

- 面向自适应攻击者的鲁棒性证明
- 面向强对手的机密性证明
- 与安全聚合协议等价的形式化隐私保证

因此，在论文里更稳妥的写法是：

- 当前审计层验证了联邦更新在传输链路中的端到端完整性。
- 当前原型展示的是传输校验能力，而不是完整形式化安全证明。

## 代表性结果应该怎么写？

代表性联邦 MNIST 运行目录为：

- `outputs/mnist_b40x5_decoder_x0_semantic100_run`

其审计汇总结果为：

- `records = 400`
- `verified = 400`
- `failed = 0`

这意味着：

- 共完成 `100` 个联邦轮次
- 每轮选择 `4` 个客户端
- 总计传输了 `400` 个可训练更新载荷
- 本次代表性运行中的完整性验证成功率达到 `100%`

## 这一部分在论文里最适合承担什么角色？

- 它证明当前系统已经不只是在“做联邦训练”，而是在“做可审计的联邦训练”。
- 它把安全传输问题从抽象说法落到可记录、可复验、可统计的工程链路上。
- 它为论文中的“系统实现工作量”提供了清晰、可量化的证据。

## 建议直接写进论文的中文表述

- “本文的审计机制验证了联邦训练中可训练更新在隐写封装与恢复之后的端到端完整性。”
- “代表性 MNIST 联邦运行共记录 400 条更新载荷，全部通过 SHA-256 一致性校验，表明原型系统在当前链路下能够稳定完成无损恢复。”
- “需要强调的是，本文原型主要证明了传输完整性，而非对强攻击者模型下机密性的完整形式化证明。”

## 相关文献出处

- [McMahan et al., 2017, Communication-Efficient Learning of Deep Networks from Decentralized Data](https://proceedings.mlr.press/v54/mcmahan17a.html)
- [Bonawitz et al., Practical Secure Aggregation for Federated Learning on User-Held Data](https://research.google/pubs/practical-secure-aggregation-for-federated-learning-on-user-held-data/)
- [Kairouz et al., Advances and Open Problems in Federated Learning](https://arxiv.org/abs/1912.04977)
