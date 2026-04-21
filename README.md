# FedDiffSecure Minimal Prototype

这是一个**能直接跑起来**的最小原型项目，目标是先把你的毕业设计主线搭起来：

- **联邦学习范式**：单机模拟多个客户端，本地训练、服务端聚合
- **扩散模型**：使用一个轻量 Tiny U-Net 作为 DDPM 噪声预测器
- **LoRA 轻量训练**：冻结基础卷积，只训练 LoRA 分支，减少上传量
- **安全审计**：每个客户端更新都会记录 SHA256、载荷大小、校验结果
- **参数载荷图片化传输**：客户端把 LoRA 参数序列化后，加密/XOR 后嵌入 PNG，再由服务端解码恢复
- **公开数据集自动下载**：默认使用 `torchvision.datasets.FashionMNIST(download=True)`，会自动联网下载；TorchVision 数据集对象在 `download=True` 时会从互联网下载并放到 root 目录下。citeturn774120search0turn774120search2

> 这不是你最终论文的“大模型完整版”，而是一个**为了先跑通主流程**而设计的最小实现。
> 你后续可以把 Tiny U-Net 替换成 Stable Diffusion / Latent Diffusion 的 UNet，把当前 PNG 载荷嵌入模块替换成更正式的可逆隐写模块。

---

## 1. 项目结构

```text
feddiffsecure_minimal/
├─ configs/
│  ├─ smoke.yaml
│  ├─ quick.yaml
│  ├─ stabilization.yaml
│  └─ default.yaml
├─ feddiffsecure/
│  ├─ audit.py
│  ├─ client.py
│  ├─ data.py
│  ├─ diffusion.py
│  ├─ lora.py
│  ├─ model.py
│  ├─ server.py
│  ├─ stego.py
│  └─ utils.py
├─ run_federated.py
├─ requirements.txt
└─ README.md
```

---

## 2. 安装依赖

建议使用 Python 3.10+。

```bash
pip install -r requirements.txt
```

如果你还没有安装 PyTorch，可以先按你自己的 CUDA / CPU 环境安装 PyTorch，再执行上面的命令。

---

## 3. 直接运行

### 快速版（先确认环境和流程能跑通）

```bash
python run_federated.py --config configs/quick.yaml
```

这个配置是**极小规模 smoke test**，重点是先验证：数据下载 → 客户端本地训练 → 参数打包成 PNG → 服务端解码校验 → 聚合，这条链路能完整跑通。

### 默认版

```bash
python run_federated.py --config configs/default.yaml
```

---

## 4. 运行后会得到什么

输出目录默认在配置文件里的 `system.output_dir`，里面会有：

- `client_splits.json`：每个客户端的数据分布
- `audit_log.jsonl`：每轮每个客户端上传参数的审计记录
- `audit_summary.json`：审计汇总
- `carriers/`：客户端上传的参数载荷 PNG
- `samples/`：如果你在配置里打开采样开关，会保存每轮或最终的生成样本图
- `global_model_final.pt`：最终模型权重
- `run_summary.json`：本次运行配置摘要

---

## 5. 核心流程

### 5.1 联邦训练

1. 服务端初始化全局扩散模型
2. 将训练集按 Dirichlet 分布切成多个非 IID 客户端
3. 每轮把全局模型发给客户端
4. 客户端只更新 LoRA 参数
5. 服务端接收各客户端 LoRA 更新并做加权平均

### 5.2 参数“图片化”传输

客户端本轮训练结束后，会：

1. 把 LoRA 参数 `state_dict` 序列化
2. 用简单密钥流做 XOR 加密
3. 把密文 payload 嵌入 PNG 图像的最低位
4. 上传该 PNG 给服务端
5. 服务端从 PNG 中提取 payload，解密、反序列化、校验 SHA256
6. 只有校验通过的更新才进入聚合

这个模块是**最小可运行版**，目的是先把“参数不裸传、具备审计记录”的主流程跑通。

---

## 6. 你后面可以怎么扩展

### 扩散模型替换
- 把 `feddiffsecure/model.py` 的 Tiny U-Net 替换成更大的 UNet
- 再进一步换成 `diffusers` 的 Stable Diffusion UNet / VAE / Scheduler

### 联邦算法替换
- 当前是 LoRA 参数的 FedAvg
- 后面可以替换成 FedProx、SCAFFOLD、MOON、FedAvgM
- 如果你想贴近你毕设里的“异构数据”方向，还可以引入扩散特征引导项

### 安全方向替换
- 当前是 toy 版 PNG-LSB 载荷封装
- 后面可以替换成：
  - 可逆隐写网络
  - 参数水印
  - 签名验签
  - 差分隐私 / 安全聚合

### 数据集替换
- 现在默认是 Fashion-MNIST
- 代码里已经支持切到 MNIST
- 如果你后面要扩到 CIFAR10 / MedMNIST，我建议在这个骨架上继续加，不要一开始就把框架做太重

---

## 7. 和你毕设主题的对应关系

这个最小原型是按你当前课题主线收敛出来的：你的题目和开题材料明确是把**联邦学习、扩散模型、可逆隐写/安全传输、责任审计**放到一个统一框架里，并强调 LoRA 这类轻量参数更新与“参数即水印/可追溯”的思路。fileciteturn1file3 你的开题报告也明确写到：各节点在本地训练扩散模型，将参数更新嵌入生成载体后上传，服务器解码恢复后再做聚合，并希望同时兼顾隐私、安全传输、内容可追溯和非 IID 下的稳定性。fileciteturn1file4 另外，你的答辩记录里把“责任审计能力”直接解释为：围绕联邦节点上传参数后的安全确认与后续问题追踪。fileciteturn1file1

所以这份代码并不是随便写的 demo，而是**针对你的毕设方向做的第一版可运行骨架**。

---

## 8. 推荐的下一步

你可以按这个顺序继续往上迭代：

1. 先把 `quick.yaml` 跑通
2. 再跑 `stabilization.yaml`，优先验证聚合稳定性
3. 再把数据集切到 CIFAR10
4. 接着把扩散模型换成更强的 UNet
5. 然后把当前 toy stego 替换成更正式的可逆隐写模块
6. 最后再补 FID / PSNR / 通信开销 / 参数恢复率等实验指标

如果你要继续，我下一步最适合直接给你做的是：**把这个最小原型升级成“更贴近论文表述”的版本，加入 FedAvg/FedProx 对比、FID/PSNR 统计和更规范的实验脚本。**

---

## 9. Experiment Presets

- `configs/smoke.yaml`: one-round pipeline smoke test with `fake_data`.
- `configs/quick.yaml`: the current quick Fashion-MNIST baseline.
- `configs/stabilization.yaml`: a stability-first run before scaling the U-Net. It uses all 4 clients each round, lowers `lr` to `5e-4`, and reduces `max_steps_per_epoch` to `25`.
- `configs/stabilization_best.yaml`: the current sweep-picked stable baseline with `lr=5e-4` and `max_steps_per_epoch=20`.
- `configs/mnist_large_unet.yaml`: a larger multi-level UNet on MNIST, paired with the current stable federated training settings.

```bash
python run_federated.py --config configs/stabilization.yaml
```

For a small stability sweep around that setup:

```bash
python sweep_stabilization.py
```
