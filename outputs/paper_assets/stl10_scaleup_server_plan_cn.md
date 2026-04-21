# STL10 扩展实验建议

## 当前结果解读

- 当前完成的 `stl10_b40x5_centralized_fulltrain_v3_run` 已经证明 `96x96` 高分辨率训练可以跑通。
- 但从结果看，当前瓶颈更像是“自然图像语义质量还不够强”，而不只是采样图像噪点问题。
- 代表信号：
  - 最佳 `classifier_fid` 出现在 `round 135`，为 `115.63`
  - 最佳 `proxy_fid` 出现在 `round 45`，为 `17.17`
  - `top1_confidence_mean` 仍然偏低，说明生成图像的语义可辨识度还不强

## 推荐的服务器实验顺序

### 1. 先跑：更长扩散链

- 配置：`configs/stl10_b40x5_centralized_t500_semantic10.yaml`
- 目的：
  - 保持当前 `22.4M` 模型结构不变
  - 把扩散链从 `300 -> 500`
  - 验证更长 diffusion process 是否能提升高分辨率图像质量
- 预期：
  - 训练耗时与当前 run 接近
  - 采样耗时会明显变长
  - 这是最干净的“只加步数”实验

### 2. 再跑：更大模型 + 适度更长扩散链

- 配置：`configs/stl10_b48x5_centralized_t400_semantic10.yaml`
- 目的：
  - 参数量从约 `22.4M -> 32.3M`
  - 扩散链从 `300 -> 400`
  - 同时提升模型容量与采样链长度
- 预期：
  - 单轮训练时间会明显高于当前 run
  - 更适合作为“服务器长跑的主升级实验”

## 命令

```powershell
.venv\Scripts\python.exe run_centralized.py --config configs/stl10_b40x5_centralized_t500_semantic10.yaml
```

```powershell
.venv\Scripts\python.exe run_centralized.py --config configs/stl10_b48x5_centralized_t400_semantic10.yaml
```

## 说明

- 这两条新配置都不会覆盖当前已经完成的 `stl10_b40x5_centralized_fulltrain_v3_run`
- 它们会写入新的输出目录，便于后续论文里做“原模型 / 加步数 / 加参数”三组对比
- `semantic_eval_every` 调成了 `10`，避免高分辨率长链采样把评估频率拖得太重
