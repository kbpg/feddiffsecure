# 服务器顺序长跑说明

这份项目已经支持把多个配置按顺序排队运行，并自动记录日志与汇总结果。

## 1. 推荐实验

当前建议优先跑这两条高分辨率 `STL10` 实验：

1. `configs/stl10_b40x5_centralized_t500_semantic10.yaml`
   - 保持模型容量不变
   - 将训练/采样步数从 `300 -> 500`
   - 用来回答“更长扩散链是否提升高分辨率图像质量”

2. `configs/stl10_b48x5_centralized_t400_semantic10.yaml`
   - 将 `base_channels` 从 `40 -> 48`
   - 总参数约 `32.3M`
   - 训练/采样步数提升到 `400`
   - 用来回答“增大模型参数后能否进一步提升图像质量”

## 2. 服务器启动命令

假设你已经把整个项目上传到服务器并进入项目根目录：

```bash
cd ~/feddiffsecure_minimal
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

先单独跑一条：

```bash
python run_centralized.py --config configs/stl10_b40x5_centralized_t500_semantic10.yaml
```

或者直接顺序排队跑两条：

```bash
python run_config_queue.py \
  --config-list configs/stl10_server_scaleup_queue.txt \
  --name stl10_scaleup_batch
```

如果你希望某一条失败后继续跑后面的：

```bash
python run_config_queue.py \
  --config-list configs/stl10_server_scaleup_queue.txt \
  --name stl10_scaleup_batch \
  --continue-on-error
```

## 3. 日志与结果位置

队列启动后会自动把信息写到：

- 总汇总：`outputs/launch_logs/<queue_name>_summary.json`
- 每个配置的标准输出：`outputs/launch_logs/<queue_name>_<idx>_<config>.out.log`
- 每个配置的错误输出：`outputs/launch_logs/<queue_name>_<idx>_<config>.err.log`

每条实验自己的正式结果仍然在各自输出目录：

- `metrics/round_metrics.json`
- `metrics/round_metrics.csv`
- `metrics/final_metrics.json`
- `run_summary.json`

## 4. 恢复与跳过

- 如果某条实验已经完整结束，并且输出目录里已有 `final_metrics.json`，队列会默认跳过它。
- 如果你明确想重跑，给队列加 `--restart`。
- 如果是未跑完的实验，直接再次执行同一条 `run_centralized.py --config ...` 即可自动断点续跑。

## 5. 建议口径

论文里建议把这批实验描述为两组：

1. 固定容量，仅增加扩散步数
2. 同时增加模型参数与扩散步数

这样实验结构会更清楚，也更容易支撑“高分辨率数据集上的泛化能力提升”这一段。
