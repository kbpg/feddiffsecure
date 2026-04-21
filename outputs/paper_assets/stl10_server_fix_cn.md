# STL10 服务器修复说明

如果服务器上出现下面这种错误：

```text
ValueError: cannot reshape array of size ... into shape (3,96,96)
```

这通常不是模型或显存问题，而是 `STL10` 数据文件损坏或下载不完整。

## 1. 先清理损坏的数据

在项目根目录执行：

```bash
rm -rf data/stl10
```

## 2. 单独测试 STL10 是否能重新正常加载

```bash
.venv/bin/python run_centralized.py --config configs/stl10_b40x5_centralized_t500_semantic10.yaml
```

现在项目已经加入了 `STL10` 完整性检查：

- 如果解压后的 `train_X.bin / test_X.bin / train_y.bin / test_y.bin` 大小不对，会自动删掉并重新下载
- 如果重下之后仍然不对，程序会直接报出更明确的错误信息

## 3. 再启动顺序队列

确认单条能正常进入训练后，再启动队列：

```bash
nohup .venv/bin/python run_config_queue.py \
  --config-list configs/stl10_server_scaleup_queue.txt \
  --name stl10_scaleup_batch_v2 \
  --continue-on-error \
  > outputs/launch_logs/stl10_scaleup_batch_v2.console.log 2>&1 &
```

## 4. 查看状态

```bash
tail -f outputs/launch_logs/stl10_scaleup_batch_v2.console.log
cat outputs/launch_logs/stl10_scaleup_batch_v2_summary.json
```

## 5. 额外提醒

如果你看到下面这个警告：

```text
CUDA initialization: The NVIDIA driver on your system is too old ...
```

那说明当前 PyTorch 和服务器驱动版本可能不匹配。即使数据修好了，训练也可能退回 CPU，速度会非常慢。

建议先检查：

```bash
.venv/bin/python - <<'PY'
import torch
print("torch:", torch.__version__)
print("torch cuda:", torch.version.cuda)
print("cuda available:", torch.cuda.is_available())
print("device count:", torch.cuda.device_count())
PY
```

如果 `cuda available: False`，需要把服务器上的 PyTorch 版本换成和当前 NVIDIA 驱动兼容的版本，再正式长跑。
