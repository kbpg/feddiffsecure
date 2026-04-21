# STL10 服务器续跑说明

## 当前已保存进度

- 运行目录：`outputs/stl10_b40x5_centralized_fulltrain_v3_run`
- 当前已完成：`round 4`
- 当前最佳 proxy_fid：`round 2`, `24.7238`
- 当前没有语义评估结果，因为 `semantic_eval_every = 5`

## 需要一并迁移的文件

1. `configs/stl10_b40x5_centralized_fulltrain_v3.yaml`
2. 整个 `outputs/stl10_b40x5_centralized_fulltrain_v3_run`
3. `outputs/evaluators/stl10_label_classifier_v2.pt`
4. `outputs/evaluators/stl10_label_classifier_v2.json`
5. `data/stl10`
6. 代码目录 `feddiffsecure`
7. `run_centralized.py`
8. `requirements.txt`

## 服务器上继续训练的命令

```powershell
.venv\Scripts\python.exe run_centralized.py --config configs\stl10_b40x5_centralized_fulltrain_v3.yaml
```

## 说明

- 这条命令会自动读取 `outputs/stl10_b40x5_centralized_fulltrain_v3_run/training_state.pt`
- 如果输出目录和 `training_state.pt` 都保留完整，就会从 `round 4` 继续
- 不要加 `--restart`，否则会从头训练
