# Portal Content Registry

前端门户现在不再把实验、对比页和论文资料硬编码在 `repository.py` 里，而是通过这一组 YAML 注册表统一接入。

## 文件说明

- `portal_runs.yaml`
  - 注册单个实验运行。
  - 每条记录至少包含 `id / title / run_dir / dataset / mode / role / accent`。
  - 只要 `run_dir` 下出现 `run_summary.json`、`metrics/final_metrics.json` 或 `metrics/round_metrics.json`，门户就会自动读取。

- `comparison_groups.yaml`
  - 注册跨实验对比页。
  - 每条记录至少包含 `id / title / dataset / dir / summary_csv / report / hero_image`。
  - 对比页会自动出现在“结果展示中心”和 `/comparisons/<id>` 页面。

- `research_library.yaml`
  - 注册“参考与思想”页面里的中文资料。
  - 包含站点总述、项目模型图、外部参考文件映射、论文 section、公式和启发说明。

## 新增实验的最短路径

1. 先跑出一个新的输出目录，例如 `outputs/my_new_run/`。
2. 把它补进 `portal_runs.yaml`。
3. 重启门户。

## 新增对比页的最短路径

1. 使用 `compare_runs.py` 生成 `summary.csv / report.md / comparison_samples.png`。
2. 把这些文件路径补进 `comparison_groups.yaml`。
3. 重启门户。

## 新增论文参考资料的最短路径

1. 在 `research_library.yaml` 的 `reference_files` 里登记原始 PDF / PPT。
2. 在 `sections` 里补中文摘要、启发点和公式。
3. 重启门户。

## 约定

- 路径默认相对仓库根目录解释；绝对路径适合外部 PDF / PPT 文件。
- 原始图片/报告如果位于仓库 `outputs/` 目录下，门户会自动提供预览或下载入口。
- 外部参考文件不会直接开放任意路径访问，而是通过 `reference_files` 的显式映射提供。
