# Paper Experiment Registry

This directory records the representative experiments that should be preserved for paper writing and later reproduction.

## Core preserved experiments

### 1. MNIST federated secure training baseline

- Run directory: `outputs/mnist_b40x5_decoder_x0_semantic100_run`
- Purpose: Representative federated diffusion experiment with non-IID clients, secure payload transport, and audit verification.
- Why it matters: This is the main evidence that the federated learning pipeline, steganographic transport, and audit verification all work together end-to-end.
- Key numbers:
  - dataset: `MNIST`
  - training mode: `federated`
  - rounds: `100`
  - clients per round: `4`
  - trainable params: `825,729 / 22,444,929`
  - best checkpoint classifier-FID in current comparison: `575.95`
  - audit records: `400/400 verified`
- Keep for paper:
  - `outputs/mnist_b40x5_decoder_x0_semantic100_run/metrics/final_metrics.json`
  - `outputs/mnist_b40x5_decoder_x0_semantic100_run/run_summary.json`
  - `outputs/mnist_b40x5_decoder_x0_semantic100_run/audit_summary.json`
  - `outputs/mnist_b40x5_decoder_x0_semantic100_run/audit_log.jsonl`
  - `outputs/mnist_b40x5_decoder_x0_semantic100_run/samples/best.png`
  - `outputs/mnist_b40x5_decoder_x0_semantic100_run/samples/final.png`

### 2. MNIST centralized upper-bound baseline

- Run directory: `outputs/mnist_b40x5_decoder_x0_centralized_fulltrain150_v2_run`
- Purpose: Centralized full-parameter diffusion baseline on the same architecture for quality upper-bound comparison.
- Why it matters: This run shows how much sample quality improves when the same `b40x5` architecture is trained centrally with all parameters enabled.
- Key numbers:
  - dataset: `MNIST`
  - training mode: `centralized`
  - rounds: `150`
  - trainable params: `22,444,929 / 22,444,929`
  - best checkpoint classifier-FID: `357.87`
  - best semantic round in training log: `round 40`, `classifier_fid=187.47`
- Keep for paper:
  - `outputs/mnist_b40x5_decoder_x0_centralized_fulltrain150_v2_run/metrics/final_metrics.json`
  - `outputs/mnist_b40x5_decoder_x0_centralized_fulltrain150_v2_run/run_summary.json`
  - `outputs/mnist_b40x5_decoder_x0_centralized_fulltrain150_v2_run/samples/best.png`
  - `outputs/mnist_b40x5_decoder_x0_centralized_fulltrain150_v2_run/samples/final.png`
  - `outputs/mnist_b40x5_decoder_x0_centralized_fulltrain150_v2_run/samples/best_compare.png`

### 3. MNIST federated vs centralized comparison page

- Run directory: `outputs/mnist_b40x5_fed_vs_central_fulltrain_best`
- Purpose: Paper-ready side-by-side comparison between federated best checkpoint and centralized best checkpoint.
- Why it matters: This is the compact figure/table bundle that directly supports the comparison section of the paper.
- Keep for paper:
  - `outputs/mnist_b40x5_fed_vs_central_fulltrain_best/comparison_samples.png`
  - `outputs/mnist_b40x5_fed_vs_central_fulltrain_best/summary.csv`
  - `outputs/mnist_b40x5_fed_vs_central_fulltrain_best/paper_report.md`

### 4. STL10 high-resolution exploratory baseline

- Run directory: `outputs/stl10_b32_centralized_fulltrain_run`
- Purpose: High-resolution exploratory baseline beyond MNIST.
- Why it matters: This run is not yet the main quality result, but it demonstrates that the pipeline can be extended to `96x96` natural images and provides a first high-resolution reference point.
- Key numbers:
  - dataset: `STL10`
  - training mode: `centralized`
  - rounds: `120`
  - trainable params: `14,383,099 / 14,383,099`
  - best semantic round: `round 85`, `classifier_fid=117.63`
  - evaluator accuracy is only `0.4375`, so semantic scores should be interpreted cautiously
- Keep for paper:
  - `outputs/stl10_b32_centralized_fulltrain_run/metrics/final_metrics.json`
  - `outputs/stl10_b32_centralized_fulltrain_run/run_summary.json`
  - `outputs/stl10_b32_centralized_fulltrain_run/samples/best.png`
  - `outputs/stl10_b32_centralized_fulltrain_run/samples/final.png`

### 5. MNIST higher-timestep centralized upper bound

- Run directory: `outputs/mnist_b40x5_decoder_x0_centralized_t500_long_run`
- Purpose: Higher-timestep centralized MNIST upper bound with a longer diffusion chain.
- Why it matters: This is the current strongest MNIST result in the project and directly answers whether extending the diffusion process beyond `300` steps improves sample quality.
- Key numbers:
  - dataset: `MNIST`
  - training mode: `centralized`
  - rounds: `180`
  - trainable params: `22,444,929 / 22,444,929`
  - best semantic round: `round 175`, `classifier_fid=86.57`
  - best proxy round: `round 67`, `proxy_fid=0.4305`
- Keep for paper:
  - `outputs/mnist_b40x5_decoder_x0_centralized_t500_long_run/metrics/final_metrics.json`
  - `outputs/mnist_b40x5_decoder_x0_centralized_t500_long_run/run_summary.json`
  - `outputs/mnist_b40x5_decoder_x0_centralized_t500_long_run/samples/best.png`
  - `outputs/mnist_b40x5_decoder_x0_centralized_t500_long_run/samples/final.png`

### 6. MNIST three-way final comparison page

- Run directory: `outputs/mnist_paper_final_comparison`
- Purpose: Paper-ready comparison across federated MNIST, centralized `t300`, and centralized `t500`.
- Why it matters: This is the cleanest single figure/table bundle for the final MNIST experiment section.
- Keep for paper:
  - `outputs/mnist_paper_final_comparison/comparison_samples.png`
  - `outputs/mnist_paper_final_comparison/summary.csv`
  - `outputs/mnist_paper_final_comparison/report.md`

## Important interpretation note

- The current best MNIST sample already uses `sample_steps=300`, which matches the diffusion training timesteps.
- Therefore, improving MNIST further by "adding more sampling steps" requires retraining a higher-timestep model rather than only changing inference on the existing checkpoint.

## In-progress follow-up

### 7. Clean MNIST t500 reproduction

- Config path: `configs/mnist_b40x5_decoder_x0_centralized_t500_repro_clean_v2.yaml`
- Planned run directory: `outputs/mnist_b40x5_decoder_x0_centralized_t500_repro_clean_v2_run`
- Purpose: Reproduce the `round 175` quality tier with a clean single-run output directory and `num_workers=0` to keep the paper result traceable.

### 8. Stronger STL10 generalization run

- Config path: `configs/stl10_b40x5_centralized_fulltrain_v3.yaml`
- Planned run directory: `outputs/stl10_b40x5_centralized_fulltrain_v3_run`
- Purpose: Validate cross-dataset generalization with a larger high-resolution model than the earlier `STL10 b32` baseline.
