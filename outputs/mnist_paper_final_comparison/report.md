# MNIST federated vs centralized t300 vs centralized t500 (best checkpoints)

![comparison_samples](comparison_samples.png)

## Key findings

- Best best-proxy-FID: `mnist_b40x5_decoder_x0_centralized_t500_long_run` (0.4305)
- Best final-round proxy-FID: `mnist_b40x5_decoder_x0_centralized_t500_long_run` (0.6018)
- Best late-stage stability: `mnist_b40x5_decoder_x0_centralized_t500_long_run` (rebound=0.1712, tail_std=0.2327)
- Best semantic quality by classifier-FID: `mnist_b40x5_decoder_x0_centralized_t500_long_run` (103.4483)

## Metrics

| run | checkpoint | rounds | lr | local_steps | sample_steps | best_proxy_fid | last_proxy_fid | proxy_eval_fid | classifier_fid | conf_mean | confident@0.9 | rebound | payload_bytes_raw |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| mnist_b40x5_decoder_x0_centralized_t500_long_run | best | 180 | 1e-04 | 80 | 500 | 0.4305 | 0.6018 | 0.7183 | 103.4483 | 0.9660 | 0.9102 | 0.1712 | 0 |
| mnist_b40x5_decoder_x0_centralized_fulltrain150_v2_run | best | 150 | 1e-04 | 80 | 300 | 0.6504 | 2.4386 | 1.4076 | 322.8146 | 0.9560 | 0.8750 | 1.7882 | 0 |
| mnist_b40x5_decoder_x0_semantic100_run | best | 100 | 1e-04 | 20 | 300 | 3.4040 | 4.9393 | 5.3461 | 500.0861 | 0.8314 | 0.5195 | 1.5353 | 3379727 |

## Qualitative panels

The real reference row comes from the evaluation split. Each run row uses the same `sample_steps` value shown in the table, so the visual comparison and the recomputed proxy FID share the same sampling budget.
Checkpoint selection for this report: `best`.

### mnist_b40x5_decoder_x0_centralized_t500_long_run

![mnist_b40x5_decoder_x0_centralized_t500_long_run](mnist_b40x5_decoder_x0_centralized_t500_long_run_samples.png)

### mnist_b40x5_decoder_x0_centralized_fulltrain150_v2_run

![mnist_b40x5_decoder_x0_centralized_fulltrain150_v2_run](mnist_b40x5_decoder_x0_centralized_fulltrain150_v2_run_samples.png)

### mnist_b40x5_decoder_x0_semantic100_run

![mnist_b40x5_decoder_x0_semantic100_run](mnist_b40x5_decoder_x0_semantic100_run_samples.png)
