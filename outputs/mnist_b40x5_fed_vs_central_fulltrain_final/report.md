# MNIST b40x5 federated vs centralized full-train final checkpoint comparison

![comparison_samples](comparison_samples.png)

## Key findings

- Best best-proxy-FID: `mnist_b40x5_decoder_x0_centralized_fulltrain150_v2_run` (0.6504)
- Best final-round proxy-FID: `mnist_b40x5_decoder_x0_centralized_fulltrain150_v2_run` (2.4386)
- Best late-stage stability: `mnist_b40x5_decoder_x0_semantic100_run` (rebound=1.5353, tail_std=0.8640)
- Best semantic quality by classifier-FID: `mnist_b40x5_decoder_x0_centralized_fulltrain150_v2_run` (330.7015)

## Metrics

| run | checkpoint | rounds | lr | local_steps | sample_steps | best_proxy_fid | last_proxy_fid | proxy_eval_fid | classifier_fid | conf_mean | confident@0.9 | rebound | payload_bytes_raw |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| mnist_b40x5_decoder_x0_centralized_fulltrain150_v2_run | final | 150 | 1e-04 | 80 | 300 | 0.6504 | 2.4386 | 2.5957 | 330.7015 | 0.9790 | 0.9297 | 1.7882 | 0 |
| mnist_b40x5_decoder_x0_semantic100_run | final | 100 | 1e-04 | 20 | 300 | 3.4040 | 4.9393 | 5.3461 | 575.9547 | 0.8354 | 0.5625 | 1.5353 | 3379727 |

## Qualitative panels

The real reference row comes from the evaluation split. Each run row uses the same `sample_steps` value shown in the table, so the visual comparison and the recomputed proxy FID share the same sampling budget.
Checkpoint selection for this report: `final`.

### mnist_b40x5_decoder_x0_centralized_fulltrain150_v2_run

![mnist_b40x5_decoder_x0_centralized_fulltrain150_v2_run](mnist_b40x5_decoder_x0_centralized_fulltrain150_v2_run_samples.png)

### mnist_b40x5_decoder_x0_semantic100_run

![mnist_b40x5_decoder_x0_semantic100_run](mnist_b40x5_decoder_x0_semantic100_run_samples.png)
