# MNIST federated vs centralized full-train best-checkpoint report

## Overview

![comparison_samples](comparison_samples.png)

## Main takeaways

- Best quality by best proxy FID: `mnist_b40x5_decoder_x0_centralized_fulltrain150_v2_run` with `best_proxy_fid=0.6504`.
- Best final-round quality: `mnist_b40x5_decoder_x0_centralized_fulltrain150_v2_run` with `last_proxy_fid=2.4386`.
- Best late-stage stability: `mnist_b40x5_decoder_x0_semantic100_run` with `rebound=1.5353` and `tail_std=0.8640`.
- Best semantic quality by classifier-FID: `mnist_b40x5_decoder_x0_centralized_fulltrain150_v2_run` with `classifier_fid=357.8714`.
- Training time contrast: `mnist_b40x5_decoder_x0_centralized_fulltrain150_v2_run` took `2.46h`, `mnist_b40x5_decoder_x0_semantic100_run` took `2.34h`.

## Model comparison

| run | checkpoint | rounds | lr | local_steps | sample_steps | params | trainable | trainable_pct | time | best_proxy_fid | last_proxy_fid | proxy_eval_fid | classifier_fid | conf_mean | confident@0.9 | rebound | payload_bytes_raw |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| mnist_b40x5_decoder_x0_centralized_fulltrain150_v2_run | best | 150 | 1e-04 | 80 | 300 | 22444929 | 22444929 | 100.00% | 2.46h | 0.6504 | 2.4386 | 1.4076 | 357.8714 | 0.9559 | 0.8711 | 1.7882 | 0 |
| mnist_b40x5_decoder_x0_semantic100_run | best | 100 | 1e-04 | 20 | 300 | 22444929 | 825729 | 3.68% | 2.34h | 3.4040 | 4.9393 | 5.3461 | 575.9547 | 0.8354 | 0.5625 | 1.5353 | 3379727 |