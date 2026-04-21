# Audit And Federated Learning Notes

## What the audit currently proves

The current audit layer verifies the integrity of each client update after secure packaging and recovery.

For every selected client in every federated round, the pipeline logs:

- client id
- round id
- number of local samples
- average local training loss
- raw payload size
- compressed payload size
- carrier PSNR
- original payload SHA-256
- recovered payload SHA-256
- verification status

This means the audit is answering the question:

- "Did the trainable update that left the client survive steganographic packaging and recovery without corruption?"

It does **not** yet prove stronger security properties such as resistance to adaptive attacks or confidentiality against a powerful adversary. So in the paper, the safest wording is:

- the audit layer validates end-to-end integrity of transmitted federated updates
- the current prototype demonstrates transport verification, not a full formal security proof

## Representative audit result

The representative federated MNIST run is:

- `outputs/mnist_b40x5_decoder_x0_semantic100_run`

Its audit summary is:

- `records = 400`
- `verified = 400`
- `failed = 0`

This corresponds to:

- `100` federated rounds
- `4` clients selected per round
- `400` transmitted and recovered payloads
- `100%` integrity verification success in this run

## What the federated learning paradigm contributes

The federated setting contributes several paper-relevant properties that are absent in centralized diffusion training:

- training is performed on distributed client subsets rather than on pooled raw data
- the system must aggregate non-IID local updates across clients
- communication cost becomes part of the method, because updates must be transmitted every round
- client drift and aggregation stability become explicit optimization problems
- secure transport and audit verification become meaningful, because the method actually moves model updates across parties

In this project, the federated run also provides a concrete systems tradeoff:

- it achieves lower sample quality than the centralized upper bound
- but it demonstrates decentralized training, stego-wrapped update transfer, and perfect integrity verification over all logged payloads

## How to describe the federated role in the paper

Suggested concise wording:

- "The federated baseline serves as the distributed-learning reference setting. It evaluates whether diffusion training remains viable under non-IID client partitioning, communication constraints, and secure update transport."

- "Compared with centralized training, the federated setting introduces client drift, aggregation instability, and payload transmission cost, but enables privacy-preserving decentralized optimization without centralizing raw client data."

- "Our audit mechanism verifies that each transmitted trainable update is recovered losslessly after steganographic packaging, yielding 400/400 successful integrity checks in the representative MNIST federated run."

## Recommended comparison framing

For the current paper draft, the cleanest comparison story is:

1. `MNIST federated secure baseline`: demonstrates distributed diffusion, stego payload transport, and audit verification.
2. `MNIST centralized upper bound`: demonstrates what the same architecture can achieve when all parameters are trained centrally.
3. `STL10 high-resolution exploratory run`: demonstrates extension to higher-resolution natural images, while also showing that high-resolution quality remains the next open challenge.
