# Dense single-query recall curriculum

This experiment asks how many distinct key-value pairs a tied vanilla RNN can
learn to retain when trained by standardized EGGROLL.

## Task

At stage `N`, every example contains `N` distinct key-value pairs. One stored
key is selected uniformly as the query:

```text
k1, v1, ..., kN, vN, query_key -> query_value
```

The RNN receives the tokens through `query_key`; the final answer is omitted
from its input and predicted autoregressively. Reported sequence length includes
that answer token, so stage `N` has full length `2N+2` and model input length
`2N+1`. For example, stage 1 is reported as the four-token problem
`K,V,Q,V_answer`, although the input tensor contains only `K,V,Q`.

Keys and values are sampled without replacement within each example. Only the
single answer prediction contributes to loss and accuracy.

## Protocol

The initial curriculum started at one pair and advanced by exactly one pair
after one fixed 512-example frontier probe exceeded 90% accuracy. This exposed
a discontinuity: one pair can be solved by retaining the only value without
using the query, whereas two pairs require key-conditioned retrieval. The
replacement schedule therefore starts at two pairs, where the intended
operation is required from initialization, and then advances by one pair. It
stops at 1,024 pairs or 2,000,000 generations, whichever comes first. Training
and evaluation cover only the current frontier.

The initial run uses a tied, single-layer 64-unit tanh RNN; population 8,192;
batch 64; Cartesian candidate evaluation; BF16 candidate forwards; rank-1
antithetic perturbations; `sigma=0.005`; z-score fitness; and no weight decay.
The EGGROLL learning rate cosine-decays from 0.4 to 0.01 over the full ceiling.

Launch or resume it on one scheduled GPU with:

```bash
bash scripts/run_dense_recall_curriculum.sh
```

W&B training metrics are logged every generation, frontier evaluation every
100 generations, and atomic resumable checkpoints every 20,000 generations.
The initial from-one run
[`densekv07`](https://wandb.ai/wobrob101/rnn-bptt-vs-egroll/runs/densekv07)
was stopped after 22,300 generations. It promoted at generation 17,500 with
90.82% one-pair accuracy, but the two-pair probe immediately fell to 37.11%
and had declined to 32.62% when stopped. Its generation-20,000 checkpoint is
retained as negative evidence. The replacement from-two run uses a separate
output directory and W&B identity so it cannot resume the failed trajectory:
[`densekv2a`](https://wandb.ai/wobrob101/rnn-bptt-vs-egroll/runs/densekv2a).

The from-two EGGROLL run was stopped at generation 9,800 after its fixed probe
peaked at 23.83% and declined to 18.36%. A fixed-`N=2` BPTT control uses the
same seed, examples, tied 64-unit RNN, vocabulary, batch size, and evaluation
set. It changes only the optimizer and disables promotion, testing whether the
plateau comes from EGGROLL or from the task/model combination:

```bash
bash scripts/run_dense_recall_bptt_n2.sh
```
