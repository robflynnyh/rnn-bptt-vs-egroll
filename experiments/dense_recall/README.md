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

The curriculum starts at one pair and advances by exactly one pair after one
fixed 512-example frontier probe exceeds 90% accuracy. It stops at 1,024 pairs
or 2,000,000 generations, whichever comes first. Training and evaluation cover
only the current frontier.

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
The run is tracked as `densekv07` in the `rnn-bptt-vs-eggroll` project.
