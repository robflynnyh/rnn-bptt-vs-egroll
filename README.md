# RNN BPTT vs EGGROLL

A controlled comparison of how far a vanilla recurrent network can learn
multi-query associative recall (MQAR) using backpropagation through time or
forward-only EGGROLL.

This repository is an experiment scaffold, not a result. BPTT and EGGROLL run
as independent processes with independent curricula and run times.

## Task

The task follows the released generator from
[Zoology](https://github.com/HazyResearch/zoology). Each example begins with
unique alternating key/value tokens. Every key then appears once as a query in
the remaining sequence and must retrieve its associated value:

```text
k1, v1, k2, v2, ..., filler, k2, filler, ..., k1, filler
                         targets: v2              v1
```

Keys come from the lower half of an 8,192-token vocabulary and values from the
upper half. Both are sampled without replacement within an example, so the
mapping cannot be memorized across examples. Query positions use Zoology's
power-law gap distribution with `power_a=0.01`. Following the paper's Figure 2
configuration, token `0` fills non-query positions. Loss and accuracy are
computed only at query positions.

This is benchmark-aligned rather than an exact Zoology reproduction. Zoology
compares language-model sequence mixers in separate fixed-length runs. Here a
single-layer tanh Elman RNN persists through an accuracy-gated curriculum so
the two optimizers can be compared.

## Curriculum

The reference curriculum begins below Zoology's shortest headline setting and
then uses its density of one key/value pair per 16 tokens:

| Stage | Sequence length | KV pairs |
| ---: | ---: | ---: |
| 0 | 16 | 1 |
| 1 | 32 | 2 |
| 2 | 64 | 4 |
| 3 | 128 | 8 |
| 4 | 256 | 16 |
| 5 | 512 | 32 |
| 6 | 1,024 | 64 |

Half of training batches use the current frontier; the other half rehearse
previously reached stages. One fixed validation probe above 90% accuracy
advances the frontier. Validation and final testing cover only stages reached
during training, so W&B does not contain out-of-distribution grid cells.

The schedule is configurable:

```bash
rnn-memory-compare \
  --method bptt \
  --preset reference \
  --curriculum-sequence-lengths 16,32,64,128,256,512,1024 \
  --curriculum-num-kv-pairs 1,2,4,8,16,32,64 \
  --curriculum-accuracy-threshold 0.9
```

`--no-curriculum` trains only the final configured stage. Every validation
probe and transition is recorded in `metrics.json` and W&B.
Training metrics log independently every generation by default; use
`--log-interval` to reduce their frequency without changing validation.

## Methods

Matching seeds produce byte-identical initial parameters in the same RNN.
BPTT uses AdamW. EGGROLL uses antithetic low-rank parameter perturbations,
global fitness shaping, and no loss backward pass.

The comparison does not constrain the methods to equal data, updates,
wall-clock time, or forward compute. The objective is to give each optimizer a
strong opportunity to reach its maximum learnable MQAR stage.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'
python -m pytest
```

Run the CPU-safe integration checks:

```bash
rnn-memory-compare \
  --method bptt \
  --preset smoke \
  --output-dir artifacts/smoke_bptt \
  --log-progress

rnn-memory-compare \
  --method eggroll \
  --preset smoke \
  --output-dir artifacts/smoke_eggroll \
  --log-progress
```

Run each research method independently:

```bash
rnn-memory-compare \
  --method bptt \
  --preset reference \
  --output-dir artifacts/reference_bptt_seed7 \
  --wandb \
  --wandb-run-name reference-bptt-seed7 \
  --wandb-group reference-seed7 \
  --log-progress

rnn-memory-compare \
  --method eggroll \
  --preset reference \
  --output-dir artifacts/reference_eggroll_seed7 \
  --wandb \
  --wandb-run-name reference-eggroll-seed7 \
  --wandb-group reference-seed7 \
  --log-progress
```

Population evaluation is chunked to control the large vocabulary readout. It
can also be sharded while retaining global fitness shaping:

```bash
torchrun --standalone --nproc_per_node=4 \
  -m rnn_bptt_vs_eggroll.experiment \
  --method eggroll \
  --preset reference \
  --output-dir artifacts/reference_eggroll_seed7 \
  --wandb \
  --wandb-run-name reference-eggroll-seed7 \
  --wandb-group reference-seed7 \
  --log-progress
```

The reference candidate chunk is 1,024. Candidate networks share the base
weights and apply their rank-1 corrections in a batched population dimension;
query losses are accumulated as each readout position is produced so the full
population-by-query-by-vocabulary tensor is never retained. Rank-1 corrections
are fused into the base activations, and nearby query states share larger
readout projections.

## Tracking

W&B runs use the
[`rnn-bptt-vs-eggroll`](https://wandb.ai/wobrob101/rnn-bptt-vs-eggroll)
project. Tracking includes optimizer diagnostics, sampled sequence length and
pair count, curriculum transitions, reached-stage validation, timing,
configuration, `metrics.json`, and `model.pt`. Only rank 0 logs distributed
EGGROLL runs.

Each run also writes:

- `metrics.json`: configuration, validation history, reached-stage test grid,
  timing, budgets, and initialization checksum;
- `model.pt`: final model state dictionary.

## Prior Work

[Zoology](https://arxiv.org/abs/2312.04927) introduced MQAR to test multiple
in-context recalls at varied positions and realistic vocabulary scale.

[Gomez and Schmidhuber (2005)](https://sferics.idsia.ch/pub/juergen/gecco05gomez.pdf)
compared evolved recurrent policies with BPTT and LSTM baselines on delayed-cue
T-mazes. Their memory content is essentially one cue bit; MQAR instead requires
dynamic key/value binding.

[Qu et al. (2026)](https://www.biorxiv.org/content/10.64898/2026.07.09.737022v1.full)
compare BPTT, evolution strategies, and genetic algorithms using the same RNN
on short `n`-back tasks. The remaining question here is the maximum trainable
MQAR stage for each optimizer.

## Next Experiments

1. Tune each method using reached-stage validation and multiple seeds.
2. Let each curriculum progress independently to a reproducible learning limit.
3. Sweep recurrent initialization radius and add gated-RNN controls.
4. Measure hidden-state decodability, recurrent Jacobian singular values,
   fixed points, and robustness to state noise.
