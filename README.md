# RNN BPTT vs EGGROLL

An initial benchmark for asking how long an associative memory a vanilla
recurrent network can learn with either backpropagation through time or
gradient-free evolution.

This repository is an experiment scaffold, not a result. The first milestone is
to establish that both methods learn the short-delay task before spending a
large compute budget on memory-horizon and dynamics comparisons.

## Experiment

Each example contains random one-to-one key-value associations, a configurable
distractor delay, and a query for one stored key:

```text
STORE(k1, v1), ..., STORE(km, vm), DISTRACTOR * d, QUERY(ki) -> vi
```

Keys and values are sampled without replacement within an example. The mapping
changes on every example, so a model cannot solve the task by memorising a
global key-to-value lookup. Supervision is applied only after the query. The
main evaluation records:

- the longest delay `d` each method learns during its curriculum;
- the number of stored pairs `m`;
- later, state noise and numerical precision.

Each process trains exactly one method. Matching seeds produce byte-identical
initial parameters in the same tanh Elman RNN, but BPTT and EGGROLL have
independent data streams, curricula, run times, and output directories. BPTT
uses AdamW. EGGROLL uses forward-only antithetic low-rank perturbations; no
loss backward pass is performed for that model.

### Accuracy-gated delay curriculum

The reference preset does not expose the model to the full delay range from the
start. It begins at delay 0 and uses the frontier sequence
`0, 2, 4, 8, 16, 32`. Half of training batches use the current frontier; the
rest rehearse previously introduced delays. At each evaluation, the active
model is measured on a fixed validation probe at its frontier. Its frontier
advances immediately after one probe exceeds 90% accuracy. The threshold,
frontier sampling probability, and milestones are configurable:

```bash
rnn-memory-compare \
  --method bptt \
  --preset reference \
  --curriculum-delays 0,1,2,4,8,16,32 \
  --curriculum-accuracy-threshold 0.9
```

`--no-curriculum` restores uniform sampling over the full configured delay
range. Every probe and transition is stored in `metrics.json`.

The initial preset trains with two associations drawn from four keys and four
values. This is already a genuine binding task: ignoring the query and returning
one of the two stored values is capped at 50%. The evaluation grid also tests
one and four associations. Once both methods reliably traverse the delay
curriculum, vocabulary size and binding capacity should be scaled separately.

The setup reports unique labelled sequences and EGGROLL candidate-forward
sequences for context, but does not constrain the methods to equal data,
updates, wall-clock time, or forward compute. The primary objective is to give
each method a strong opportunity to reach its maximum learnable delay.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'
pytest
```

Run the tiny CPU-safe integration check:

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

The research preset carries over the working population EGGROLL setup from
[`spiral-extrapolation`](https://github.com/robflynnyh/spiral-extrapolation):

| Setting | Value |
| --- | ---: |
| Global population | 16,384 |
| Batch | 256 |
| Updates | 3,000 |
| Training associations | 2 of 4 keys/values |
| Delay curriculum | 0, 2, 4, 8, 16, 32 |
| Perturbation | antithetic rank 1 |
| Sigma | 0.005 |
| Fitness shaping | global z-score |
| EGGROLL update | SGD, lr 0.3, wd 0.001 |
| BPTT update | AdamW, lr 0.003, wd 0.001 |

That is a starting point, not an assertion that the spiral hyperparameters are
optimal for an RNN. A single-device run is:

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

Population evaluation is chunked to control activation memory. It can also be
sharded over four GPUs while retaining global fitness shaping:

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

The reference run is intentionally expensive: 12.58 billion candidate-sequence
forwards before accounting for recurrent timesteps. Start with the smoke preset
and a reduced population while checking whether the task and learning curves
behave sensibly.

### W&B tracking

Research commands enable W&B under the
[`rnn-bptt-vs-eggroll`](https://wandb.ai/wobrob101/rnn-bptt-vs-eggroll)
project. Tracking includes optimizer diagnostics, sampled delays, curriculum
frontiers and transitions, every validation-grid cell, the final test grid,
timings, configuration, `metrics.json`, and `model.pt`. BPTT and EGGROLL use
separate W&B runs; `--wandb-group` can group runs that belong to the same
comparison. In distributed EGGROLL, only rank 0 initializes and uploads to
W&B.

## Outputs

Each run writes:

- `metrics.json`: exact configuration, validation history, final test grid,
  timings, sample budgets, and initialization checksums;
- `model.pt`: the final state dictionary for the selected method.

The test grid keeps `num_pairs` and `delay` explicit rather than averaging them
into one score. This matters because remembering one cue for a long time and
dynamically binding several arbitrary associations are different capabilities.

## Relation to prior work

[Gomez and Schmidhuber (2005)](https://sferics.idsia.ch/pub/juergen/gecco05gomez.pdf)
compared evolved recurrent policies with BPTT and LSTM baselines on delayed-cue
T-mazes, including extremely long corridors. Their experiment changes the
training framework and network configuration as well as the optimiser, and the
memory content is essentially one cue bit. This repository targets the narrower
optimizer-controlled, dynamic-association comparison.

[Qu et al. (2026)](https://www.biorxiv.org/content/10.64898/2026.07.09.737022v1.full)
compare BPTT, evolution strategies, and genetic algorithms using the same RNN
on short `n`-back tasks. The remaining question here is the maximum trainable
memory horizon for arbitrary key-value bindings.

## Next experiments

1. Tune each method using only short-delay validation data and multiple seeds.
2. Let each independently progressing curriculum run until it reaches a
   reproducible learning limit.
3. Sweep recurrent initialization radius and include orthogonal/unitary and
   gated-RNN controls.
4. Measure hidden-state decodability, recurrent Jacobian singular values,
   fixed points, and robustness to state noise.
5. Near each method's curriculum limit, verify the result across seeds and
   inspect whether the recurrent dynamics retain information throughout the
   trained delay.
