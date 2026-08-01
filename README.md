# RNN BPTT vs EGGROLL

An initial controlled benchmark for asking whether gradient-free evolution finds
longer-lived associative memory dynamics than backpropagation through time in
the **same vanilla recurrent network**.

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
main evaluation independently sweeps:

- the delay `d`, including delays longer than any seen during training;
- the number of stored pairs `m`;
- later, state noise and numerical precision.

Both methods use byte-identical initial parameters, the same tanh Elman RNN,
and the same fresh labelled batch at every update. BPTT uses AdamW. EGGROLL
uses forward-only antithetic low-rank perturbations; no loss backward pass is
performed for that model.

### Accuracy-gated delay curriculum

The reference preset does not expose the model to the full delay range from the
start. It begins at delay 0 and uses the frontier sequence
`0, 2, 4, 8, 16, 32`. Half of training batches use the current frontier; the
rest rehearse previously introduced delays. At each evaluation, both models
are measured on a fixed validation probe at the frontier. The shared frontier
advances after both models exceed 90% accuracy on two consecutive probes.

Gating on both models is the default because it keeps their labelled stream
identical. The threshold, patience, frontier sampling probability, milestones,
and gate are configurable:

```bash
rnn-memory-compare \
  --preset reference \
  --curriculum-delays 0,1,2,4,8,16,32 \
  --curriculum-accuracy-threshold 0.9 \
  --curriculum-consecutive-probes 2 \
  --curriculum-gate all
```

`--curriculum-gate bptt`, `eggroll`, or `mean` supports diagnostic schedules,
and `--no-curriculum` restores uniform sampling over the full configured delay
range. Every probe and transition is stored in `metrics.json`.

The initial preset trains with two associations drawn from four keys and four
values. This is already a genuine binding task: ignoring the query and returning
one of the two stored values is capped at 50%. The evaluation grid also tests
one and four associations. Once both methods reliably traverse the delay
curriculum, vocabulary size and binding capacity should be scaled separately.

The setup deliberately reports two budgets:

- **unique labelled sequences**, which are shared between the methods;
- **candidate-forward sequences**, which exposes EGGROLL's much larger compute
  cost instead of hiding it behind an equal-update comparison.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'
pytest
```

Run the tiny CPU-safe integration check:

```bash
rnn-memory-compare --preset smoke --output-dir artifacts/smoke --log-progress
```

The research preset carries over the working population EGGROLL setup from
[`spiral-extrapolation`](https://github.com/robflynnyh/spiral-extrapolation):

| Setting | Value |
| --- | ---: |
| Global population | 16,384 |
| Shared batch | 256 |
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
  --preset reference \
  --output-dir artifacts/reference_seed7 \
  --log-progress
```

Population evaluation is chunked to control activation memory. It can also be
sharded over four GPUs while retaining global fitness shaping:

```bash
torchrun --standalone --nproc_per_node=4 \
  -m rnn_bptt_vs_eggroll.experiment \
  --preset reference \
  --output-dir artifacts/reference_seed7 \
  --log-progress
```

The reference run is intentionally expensive: 12.58 billion candidate-sequence
forwards before accounting for recurrent timesteps. Start with the smoke preset
and a reduced population while checking whether the task and learning curves
behave sensibly.

## Outputs

Each run writes:

- `metrics.json`: exact configuration, validation history, final test grid,
  timings, sample budgets, and initialization checksums;
- `bptt.pt` and `eggroll.pt`: final state dictionaries.

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
on short `n`-back tasks. The remaining question here is horizon extrapolation
on arbitrary key-value bindings.

## Next experiments

1. Tune each method using only short-delay validation data and multiple seeds.
2. Add matched wall-clock and matched forward-FLOP comparisons alongside the
   shared-data comparison.
3. Sweep recurrent initialization radius and include orthogonal/unitary and
   gated-RNN controls.
4. Measure hidden-state decodability, recurrent Jacobian singular values,
   fixed points, and robustness to state noise.
5. Train on bounded delays and test far beyond them to distinguish interpolation
   from genuinely stable memory dynamics.
