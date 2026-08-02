# Bounded EGGROLL tuning

This study tests whether optimizer hyperparameters or an elite-centroid update
improve early MQAR convergence over the Cartesian standardized-fitness control.
It is deliberately bounded: ten 2,000-generation screens, at most two
10,000-generation finalists, and one matched confirmation seed if a finalist
qualifies.

## Control

The historical control is W&B run
[`ceijp8r1`](https://wandb.ai/wobrob101/rnn-bptt-vs-egroll/runs/ceijp8r1):
Cartesian `P=8192` by batch 64, BF16 candidate forwards, rank-1 perturbations,
`sigma=0.005`, z-score fitness, SGD learning rate `0.3`, no weight decay, and
seed 7. Its predeclared validation losses at generations 1,600--2,000 are:

| Generation | Length 16, one-pair loss |
|---:|---:|
| 1,600 | 8.703370 |
| 1,700 | 8.697623 |
| 1,800 | 8.686286 |
| 1,900 | 8.678341 |
| 2,000 | 8.661924 |

The historical mean is `8.685509`; a 1% improvement requires a score below
`8.598654`. The bounded study reruns the control under the instrumented code
and uses that reproduction as the formal promotion baseline.

## Score

The screening score is declared before running: mean
`validation_grid/seq_len_16/kv_pairs_1/loss` at generations 1,600, 1,700,
1,800, 1,900, and 2,000. Accuracy and the last-500 update diagnostics are
secondary. A screen is eligible for promotion only if its score is at least 1%
lower than the reproduced control. Transient best checkpoints are not used.

## Configurations

All screens keep the task, model, seed, population, batch, data mode, candidate
precision, rank, and fixed validation examples unchanged.

| Run | Rule | Sigma | LR / commit | Elite K | Purpose |
|---|---|---:|---:|---:|---|
| `screen-zscore-s5e-3-lr3e-1-control` | standardized | 0.005 | 0.3 | - | control reproduction |
| `screen-zscore-s2p5e-3-lr6e-1` | standardized | 0.0025 | 0.6 | - | smaller radius, matched nominal step |
| `screen-zscore-s1e-2-lr1p5e-1` | standardized | 0.01 | 0.15 | - | larger radius, matched nominal step |
| `screen-zscore-s2e-2-lr7p5e-2` | standardized | 0.02 | 0.075 | - | much larger radius, matched nominal step |
| `screen-zscore-s5e-3-lr1p5e-1` | standardized | 0.005 | 0.15 | - | half update scale |
| `screen-zscore-s5e-3-lr6e-1` | standardized | 0.005 | 0.6 | - | double update scale |
| `screen-elite-k8-c1p198` | elite centroid | 0.005 | 1.1976 | 8 | narrow elite set |
| `screen-elite-k64-c3p371` | elite centroid | 0.005 | 3.3709 | 64 | medium elite set |
| `screen-elite-k512-c9p512` | elite centroid | 0.005 | 9.512 | 512 | broad elite set |
| `screen-elite-k1024-c13p527` | elite centroid | 0.005 | 13.527 | 1,024 | top quarter of directions |

For elite updates, the fitter sign from every antithetic pair is retained,
the unique winners are ranked by loss, and the selected signed directions are
averaged. The commit scales were calibrated so generation-one parameter-update
RMS matches the control's approximately `0.002126`. Values above one therefore
mean extrapolation past the elite centroid; they avoid confounding elite count
with a progressively smaller update.

## Execution

The launcher acquires exactly one GPU and runs the ten configurations
sequentially:

```bash
bash scripts/run_eggroll_screening.sh
```

Results are scored with:

```bash
python scripts/summarize_eggroll_screening.py
```

All substantive runs use the W&B group
`eggroll-mqar-bounded-screening-seed7`.

## Results

The table is ordered by the predeclared screening score. `Change` is the
relative reduction in loss versus the reproduced control, so positive is
better. Update diagnostics are means over generations 1,501--2,000. Runtime is
GPU training time and excludes final evaluation and W&B upload overhead.

| Run | W&B | Score | Change | Update RMS | Update / parameter RMS | Runtime (s) | Decision |
|---|---|---:|---:|---:|---:|---:|---|
| `screen-zscore-s2e-2-lr7p5e-2` | [`6fpy9bvu`](https://wandb.ai/wobrob101/rnn-bptt-vs-egroll/runs/6fpy9bvu) | 8.682892 | +0.030% | 0.001882 | 0.014377 | 893.8 | reject |
| `screen-zscore-s1e-2-lr1p5e-1` | [`x76w5dx0`](https://wandb.ai/wobrob101/rnn-bptt-vs-egroll/runs/x76w5dx0) | 8.684152 | +0.016% | 0.002062 | 0.015382 | 894.3 | reject |
| `screen-zscore-s5e-3-lr3e-1-control` | [`pnqecpy3`](https://wandb.ai/wobrob101/rnn-bptt-vs-egroll/runs/pnqecpy3) | 8.685509 | baseline | 0.002114 | 0.015658 | 892.6 | control |
| `screen-zscore-s2p5e-3-lr6e-1` | [`jrll44ak`](https://wandb.ai/wobrob101/rnn-bptt-vs-egroll/runs/jrll44ak) | 8.689900 | -0.051% | 0.002120 | 0.015697 | 894.4 | reject |
| `screen-elite-k1024-c13p527` | [`56338xqr`](https://wandb.ai/wobrob101/rnn-bptt-vs-egroll/runs/56338xqr) | 8.715595 | -0.346% | 0.002122 | 0.015700 | 892.5 | reject |
| `screen-zscore-s5e-3-lr1p5e-1` | [`fe39srcg`](https://wandb.ai/wobrob101/rnn-bptt-vs-egroll/runs/fe39srcg) | 8.762033 | -0.881% | 0.001056 | 0.009504 | 894.1 | reject |
| `screen-elite-k512-c9p512` | [`d1na9x2y`](https://wandb.ai/wobrob101/rnn-bptt-vs-egroll/runs/d1na9x2y) | 8.791926 | -1.225% | 0.002112 | 0.015678 | 892.1 | reject |
| `screen-zscore-s5e-3-lr6e-1` | [`kmq42s2g`](https://wandb.ai/wobrob101/rnn-bptt-vs-egroll/runs/kmq42s2g) | 8.794922 | -1.260% | 0.004229 | 0.020768 | 894.3 | reject |
| `screen-elite-k64-c3p371` | [`simvjwyf`](https://wandb.ai/wobrob101/rnn-bptt-vs-egroll/runs/simvjwyf) | 8.990415 | -3.511% | 0.002114 | 0.015679 | 891.6 | reject |
| `screen-elite-k8-c1p198` | [`s0eqvhz5`](https://wandb.ai/wobrob101/rnn-bptt-vs-egroll/runs/s0eqvhz5) | 9.072038 | -4.450% | 0.002132 | 0.015745 | 891.9 | reject |

The ten screens used 8,931.7 seconds (2 h 28 min 51.7 s) of measured GPU
training time. Every run completed exactly 2,000 generations. Validation
accuracy remained zero at all five score points, so it did not provide a useful
secondary discriminator at this early horizon.

For the elite runs, mean selected fractions over the final 500 generations
were 0.001953, 0.015625, 0.125, and 0.25 for K=8, 64, 512, and 1,024. Mean
positive-sign fractions were 0.500, 0.502, 0.500, and 0.500 respectively. This
confirms that antithetic deduplication selected the configured number of unique
pairs without a persistent sign imbalance.

## Decision

The formal promotion threshold was `8.598654`. No configuration improved on
the reproduced control by the required 1%, so all nine alternatives were
rejected. Under the declared protocol there are therefore no 10,000-generation
finalists and no matched confirmation-seed runs.

The reproduced control exactly matched historical run `ceijp8r1` at every
scored generation. Matched `sigma`/learning-rate combinations also produced
nearly identical update RMS and scores, which suggests z-score EGGROLL is
locally insensitive to those radius changes when the effective step is held
approximately constant. Halving or doubling the update was worse. The
elite-centroid rule was also worse at every tested elite count despite matched
update RMS; performance approached the control as more directions were
averaged.

This is a bounded negative result, not evidence that elite selection can never
help. The study covers one model/task configuration, one screening seed, rank-1
perturbations, four elite counts, and only the first 2,000 generations. The
protocol intentionally stops here rather than selecting transient checkpoints
or expanding the search after observing the results.
