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
`eggroll-mqar-bounded-screening-seed7`. Final screening results, promotion
decisions, finalist links, confirmation evidence, and the scientific conclusion
will be added here after the bounded jobs finish.
