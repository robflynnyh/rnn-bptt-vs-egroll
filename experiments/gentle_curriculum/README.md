# Gentle MQAR curriculum

This bounded study tests whether standardized EGGROLL can master longer MQAR
dependencies when training starts below the reference curriculum's raw token
length of 16.

## Length semantics

MQAR needs at least four raw tokens: key, value, query, and one non-query slot.
For one KV pair, this study calls the number of available query-distance slots
the **logical query span**:

```text
raw token length = 2 + 2 * logical query span
```

Logical spans 1, 2, 3, and 4 therefore mean raw token lengths 4, 6, 8, and 10.
They do not mean one-token through four-token MQAR examples.

## Schedules

The reference schedule is unchanged:

```text
(16,1), (32,2), (64,4), (128,8), (256,16), (512,32), (1024,64)
```

The gentle schedule adds a one-pair prefix and then rejoins the reference
milestones:

```text
(4,1), (6,1), (8,1), (10,1), (12,1), (14,1), (16,1),
(32,2), (64,4), (128,8), (256,16), (512,32), (1024,64)
```

One fixed 512-example frontier probe at or above 90% accuracy advances one
stage. Half of training batches use the frontier and half rehearse reached
earlier stages.

## Protocol

Both schedules use seed 7 and exactly 20,000 generations of the reproduced
standardized EGGROLL setup: Cartesian population 8,192 by batch 64, BF16
candidate forwards, rank-1 perturbations, `sigma=0.005`, z-score fitness, SGD
learning rate 0.3 with the existing cosine schedule, no weight decay, and
hidden size 64. The schedule is the intended experimental difference.

The primary outcome is the highest common milestone mastered. A confirmation
pair at seed 8 is run only if gentle masters a strictly higher common milestone
at seed 7. At most four substantive runs are permitted.

Run seed 7 with:

```bash
bash scripts/run_gentle_curriculum_comparison.sh
```

To run only one side while preserving the same configuration, set
`SCHEDULES=reference` or `SCHEDULES=gentle`.

If and only if the strict promotion rule passes, run seed 8 with:

```bash
SEED=8 bash scripts/run_gentle_curriculum_comparison.sh
```

Substantive runs use W&B group
`eggroll-mqar-gentle-curriculum-seed<seed>`. Results, links, transition tables,
decisions, and conclusions will be added after the bounded runs complete.

## Run ledger

| Role | Seed | Schedule | Generations | W&B | Status |
|---|---:|---|---:|---|---|
| Existing reference evidence | 7 | Reference | 83,662 | [`gunzsjri`](https://wandb.ai/wobrob101/rnn-bptt-vs-egroll/runs/gunzsjri) | Completed until manually stopped |
| Exact duplicate reference | 7 | Reference | 4,290 | [`59vrqlqz`](https://wandb.ai/wobrob101/rnn-bptt-vs-egroll/runs/59vrqlqz) | Intentionally stopped as redundant |
| Bounded treatment | 7 | Gentle | 20,000 target | [`is9ms5za`](https://wandb.ai/wobrob101/rnn-bptt-vs-egroll/runs/is9ms5za) | Running |

The existing reference used the same seed, model, Cartesian population,
batch, BF16 candidate forwards, rank-1 perturbations, `sigma`, z-score update,
SGD learning rate, no weight decay, fixed 512-example probe, threshold, and
reference schedule. It used constant learning rate 0.3 and frontier sampling
probability 1.0. The latter difference has no effect while stage zero is the
only reached stage. At generation 20,000 its `(16,1)` probe accuracy was 0%,
and it never advanced from `(16,1)` over 83,662 generations (38,114.9 seconds,
final probe accuracy 0.586%). This is stronger negative evidence for the
reference schedule than spending more compute on the duplicate.

The exact duplicate included the cosine learning-rate schedule and 0.5
frontier rehearsal specified above. It also remained at `(16,1)` through
generation 4,290. It was stopped once the prior long reference run was
identified, before the launcher could start the gentle treatment. The gentle
run is therefore the only remaining seed-7 compute. This execution decision
was made to avoid repeating an already established negative control; the
learning-rate difference is retained as a comparison limitation.

## Interim matched diagnostics

These values compare the existing reference and gentle treatment at the same
generation. Accuracy is from each run's fixed 512-example frontier probe. The
tasks differ here by design: reference probes `(16,1)`, while gentle is still
probing `(4,1)`. They therefore diagnose optimization trajectory rather than
the predeclared common-milestone outcome.

| Generation | Reference train loss | Reference probe | Gentle train loss | Gentle probe |
|---:|---:|---:|---:|---:|
| 1,000 | 8.8205 | 0.000% | 8.7043 | 0.195% |
| 2,000 | 8.5307 | 0.000% | 8.4401 | 0.000% |
| 4,000 | 8.4220 | 0.000% | 8.0334 | 0.781% |
| 6,000 | 8.3976 | 0.000% | 7.6883 | 2.148% |

The gentle run has a clearly better early optimization trajectory, but this is
not yet evidence that it masters a higher common milestone. The primary result
remains gated on the full 20,000-generation run and the fixed 90% criterion.
