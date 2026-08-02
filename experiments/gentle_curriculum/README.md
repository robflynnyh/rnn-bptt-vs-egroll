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

If and only if the strict promotion rule passes, run seed 8 with:

```bash
SEED=8 bash scripts/run_gentle_curriculum_comparison.sh
```

Substantive runs use W&B group
`eggroll-mqar-gentle-curriculum-seed<seed>`. Results, links, transition tables,
decisions, and conclusions will be added after the bounded runs complete.
