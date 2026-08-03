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

The exact input/output weight-tying follow-up uses the same command with:

```bash
SCHEDULES=gentle TIE_INPUT_OUTPUT=1 bash scripts/run_gentle_curriculum_comparison.sh
```

This reuses the input table, transposed, as the output classifier. Candidate
perturbations are tied too: EGGROLL samples one rank-1 change to the shared
table and uses its exact transpose at the output. The tied run has 536,640
parameters rather than 1,060,928. It is an architecture ablation, not the
schedule-only treatment specified by the original comparison.

An additional tied follow-up learns separate relative mutation scales for the
shared token matrix, recurrent matrix, hidden bias, and output bias:

```bash
SCHEDULES=gentle TIE_INPUT_OUTPUT=1 ADAPTIVE_MUTATION_SCALES=1 \
  bash scripts/run_gentle_curriculum_comparison.sh
```

It uses a separable NES natural-gradient estimate from each block's normalized
perturbation energy and candidate fitness. Log scales start at 1, use learning
rate 0.5, and are bounded to `[0.1, 10]`. The mean-gradient estimate divides
out each relative radius, so scale adaptation changes exploration without
silently changing the EGGROLL commit learning rate. This is a follow-up
optimization ablation and does not replace the predeclared schedule comparison.

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
| Scheduled-LR reference evidence | 7 | Reference | 14,392 | [`ceijp8r1`](https://wandb.ai/wobrob101/rnn-bptt-vs-egroll/runs/ceijp8r1) | Ran until manually stopped |
| Long reference evidence | 7 | Reference | 83,662 | [`gunzsjri`](https://wandb.ai/wobrob101/rnn-bptt-vs-egroll/runs/gunzsjri) | Ran until manually stopped |
| Exact duplicate reference | 7 | Reference | 4,290 | [`59vrqlqz`](https://wandb.ai/wobrob101/rnn-bptt-vs-egroll/runs/59vrqlqz) | Intentionally stopped as redundant |
| Untied treatment | 7 | Gentle | 13,200 | [`is9ms5za`](https://wandb.ai/wobrob101/rnn-bptt-vs-egroll/runs/is9ms5za) | Intentionally stopped for tied ablation |
| Tied follow-up | 7 | Gentle, tied input/output | 20,000 | [`okos5dn0`](https://wandb.ai/wobrob101/rnn-bptt-vs-egroll/runs/okos5dn0) | Complete; reached raw length 32 |
| Adaptive-scale follow-up | 7 | Gentle, tied input/output, learned block scales | 3,180 | [`0fjtgz6y`](https://wandb.ai/wobrob101/rnn-bptt-vs-egroll/runs/0fjtgz6y) | Intentionally stopped after scale runaway |
| Long tied continuation, initial logging | 7 | Gentle, tied input/output | 1,600 continuation generations | [`tied2m07`](https://wandb.ai/wobrob101/rnn-bptt-vs-egroll/runs/tied2m07) | Superseded to log every update |
| Long tied continuation, original LR | 7 | Gentle, tied input/output | 70,400 total | [`tied2m7b`](https://wandb.ai/wobrob101/rnn-bptt-vs-egroll/runs/tied2m7b) | Stopped after sustained regression; continuation checkpoints removed |
| Long tied continuation, LR 0.1 | 7 | Gentle, tied input/output | 2,000,000 ceiling | [`tiedlr10`](https://wandb.ai/wobrob101/rnn-bptt-vs-egroll/runs/tiedlr10) | Active; bootstrapped at generation 20,000 |

The scheduled-LR reference used the same seed, model, Cartesian population,
batch, BF16 candidate forwards, rank-1 perturbations, `sigma`, z-score update,
cosine learning-rate schedule, no weight decay, fixed 512-example probe,
threshold, and reference schedule. Its frontier sampling probability was 1.0,
which has no effect while stage zero is the only reached stage. It never
advanced from `(16,1)` over 14,392 generations (6,555.0 seconds; final probe
accuracy 0.195%).

The longer reference was identical except that its learning rate stayed at
0.3. At generation 20,000 its `(16,1)` probe accuracy was 0%, and it never
advanced over 83,662 generations (38,114.9 seconds; final probe accuracy
0.586%). Together these runs are stronger negative evidence for the reference
schedule than spending more compute on the duplicate. The constant learning
rate in the longer run remains a comparison limitation, so matched early
diagnostics use the scheduled-LR run where relevant.

The exact duplicate included the cosine learning-rate schedule and 0.5
frontier rehearsal specified above. It also remained at `(16,1)` through
generation 4,290. It was stopped once the prior long reference run was
identified, before the launcher could start the gentle treatment. The gentle
run is therefore the only remaining seed-7 compute. This execution decision
was made to avoid repeating an already established negative control; the
learning-rate difference is retained as a comparison limitation. The untied
gentle treatment then remained at `(4,1)` through generation 13,200. Its final
fixed-probe accuracy was 9.18%, and its best observed fixed-probe accuracy was
10.94%. It was stopped at the user's request to test exact input/output weight
tying, which directly targets the coordination bottleneck described below.

The tied follow-up passed seven curriculum gates between generations 17,400
and 18,400, ending at raw length 32 with two key-value pairs and logical query
span 14. Its final fixed-probe accuracy was 91.60% at raw length 4, 73.24% at
raw length 16, and 17.29% at the newly reached raw length 32. This establishes
that exact input/output tying removed the early optimization bottleneck, but
the 20,000-generation boundary arrived shortly after promotion to length 32.

The original-LR continuation regressed: length-32 fixed-probe accuracy fell
from 17.29% at generation 20,000 to 11.91% at generation 70,400, while mean
accuracy across reached stages fell from 72.81% to 54.98%. The parameter norm
also rose steadily. It was stopped and its generation-40,000 and -60,000
checkpoints were removed; the successful generation-20,000 parent is retained.

The replacement continuation loads that parent model, curriculum stage,
transition history, mutation scales, and cumulative training time. It lowers
the initial EGGROLL learning rate from 0.3 to 0.1 and cosine-decays it to 0.01
between generations 20,000 and 100,000. The old parent did not save RNG or
optimizer state, so its first continuation segment uses new, deterministic data
and perturbation streams. From generation 40,000 onward, atomic full-state
checkpoints every 20,000 generations preserve exact model, optimizer,
curriculum, mutation-scale, and RNG continuity. Relaunching
`scripts/run_tied_long_continuation.sh` automatically selects the newest such
checkpoint. The 2,000,000-generation value is a manual-stop ceiling, not a
claim that all of that compute is required. Training metrics are sent to W&B
after every update; local in-memory and JSON history remains sampled every 100
generations so it stays bounded.

## Interim matched diagnostics

These values compare the scheduled-LR reference and gentle treatment at the
same generation. Accuracy is from each run's fixed 512-example frontier probe.
The tasks differ here by design: reference probes `(16,1)`, while gentle is
still probing `(4,1)`. They therefore diagnose optimization trajectory rather
than the predeclared common-milestone outcome.

| Generation | Reference train loss | Reference probe | Gentle train loss | Gentle probe |
|---:|---:|---:|---:|---:|
| 1,000 | 8.8186 | 0.000% | 8.7043 | 0.195% |
| 2,000 | 8.5271 | 0.000% | 8.4401 | 0.000% |
| 4,000 | 8.4172 | 0.195% | 8.0334 | 0.781% |
| 6,000 | 8.4206 | 0.000% | 7.6883 | 2.148% |

The gentle run has a clearly better early optimization trajectory, but this is
not yet evidence that it masters a higher common milestone. The primary result
remains gated on the full 20,000-generation run and the fixed 90% criterion.

## Preliminary interpretation

The gentle trajectory should not be described as efficient learning merely
because it improves on the failed reference. Prior BPTT runs mastered the
harder initial `(16,1)` stage in 500--4,500 updates, including
[`6llb1eju`](https://wandb.ai/wobrob101/rnn-bptt-vs-egroll/runs/6llb1eju)
at generation 500 and
[`4zaw3lt7`](https://wandb.ai/wobrob101/rnn-bptt-vs-egroll/runs/4zaw3lt7)
at generation 1,100. This establishes that the task and model architecture are
learnable; the slow gentle result is specific to the EGGROLL optimization.

The model has 1,060,928 parameters. Its independent 8,192-token input and
output weight tables contain 1,048,576 of them (98.84%). Generic MQAR retrieval
requires the hidden representation induced by a random value token to align
with that same token's output row. Standardized EGGROLL estimates this
coordination from one scalar fitness per candidate while applying independent
rank-1 perturbations to both large matrices. A plausible explanation is that
useful changes on either side have weak reward until the other side is already
aligned. This hypothesis fits the observed slow loss reduction, but the
curriculum experiment does not isolate or prove it. Weight tying would be a
useful follow-up ablation after this bounded protocol, not a change to the
active run.

The 8,192-token vocabulary is inherited from Zoology to retain a realistic
language-model readout scale; it is not required by associative recall itself.
Under this repository's current generator, an even vocabulary just above the
maximum raw length of 1,024 would suffice, making 2,048 a natural reduced-vocab
control. Such a control would reduce the dominant token tables fourfold and
better isolate recurrent-memory optimization, at the cost of no longer
matching Zoology's vocabulary scale. It is a follow-up, not part of this fixed
protocol.

## Adaptive-scale ablation

The adaptive run was stopped at generation 3,180 rather than spending the full
budget on a clearly unhealthy search distribution. Its shared input/output
table radius had risen to 9.303 times the base `sigma`, close to the declared
10-times cap, while the recurrent and hidden-bias radii fell to 0.715 and
0.718. The output-bias radius remained 1.005. At the matched generation 3,180,
fixed and adaptive runs both had 6.25% batch accuracy, but fixed-scale training
loss was 6.193 versus 7.866 adaptive. Their nearby fixed-probe accuracies were
also similar rather than improved by adaptation.

The separable NES objective rewards mutation radii that produce better sampled
offspring. Here it consistently favored a much wider shared-table search. The
mean-gradient estimator must divide out that relative radius to keep mutation
scale separate from the commit learning rate, so this expansion weakened the
shared table's committed update and slowed optimization. This negative result
does not show that every adaptive-radius method will fail, but this estimator
and learning rate did not improve the working fixed-scale setup. No post-hoc
retuning run was launched.
