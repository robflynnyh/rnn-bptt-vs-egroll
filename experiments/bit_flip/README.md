# One-bit flip curriculum

This experiment separates long recurrent credit assignment from growing memory
and output-class requirements. Each example has the form:

```text
<START_1> HOLD FLIP HOLD FLIP <QUERY> -> 1
```

The hidden state only needs to represent one bit. `HOLD` preserves it and
`FLIP` toggles it. The input vocabulary always has five symbols and the output
always has two classes. Increasing the operation count therefore lengthens the
recurrent computation without increasing the information stored or the number
of output alternatives.

The initial BPTT baseline uses a single-layer, 16-unit tanh RNN, Adam at
`3e-3`, batch size 256, no weight decay, and gradient clipping at 1. Training
uses only the current curriculum frontier. A stage advances the first time its
fixed 2,048-example validation probe reaches 95% accuracy. Operation counts are
dense from 1 through 32 and then grow by 25% up to 16,384. Training stops after
100,000 updates without promotion, after passing the maximum operation count,
or at the global two-million-update limit.

```bash
bash scripts/run_bit_flip_bptt.sh
```

The run keeps one overwritten `checkpoint.pt`, rather than accumulating
generation snapshots. Final weights and metrics are written as `model.pt` and
`metrics.json`.

## Initial result

The seed-7 BPTT run passed every one of the 60 curriculum stages, reaching
100% fixed-validation accuracy at 16,384 operations after 6,000 updates. No
BPTT length limit was observed. This is a useful negative result for the
optimizer comparison: once the tanh RNN discovered the exact one-bit state
transition, increasing the number of repeated operations did not make the task
harder for BPTT under curriculum.

[W&B run](https://wandb.ai/wobrob101/rnn-bptt-vs-eggroll/runs/bitflipb1)
