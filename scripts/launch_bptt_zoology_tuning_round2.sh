#!/usr/bin/env bash
set -euo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$repo_root"

export PYTHONPATH=src
mkdir -p .scratch artifacts/bptt_zoology_tuning_round2

pids=()

launch() {
    local name=$1
    local batch_size=$2
    local learning_rate=$3
    local hidden_size=$4

    python -u -m rnn_bptt_vs_eggroll.experiment \
        --method bptt \
        --preset reference \
        --device cuda \
        --seed 7 \
        --generations 20000 \
        --batch-size "$batch_size" \
        --hidden-size "$hidden_size" \
        --bptt-learning-rate "$learning_rate" \
        --curriculum-sequence-lengths 16,32,64 \
        --curriculum-num-kv-pairs 1,2,4 \
        --curriculum-frontier-probability 1.0 \
        --evaluation-interval 500 \
        --evaluation-examples 256 \
        --curriculum-probe-examples 512 \
        --test-examples 2048 \
        --output-dir "artifacts/bptt_zoology_tuning_round2/$name" \
        --wandb \
        --wandb-project rnn-bptt-vs-eggroll \
        --wandb-entity wobrob101 \
        --wandb-run-name "$name" \
        --wandb-group bptt-zoology-tuning-round2-seed7 \
        --log-progress \
        > ".scratch/$name.log" 2>&1 &
    pids+=("$!")
}

launch bptt-zoology-r2-d64-b1024-lr3e-3 1024 0.003 64
launch bptt-zoology-r2-d64-b4096-lr3e-3 4096 0.003 64
launch bptt-zoology-r2-d64-b4096-lr1e-2 4096 0.01 64
launch bptt-zoology-r2-d128-b1024-lr1e-3 1024 0.001 128

status=0
for pid in "${pids[@]}"; do
    if ! wait "$pid"; then
        status=1
    fi
done
exit "$status"
