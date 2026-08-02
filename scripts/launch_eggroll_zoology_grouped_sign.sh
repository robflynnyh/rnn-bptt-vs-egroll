#!/usr/bin/env bash
set -euo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$repo_root"

export PYTHONPATH=src
mkdir -p .scratch artifacts/eggroll_zoology

generations="${GENERATIONS:-300000}"
name="eggroll-zoology-grouped-p131072-b16-bf16-sign-lr3e-1-to1e-2-by100k-g${generations}-seed7"
python -u -m rnn_bptt_vs_eggroll.experiment \
    --method eggroll \
    --preset reference \
    --device cuda \
    --seed 7 \
    --generations "$generations" \
    --batch-size 16 \
    --hidden-size 64 \
    --population-size 131072 \
    --population-chunk-size 8192 \
    --population-data-mode grouped \
    --population-precision bfloat16 \
    --perturbation-rank 1 \
    --sigma 0.005 \
    --fitness-shaping antithetic-sign \
    --eggroll-learning-rate 0.3 \
    --eggroll-learning-rate-final 0.01 \
    --eggroll-learning-rate-decay-start 0 \
    --eggroll-learning-rate-decay-end 100000 \
    --eggroll-weight-decay 0 \
    --curriculum-sequence-lengths 16,32,64,128,256,512,1024 \
    --curriculum-num-kv-pairs 1,2,4,8,16,32,64 \
    --curriculum-frontier-probability 1.0 \
    --evaluation-interval 100 \
    --log-interval 1 \
    --evaluation-examples 512 \
    --curriculum-probe-examples 512 \
    --test-examples 4096 \
    --output-dir "artifacts/eggroll_zoology/$name" \
    --wandb \
    --wandb-project rnn-bptt-vs-eggroll \
    --wandb-entity wobrob101 \
    --wandb-run-name "$name" \
    --wandb-group eggroll-zoology-seed7 \
    --log-progress
