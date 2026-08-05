#!/usr/bin/env bash
set -euo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
with_gpu="${WITH_GPU:-/store/store5/software/simple-gpu-schedule/with-gpu}"
gpu_pool="${GPU_POOL:-all}"
python_bin="${PYTHON_BIN:-/store/store4/software/bin/anaconda3/envs/flash_attn_pytorch2/bin/python}"
target_generations="${GENERATIONS:-2100000}"
parent_dir="${PARENT_DIR:-$repo_root/artifacts/dense_recall/seed-7/bptt-lstm-tied-coupled-vocab}"
output_dir="${OUTPUT_DIR:-$repo_root/artifacts/dense_recall/seed-7/eggroll-from-bptt-2m-lr0p03-decay10k}"
wandb_run_id="${WANDB_RUN_ID:-bpttegg3}"

if [[ ! -x "$python_bin" ]]; then
    echo "Set PYTHON_BIN to an executable Python interpreter" >&2
    exit 2
fi
if [[ ! "$target_generations" =~ ^[0-9]+$ ]] || (( target_generations <= 2000000 )); then
    echo "GENERATIONS must be an integer greater than the 2000000-step parent" >&2
    exit 2
fi
if [[ ! -f "$parent_dir/model.pt" || ! -f "$parent_dir/metrics.json" ]]; then
    echo "The completed 2M-step BPTT model and metrics are required" >&2
    exit 2
fi

if [[ "${DENSE_RECALL_EGGROLL_CONTINUATION_ON_GPU:-0}" != 1 ]]; then
    exec "$with_gpu" "$gpu_pool" --num 1 -- env \
        DENSE_RECALL_EGGROLL_CONTINUATION_ON_GPU=1 \
        PYTHON_BIN="$python_bin" \
        GENERATIONS="$target_generations" \
        PARENT_DIR="$parent_dir" \
        OUTPUT_DIR="$output_dir" \
        WANDB_RUN_ID="$wandb_run_id" \
        bash "$0" "$@"
fi

cd "$repo_root"
export PYTHONPATH=src
export WANDB__SERVICE_WAIT="${WANDB__SERVICE_WAIT:-300}"
mkdir -p "$output_dir/checkpoints"

start_args=()
latest_checkpoint=$(
    find "$output_dir/checkpoints" -maxdepth 1 -type f \
        -name 'generation_*.pt' -print | sort | tail -n 1
)
if [[ -n "$latest_checkpoint" ]]; then
    echo "Resuming EGGROLL state from $latest_checkpoint"
    start_args+=(--resume-checkpoint "$latest_checkpoint")
else
    echo "Bootstrapping EGGROLL from the completed 2M-step BPTT model"
    start_args+=(
        --bootstrap-model-path "$parent_dir/model.pt"
        --bootstrap-metrics-path "$parent_dir/metrics.json"
        --bootstrap-allow-method-override
    )
fi

exec "$python_bin" -u -m rnn_bptt_vs_eggroll.experiment \
    --method eggroll \
    --preset reference \
    --architecture lstm \
    --curriculum-schedule dense-recall-from-2 \
    --dense-recall-coupled-vocab \
    --device cuda \
    --seed 7 \
    --generations "$target_generations" \
    --batch-size 64 \
    --hidden-size 128 \
    --tie-input-output \
    --population-size 8192 \
    --population-chunk-size 1024 \
    --population-data-mode cartesian \
    --population-precision bfloat16 \
    --perturbation-rank 1 \
    --sigma 0.005 \
    --no-adaptive-mutation-scales \
    --fitness-shaping zscore \
    --eggroll-update-rule standardized \
    --eggroll-learning-rate 0.03 \
    --eggroll-learning-rate-final 0.003 \
    --eggroll-learning-rate-decay-start 2000000 \
    --eggroll-learning-rate-decay-end 2010000 \
    --eggroll-weight-decay 0 \
    --bptt-learning-rate 0.001 \
    --bptt-weight-decay 0 \
    --bptt-gradient-clip 5 \
    --curriculum-accuracy-threshold 0.9 \
    --curriculum-frontier-probability 1 \
    --curriculum-probe-examples 512 \
    --no-final-full-curriculum-probe \
    --evaluation-frontier-only \
    --evaluation-interval 100 \
    --evaluation-examples 512 \
    --test-examples 4096 \
    --log-interval 1 \
    --wandb-log-interval 1 \
    --checkpoint-interval 1000 \
    --output-dir "$output_dir" \
    --wandb \
    --wandb-project rnn-bptt-vs-eggroll \
    --wandb-entity wobrob101 \
    --wandb-run-name dense-recall-eggroll-from-bptt-2m-lr0p03-decay10k-seed7 \
    --wandb-group dense-recall-optimizer-handoff \
    --wandb-run-id "$wandb_run_id" \
    --wandb-resume allow \
    --log-progress \
    "${start_args[@]}" \
    "$@"
