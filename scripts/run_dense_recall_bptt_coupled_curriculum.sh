#!/usr/bin/env bash
set -euo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
with_gpu="${WITH_GPU:-/store/store5/software/simple-gpu-schedule/with-gpu}"
gpu_pool="${GPU_POOL:-all}"
python_bin="${PYTHON_BIN:-/store/store4/software/bin/anaconda3/envs/flash_attn_pytorch2/bin/python}"
target_generations="${GENERATIONS:-2000000}"
output_dir="${OUTPUT_DIR:-$repo_root/artifacts/dense_recall/seed-7/bptt-lstm-tied-coupled-vocab}"
wandb_run_id="${WANDB_RUN_ID:-bpttcv01}"

if [[ ! -x "$python_bin" ]]; then
    echo "Set PYTHON_BIN to an executable Python interpreter" >&2
    exit 2
fi
if [[ ! "$target_generations" =~ ^[0-9]+$ ]] || (( target_generations < 1 )); then
    echo "GENERATIONS must be a positive integer" >&2
    exit 2
fi

if [[ "${DENSE_RECALL_BPTT_ON_GPU:-0}" != 1 ]]; then
    exec "$with_gpu" "$gpu_pool" --num 1 -- env \
        DENSE_RECALL_BPTT_ON_GPU=1 \
        PYTHON_BIN="$python_bin" \
        GENERATIONS="$target_generations" \
        OUTPUT_DIR="$output_dir" \
        WANDB_RUN_ID="$wandb_run_id" \
        bash "$0" "$@"
fi

cd "$repo_root"
export PYTHONPATH=src
export WANDB__SERVICE_WAIT="${WANDB__SERVICE_WAIT:-300}"
mkdir -p "$output_dir/checkpoints"

resume_args=()
latest_checkpoint=$(
    find "$output_dir/checkpoints" -maxdepth 1 -type f \
        -name 'generation_*.pt' -print | sort | tail -n 1
)
if [[ -n "$latest_checkpoint" ]]; then
    echo "Resuming full state from $latest_checkpoint"
    resume_args+=(--resume-checkpoint "$latest_checkpoint")
else
    echo "Starting coupled sequence/vocabulary curriculum from scratch"
fi

exec "$python_bin" -u -m rnn_bptt_vs_eggroll.experiment \
    --method bptt \
    --preset reference \
    --architecture lstm \
    --curriculum-schedule dense-recall-from-2 \
    --dense-recall-coupled-vocab \
    --device cuda \
    --seed 7 \
    --generations "$target_generations" \
    --batch-size 256 \
    --hidden-size 128 \
    --tie-input-output \
    --bptt-learning-rate 0.001 \
    --bptt-weight-decay 0 \
    --bptt-gradient-clip 5 \
    --curriculum-accuracy-threshold 0.9 \
    --curriculum-max-updates-per-stage 20000 \
    --curriculum-frontier-probability 1 \
    --curriculum-probe-examples 512 \
    --no-final-full-curriculum-probe \
    --evaluation-frontier-only \
    --evaluation-interval 100 \
    --evaluation-examples 512 \
    --test-examples 4096 \
    --log-interval 100 \
    --wandb-log-interval 1 \
    --checkpoint-interval 20000 \
    --output-dir "$output_dir" \
    --wandb \
    --wandb-project rnn-bptt-vs-eggroll \
    --wandb-entity wobrob101 \
    --wandb-run-name dense-recall-bptt-lstm-tied-coupled-vocab-seed7 \
    --wandb-group dense-single-query-recall-coupled-curriculum \
    --wandb-run-id "$wandb_run_id" \
    --wandb-resume allow \
    --log-progress \
    "${resume_args[@]}" \
    "$@"
