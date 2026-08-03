#!/usr/bin/env bash
set -euo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
with_gpu="${WITH_GPU:-/store/store5/software/simple-gpu-schedule/with-gpu}"
gpu_pool="${GPU_POOL:-all}"
python_bin="${PYTHON_BIN:-/store/store4/software/bin/anaconda3/envs/flash_attn_pytorch2/bin/python}"
target_generations="${GENERATIONS:-2000000}"
curriculum_schedule="${CURRICULUM_SCHEDULE:-dense-recall-from-2}"

case "$curriculum_schedule" in
    dense-recall)
        run_slug="eggroll-tied-from-1"
        wandb_run_id="densekv07"
        wandb_run_name="dense-single-query-recall-from-1-eggroll-tied-seed7"
        ;;
    dense-recall-from-2)
        run_slug="eggroll-tied-from-2"
        wandb_run_id="densekv2a"
        wandb_run_name="dense-single-query-recall-from-2-eggroll-tied-seed7"
        ;;
    *)
        echo "Unsupported CURRICULUM_SCHEDULE: $curriculum_schedule" >&2
        exit 2
        ;;
esac

output_dir="${OUTPUT_DIR:-$repo_root/artifacts/dense_recall/seed-7/$run_slug}"

if [[ ! -x "$python_bin" ]]; then
    echo "Set PYTHON_BIN to an executable Python interpreter" >&2
    exit 2
fi
if [[ ! "$target_generations" =~ ^[0-9]+$ ]] || (( target_generations < 1 )); then
    echo "GENERATIONS must be a positive integer" >&2
    exit 2
fi

if [[ "${DENSE_RECALL_ON_GPU:-0}" != 1 ]]; then
    exec "$with_gpu" "$gpu_pool" --num 1 -- env \
        DENSE_RECALL_ON_GPU=1 \
        PYTHON_BIN="$python_bin" \
        GENERATIONS="$target_generations" \
        CURRICULUM_SCHEDULE="$curriculum_schedule" \
        OUTPUT_DIR="$output_dir" \
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
    echo "Starting dense-recall curriculum from a fresh initialization"
fi

exec "$python_bin" -u -m rnn_bptt_vs_eggroll.experiment \
    --method eggroll \
    --preset reference \
    --curriculum-schedule "$curriculum_schedule" \
    --device cuda \
    --seed 7 \
    --generations "$target_generations" \
    --batch-size 64 \
    --hidden-size 64 \
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
    --eggroll-learning-rate 0.4 \
    --eggroll-learning-rate-final 0.01 \
    --eggroll-learning-rate-decay-start 0 \
    --eggroll-learning-rate-decay-end "$target_generations" \
    --eggroll-weight-decay 0 \
    --curriculum-accuracy-threshold 0.9 \
    --curriculum-frontier-probability 1 \
    --curriculum-probe-examples 512 \
    --no-final-full-curriculum-probe \
    --evaluation-frontier-only \
    --evaluation-interval 100 \
    --evaluation-examples 512 \
    --test-examples 512 \
    --log-interval 100 \
    --wandb-log-interval 1 \
    --checkpoint-interval 20000 \
    --output-dir "$output_dir" \
    --wandb \
    --wandb-project rnn-bptt-vs-eggroll \
    --wandb-entity wobrob101 \
    --wandb-run-name "$wandb_run_name" \
    --wandb-group eggroll-dense-single-query-recall \
    --wandb-run-id "$wandb_run_id" \
    --wandb-resume allow \
    --log-progress \
    "${resume_args[@]}" \
    "$@"
