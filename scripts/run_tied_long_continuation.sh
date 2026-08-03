#!/usr/bin/env bash
set -euo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
with_gpu="${WITH_GPU:-/store/store5/software/simple-gpu-schedule/with-gpu}"
gpu_pool="${GPU_POOL:-all}"
python_bin="${PYTHON_BIN:-$(command -v python || true)}"
target_generations="${GENERATIONS:-2000000}"
output_dir="$repo_root/artifacts/gentle_curriculum/seed-7/gentle-tied-lr0p1-long-2m"
parent_dir="$repo_root/artifacts/gentle_curriculum/seed-7/gentle-tied"

if [[ ! -x "$python_bin" ]]; then
    echo "Set PYTHON_BIN to an executable Python interpreter" >&2
    exit 2
fi
if [[ ! "$target_generations" =~ ^[0-9]+$ ]] || (( target_generations <= 20000 )); then
    echo "GENERATIONS must be an integer greater than 20000" >&2
    exit 2
fi
if [[ ! -f "$parent_dir/model.pt" || ! -f "$parent_dir/metrics.json" ]]; then
    echo "The completed tied 20k parent artifacts are required" >&2
    exit 2
fi

if [[ "${TIED_LONG_ON_GPU:-0}" != 1 ]]; then
    exec "$with_gpu" "$gpu_pool" --num 1 -- env \
        TIED_LONG_ON_GPU=1 \
        PYTHON_BIN="$python_bin" \
        GENERATIONS="$target_generations" \
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
    echo "Bootstrapping from the completed tied 20k run"
    resume_args+=(
        --bootstrap-model-path "$parent_dir/model.pt"
        --bootstrap-metrics-path "$parent_dir/metrics.json"
        --bootstrap-allow-optimizer-override
    )
fi

exec "$python_bin" -u -m rnn_bptt_vs_eggroll.experiment \
    --method eggroll \
    --preset reference \
    --curriculum-schedule gentle \
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
    --eggroll-learning-rate 0.1 \
    --eggroll-learning-rate-final 0.01 \
    --eggroll-learning-rate-decay-start 20000 \
    --eggroll-learning-rate-decay-end 100000 \
    --eggroll-weight-decay 0 \
    --curriculum-accuracy-threshold 0.9 \
    --curriculum-frontier-probability 0.5 \
    --curriculum-probe-examples 512 \
    --final-full-curriculum-probe \
    --evaluation-interval 100 \
    --evaluation-examples 512 \
    --test-examples 512 \
    --log-interval 100 \
    --checkpoint-interval 20000 \
    --output-dir "$output_dir" \
    --wandb \
    --wandb-project rnn-bptt-vs-eggroll \
    --wandb-entity wobrob101 \
    --wandb-run-name gentle-curriculum-gentle-tied-seed7-lr0p1-2m \
    --wandb-group eggroll-mqar-gentle-curriculum-long-seed7 \
    --wandb-run-id tiedlr10 \
    --wandb-resume allow \
    --wandb-log-interval 1 \
    --log-progress \
    "${resume_args[@]}"
