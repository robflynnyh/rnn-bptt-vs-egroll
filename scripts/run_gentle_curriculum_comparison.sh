#!/usr/bin/env bash
set -euo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
with_gpu="${WITH_GPU:-/store/store5/software/simple-gpu-schedule/with-gpu}"
gpu_pool="${GPU_POOL:-all}"
seed="${SEED:-7}"
schedules="${SCHEDULES:-reference gentle}"
tie_input_output="${TIE_INPUT_OUTPUT:-0}"
adaptive_mutation_scales="${ADAPTIVE_MUTATION_SCALES:-0}"
python_bin="${PYTHON_BIN:-$(command -v python || true)}"

if [[ "$seed" != 7 && "$seed" != 8 ]]; then
    echo "SEED must be 7 or the predeclared confirmation seed 8" >&2
    exit 2
fi
if [[ "$tie_input_output" != 0 && "$tie_input_output" != 1 ]]; then
    echo "TIE_INPUT_OUTPUT must be 0 or 1" >&2
    exit 2
fi
if [[ "$adaptive_mutation_scales" != 0 && "$adaptive_mutation_scales" != 1 ]]; then
    echo "ADAPTIVE_MUTATION_SCALES must be 0 or 1" >&2
    exit 2
fi
if [[ -z "$python_bin" || ! -x "$python_bin" ]]; then
    echo "Set PYTHON_BIN to an executable Python interpreter" >&2
    exit 2
fi

if [[ "${GENTLE_CURRICULUM_ON_GPU:-0}" != 1 ]]; then
    exec "$with_gpu" "$gpu_pool" --num 1 -- env \
        GENTLE_CURRICULUM_ON_GPU=1 SEED="$seed" \
        TIE_INPUT_OUTPUT="$tie_input_output" \
        ADAPTIVE_MUTATION_SCALES="$adaptive_mutation_scales" \
        PYTHON_BIN="$python_bin" \
        bash "$0" "$@"
fi

cd "$repo_root"
export PYTHONPATH=src
export WANDB__SERVICE_WAIT="${WANDB__SERVICE_WAIT:-300}"

output_root="artifacts/gentle_curriculum/seed-${seed}"
mkdir -p "$output_root"

run_schedule() {
    local schedule=$1
    local variant="$schedule"
    local tied_args=()
    local adaptive_args=()
    if [[ "$tie_input_output" == 1 ]]; then
        variant="${schedule}-tied"
        tied_args+=(--tie-input-output)
    fi
    if [[ "$adaptive_mutation_scales" == 1 ]]; then
        variant="${variant}-adaptive-scales"
        adaptive_args+=(
            --adaptive-mutation-scales
            --mutation-scale-learning-rate 0.5
            --mutation-scale-min 0.1
            --mutation-scale-max 10
        )
    fi
    local name="gentle-curriculum-${variant}-seed${seed}-20k"
    local output_dir="$output_root/$variant"

    if [[ -f "$output_dir/metrics.json" ]]; then
        echo "Skipping completed run: $name"
        return
    fi
    rm -rf "$output_dir"

    "$python_bin" -u -m rnn_bptt_vs_eggroll.experiment \
        --method eggroll \
        --preset reference \
        --curriculum-schedule "$schedule" \
        --device cuda \
        --seed "$seed" \
        --generations 20000 \
        --batch-size 64 \
        --hidden-size 64 \
        "${tied_args[@]}" \
        --population-size 8192 \
        --population-chunk-size 1024 \
        --population-data-mode cartesian \
        --population-precision bfloat16 \
        --perturbation-rank 1 \
        --sigma 0.005 \
        "${adaptive_args[@]}" \
        --fitness-shaping zscore \
        --eggroll-update-rule standardized \
        --eggroll-learning-rate 0.3 \
        --eggroll-learning-rate-final 0.01 \
        --eggroll-learning-rate-decay-start 0 \
        --eggroll-learning-rate-decay-end 100000 \
        --eggroll-weight-decay 0 \
        --curriculum-accuracy-threshold 0.9 \
        --curriculum-frontier-probability 0.5 \
        --curriculum-probe-examples 512 \
        --final-full-curriculum-probe \
        --evaluation-interval 100 \
        --evaluation-examples 512 \
        --test-examples 512 \
        --log-interval 10 \
        --output-dir "$output_dir" \
        --wandb \
        --wandb-project rnn-bptt-vs-eggroll \
        --wandb-entity wobrob101 \
        --wandb-run-name "$name" \
        --wandb-group "eggroll-mqar-gentle-curriculum-seed${seed}" \
        --log-progress
}

for schedule in $schedules; do
    case "$schedule" in
        reference|gentle) run_schedule "$schedule" ;;
        *)
            echo "SCHEDULES entries must be 'reference' or 'gentle'" >&2
            exit 2
            ;;
    esac
done

if [[ -f "$output_root/reference/metrics.json" \
    && -f "$output_root/gentle/metrics.json" ]]; then
    "$python_bin" scripts/summarize_gentle_curriculum.py --seed "$seed"
fi
