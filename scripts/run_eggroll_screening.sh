#!/usr/bin/env bash
set -euo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
with_gpu="${WITH_GPU:-/store/store5/software/simple-gpu-schedule/with-gpu}"
gpu_pool="${GPU_POOL:-all}"

if [[ "${EGGROLL_SCREENING_ON_GPU:-0}" != 1 ]]; then
    exec "$with_gpu" "$gpu_pool" --num 1 -- env \
        EGGROLL_SCREENING_ON_GPU=1 \
        bash "$0" "$@"
fi

cd "$repo_root"
export PYTHONPATH=src
export WANDB__SERVICE_WAIT="${WANDB__SERVICE_WAIT:-300}"

generations=2000
output_root="artifacts/eggroll_tuning/screening"
mkdir -p "$output_root"

run_screen() {
    local name=$1
    local update_rule=$2
    local sigma=$3
    local learning_rate=$4
    local elite_count=$5
    local elite_commit_scale=$6
    local output_dir="$output_root/$name"

    if [[ -f "$output_dir/metrics.json" ]]; then
        echo "Skipping completed screen: $name"
        return
    fi
    rm -rf "$output_dir"

    local rule_args=(
        --eggroll-update-rule "$update_rule"
        --elite-count "$elite_count"
        --elite-commit-scale "$elite_commit_scale"
    )
    if [[ "$update_rule" == standardized ]]; then
        rule_args+=(
            --eggroll-learning-rate "$learning_rate"
            --eggroll-learning-rate-final 0.01
            --eggroll-learning-rate-decay-start 0
            --eggroll-learning-rate-decay-end 100000
        )
    fi

    python -u -m rnn_bptt_vs_eggroll.experiment \
        --method eggroll \
        --preset reference \
        --device cuda \
        --seed 7 \
        --generations "$generations" \
        --batch-size 64 \
        --hidden-size 64 \
        --population-size 8192 \
        --population-chunk-size 1024 \
        --population-data-mode cartesian \
        --population-precision bfloat16 \
        --perturbation-rank 1 \
        --sigma "$sigma" \
        --fitness-shaping zscore \
        "${rule_args[@]}" \
        --eggroll-weight-decay 0 \
        --curriculum-sequence-lengths 16,32,64,128,256,512,1024 \
        --curriculum-num-kv-pairs 1,2,4,8,16,32,64 \
        --curriculum-frontier-probability 1.0 \
        --evaluation-interval 100 \
        --log-interval 1 \
        --evaluation-examples 512 \
        --curriculum-probe-examples 512 \
        --test-examples 512 \
        --output-dir "$output_dir" \
        --wandb \
        --wandb-project rnn-bptt-vs-eggroll \
        --wandb-entity wobrob101 \
        --wandb-run-name "$name" \
        --wandb-group eggroll-mqar-bounded-screening-seed7 \
        --log-progress
}

# Standardized-fitness control and perturbation/update-scale ablations.
run_screen screen-zscore-s5e-3-lr3e-1-control standardized 0.005 0.3 8 0.03
run_screen screen-zscore-s2p5e-3-lr6e-1 standardized 0.0025 0.6 8 0.03
run_screen screen-zscore-s1e-2-lr1p5e-1 standardized 0.01 0.15 8 0.03
run_screen screen-zscore-s2e-2-lr7p5e-2 standardized 0.02 0.075 8 0.03
run_screen screen-zscore-s5e-3-lr1p5e-1 standardized 0.005 0.15 8 0.03
run_screen screen-zscore-s5e-3-lr6e-1 standardized 0.005 0.6 8 0.03

# Elite commit scales match the control's generation-1 update RMS (~0.002126).
run_screen screen-elite-k8-c1p198 elite-centroid 0.005 0.3 8 1.1976
run_screen screen-elite-k64-c3p371 elite-centroid 0.005 0.3 64 3.3709
run_screen screen-elite-k512-c9p512 elite-centroid 0.005 0.3 512 9.512
run_screen screen-elite-k1024-c13p527 elite-centroid 0.005 0.3 1024 13.527

python scripts/summarize_eggroll_screening.py
