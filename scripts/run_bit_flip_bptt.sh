#!/usr/bin/env bash
set -euo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
with_gpu="${WITH_GPU:-/store/store5/software/simple-gpu-schedule/with-gpu}"
gpu_pool="${GPU_POOL:-all}"
python_bin="${PYTHON_BIN:-/store/store4/software/bin/anaconda3/envs/flash_attn_pytorch2/bin/python}"
output_dir="${OUTPUT_DIR:-$repo_root/artifacts/bit_flip/seed-7/bptt-elman-h16}"
wandb_run_id="${WANDB_RUN_ID:-bitflipb1}"

if [[ "${BIT_FLIP_BPTT_ON_GPU:-0}" != 1 ]]; then
    exec "$with_gpu" "$gpu_pool" --num 1 -- env \
        BIT_FLIP_BPTT_ON_GPU=1 \
        PYTHON_BIN="$python_bin" \
        OUTPUT_DIR="$output_dir" \
        WANDB_RUN_ID="$wandb_run_id" \
        bash "$0" "$@"
fi

cd "$repo_root"
export PYTHONPATH=src
export WANDB__SERVICE_WAIT="${WANDB__SERVICE_WAIT:-300}"
mkdir -p "$output_dir"

exec "$python_bin" -u -m rnn_bptt_vs_eggroll.bit_flip \
    --output-dir "$output_dir" \
    --device cuda \
    --seed 7 \
    --max-updates 2000000 \
    --batch-size 256 \
    --hidden-size 16 \
    --recurrent-radius 0.9 \
    --learning-rate 0.003 \
    --weight-decay 0 \
    --gradient-clip 1 \
    --promotion-accuracy 0.95 \
    --evaluation-interval 100 \
    --evaluation-examples 2048 \
    --evaluation-batch-size 256 \
    --stage-patience 100000 \
    --curriculum-dense-until 32 \
    --curriculum-max-operations 16384 \
    --curriculum-growth 1.25 \
    --checkpoint-interval 10000 \
    --wandb \
    --wandb-project rnn-bptt-vs-eggroll \
    --wandb-entity wobrob101 \
    --wandb-run-name bit-flip-bptt-elman-h16-seed7 \
    --wandb-group bit-flip-bptt \
    --wandb-run-id "$wandb_run_id" \
    --wandb-resume allow \
    "$@"
