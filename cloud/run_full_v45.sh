#!/bin/bash
# Run ON the GPU instance, from the repo root, after setup_gpu.sh and
#   bash cloud/upload_data.sh <user>@<ip> train_full_v1      (from your Mac)
# inside tmux:  tmux new -s v45 'bash cloud/run_full_v45.sh'
# Outputs go to v45/ (checkpoints, pos_weight files, log, GPU utilization).
set -uo pipefail
source ~/geo-venv/bin/activate
OUT=v45
mkdir -p "$OUT"
nvidia-smi --query-gpu=timestamp,utilization.gpu,memory.used --format=csv -l 10 > "$OUT/gpu_util.csv" &
SMI_PID=$!
trap 'kill $SMI_PID 2>/dev/null' EXIT

sed "s|model_save_name: .*|model_save_name: \"$OUT/focal_full_model_v45.keras\"|" focal_full_config_v45.yml > "$OUT/focal_full_config_v45.yml"
FOCAL=$(python -c "import numpy as np; print(','.join(str(t) for t in np.load('group500_taxa_v41.npy')))")
/usr/bin/time -v python train_tf_sinr_test.py --config_file "$OUT/focal_full_config_v45.yml" \
    --train_npy train_full_v1 --bg_sampling target_group --bg_tg_fraction 0.65 \
    --num_epochs 20 --focal_taxa "$FOCAL" --pos_weight 2.25 --independent_head_dim 16 \
    --bg_base_weight 1.5 --range_penalty_scale 3.0 --oversample_target_exponent 1.0 --oversample_max 20 \
    --grad_clip_value 1.0 --no_wandb --pos_weight_adapt --pos_weight_adapt_every 3 \
    --pos_weight_adapt_threshold 0.05 --pos_weight_adapt_factor 2.0 --pos_weight_adapt_max 50.0 \
    --full_shuffle --jit_compile --steps_per_call 50 > "$OUT/train_v45.log" 2>&1
echo "TRAIN_EXIT $?" >> "$OUT/train_v45.log"
