#!/bin/bash
# Short test runs on the full dataset with the v45 recipe plus overrides, run back to back on one
# GPU instance. Run ON the instance, from the repo root, after setup_gpu.sh and
#   bash cloud/upload_data.sh <user>@<ip> train_full_v1      (from your Mac)
# inside tmux:
#   tmux new -s tests 'bash cloud/run_full_test.sh v46a 1 "--pos_weight 9.8"; \
#                      HIDDEN_DIM=1024 bash cloud/run_full_test.sh v46b 1'
# Arguments: NAME EPOCHS [EXTRA_TRAIN_ARGS]. Extra args come after the v45 ones, so they win.
# HIDDEN_DIM and INITIAL_LR override the config's hidden_dim and initial_lr. Outputs go to NAME/; each epoch is then scored
# with check_population.py (on the GPU, after training) against train_full_v1's absence cells.
set -uo pipefail
source ~/geo-venv/bin/activate
NAME="$1"; EPOCHS="$2"; EXTRA="${3:-}"
mkdir -p "$NAME"
nvidia-smi --query-gpu=timestamp,utilization.gpu,memory.used --format=csv -l 10 > "$NAME/gpu_util.csv" &
SMI_PID=$!
trap 'kill $SMI_PID 2>/dev/null' EXIT

sed -e "s|model_save_name: .*|model_save_name: \"$NAME/focal_full_model_$NAME.keras\"|" \
    -e "s|hidden_dim: .*|hidden_dim: ${HIDDEN_DIM:-512}|" \
    -e "s|initial_lr: .*|initial_lr: ${INITIAL_LR:-0.0005}|" \
    focal_full_config_v45.yml > "$NAME/config_$NAME.yml"
FOCAL=$(python -c "import numpy as np; print(','.join(str(t) for t in np.load('group500_taxa_v41.npy')))")
# shellcheck disable=SC2086
/usr/bin/time -v python train_tf_sinr_test.py --config_file "$NAME/config_$NAME.yml" \
    --train_npy train_full_v1 --bg_sampling target_group --bg_tg_fraction 0.65 \
    --num_epochs "$EPOCHS" --focal_taxa "$FOCAL" --pos_weight 2.25 --independent_head_dim 16 \
    --bg_base_weight 1.5 --range_penalty_scale 3.0 --oversample_target_exponent 1.0 --oversample_max 20 \
    --grad_clip_value 1.0 --no_wandb --pos_weight_adapt --pos_weight_adapt_every 3 \
    --pos_weight_adapt_threshold 0.05 --pos_weight_adapt_factor 2.0 --pos_weight_adapt_max 50.0 \
    --full_shuffle --jit_compile --steps_per_call 50 $EXTRA > "$NAME/train_$NAME.log" 2>&1
echo "TRAIN_EXIT $?" >> "$NAME/train_$NAME.log"

for ckpt in "$NAME"/focal_full_model_"$NAME".epoch*.keras; do
    e=$(basename "$ckpt" .keras | sed 's/.*\.epoch//')
    python check_population.py --model "$ckpt" --config "$NAME/config_$NAME.yml" --train_npy train_full_v1 \
        --group group500_taxa_v41.npy --out_csv "$NAME/population_${NAME}_e$e.csv" \
        > "$NAME/check_population_${NAME}_e$e.log" 2>&1
done
echo "ALL_DONE" >> "$NAME/train_$NAME.log"
