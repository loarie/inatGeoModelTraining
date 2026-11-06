#!/bin/bash
# Local regional experiment (California by default) with the v48 recipe plus overrides, then
# eval_regional.py on every epoch for the Calochortus species.
#   bash run_regional.sh ca_base 4
#   bash run_regional.sh ca_heads 4 "--head_all_classes"
# Arguments: NAME EPOCHS [EXTRA_TRAIN_ARGS]. DATA (train_ca_v1), INITIAL_LR (0.0005) and
# EFFORT_SCALE (5.0) and HIDDEN_DIM (512) override.
set -uo pipefail
NAME="$1"; EPOCHS="$2"; EXTRA="${3:-}"; DATA="${DATA:-train_ca_v1}"
mkdir -p "ca/$NAME"
sed -e "s|model_save_name: .*|model_save_name: \"ca/$NAME/model_$NAME.keras\"|" \
    -e "s|initial_lr: .*|initial_lr: ${INITIAL_LR:-0.0005}|" \
    -e "s|effort_scale: .*|effort_scale: ${EFFORT_SCALE:-5.0}|" \
    -e "s|hidden_dim: .*|hidden_dim: ${HIDDEN_DIM:-512}|" \
    focal_full_config_v45.yml > "ca/$NAME/config_$NAME.yml"
# shellcheck disable=SC2086
python train_tf_sinr_test.py --config_file "ca/$NAME/config_$NAME.yml" \
    --train_npy "$DATA" --bg_sampling target_group --bg_tg_fraction 0.65 \
    --num_epochs "$EPOCHS" --focal_taxa "$(cat "${DATA}_focal.txt")" --pos_weight 2.25 --independent_head_dim 16 \
    --bg_base_weight 1.5 --range_penalty_scale 3.0 --oversample_target_exponent 1.0 --oversample_max 20 \
    --grad_clip_value 1.0 --no_wandb --pos_weight_adapt --pos_weight_adapt_every 3 \
    --pos_weight_adapt_threshold 0.05 --pos_weight_adapt_factor 2.0 --pos_weight_adapt_max 50.0 \
    --full_shuffle --steps_per_call 50 --focal_oversample_scale 5 --no_effort_input $EXTRA \
    > "ca/$NAME/train_$NAME.log" 2>&1
echo "TRAIN_EXIT $?" >> "ca/$NAME/train_$NAME.log"
for ckpt in ca/"$NAME"/model_"$NAME".epoch*.keras; do
    e=$(basename "$ckpt" .keras | sed 's/.*\.epoch//')
    python eval_regional.py --model "$ckpt" --config "ca/$NAME/config_$NAME.yml" --data "$DATA" \
        --taxa "${DATA}_calochortus.npy" --out_csv "ca/$NAME/eval_${NAME}_e$e.csv" 2>&1 \
        | grep -E "scored|well-sampled|Error" | sed "s/^/e$e /" >> "ca/$NAME/eval_$NAME.log"
done
# far-off strays of 500 world species (the global failure the Calochortus scores can't see)
if [ -f caworld_world500.npy ] && [ "$DATA" = train_caworld_v1 ]; then
    for ckpt in ca/"$NAME"/model_"$NAME".epoch*.keras; do
        e=$(basename "$ckpt" .keras | sed 's/.*\.epoch//')
        python check_population.py --model "$ckpt" --config "ca/$NAME/config_$NAME.yml" --train_npy "$DATA" \
            --absence_npy train_full_v1 --group caworld_world500.npy --out_csv "ca/$NAME/world500_${NAME}_e$e.csv" 2>&1 \
            | grep -E "^specificity >= 0.95|mean stray|stray share >" | sed "s/^/e$e /" >> "ca/$NAME/eval_$NAME.log"
    done
fi
echo "ALL_DONE" >> "ca/$NAME/train_$NAME.log"
