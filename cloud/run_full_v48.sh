#!/bin/bash
# v48: the v47 recipe (lr 2e-4, --focal_oversample_scale 5) plus --no_effort_input, so the network
# can't learn "few iNat observations -> absent" (v47's stacked ranges tracked effort: Congo ~1
# species). 3 epochs (v47 was best at epoch 3), each scored with check_population.py, then the
# epoch-3 stacked-range diversity maps. Run ON the instance inside tmux:
#   tmux new -s v48 'bash cloud/run_full_v48.sh'
INITIAL_LR=0.0002 bash cloud/run_full_test.sh v48 3 "--focal_oversample_scale 5 --no_effort_input"
source ~/geo-venv/bin/activate
python diversity_maps.py --model v48/focal_full_model_v48.epoch003.keras --config v48/config_v48.yml \
    --train_npy train_full_v1 --iconic_csv diversity/iconic_train_full_v1.csv --out_dir v48/diversity \
    > v48/diversity.log 2>&1 && python plot_diversity_maps.py v48/diversity >> v48/diversity.log 2>&1 \
    && python plot_diversity_maps.py v48/diversity --despike >> v48/diversity.log 2>&1
echo "DIVERSITY_DONE" >> v48/train_v48.log
