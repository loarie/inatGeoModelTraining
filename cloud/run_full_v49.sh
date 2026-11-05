#!/bin/bash
# v49: v48 (lr 2e-4, --focal_oversample_scale 5, --no_effort_input) plus effort_scale 20,
# nearby negatives (--near_bg_weight 1.0) and look-alike negatives. Free local tests on California + 20k world species
# (ca/, eval_regional.py) reproduced v48's regional spill (C. amabilis into the Sierra foothills);
# effort 20 recovered ~70% of it, nearby negatives more, and look-alike negatives (similar climate
# 100-500 km away, --lookalike_bg_weight 1.0) closed nearly all the rest. Separate heads for every
# species and a 1024-wide layer did not help there. 3 epochs, each scored with
# check_population.py, then the epoch-3 diversity maps and the Calochortus range-map sheets.
# Run ON the instance inside tmux:
#   tmux new -s v49 'bash cloud/run_full_v49.sh'
INITIAL_LR=0.0002 EFFORT_SCALE=20.0 bash cloud/run_full_test.sh v49 3 \
    "--focal_oversample_scale 5 --no_effort_input --near_bg_weight 1.0 --lookalike_bg_weight 1.0"
source ~/geo-venv/bin/activate
pip install -q matplotlib 'geopandas<1'
python diversity_maps.py --model v49/focal_full_model_v49.epoch003.keras --config v49/config_v49.yml \
    --train_npy train_full_v1 --iconic_csv diversity/iconic_train_full_v1.csv --out_dir v49/diversity \
    > v49/diversity.log 2>&1 && python plot_diversity_maps.py v49/diversity >> v49/diversity.log 2>&1 \
    && python plot_diversity_maps.py v49/diversity --despike >> v49/diversity.log 2>&1
echo "DIVERSITY_DONE" >> v49/train_v49.log
mkdir -p v49/maps_calochortus
n=0
while read -r panels; do
    n=$((n + 1))
    python visualize_combined.py --train_npy train_full_v1 --model v49/focal_full_model_v49.epoch003.keras \
        --config v49/config_v49.yml --focal_taxa "$panels" \
        --out "v49/maps_calochortus/calochortus_v49_sheet$n.png" >> v49/maps.log 2>&1
done < diversity/calochortus_panels.txt
echo "MAPS_DONE" >> v49/train_v49.log
