#!/bin/bash
# v50: v49 plus --own_random_bg_weight 3.0. v49 (stopped after epoch 1) tightened the Calochortus
# but tripled far-off strays (Cakile maritima in mid-ocean): nearby and look-alike negatives only
# push a species down where people survey. A random point anywhere on the globe per record, for
# its own class, balances that - on California + 20k world species (ca/, run_regional2.sh) it kept
# the Calochortus gain (local .947/.898) with world strays back at the no-new-negatives level and
# total stray area a third lower. Paired negatives now score only their own class (~10% slower
# than none, vs ~4x in v49). 3 epochs, each scored with check_population.py, then the epoch-3
# diversity maps and the Calochortus range-map sheets.
# Run ON the instance inside tmux:
#   tmux new -s v50 'bash cloud/run_full_v50.sh'
INITIAL_LR=0.0002 EFFORT_SCALE=20.0 bash cloud/run_full_test.sh v50 3 \
    "--focal_oversample_scale 5 --no_effort_input --near_bg_weight 1.0 --lookalike_bg_weight 1.0 --own_random_bg_weight 3.0"
source ~/geo-venv/bin/activate
pip install -q matplotlib 'geopandas<1'
python diversity_maps.py --model v50/focal_full_model_v50.epoch003.keras --config v50/config_v50.yml \
    --train_npy train_full_v1 --iconic_csv diversity/iconic_train_full_v1.csv --out_dir v50/diversity \
    > v50/diversity.log 2>&1 && python plot_diversity_maps.py v50/diversity >> v50/diversity.log 2>&1 \
    && python plot_diversity_maps.py v50/diversity --despike >> v50/diversity.log 2>&1
echo "DIVERSITY_DONE" >> v50/train_v50.log
mkdir -p v50/maps_calochortus
n=0
while read -r panels; do
    n=$((n + 1))
    python visualize_combined.py --train_npy train_full_v1 --model v50/focal_full_model_v50.epoch003.keras \
        --config v50/config_v50.yml --focal_taxa "$panels" \
        --out "v50/maps_calochortus/calochortus_v50_sheet$n.png" >> v50/maps.log 2>&1
done < diversity/calochortus_panels.txt
echo "MAPS_DONE" >> v50/train_v50.log
