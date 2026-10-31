#!/bin/bash
# The full 120.5k-species run with the recipe from the v46 tests: the v45 settings plus
# initial_lr 2e-4 and --focal_oversample_scale 5 (focal species back to ~14% of each epoch;
# v46g matched the 25k pilot after 3 epochs). 20 epochs, ~1h28 each on an A100 40GB; every
# epoch is saved, then all are scored with check_population.py. Run ON the instance inside tmux:
#   tmux new -s v47 'bash cloud/run_full_v47.sh'
INITIAL_LR=0.0002 bash cloud/run_full_test.sh v47 20 "--focal_oversample_scale 5"
