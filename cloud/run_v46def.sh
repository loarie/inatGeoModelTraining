#!/bin/bash
# The v46d-f full-dataset tests, back to back on one GPU instance (~8h on an A100 40GB). Run ON
# the instance, from the repo root, after setup_gpu.sh and the train_full_v1 upload, inside tmux:
#   tmux new -s v46 'bash cloud/run_v46def.sh'
# The one-epoch tests go first so their results arrive early:
#   v46d  initial_lr 1e-4, 1 epoch            - does going below v46c's 2e-4 keep helping?
#   v46f  initial_lr 2e-4, focal share ~14%, 1 epoch  (v44's share was 13%, v45's 3%)
#   v46e  initial_lr 2e-4, 3 epochs           - does the lower lr also stop v45's per-epoch decline?
# Each NAME/train_NAME.log ends with ALL_DONE once its checkpoints are scored.
INITIAL_LR=0.0001 bash cloud/run_full_test.sh v46d 1
INITIAL_LR=0.0002 bash cloud/run_full_test.sh v46f 1 "--focal_oversample_scale 5"
INITIAL_LR=0.0002 bash cloud/run_full_test.sh v46e 3
