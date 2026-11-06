#!/bin/bash
# Run on your Mac, from the repo root, once the instance is up:  bash cloud/launch_v50.sh <ip>
# Sets up the environment, uploads train_full_v1 and starts cloud/run_full_v50.sh in tmux "v50".
set -euo pipefail
HOST="ubuntu@$1"
SSH="ssh -o StrictHostKeyChecking=accept-new $HOST"
$SSH 'curl -fsSL https://raw.githubusercontent.com/loarie/inatGeoModelTraining/gpu-ready-training/cloud/setup_gpu.sh | bash'
bash cloud/upload_data.sh "$HOST" train_full_v1
$SSH 'mkdir -p inatGeoModelTraining/diversity'
scp diversity/iconic_train_full_v1.csv diversity/calochortus_panels.txt "$HOST:inatGeoModelTraining/diversity/"
$SSH 'cd inatGeoModelTraining && git pull -q && tmux new -d -s v50 "bash cloud/run_full_v50.sh"'
echo "started; check with: $SSH 'tmux ls'"
