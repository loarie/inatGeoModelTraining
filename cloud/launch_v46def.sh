#!/bin/bash
# Run on your Mac, from the repo root, once the instance is up:  bash cloud/launch_v46def.sh <ip>
# Sets up the environment, uploads train_full_v1 and starts cloud/run_v46def.sh in tmux "v46".
set -euo pipefail
HOST="ubuntu@$1"
SSH="ssh -o StrictHostKeyChecking=accept-new $HOST"
$SSH 'curl -fsSL https://raw.githubusercontent.com/loarie/inatGeoModelTraining/gpu-ready-training/cloud/setup_gpu.sh | bash'
bash cloud/upload_data.sh "$HOST" train_full_v1
$SSH 'cd inatGeoModelTraining && git pull -q && tmux new -d -s v46 "bash cloud/run_v46def.sh"'
echo "started; check with: $SSH 'tmux ls'"
