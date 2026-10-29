#!/bin/bash
# Run ON the GPU instance (Ubuntu + NVIDIA driver, e.g. Lambda or RunPod) to set up a training
# environment for the gpu-ready-training branch.
#   bash setup_gpu.sh            # clones into ~/inatGeoModelTraining
set -euo pipefail

REPO_DIR="${REPO_DIR:-$HOME/inatGeoModelTraining}"
BRANCH="${BRANCH:-gpu-ready-training}"

nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv

if [ ! -d "$REPO_DIR" ]; then
    git clone --branch "$BRANCH" https://github.com/loarie/inatGeoModelTraining.git "$REPO_DIR"
fi
cd "$REPO_DIR"

sudo apt-get update -qq && sudo apt-get install -y -qq python3-venv tmux time > /dev/null
python3 -m venv ~/geo-venv
source ~/geo-venv/bin/activate
pip install -q --upgrade pip
# --timeout/--retries: a stalled download otherwise hangs pip indefinitely
pip install -q --timeout 60 --retries 10 -r cloud/requirements-gpu.txt

python - <<'EOF'
import tensorflow as tf
gpus = tf.config.list_physical_devices("GPU")
print("TF", tf.__version__, "GPUs:", gpus)
assert gpus, "TensorFlow does not see a GPU - check the NVIDIA driver / CUDA install"
EOF

echo
echo "Setup done. Next, from your Mac: bash cloud/upload_data.sh <user>@<instance-ip>"
echo "Then here:  source ~/geo-venv/bin/activate && tmux new -s bench 'bash cloud/benchmark.sh'"
