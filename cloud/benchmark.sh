#!/bin/bash
# Run ON the GPU instance, from the repo root, after setup_gpu.sh and upload_data.sh.
# Run it inside tmux so a dropped SSH connection doesn't kill it:
#   source ~/geo-venv/bin/activate && tmux new -s bench 'bash cloud/benchmark.sh'
# Results go to bench_results/; copy that folder back when it's done.
set -uo pipefail

OUT=bench_results
mkdir -p "$OUT"
nvidia-smi --query-gpu=name,memory.total --format=csv | tee "$OUT/gpu.txt"

# GPU utilization log for the whole benchmark (1 sample / 5s)
nvidia-smi --query-gpu=timestamp,utilization.gpu,memory.used --format=csv -l 5 > "$OUT/gpu_util.csv" &
SMI_PID=$!
trap 'kill $SMI_PID 2>/dev/null' EXIT

echo "=== 1. synthetic scaling sweep (a few minutes) ==="
python cloud/bench_scale.py        2>&1 | grep -v -i warning | tee "$OUT/scale_nojit.txt"
python cloud/bench_scale.py --jit  2>&1 | grep -v -i warning | tee "$OUT/scale_jit.txt"

# v42 recipe, 1 epoch, checkpoints written into bench_results/ (never over real models)
FOCAL=$(python -c "import numpy as np; print(','.join(str(t) for t in np.load('group500_taxa_v41.npy')))")
sed "s|model_save_name: .*|model_save_name: \"$OUT/bench_v42.keras\"|" focal_10k_config_v42.yml > "$OUT/bench_v42.yml"
V42_ARGS=(--config_file "$OUT/bench_v42.yml" --train_npy train_5007_clean_matched_v2
    --bg_sampling target_group --bg_tg_fraction 0.65 --num_epochs 1 --focal_taxa "$FOCAL"
    --pos_weight 2.25 --independent_head_dim 16 --bg_base_weight 1.5 --range_penalty_scale 3.0
    --oversample_target_exponent 1.0 --oversample_max 20 --grad_clip_value 1.0 --no_wandb)

echo "=== 2. real v42 epoch, no XLA (compare epoch mean loss to 2.1543) ==="
/usr/bin/time -v python train_tf_sinr_test.py "${V42_ARGS[@]}" > "$OUT/v42_nojit.log" 2>&1
grep -E "Epoch mean loss|Traceback|Error" "$OUT/v42_nojit.log"

echo "=== 3. real v42 epoch, XLA ==="
/usr/bin/time -v python train_tf_sinr_test.py "${V42_ARGS[@]}" --jit_compile > "$OUT/v42_jit.log" 2>&1
grep -E "Epoch mean loss|Traceback|Error" "$OUT/v42_jit.log"

echo "=== done: results in $OUT/ ==="
for f in "$OUT"/v42_*.log; do
    echo "$f: $(tr '\r' '\n' < "$f" | grep -E '100%\|' | tail -1 | grep -oE '\[[^]]*\]')  peak RSS $(grep 'Maximum resident' "$f" | awk '{print $NF/1e6 " GB"}')"
done
