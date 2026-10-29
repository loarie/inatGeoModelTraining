#!/bin/bash
# Two-pass clean-data build, run directly against the real production spatial_data.csv.
# v2 (single-pass, accumulates points in Python lists as it streams) OOM-killed twice on the
# real 345M-row file around 85-110M rows in. v3 fixes this with the same two-pass,
# pre-allocated-numpy-array pattern extract_random_species_dataset.py already proved works on
# this exact file/machine: pass 1 counts only, pass 2 writes straight into sized arrays.
set -euo pipefail

INPUT_CSV="$1"          # path to a production spatial_data.csv export
OUTDIR="${2:-.}"
cd "$(dirname "$0")"

mkdir -p "$OUTDIR"

echo "=== preprocess_training_data_v3.py: captive/accuracy filter + hard_cap + guarantee_taxa ==="
date
time python3 preprocess_training_data_v3.py "$INPUT_CSV" \
    --hard_cap 100000 \
    --guarantee_taxa_file group500_taxa.npy \
    --chunk_size 5000000 \
    --output_prefix "$OUTDIR/train_5007_clean"

echo ""
echo "=== Done ==="
date
ls -la "$OUTDIR"/train_5007_clean*
