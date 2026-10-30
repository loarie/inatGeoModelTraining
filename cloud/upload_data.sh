#!/bin/bash
# Run on your Mac, from the repo root, to copy a training dataset plus the rasters to the GPU
# instance (the bioclim raster alone is ~1.6GB).
#   bash cloud/upload_data.sh ubuntu@1.2.3.4                      # v42/v43 data
#   bash cloud/upload_data.sh ubuntu@1.2.3.4 train_pilot25k_v1    # any <prefix>_*.npy dataset
set -euo pipefail

HOST="$1"
PREFIX="${2:-train_5007_clean_matched_v2}"
REMOTE_DIR="${REMOTE_DIR:-inatGeoModelTraining}"

rsync -avP --compress \
    bioclim_coast_effort_landmask_interact_scaled.npy \
    elev_scaled.npy \
    "${PREFIX}_locs.npy" \
    "${PREFIX}_class_ids.npy" \
    "${PREFIX}_unique_taxa.npy" \
    "$HOST:$REMOTE_DIR/"
