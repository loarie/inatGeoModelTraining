#!/bin/bash
# Run on your Mac, from the repo root, to copy the data the v42 benchmark needs (~1.7GB, mostly
# the bioclim raster) to the GPU instance.
#   bash cloud/upload_data.sh ubuntu@1.2.3.4
set -euo pipefail

HOST="$1"
REMOTE_DIR="${REMOTE_DIR:-inatGeoModelTraining}"

rsync -avP --compress \
    bioclim_coast_effort_landmask_interact_scaled.npy \
    elev_scaled.npy \
    train_5007_clean_matched_v2_locs.npy \
    train_5007_clean_matched_v2_class_ids.npy \
    train_5007_clean_matched_v2_unique_taxa.npy \
    "$HOST:$REMOTE_DIR/"
