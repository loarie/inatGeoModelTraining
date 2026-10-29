"""
Specificity/false-positive check at genuine high-confidence true absences.

The standard "95% sensitivity" threshold (see visualize_combined.py) is calibrated ONLY from
real presence points - it controls false absences but says nothing about false positives
(over-prediction into non-habitat). This script complements it: for each focal species, finds
H3-4 cells with a large total observation count (any species) where the focal species was
NEVER recorded - a defensible true-absence set - and reports what fraction of those cells the
model correctly predicts absent (specificity) at that same 95%-sensitivity threshold.

Usage:
    python3 check_specificity.py --model focal_10k_model_v11.keras --config focal_10k_config_v11.yml
    python3 check_specificity.py --model focal_10k_model_v18.epoch020.keras --config focal_10k_config_v18.yml --min_obs 3000
"""
import argparse
from collections import defaultdict

import h3
import numpy as np
import tensorflow as tf
import yaml

from lib.models.encoders import CoordEncoder

FOCAL = [
    ("Pika", 43188), ("Millipede", 47737), ("Lily", 51315), ("Crab", 54952),
    ("Lewisia", 47360), ("Amorph", 57358), ("Ant", 472214),
]
H3_RES = 4


def build_cell_stats(locs, cids):
    cell_to_species = defaultdict(set)
    cell_count = defaultdict(int)
    for i in range(len(locs)):
        cell = h3.geo_to_h3(float(locs[i, 1]), float(locs[i, 0]), H3_RES)
        cell_to_species[cell].add(int(cids[i]))
        cell_count[cell] += 1
    return cell_to_species, cell_count


def get_threshold(model, enc_presence, cidx, percentile=5):
    scores = []
    for start in range(0, len(enc_presence), 8192):
        scores.append(model(enc_presence[start:start + 8192, :-1], training=False).numpy()[:, cidx])
    scores = np.concatenate(scores)
    return float(np.percentile(scores, percentile))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--train_npy", default="train_10k_focal_clean")
    ap.add_argument("--min_obs", type=int, default=3000,
                     help="Minimum total observation count (any species) for a cell to count as a confident true-absence candidate.")
    ap.add_argument("--percentile", type=float, default=5,
                     help="Sensitivity threshold percentile (5 = 95%% sensitivity, matching visualize_combined.py).")
    args = ap.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)
    raster = np.load(config["bioclim_data"]).astype(np.float32)
    encoder = CoordEncoder(config["inputs"]["loc_feat_encoding"], raster)

    unique_taxa = np.load(f"{args.train_npy}_unique_taxa.npy")
    locs = np.load(f"{args.train_npy}_locs.npy")
    cids = np.load(f"{args.train_npy}_class_ids.npy")
    class_idx = {name: int(np.where(unique_taxa == tid)[0][0]) for name, tid in FOCAL}
    focal_indices = list(class_idx.values())

    print("building H3-4 cell -> species set + total obs count...", flush=True)
    cell_to_species, cell_count = build_cell_stats(locs, cids)
    high_count_cells = [c for c, n in cell_count.items() if n >= args.min_obs]
    print(f"  {len(cell_to_species):,} occupied cells, {len(high_count_cells)} with >={args.min_obs} total obs", flush=True)

    cell_latlon = np.array([h3.h3_to_geo(c) for c in high_count_cells])
    cell_locs = np.stack([cell_latlon[:, 1], cell_latlon[:, 0]], axis=1)
    encoded = encoder.encode(cell_locs)

    print(f"loading model {args.model} ...", flush=True)
    model = tf.keras.models.load_model(args.model)

    all_preds = []
    for start in range(0, len(encoded), 4096):
        batch = model(encoded[start:start + 4096, :-1], training=False).numpy()
        all_preds.append(batch[:, focal_indices])
    all_preds = np.concatenate(all_preds)

    print()
    print(f"{'species':10s} {'threshold':>10s} {'n_abs_cells':>12s} {'specificity':>12s} {'fp_rate':>10s} {'mean_pred':>10s}")
    for i, (name, tid) in enumerate(FOCAL):
        cidx = class_idx[name]
        sp_locs = locs[cids == cidx]
        enc_presence = encoder.encode(sp_locs)
        thresh = get_threshold(model, enc_presence, cidx, args.percentile)

        absent_mask = np.array([cidx not in cell_to_species[c] for c in high_count_cells])
        preds_here = all_preds[absent_mask, i]
        fp_rate = (preds_here >= thresh).mean()
        specificity = 1 - fp_rate
        print(f"{name:10s} {thresh:10.4f} {len(preds_here):12d} {specificity:12.4f} {fp_rate:10.4f} {preds_here.mean():10.4f}")


if __name__ == "__main__":
    main()
