"""
Population-wide health check combining mean-prediction bucketing with the specificity metric
that actually matters (same standard used to validate v28's accepted baseline).

Why both: a class with low mean prediction at its own presence
points isn't necessarily broken - it may have learned a clean, well-discriminating decision
boundary that's just never been scaled up toward 1.0 (see taxon 9296, mean=0.02 but
specificity=1.0, perfect). Mean prediction measures output calibration; specificity measures
whether the model can actually tell presence from true absence, which is the thing that matters
for a usable range map. Report both, don't conflate them.

Usage:
  python3 check_population.py --model focal_10k_model_v40.epoch006.keras --config focal_10k_config_v40.yml
"""
import argparse
import pickle
from collections import defaultdict

import h3
import numpy as np
import tensorflow as tf
import yaml

from lib.models.encoders import CoordEncoder
from lib.models.geo_model_net import FocalScatterAdd, IndependentPerClassHead

H3_RES = 4


def build_or_load_cell_stats(locs, cids, train_npy):
    # keyed by dataset: class ids index into that dataset's unique_taxa, so a cache built from
    # one train_npy gives wrong absence sets for another (e.g. train_5007 vs clean_matched_v2)
    CELL_STATS_CACHE = f"h3_cell_stats_{train_npy}.pkl"
    try:
        with open(CELL_STATS_CACHE, "rb") as f:
            cell_to_species, cell_count = pickle.load(f)
        print(f"loaded cached H3 cell stats ({len(cell_to_species):,} cells)")
        return cell_to_species, cell_count
    except FileNotFoundError:
        pass
    print("building H3-4 cell -> species set + total obs count (one-time, ~slow)...", flush=True)
    cell_to_species = defaultdict(set)
    cell_count = defaultdict(int)
    for i in range(len(locs)):
        cell = h3.geo_to_h3(float(locs[i, 1]), float(locs[i, 0]), H3_RES)
        cell_to_species[cell].add(int(cids[i]))
        cell_count[cell] += 1
    with open(CELL_STATS_CACHE, "wb") as f:
        pickle.dump((cell_to_species, cell_count), f)
    print(f"  {len(cell_to_species):,} cells, cached to {CELL_STATS_CACHE}")
    return cell_to_species, cell_count


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--train_npy", default="train_5007")
    ap.add_argument("--min_obs", type=int, default=3000)
    ap.add_argument("--percentile", type=float, default=5)
    ap.add_argument("--group", default="group500_taxa.npy",
                    help="focal 500-taxa group the model was trained with (v41+: group500_taxa_v41.npy)")
    ap.add_argument("--out_csv", default=None, help="optional per-taxon taxon_id,mean,specificity dump")
    args = ap.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)
    raster = np.load(config["bioclim_data"]).astype(np.float32)
    encoder = CoordEncoder(config["inputs"]["loc_feat_encoding"], raster)

    unique_taxa = np.load(f"{args.train_npy}_unique_taxa.npy")
    locs = np.load(f"{args.train_npy}_locs.npy")
    cids = np.load(f"{args.train_npy}_class_ids.npy")
    group500 = np.load(args.group)
    focal_class_ids = [int(np.where(unique_taxa == tid)[0][0]) for tid in group500 if tid in unique_taxa]

    cell_to_species, cell_count = build_or_load_cell_stats(locs, cids, args.train_npy)
    high_count_cells = [c for c, n in cell_count.items() if n >= args.min_obs]
    cell_latlon = np.array([h3.h3_to_geo(c) for c in high_count_cells])
    cell_locs = np.stack([cell_latlon[:, 1], cell_latlon[:, 0]], axis=1)
    encoded_cells = encoder.encode(cell_locs)
    print(f"{len(high_count_cells)} high-confidence true-absence-candidate cells")

    print(f"loading model {args.model} ...", flush=True)
    model = tf.keras.models.load_model(
        args.model, custom_objects={"FocalScatterAdd": FocalScatterAdd, "IndependentPerClassHead": IndependentPerClassHead}
    )

    all_cell_preds = []
    for start in range(0, len(encoded_cells), 4096):
        batch = model(encoded_cells[start:start + 4096, :-1], training=False).numpy()
        all_cell_preds.append(batch[:, focal_class_ids])
    all_cell_preds = np.concatenate(all_cell_preds)

    means, specs = {}, {}
    for i, cidx in enumerate(focal_class_ids):
        sp_locs = locs[cids == cidx]
        if len(sp_locs) == 0:
            continue
        enc_presence = encoder.encode(sp_locs)
        scores = []
        for start in range(0, len(enc_presence), 8192):
            scores.append(model(enc_presence[start:start + 8192, :-1], training=False).numpy()[:, cidx])
        scores = np.concatenate(scores)
        means[cidx] = scores.mean()

        thresh = float(np.percentile(scores, args.percentile))
        absent_mask = np.array([cidx not in cell_to_species[c] for c in high_count_cells])
        preds_here = all_cell_preds[absent_mask, i]
        if len(preds_here) > 0:
            specs[cidx] = 1 - (preds_here >= thresh).mean()

    mean_vals = np.array(list(means.values()))
    spec_vals = np.array([specs[c] for c in means if c in specs])
    print(f"\n=== mean-prediction bucketing (n={len(mean_vals)}) ===")
    print(f"dead (==0.0): {(mean_vals==0.0).sum()}")
    print(f"low mean (<0.05): {(mean_vals<0.05).sum()} ({100*(mean_vals<0.05).mean():.1f}%)")
    print(f"partial (0.05-0.5): {((mean_vals>=0.05)&(mean_vals<0.5)).sum()}")
    print(f"high mean (>=0.5): {(mean_vals>=0.5).sum()} ({100*(mean_vals>=0.5).mean():.1f}%)")

    print(f"\n=== specificity (n={len(spec_vals)}, the metric that actually matters) ===")
    print(f"specificity >= 0.95: {(spec_vals>=0.95).sum()} ({100*(spec_vals>=0.95).mean():.1f}%)")
    print(f"specificity >= 0.8: {(spec_vals>=0.8).sum()} ({100*(spec_vals>=0.8).mean():.1f}%)")
    print(f"specificity < 0.5 (genuinely poor): {(spec_vals<0.5).sum()} ({100*(spec_vals<0.5).mean():.1f}%)")
    print(f"mean specificity: {spec_vals.mean():.4f}  median: {np.median(spec_vals):.4f}")

    if args.out_csv:
        with open(args.out_csv, "w") as f:
            f.write("taxon_id,mean,specificity\n")
            for c in means:
                f.write(f"{int(unique_taxa[c])},{means[c]:.6f},{specs.get(c, float('nan')):.6f}\n")

    print(f"\n=== cross-tab: low mean (<0.05) AND poor specificity (<0.5) = genuinely broken ===")
    genuinely_broken = [c for c in means if means[c] < 0.05 and c in specs and specs[c] < 0.5]
    print(f"{len(genuinely_broken)}/{len(means)} ({100*len(genuinely_broken)/len(means):.1f}%)")


if __name__ == "__main__":
    main()
