"""
Select an automatically-chosen, stratified group of species from an existing npy dataset, for
scaling the dedicated-head mechanism (see lib/models/geo_model_net.py:make_geo_model_net_focal_head)
beyond a hand-picked few species without curation-by-name.

Stratifies the OTHER species by log10(observation count) into --n_bins bins and samples evenly
from each (so the group spans the full sparse-to-common spectrum, not just whichever species
happen to be common), then unions in any --force_include taxa so known comparison species are
always present regardless of the random draw.

Usage:
  python3 select_stratified_group.py --train_npy train_5007 --group_size 500 \\
      --force_include 43188,47360,47737,51315,54952,57358,472214 \\
      --seed 42 --output group500_taxa.npy
"""

import argparse
import numpy as np


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--train_npy", required=True, help="Prefix for the source *_class_ids.npy/_unique_taxa.npy")
    p.add_argument("--group_size", type=int, default=500)
    p.add_argument("--n_bins", type=int, default=10, help="log10(count) strata to sample evenly from")
    p.add_argument("--force_include", type=str, default=None,
                   help="Comma-separated taxon_ids always included, counts toward --group_size")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output", required=True, help="Output .npy path for the selected taxon_id array")
    return p.parse_args()


def main():
    args = parse_args()
    rng = np.random.default_rng(args.seed)

    class_ids = np.load(f"{args.train_npy}_class_ids.npy")
    taxa = np.load(f"{args.train_npy}_unique_taxa.npy")
    counts = np.bincount(class_ids, minlength=len(taxa))

    forced = []
    if args.force_include:
        forced = [int(t.strip()) for t in args.force_include.split(",")]
        missing = [t for t in forced if t not in taxa]
        if missing:
            print(f"WARNING: force_include taxa not found in {args.train_npy}: {missing}")
        forced = [t for t in forced if t in taxa]
    forced_idx = set(np.where(np.isin(taxa, forced))[0].tolist())

    n_random = args.group_size - len(forced_idx)
    if n_random < 0:
        raise ValueError(f"force_include ({len(forced_idx)}) exceeds group_size ({args.group_size})")

    remaining_idx = np.array([i for i in range(len(taxa)) if i not in forced_idx])
    remaining_counts = counts[remaining_idx]
    log_counts = np.log10(np.maximum(remaining_counts, 1))
    bins = np.linspace(log_counts.min(), log_counts.max(), args.n_bins + 1)
    bin_ids = np.digitize(log_counts, bins[1:-1])

    selected = []
    per_bin = n_random // args.n_bins
    for b in range(args.n_bins):
        pool = remaining_idx[bin_ids == b]
        take = min(per_bin, len(pool))
        if take > 0:
            selected.extend(rng.choice(pool, size=take, replace=False).tolist())

    selected_set = set(selected)
    leftover_pool = np.array([i for i in remaining_idx if i not in selected_set])
    need = n_random - len(selected)
    if need > 0:
        selected.extend(rng.choice(leftover_pool, size=need, replace=False).tolist())

    all_selected_idx = sorted(set(selected) | forced_idx)
    sel_taxa = taxa[all_selected_idx]
    sel_counts = counts[all_selected_idx]

    print(f"Selected {len(sel_taxa)} species (seed={args.seed}, {len(forced_idx)} forced)")
    print(f"  count percentiles [0,10,25,50,75,90,100]: "
          f"{np.percentile(sel_counts, [0,10,25,50,75,90,100])}")
    print(f"  all force_include taxa present: {all(t in sel_taxa for t in forced)}")

    np.save(args.output, sel_taxa)
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
