"""
Add the rest of the world to a regional training set (make_regional_subset.py), to test whether
crowding and diluted background are what make a global model worse than a regional one. The
regional set is kept exactly (same class ids, training and test observations); --n_species random
species from outside it are appended with up to --cap observations each, from anywhere.

  python make_world_mix.py --src train_full_v1 --region train_ca_v1 --out train_caworld_v1 --n_species 20000

Writes <out>_{locs,class_ids,unique_taxa}.npy and copies the region's test set, focal list,
Calochortus list and eval absence-cell cache under the new name.
"""
import argparse
import shutil

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--region", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n_species", type=int, default=20000)
    ap.add_argument("--cap", type=int, default=500)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)

    r_locs = np.load(f"{args.region}_locs.npy")
    r_cids = np.load(f"{args.region}_class_ids.npy")
    r_taxa = np.load(f"{args.region}_unique_taxa.npy")
    taxa = np.load(f"{args.src}_unique_taxa.npy")
    cids = np.load(f"{args.src}_class_ids.npy", mmap_mode="r")
    others = np.flatnonzero(~np.isin(taxa, r_taxa))
    pick = np.sort(rng.choice(others, args.n_species, replace=False))
    new_id = np.full(len(taxa), -1, np.int64)
    new_id[pick] = len(r_taxa) + np.arange(len(pick))

    rows = np.flatnonzero(new_id[np.asarray(cids)] >= 0)
    rows = rows[rng.permutation(len(rows))]
    c = new_id[np.asarray(cids[rows])]
    order = np.argsort(c, kind="stable")
    rank = np.arange(len(order)) - np.searchsorted(c[order], c[order])
    rows = np.sort(rows[order[rank < args.cap]])
    w_locs = np.asarray(np.load(f"{args.src}_locs.npy", mmap_mode="r")[rows]).astype(np.float32)
    w_cids = new_id[np.asarray(cids[rows])]

    np.save(f"{args.out}_locs.npy", np.concatenate([r_locs, w_locs]))
    np.save(f"{args.out}_class_ids.npy", np.concatenate([r_cids, w_cids]))
    np.save(f"{args.out}_unique_taxa.npy", np.concatenate([r_taxa, taxa[pick]]))
    for suf in ["_test_locs.npy", "_test_class_ids.npy", "_focal.txt", "_calochortus.npy"]:
        shutil.copy(f"{args.region}{suf}", f"{args.out}{suf}")
    shutil.copy(f"h3r5_cells_{args.region}_min500.pkl", f"h3r5_cells_{args.out}_min500.pkl")
    print(f"region {len(r_taxa):,} species / {len(r_cids):,} obs + world {len(pick):,} species / "
          f"{len(w_cids):,} obs -> region share of observations {len(r_cids) / (len(r_cids) + len(w_cids)):.0%}")


if __name__ == "__main__":
    main()
