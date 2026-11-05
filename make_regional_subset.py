"""
Cut a regional training set out of a global one, for cheap local experiments (a California model
trains on a laptop in about an hour per epoch). Keeps species with at least --min_obs observations
inside the box, caps each at --cap (random), and holds out every observation in a random
--holdout share of H3 cells (spatial blocks) as a test set.

  python make_regional_subset.py --src train_full_v1 --out train_ca_v1 --bbox -124.5,-114,32.5,42

Writes <out>_{locs,class_ids,unique_taxa}.npy (train) and <out>_test_{locs,class_ids}.npy (test,
same class ids). Class ids are renumbered; unique_taxa maps them back to taxon ids.
"""
import argparse

import h3
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--bbox", required=True, help="lon_min,lon_max,lat_min,lat_max")
    ap.add_argument("--min_obs", type=int, default=20)
    ap.add_argument("--cap", type=int, default=500)
    ap.add_argument("--holdout", type=float, default=0.2)
    ap.add_argument("--holdout_res", type=int, default=5, help="H3 resolution of the held-out blocks")
    ap.add_argument("--cell_cap", type=int, default=0,
                    help="spatial thinning: keep at most this many training observations per species per "
                         "H3 cell (--cell_cap_res), before --cap. 0 disables. The test set is unchanged.")
    ap.add_argument("--cell_cap_res", type=int, default=6)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    x0, x1, y0, y1 = (float(v) for v in args.bbox.split(","))
    rng = np.random.default_rng(args.seed)

    locs = np.load(f"{args.src}_locs.npy", mmap_mode="r")
    cids = np.load(f"{args.src}_class_ids.npy", mmap_mode="r")
    taxa = np.load(f"{args.src}_unique_taxa.npy")
    lon, lat = np.asarray(locs[:, 0]), np.asarray(locs[:, 1])
    rows = np.flatnonzero((lon >= x0) & (lon <= x1) & (lat >= y0) & (lat <= y1))
    del lon, lat
    sub_locs = np.asarray(locs[rows]).astype(np.float32)
    sub_cids = np.asarray(cids[rows])
    counts = np.bincount(sub_cids, minlength=len(taxa))
    keep_cls = np.flatnonzero(counts >= args.min_obs)
    new_id = np.full(len(taxa), -1, np.int64)
    new_id[keep_cls] = np.arange(len(keep_cls))
    m = new_id[sub_cids] >= 0
    sub_locs, sub_cids = sub_locs[m], new_id[sub_cids[m]]
    print(f"{len(rows):,} obs in box; {len(keep_cls):,} species with >= {args.min_obs}: {len(sub_cids):,} obs")

    # spatial holdout: every observation in a random share of H3 cells
    cells = np.array([h3.geo_to_h3(float(la), float(lo), args.holdout_res) for lo, la in sub_locs])
    uniq, inv = np.unique(cells, return_inverse=True)
    test_cells = rng.random(len(uniq)) < args.holdout
    test = test_cells[inv]

    # cap each species' training observations (random)
    tr = np.flatnonzero(~test)
    tr = tr[rng.permutation(len(tr))]
    if args.cell_cap > 0:
        fine = np.array([h3.geo_to_h3(float(la), float(lo), args.cell_cap_res) for lo, la in sub_locs[tr]])
        _, key = np.unique(np.char.add(fine.astype(str), sub_cids[tr].astype(str)), return_inverse=True)
        order = np.argsort(key, kind="stable")
        rank = np.arange(len(order)) - np.searchsorted(key[order], key[order])
        n0 = len(tr)
        tr = tr[order[rank < args.cell_cap]]
        tr = tr[rng.permutation(len(tr))]
        print(f"cell cap {args.cell_cap}/species/H3-{args.cell_cap_res} cell: {n0:,} -> {len(tr):,} training obs")
    order = np.argsort(sub_cids[tr], kind="stable")
    sorted_c = sub_cids[tr][order]
    rank = np.arange(len(order)) - np.searchsorted(sorted_c, sorted_c)
    tr = np.sort(tr[order[rank < args.cap]])

    np.save(f"{args.out}_locs.npy", sub_locs[tr])
    np.save(f"{args.out}_class_ids.npy", sub_cids[tr])
    np.save(f"{args.out}_unique_taxa.npy", taxa[keep_cls])
    np.save(f"{args.out}_test_locs.npy", sub_locs[test])
    np.save(f"{args.out}_test_class_ids.npy", sub_cids[test])
    print(f"train {len(tr):,} obs (cap {args.cap}); test {int(test.sum()):,} obs in "
          f"{int(test_cells.sum()):,} of {len(uniq):,} H3-{args.holdout_res} cells")


if __name__ == "__main__":
    main()
