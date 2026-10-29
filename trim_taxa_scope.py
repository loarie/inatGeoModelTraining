"""
Post-hoc trim of a clean-rebuild dataset (e.g. train_5007_clean_{locs,class_ids,unique_taxa}.npy)
down to a target total taxa count, to match an earlier run's scale for apples-to-apples
comparison. Guaranteed taxa are always kept in full; the remaining slots go to the highest-count
non-guaranteed taxa. Pure numpy array filtering - does not touch the source CSV, so it's cheap
even at hundreds of millions of rows.

Usage:
  python3 trim_taxa_scope.py --prefix train_5007_clean --guarantee_taxa_file group500_taxa.npy \\
      --target_total_taxa 5007 --output_prefix train_5007_clean_trimmed
"""
import argparse
import numpy as np


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--prefix", required=True, help="Input prefix (expects {prefix}_locs.npy etc)")
    p.add_argument("--guarantee_taxa_file", required=True)
    p.add_argument("--target_total_taxa", type=int, required=True)
    p.add_argument("--output_prefix", required=True)
    return p.parse_args()


def main():
    args = parse_args()

    locs = np.load(f"{args.prefix}_locs.npy")
    class_ids = np.load(f"{args.prefix}_class_ids.npy")
    unique_taxa = np.load(f"{args.prefix}_unique_taxa.npy")
    guarantee_taxa = set(int(t) for t in np.load(args.guarantee_taxa_file))

    n_taxa = len(unique_taxa)
    counts = np.bincount(class_ids, minlength=n_taxa)

    guarantee_idx = [i for i, t in enumerate(unique_taxa) if int(t) in guarantee_taxa]
    guarantee_idx_set = set(guarantee_idx)
    n_needed = args.target_total_taxa - len(guarantee_idx)
    if n_needed < 0:
        raise ValueError(f"target_total_taxa={args.target_total_taxa} is smaller than the "
                          f"{len(guarantee_idx)} guaranteed taxa alone")

    rankable_idx = [i for i in range(n_taxa) if i not in guarantee_idx_set]
    rankable_sorted = sorted(rankable_idx, key=lambda i: counts[i], reverse=True)
    top_idx = rankable_sorted[:n_needed]

    keep_idx_sorted = sorted(guarantee_idx_set | set(top_idx))
    print(f"Trimming {n_taxa:,} taxa -> {len(keep_idx_sorted):,} "
          f"({len(guarantee_idx)} guaranteed + {len(top_idx)} top-by-count)")

    remap = np.full(n_taxa, -1, dtype=np.int32)
    for new_i, old_i in enumerate(keep_idx_sorted):
        remap[old_i] = new_i

    keep_mask_per_class = np.zeros(n_taxa, dtype=bool)
    keep_mask_per_class[keep_idx_sorted] = True
    row_mask = keep_mask_per_class[class_ids]

    new_locs = locs[row_mask]
    new_class_ids = remap[class_ids[row_mask]]
    new_unique_taxa = unique_taxa[keep_idx_sorted]

    print(f"Kept {row_mask.sum():,}/{len(class_ids):,} observations "
          f"(dropped {(~row_mask).sum():,} from excluded taxa)")

    np.save(f"{args.output_prefix}_locs.npy", new_locs)
    np.save(f"{args.output_prefix}_class_ids.npy", new_class_ids)
    np.save(f"{args.output_prefix}_unique_taxa.npy", new_unique_taxa)

    missing_guarantee = guarantee_taxa - set(int(t) for t in new_unique_taxa)
    print(f"Guaranteed-taxa check: {len(guarantee_taxa) - len(missing_guarantee)}/{len(guarantee_taxa)} present")
    if missing_guarantee:
        print(f"  WARNING - missing: {sorted(missing_guarantee)}")

    print(f"\nSaved: {args.output_prefix}_locs.npy, _class_ids.npy, _unique_taxa.npy")


if __name__ == "__main__":
    main()
