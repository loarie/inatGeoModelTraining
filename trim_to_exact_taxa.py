"""
Trim a clean-rebuild dataset (e.g. train_5007_clean_{locs,class_ids,unique_taxa}.npy) down to an
EXACT taxon_id list, rather than trim_taxa_scope.py's top-N-by-count approach (which was found to
skew heavily toward the platform's most-observed species overall - median 23,834 obs/taxon vs.
train_5007's original 453 - a fundamentally different, much larger composition, not a fair
before/after comparison).

Usage:
  python3 trim_to_exact_taxa.py --prefix train_5007_clean --target_taxa_file train_5007_unique_taxa.npy \\
      --output_prefix train_5007_clean_matched
"""
import argparse
import numpy as np


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--prefix", required=True)
    p.add_argument("--target_taxa_file", required=True,
                   help=".npy of the exact taxon_ids to keep (e.g. the original train_5007_unique_taxa.npy)")
    p.add_argument("--output_prefix", required=True)
    return p.parse_args()


def main():
    args = parse_args()

    locs = np.load(f"{args.prefix}_locs.npy")
    class_ids = np.load(f"{args.prefix}_class_ids.npy")
    unique_taxa = np.load(f"{args.prefix}_unique_taxa.npy")
    target_taxa = set(int(t) for t in np.load(args.target_taxa_file))

    n_taxa = len(unique_taxa)
    keep_idx_sorted = [i for i, t in enumerate(unique_taxa) if int(t) in target_taxa]

    present_taxa = set(int(unique_taxa[i]) for i in keep_idx_sorted)
    missing_taxa = target_taxa - present_taxa
    print(f"Target taxa: {len(target_taxa):,}  |  found: {len(present_taxa):,}  |  missing: {len(missing_taxa):,}")
    if missing_taxa:
        print(f"  WARNING - these target taxa have ZERO qualifying (clean) observations: {sorted(missing_taxa)}")

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
    print(f"Mean obs/taxon: {row_mask.sum()/len(keep_idx_sorted):.0f}")

    np.save(f"{args.output_prefix}_locs.npy", new_locs)
    np.save(f"{args.output_prefix}_class_ids.npy", new_class_ids)
    np.save(f"{args.output_prefix}_unique_taxa.npy", new_unique_taxa)
    print(f"\nSaved: {args.output_prefix}_locs.npy, _class_ids.npy, _unique_taxa.npy")


if __name__ == "__main__":
    main()
