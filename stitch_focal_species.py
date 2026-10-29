"""
Stitch the 7 known focal species (with their full, real, uncapped observation counts,
already present in train_10k_focal_clean) onto a randomly-extracted species dataset that's
missing them (e.g. because --force_include wasn't passed during remote extraction).

Produces a combined dataset with no overlap-handling needed (the two source datasets are
assumed disjoint in taxon_id) - used once to build train_5007_* from train_5k_random_*
+ train_10k_focal_clean's 7 focal species, for the "universal secondary head, no curation"
scaling experiment.

Usage:
  python3 stitch_focal_species.py \\
      --base_prefix train_5k_random --supplement_prefix train_10k_focal_clean \\
      --supplement_taxa 43188,47360,47737,51315,54952,57358,472214 \\
      --output_prefix train_5007
"""

import argparse
import numpy as np


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--base_prefix", required=True, help="Prefix for the base *_locs.npy/_class_ids.npy/_unique_taxa.npy")
    p.add_argument("--supplement_prefix", required=True, help="Prefix for the dataset to pull extra species from")
    p.add_argument("--supplement_taxa", required=True, help="Comma-separated taxon_ids to pull from supplement_prefix")
    p.add_argument("--output_prefix", required=True)
    return p.parse_args()


def main():
    args = parse_args()
    supplement_taxa = [int(t.strip()) for t in args.supplement_taxa.split(",")]

    locs_base = np.load(f"{args.base_prefix}_locs.npy")
    cids_base = np.load(f"{args.base_prefix}_class_ids.npy")
    taxa_base = np.load(f"{args.base_prefix}_unique_taxa.npy")

    locs_supp = np.load(f"{args.supplement_prefix}_locs.npy")
    cids_supp = np.load(f"{args.supplement_prefix}_class_ids.npy")
    taxa_supp = np.load(f"{args.supplement_prefix}_unique_taxa.npy")

    overlap = set(supplement_taxa) & set(taxa_base.tolist())
    if overlap:
        raise ValueError(f"supplement_taxa already present in base dataset: {overlap}")

    supp_class_idx = [int(np.where(taxa_supp == t)[0][0]) for t in supplement_taxa]
    supp_mask = np.isin(cids_supp, supp_class_idx)
    supp_locs = locs_supp[supp_mask]
    supp_taxon_ids_per_row = taxa_supp[cids_supp[supp_mask]]

    print(f"Pulled {len(supp_locs):,} rows for {len(supplement_taxa)} supplement taxa:")
    for t in supplement_taxa:
        print(f"  {t}: {(supp_taxon_ids_per_row == t).sum():,} obs")

    base_taxon_ids_per_row = taxa_base[cids_base]
    all_taxon_ids_per_row = np.concatenate([base_taxon_ids_per_row, supp_taxon_ids_per_row])
    all_locs = np.concatenate([locs_base, supp_locs], axis=0).astype(np.float32)

    unique_taxa_new, class_ids_new = np.unique(all_taxon_ids_per_row, return_inverse=True)
    expected_n = len(taxa_base) + len(supplement_taxa)
    print(f"\nCombined: {len(unique_taxa_new):,} species (expected {expected_n:,}), "
          f"{len(all_locs):,} observations")
    assert len(unique_taxa_new) == expected_n
    assert all(t in unique_taxa_new for t in supplement_taxa)

    np.save(f"{args.output_prefix}_locs.npy", all_locs)
    np.save(f"{args.output_prefix}_class_ids.npy", class_ids_new.astype(np.int32))
    np.save(f"{args.output_prefix}_unique_taxa.npy", unique_taxa_new.astype(np.int64))
    print(f"\nSaved {args.output_prefix}_locs.npy / _class_ids.npy / _unique_taxa.npy")


if __name__ == "__main__":
    main()
