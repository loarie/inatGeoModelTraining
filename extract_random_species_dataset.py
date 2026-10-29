"""
Two-pass extraction of a random-species subset from a huge (TB-scale) observations CSV,
for testing a non-curated architecture (no "focal species" special-casing) at a scale that
still iterates locally in a reasonable time.

Parsing uses only the Python standard library (csv, random, collections, argparse) since the
remote machine has no pandas - but numpy IS available there, so output goes straight to .npy
files in the exact format train_tf_sinr_test.py's --train_npy loader expects. No local
conversion step needed; just copy the three .npy files back.

Pass 1: stream the file once, counting rows per taxon_id only (memory: one dict entry per
        unique taxon_id seen, not the observations themselves - trivial even at hundreds of
        millions of rows). Randomly sample --n_species from the taxa meeting --min_count.
Pass 2: pre-allocate numpy arrays sized from pass 1's counts (upper bound, since pass 1 didn't
        validate coordinates), stream the file again writing directly into them for exactly the
        sampled species (up to --cap observations each if set; default uncapped), then trim to
        the actual number kept.

Deliberately does NOT single out any species for uncapped/full-count treatment the way
preprocess_training_data.py's --supplement_csvs does for focal species - every sampled
species is treated identically here, so the resulting dataset has no built-in data-volume
advantage baked in. Natural skew (some species with 100 points, some with 40,000+) is
preserved as-is, since that's the realistic distribution any final architecture needs to
handle without curation.

Usage (on the remote machine):
  python3 extract_random_species_dataset.py spatial_data.csv \\
      --n_species 5000 --min_count 100 --seed 42 --output_prefix train_5k_random

  # with a generous cap if the uncapped pull is too large to iterate on quickly:
  python3 extract_random_species_dataset.py spatial_data.csv \\
      --n_species 5000 --min_count 100 --cap 10000 --output_prefix train_5k_random_capped

Then copy train_5k_random_locs.npy / _class_ids.npy / _unique_taxa.npy back to the local
machine and train directly with --train_npy train_5k_random.

IMPORTANT: assumes the input CSV has longitude,latitude,taxon_id columns (same convention as
preprocess_training_data.py). If your remote file uses different column names, edit LON_COL /
LAT_COL / TAXON_COL below before running.
"""

import argparse
import csv
import random
import time
from collections import defaultdict

import numpy as np

LON_COL = "longitude"
LAT_COL = "latitude"
TAXON_COL = "taxon_id"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("input_csv", help=f"Path to CSV with {LON_COL},{LAT_COL},{TAXON_COL} columns")
    p.add_argument("--n_species", type=int, default=5000,
                   help="Number of species to randomly sample (default: 5000)")
    p.add_argument("--min_count", type=int, default=100,
                   help="Only consider species with at least this many observations (default: 100)")
    p.add_argument("--cap", type=int, default=None,
                   help="Optional max observations per sampled species. Default: uncapped "
                        "(every sampled species keeps ALL its real observations).")
    p.add_argument("--seed", type=int, default=42,
                   help="Random seed for species sampling (default: 42, for reproducibility)")
    p.add_argument("--force_include", type=str, default=None,
                   help="Comma-separated taxon_ids to always include, regardless of the random "
                        "draw or --min_count (e.g. known focal species you want in every extract "
                        "for direct comparison against prior runs). Counts toward --n_species.")
    p.add_argument("--output_prefix", type=str, default="train_5k_random",
                   help="Prefix for output .npy files")
    p.add_argument("--progress_every", type=int, default=5_000_000,
                   help="Print progress every this many rows (default: 5M)")
    return p.parse_args()


def pass1_count_species(input_csv, progress_every):
    """Stream once, counting observations per taxon_id. Returns {taxon_id: count}."""
    print(f"Pass 1: counting observations per species in {input_csv} ...")
    counts = defaultdict(int)
    total_rows = 0
    skipped = 0
    t0 = time.time()

    with open(input_csv, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            total_rows += 1
            if total_rows % progress_every == 0:
                elapsed = time.time() - t0
                print(f"  {total_rows:,} rows  |  {len(counts):,} unique taxa  |  {elapsed:.0f}s",
                      flush=True)
            try:
                tid = int(row[TAXON_COL])
            except (ValueError, KeyError, TypeError):
                skipped += 1
                continue
            counts[tid] += 1

    elapsed = time.time() - t0
    print(f"\nPass 1 done in {elapsed:.0f}s: {total_rows:,} total rows "
          f"({skipped:,} skipped), {len(counts):,} unique taxa")
    return counts


def pass2_extract_species(input_csv, sampled_set, taxa_to_idx, cap, upper_bound, progress_every):
    """Stream again, writing directly into pre-allocated numpy arrays. Returns trimmed arrays
    and the actual per-species kept counts."""
    print(f"\nPass 2: extracting {len(sampled_set):,} sampled species "
          f"({'uncapped' if cap is None else f'capped at {cap}'}), "
          f"pre-allocated for up to {upper_bound:,} obs ...")

    lons = np.empty(upper_bound, dtype=np.float32)
    lats = np.empty(upper_bound, dtype=np.float32)
    class_ids = np.empty(upper_bound, dtype=np.int32)

    kept_counts = defaultdict(int)
    write_idx = 0
    total_rows = 0
    skipped_invalid = 0
    t0 = time.time()

    with open(input_csv, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            total_rows += 1
            if total_rows % progress_every == 0:
                elapsed = time.time() - t0
                print(f"  {total_rows:,} rows scanned  |  {write_idx:,} kept  |  {elapsed:.0f}s",
                      flush=True)

            try:
                tid = int(row[TAXON_COL])
            except (ValueError, KeyError, TypeError):
                continue
            if tid not in sampled_set:
                continue

            try:
                lon = float(row[LON_COL])
                lat = float(row[LAT_COL])
            except (ValueError, KeyError, TypeError):
                skipped_invalid += 1
                continue
            if not (-180 <= lon <= 180 and -90 <= lat <= 90):
                skipped_invalid += 1
                continue

            if cap is not None and kept_counts[tid] >= cap:
                continue

            lons[write_idx] = lon
            lats[write_idx] = lat
            class_ids[write_idx] = taxa_to_idx[tid]
            write_idx += 1
            kept_counts[tid] += 1

    elapsed = time.time() - t0
    print(f"\nPass 2 done in {elapsed:.0f}s: {write_idx:,} observations kept "
          f"(skipped {skipped_invalid:,} invalid coords)")
    return lons[:write_idx], lats[:write_idx], class_ids[:write_idx], kept_counts


def main():
    args = parse_args()
    rng = random.Random(args.seed)

    counts = pass1_count_species(args.input_csv, args.progress_every)
    eligible = [tid for tid, c in counts.items() if c >= args.min_count]
    print(f"\n{len(eligible):,} species have >= {args.min_count} observations")

    forced = []
    if args.force_include:
        requested = [int(t.strip()) for t in args.force_include.split(",")]
        forced = [t for t in requested if t in counts]
        missing = [t for t in requested if t not in counts]
        if missing:
            print(f"WARNING: force_include taxon_ids not found in the input file at all: {missing}")
        print(f"\nForce-including {len(forced):,} taxa regardless of --min_count/random draw:")
        for t in forced:
            print(f"  {t}: {counts[t]:,} obs")
        if len(forced) > args.n_species:
            raise ValueError(f"--force_include has {len(forced)} taxa, more than --n_species={args.n_species}")

    forced_set = set(forced)
    remaining_pool = [t for t in eligible if t not in forced_set]
    n_random = args.n_species - len(forced)
    if len(remaining_pool) < n_random:
        raise ValueError(f"Only {len(remaining_pool)} eligible non-forced species, "
                          f"but need {n_random} more to reach --n_species={args.n_species}")

    sampled = forced + rng.sample(remaining_pool, n_random)
    sampled_set = set(sampled)
    sampled_counts_preview = sorted((counts[t] for t in sampled), reverse=True)
    n = len(sampled_counts_preview)
    total_preview = sum(sampled_counts_preview)
    print(f"\nSampled {args.n_species:,} species (seed={args.seed}):")
    print(f"  smallest: {sampled_counts_preview[-1]:,} obs   largest: {sampled_counts_preview[0]:,} obs")
    print(f"  mean: {total_preview/n:.0f} obs   median: {sampled_counts_preview[n//2]:,} obs")
    print(f"  expected total (uncapped): {total_preview:,} obs")

    upper_bound = sum(min(counts[t], args.cap) if args.cap is not None else counts[t] for t in sampled)

    unique_taxa = np.array(sorted(sampled), dtype=np.int64)
    taxa_to_idx = {int(t): i for i, t in enumerate(unique_taxa)}

    lons, lats, class_ids, kept_counts = pass2_extract_species(
        args.input_csv, sampled_set, taxa_to_idx, args.cap, upper_bound, args.progress_every
    )
    locs_arr = np.stack([lons, lats], axis=1)

    locs_path = f"{args.output_prefix}_locs.npy"
    class_ids_path = f"{args.output_prefix}_class_ids.npy"
    taxa_path = f"{args.output_prefix}_unique_taxa.npy"
    manifest_path = f"{args.output_prefix}_species_manifest.csv"

    np.save(locs_path, locs_arr)
    np.save(class_ids_path, class_ids)
    np.save(taxa_path, unique_taxa)
    with open(manifest_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["taxon_id", "obs_count"])
        for tid in sorted(kept_counts, key=lambda t: kept_counts[t], reverse=True):
            writer.writerow([tid, kept_counts[tid]])

    print(f"\nFinal dataset: {len(class_ids):,} observations, {len(unique_taxa):,} taxa")
    print(f"Mean obs/taxon: {len(class_ids)/len(unique_taxa):.0f}")
    print(f"\nSaved (copy these back to the local machine, then train with --train_npy {args.output_prefix}):")
    import os
    for path in (locs_path, class_ids_path, taxa_path):
        print(f"  {path}  ({os.path.getsize(path)/1e6:.1f} MB)")
    print(f"  {manifest_path}  (species list + kept obs counts, for reference)")


if __name__ == "__main__":
    main()
