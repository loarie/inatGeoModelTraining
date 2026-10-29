"""
One-time preprocessing: stream a large CSV and write compact numpy arrays.

Applies hard_cap per species and saves:
  {output_prefix}_locs.npy        float32 (N, 2)  [longitude, latitude]
  {output_prefix}_class_ids.npy   int32   (N,)
  {output_prefix}_unique_taxa.npy int64   (K,)    sorted taxon_id -> class index

Focal species can be supplied via --supplement_csvs with their full observation
counts, bypassing hard_cap. They are included unconditionally alongside the
background species from the main file. --max_taxa limits only bg species.

Usage:
  # full run
  python preprocess_training_data.py spatial_data.csv --hard_cap 5000 --output_prefix train_120k

  # 10K bg species + 7 focal species at full obs
  python preprocess_training_data.py spatial_data.csv \\
      --hard_cap 5000 --max_taxa 10000 --output_prefix train_10k \\
      --supplement_csvs pika.csv,lily.csv,lewisia.csv,millipede.csv,crab.csv,amorph.csv,ant.csv
"""

import argparse
import csv
import os
import sys
import time
import numpy as np
from collections import defaultdict


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("input_csv", help="Path to CSV with longitude,latitude,taxon_id columns")
    p.add_argument("--hard_cap", type=int, default=5000,
                   help="Max observations per bg species (default: 5000)")
    p.add_argument("--max_taxa", type=int, default=None,
                   help="Keep only the top N bg taxa by observation count. "
                        "Supplement CSV species are always included and do not count toward this limit.")
    p.add_argument("--supplement_csvs", type=str, default=None,
                   help="Comma-separated list of CSV files for focal species. "
                        "All observations are kept (no hard_cap). Species in these files "
                        "are excluded from the main file to avoid double-counting.")
    p.add_argument("--output_prefix", type=str, default="train",
                   help="Prefix for output .npy files (default: 'train')")
    p.add_argument("--chunk_size", type=int, default=1_000_000,
                   help="Rows to read per chunk for progress reporting (default: 1M)")
    return p.parse_args()


def load_supplement_csv(path):
    """Load a focal species CSV with all observations (no cap). Returns {tid: [(lon,lat),...]}."""
    species_locs = defaultdict(list)
    skipped = 0
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                lon = float(row["longitude"])
                lat = float(row["latitude"])
                tid = int(row["taxon_id"])
            except (ValueError, KeyError):
                skipped += 1
                continue
            if not (-180 <= lon <= 180 and -90 <= lat <= 90):
                skipped += 1
                continue
            species_locs[tid].append((lon, lat))
    return species_locs, skipped


def main():
    args = parse_args()

    # --- Step 1: load supplement (focal) species at full obs counts ---
    focal_species_locs = {}   # tid -> [(lon, lat), ...]
    focal_taxon_ids = set()

    if args.supplement_csvs:
        paths = [p.strip() for p in args.supplement_csvs.split(",")]
        for path in paths:
            print(f"Loading supplement CSV: {path}")
            sp_locs, skipped = load_supplement_csv(path)
            for tid, locs in sp_locs.items():
                if tid in focal_species_locs:
                    focal_species_locs[tid].extend(locs)
                else:
                    focal_species_locs[tid] = list(locs)
            focal_taxon_ids.update(sp_locs.keys())
            total = sum(len(v) for v in sp_locs.values())
            print(f"  {len(sp_locs):,} taxa, {total:,} obs (skipped {skipped})")
        print(f"Supplement total: {len(focal_species_locs):,} focal taxa, "
              f"{sum(len(v) for v in focal_species_locs.values()):,} obs\n")

    # --- Step 2: stream main file, skip focal taxa, apply hard_cap to bg ---
    print(f"Streaming {args.input_csv} ...")
    print(f"  hard_cap={args.hard_cap}, max_taxa={args.max_taxa}, "
          f"skipping {len(focal_taxon_ids)} focal taxa")

    bg_locs = defaultdict(list)   # taxon_id -> list of (lon, lat)
    bg_count = defaultdict(int)   # total obs seen (before cap)
    total_rows = 0
    skipped = 0
    t0 = time.time()

    with open(args.input_csv, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            total_rows += 1
            if total_rows % args.chunk_size == 0:
                elapsed = time.time() - t0
                print(f"  {total_rows:,} rows  |  {len(bg_locs):,} bg taxa  |  {elapsed:.0f}s",
                      flush=True)

            try:
                lon = float(row["longitude"])
                lat = float(row["latitude"])
                tid = int(row["taxon_id"])
            except (ValueError, KeyError):
                skipped += 1
                continue

            if not (-180 <= lon <= 180 and -90 <= lat <= 90):
                skipped += 1
                continue

            if tid in focal_taxon_ids:
                continue  # handled by supplement

            bg_count[tid] += 1
            if len(bg_locs[tid]) < args.hard_cap:
                bg_locs[tid].append((lon, lat))

    elapsed = time.time() - t0
    print(f"\nMain file done in {elapsed:.0f}s")
    print(f"  Total rows: {total_rows:,}  |  Skipped: {skipped:,}")
    print(f"  Unique bg taxa: {len(bg_locs):,}")

    # --- Step 3: optionally restrict bg taxa to top N ---
    all_bg_taxa = sorted(bg_locs.keys())
    if args.max_taxa is not None and len(all_bg_taxa) > args.max_taxa:
        top_bg = sorted(all_bg_taxa, key=lambda t: bg_count[t], reverse=True)[:args.max_taxa]
        top_bg_set = set(top_bg)
        bg_locs = {t: v for t, v in bg_locs.items() if t in top_bg_set}
        all_bg_taxa = sorted(bg_locs.keys())
        print(f"  Kept top {args.max_taxa:,} bg taxa → {len(all_bg_taxa):,} bg taxa")

    # --- Step 4: merge focal + bg, build arrays ---
    all_taxa = sorted(set(focal_species_locs.keys()) | set(all_bg_taxa))
    unique_taxa = np.array(all_taxa, dtype=np.int64)
    taxa_to_idx = {t: i for i, t in enumerate(unique_taxa)}

    all_lons, all_lats, all_class_ids = [], [], []

    for tid in all_taxa:
        cls = taxa_to_idx[tid]
        locs = focal_species_locs.get(tid, []) + bg_locs.get(tid, [])
        for lon, lat in locs:
            all_lons.append(lon)
            all_lats.append(lat)
            all_class_ids.append(cls)

    locs_arr = np.stack([np.array(all_lons, dtype=np.float32),
                         np.array(all_lats, dtype=np.float32)], axis=1)
    class_ids_arr = np.array(all_class_ids, dtype=np.int32)

    n_focal = len(focal_species_locs)
    n_bg = len(all_bg_taxa)
    n_focal_obs = sum(len(focal_species_locs.get(t, [])) for t in all_taxa)
    print(f"\n  Final dataset: {len(class_ids_arr):,} observations, {len(unique_taxa):,} taxa")
    print(f"    focal: {n_focal:,} taxa, {n_focal_obs:,} obs (full counts)")
    print(f"    bg:    {n_bg:,} taxa, {len(class_ids_arr)-n_focal_obs:,} obs (capped at {args.hard_cap})")
    print(f"  Mean obs/taxon: {len(class_ids_arr)/len(unique_taxa):.0f}")

    # --- Step 5: save ---
    locs_path = f"{args.output_prefix}_locs.npy"
    class_ids_path = f"{args.output_prefix}_class_ids.npy"
    taxa_path = f"{args.output_prefix}_unique_taxa.npy"

    np.save(locs_path, locs_arr)
    np.save(class_ids_path, class_ids_arr)
    np.save(taxa_path, unique_taxa)

    locs_mb = os.path.getsize(locs_path) / 1e6
    ids_mb = os.path.getsize(class_ids_path) / 1e6
    taxa_mb = os.path.getsize(taxa_path) / 1e6
    print(f"\nSaved:")
    print(f"  {locs_path}  ({locs_mb:.1f} MB)")
    print(f"  {class_ids_path}  ({ids_mb:.1f} MB)")
    print(f"  {taxa_path}  ({taxa_mb:.1f} MB)")
    print(f"  Total: {locs_mb + ids_mb + taxa_mb:.1f} MB")


if __name__ == "__main__":
    main()
