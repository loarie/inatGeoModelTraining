"""
Same output format as preprocess_training_data.py (locs/class_ids/unique_taxa .npy), but reads
directly from the real production spatial_data.csv export schema:

  observation_id,latitude,longitude,observed_on,leaf_class_id,iconic_class_id,spatial_class_id,
  taxon_id,obs_taxon_id,community,captive,positional_accuracy

NEW vs. the original script (motivated by taxon 133607, which turned out to span disjunct continents -
India/SE Asia/US East Coast/Caribbean/Brazil/Hawaii/Pacific in one "species", almost certainly
captive/cultivated contamination or a taxonomic aggregate riding on records outside the wild
range - and there is no rank/captive filtering anywhere upstream of this pipeline):

  --exclude_captive (default on): drops rows where captive is true. A row with a MISSING/blank
    captive value is KEPT (absence of the flag isn't evidence of captivity).
  --max_positional_accuracy (default 5000, meters): drops rows where positional_accuracy is
    present AND exceeds this. A row with a MISSING/blank positional_accuracy is KEPT (per user:
    nil is fine, only exclude confirmed-imprecise records). 5000m was chosen relative to the
    bioclim covariate raster's resolution (2160x4320, 0.0833deg/px, ~9.3km/px at the equator,
    narrower toward the poles) - comfortably under half a pixel width so GPS error can't
    plausibly place an observation in the wrong covariate cell, while keeping the vast majority
    of real citizen-science records (typical phone GPS accuracy is 5-50m).

Usage:
  python preprocess_training_data_v2.py spatial_data.csv --hard_cap 5000 --output_prefix train_120k_clean
  python preprocess_training_data_v2.py spatial_data.csv \\
      --hard_cap 5000 --max_taxa 10000 --output_prefix train_10k_clean \\
      --supplement_csvs pika.csv,lily.csv,lewisia.csv,millipede.csv,crab.csv,amorph.csv,ant.csv
"""

import argparse
import csv
import os
import time
import numpy as np
from collections import defaultdict

TRUE_STRINGS = {"t", "true", "1", "yes"}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("input_csv", help="Path to spatial_data.csv (real production export schema)")
    p.add_argument("--hard_cap", type=int, default=5000,
                   help="Max observations per bg species (default: 5000)")
    p.add_argument("--max_taxa", type=int, default=None,
                   help="Keep only the top N bg taxa by observation count. "
                        "Supplement CSV species are always included and do not count toward this limit.")
    p.add_argument("--supplement_csvs", type=str, default=None,
                   help="Comma-separated list of CSV files for focal species (longitude,latitude,taxon_id "
                        "format, same as before - these are assumed already clean). All observations are "
                        "kept (no hard_cap). Species in these files are excluded from the main file.")
    p.add_argument("--output_prefix", type=str, default="train")
    p.add_argument("--chunk_size", type=int, default=1_000_000)
    p.add_argument("--exclude_captive", action=argparse.BooleanOptionalAction, default=True,
                   help="Drop rows flagged captive/cultivated. Missing captive value is kept. Default: on.")
    p.add_argument("--max_positional_accuracy", type=float, default=5000.0,
                   help="Drop rows with positional_accuracy (meters) above this. Missing value is kept. "
                        "Set to 0 or negative to disable. Default: 5000m (~half a covariate pixel width).")
    p.add_argument("--guarantee_taxa", type=str, default=None,
                   help="Comma-separated taxon_ids (e.g. the group500 set) that are exempt from "
                        "--hard_cap - all their qualifying (non-captive, accuracy-ok) observations are "
                        "kept, same treatment as --supplement_csvs species but sourced from the main "
                        "file instead of needing a separate CSV per species. Use this to guarantee a "
                        "specific tracked group survives the rebuild for before/after continuity "
                        "comparisons, rather than assuming they clear a --max_taxa cutoff.")
    p.add_argument("--guarantee_taxa_file", type=str, default=None,
                   help="Path to a .npy file of taxon_ids, alternative to --guarantee_taxa for a large list.")
    return p.parse_args()


def load_supplement_csv(path):
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

    # --- Step 0: load the guaranteed-inclusion taxa set (exempt from hard_cap and max_taxa) ---
    guarantee_taxa_set = set()
    if args.guarantee_taxa:
        guarantee_taxa_set.update(int(t.strip()) for t in args.guarantee_taxa.split(","))
    if args.guarantee_taxa_file:
        guarantee_taxa_set.update(int(t) for t in np.load(args.guarantee_taxa_file))
    if guarantee_taxa_set:
        print(f"Guaranteeing {len(guarantee_taxa_set):,} taxa are included uncapped "
              f"(still subject to captive/accuracy filters)\n")

    # --- Step 1: load supplement (focal) species at full obs counts ---
    focal_species_locs = {}
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
          f"exclude_captive={args.exclude_captive}, max_positional_accuracy={args.max_positional_accuracy}, "
          f"skipping {len(focal_taxon_ids)} focal taxa")

    bg_locs = defaultdict(list)
    bg_count = defaultdict(int)
    total_rows = 0
    skipped_parse = 0
    skipped_captive = 0
    skipped_accuracy = 0
    t0 = time.time()

    with open(args.input_csv, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            total_rows += 1
            if total_rows % args.chunk_size == 0:
                elapsed = time.time() - t0
                print(f"  {total_rows:,} rows  |  {len(bg_locs):,} bg taxa  |  "
                      f"skipped: parse={skipped_parse:,} captive={skipped_captive:,} "
                      f"accuracy={skipped_accuracy:,}  |  {elapsed:.0f}s", flush=True)

            try:
                lon = float(row["longitude"])
                lat = float(row["latitude"])
                tid = int(row["taxon_id"])
            except (ValueError, KeyError):
                skipped_parse += 1
                continue

            if not (-180 <= lon <= 180 and -90 <= lat <= 90):
                skipped_parse += 1
                continue

            if args.exclude_captive:
                captive_raw = (row.get("captive") or "").strip().lower()
                if captive_raw in TRUE_STRINGS:
                    skipped_captive += 1
                    continue

            if args.max_positional_accuracy > 0:
                acc_raw = (row.get("positional_accuracy") or "").strip()
                if acc_raw:
                    try:
                        acc = float(acc_raw)
                        if acc > args.max_positional_accuracy:
                            skipped_accuracy += 1
                            continue
                    except ValueError:
                        pass  # unparseable accuracy value - treat like missing, keep the row

            if tid in focal_taxon_ids:
                continue  # handled by supplement

            bg_count[tid] += 1
            if tid in guarantee_taxa_set or len(bg_locs[tid]) < args.hard_cap:
                bg_locs[tid].append((lon, lat))

    elapsed = time.time() - t0
    print(f"\nMain file done in {elapsed:.0f}s")
    print(f"  Total rows: {total_rows:,}")
    print(f"  Skipped - parse errors: {skipped_parse:,}  captive: {skipped_captive:,}  "
          f"positional_accuracy: {skipped_accuracy:,}")
    print(f"  Unique bg taxa: {len(bg_locs):,}")

    # --- Step 3: optionally restrict bg taxa to top N (guaranteed taxa are exempt) ---
    all_bg_taxa = sorted(bg_locs.keys())
    if args.max_taxa is not None and len(all_bg_taxa) > args.max_taxa:
        rankable = [t for t in all_bg_taxa if t not in guarantee_taxa_set]
        top_bg = sorted(rankable, key=lambda t: bg_count[t], reverse=True)[:args.max_taxa]
        top_bg_set = set(top_bg) | (guarantee_taxa_set & set(all_bg_taxa))
        bg_locs = {t: v for t, v in bg_locs.items() if t in top_bg_set}
        all_bg_taxa = sorted(bg_locs.keys())
        print(f"  Kept top {args.max_taxa:,} bg taxa (+ any guaranteed taxa) → {len(all_bg_taxa):,} bg taxa")

    # --- verify the guaranteed taxa actually made it, rather than assume ---
    if guarantee_taxa_set:
        missing = guarantee_taxa_set - set(focal_species_locs.keys()) - set(all_bg_taxa)
        present = guarantee_taxa_set - missing
        print(f"\nGuaranteed-taxa check: {len(present)}/{len(guarantee_taxa_set)} present in final dataset")
        if missing:
            print(f"  WARNING - {len(missing)} guaranteed taxa have ZERO qualifying observations "
                  f"after captive/accuracy filtering: {sorted(missing)}")

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

    KNOWN_7 = [
        ("Pika", 43188), ("Millipede", 47737), ("Lily", 51315), ("Crab", 54952),
        ("Lewisia", 47360), ("Amorph", 57358), ("Ant", 472214),
    ]
    print(f"\n  Known-7 continuity check (post-filter counts, compare to train_5007's pre-filter counts):")
    for name, tid in KNOWN_7:
        n = len(focal_species_locs.get(tid, [])) + len(bg_locs.get(tid, []))
        print(f"    {name:10s} ({tid}): {n:,} obs" + ("  [NOT FOUND]" if n == 0 else ""))

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
