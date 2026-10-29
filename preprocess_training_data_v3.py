"""
Same purpose/CLI as preprocess_training_data_v2.py (real production spatial_data.csv schema,
captive/positional_accuracy filtering, per-species hard_cap, --guarantee_taxa exemption) but
fixed for the true production scale (345M rows / 24.2GB), where v2 was OOM-killed twice around
85-110M rows in.

Root cause of the OOM: v2 accumulates every kept observation as a Python (lon, lat) tuple in a
per-taxon list (bg_locs[tid].append(...)) as it streams. With tens of thousands of distinct
background taxa live at once, that's tens of millions of individual Python tuple objects
in memory simultaneously - each one costs 100+ bytes of interpreter overhead versus a few bytes
in a packed array.

Fix, ported from the already-proven extract_random_species_dataset.py (which completed two full
passes over this exact file on this exact machine with no issue): two-pass, pre-allocated numpy
arrays.
  Pass 1: stream once, apply captive/accuracy/validity filters, only COUNT observations per
          taxon_id (one dict entry per unique taxon - trivial memory even at 100k+ taxa).
  Pass 2: compute the exact final per-taxon kept count (hard_cap-limited, guarantee-exempt,
          max_taxa-restricted) from pass 1's counts, pre-allocate float32/int32 numpy arrays
          sized exactly, then stream the file again writing straight into them.

Usage: identical to v2.
  python preprocess_training_data_v3.py spatial_data.csv --hard_cap 100000 \\
      --guarantee_taxa_file group500_taxa.npy --output_prefix train_5007_clean
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
    p.add_argument("--require_community", action=argparse.BooleanOptionalAction, default=False,
                   help="Keep only rows that have reached community/research-grade ID consensus "
                        "(community truthy). Off by default (matches prior behavior). Added after "
                        "v41's clean_matched build let through ~20 wild-continent-mismatch "
                        "Millipede (taxon 47737) records - all had captive=0 (correctly not "
                        "captive) AND community=0 (not yet consensus-ID'd), a signal the pipeline "
                        "never filtered on. Missing/blank community value is treated as NOT "
                        "consensus (dropped) when this flag is on - unlike --exclude_captive's "
                        "keep-on-missing default, since here the absence of consensus is the "
                        "actual condition being filtered on, not an unrelated flag that merely "
                        "wasn't recorded.")
    p.add_argument("--max_positional_accuracy", type=float, default=5000.0,
                   help="Drop rows with positional_accuracy (meters) above this. Missing value is kept. "
                        "Set to 0 or negative to disable. Default: 5000m (~half a covariate pixel width).")
    p.add_argument("--guarantee_taxa", type=str, default=None,
                   help="Comma-separated taxon_ids (e.g. the group500 set) that are exempt from "
                        "--hard_cap - all their qualifying (non-captive, accuracy-ok) observations are "
                        "kept, same treatment as --supplement_csvs species but sourced from the main "
                        "file instead of needing a separate CSV per species.")
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


def row_passes_filters(row, exclude_captive, max_positional_accuracy, require_community=False):
    """Returns (lon, lat, tid) if the row is valid/qualifying, else None. Shared by both passes
    so pass 1's counts and pass 2's writes can never silently diverge."""
    try:
        lon = float(row["longitude"])
        lat = float(row["latitude"])
        tid = int(row["taxon_id"])
    except (ValueError, KeyError):
        return None
    if not (-180 <= lon <= 180 and -90 <= lat <= 90):
        return None
    if exclude_captive:
        captive_raw = (row.get("captive") or "").strip().lower()
        if captive_raw in TRUE_STRINGS:
            return None
    if require_community:
        community_raw = (row.get("community") or "").strip().lower()
        if community_raw not in TRUE_STRINGS:
            return None
    if max_positional_accuracy > 0:
        acc_raw = (row.get("positional_accuracy") or "").strip()
        if acc_raw:
            try:
                acc = float(acc_raw)
                if acc > max_positional_accuracy:
                    return None
            except ValueError:
                pass  # unparseable - treat like missing, keep
    return lon, lat, tid


def main():
    args = parse_args()

    guarantee_taxa_set = set()
    if args.guarantee_taxa:
        guarantee_taxa_set.update(int(t.strip()) for t in args.guarantee_taxa.split(","))
    if args.guarantee_taxa_file:
        guarantee_taxa_set.update(int(t) for t in np.load(args.guarantee_taxa_file))
    if guarantee_taxa_set:
        print(f"Guaranteeing {len(guarantee_taxa_set):,} taxa are included uncapped "
              f"(still subject to captive/community/accuracy filters)\n")

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

    # ================= Pass 1: count only (no point data held in memory) =================
    print(f"Pass 1: counting qualifying observations per bg taxon in {args.input_csv} ...")
    print(f"  exclude_captive={args.exclude_captive}, require_community={args.require_community}, "
          f"max_positional_accuracy={args.max_positional_accuracy}, "
          f"skipping {len(focal_taxon_ids)} focal taxa")

    bg_count = defaultdict(int)
    total_rows = 0
    skipped_parse = 0
    skipped_captive = 0
    skipped_community = 0
    skipped_accuracy = 0
    t0 = time.time()

    with open(args.input_csv, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            total_rows += 1
            if total_rows % args.chunk_size == 0:
                elapsed = time.time() - t0
                print(f"  {total_rows:,} rows  |  {len(bg_count):,} bg taxa so far  |  {elapsed:.0f}s",
                      flush=True)

            captive_raw = (row.get("captive") or "").strip().lower()
            community_raw = (row.get("community") or "").strip().lower()
            acc_raw = (row.get("positional_accuracy") or "").strip()
            try:
                lon = float(row["longitude"]); lat = float(row["latitude"]); tid = int(row["taxon_id"])
                valid = -180 <= lon <= 180 and -90 <= lat <= 90
            except (ValueError, KeyError):
                valid = False
            if not valid:
                skipped_parse += 1
                continue
            if args.exclude_captive and captive_raw in TRUE_STRINGS:
                skipped_captive += 1
                continue
            if args.require_community and community_raw not in TRUE_STRINGS:
                skipped_community += 1
                continue
            if args.max_positional_accuracy > 0 and acc_raw:
                try:
                    if float(acc_raw) > args.max_positional_accuracy:
                        skipped_accuracy += 1
                        continue
                except ValueError:
                    pass
            if tid in focal_taxon_ids:
                continue
            bg_count[tid] += 1

    elapsed = time.time() - t0
    print(f"\nPass 1 done in {elapsed:.0f}s")
    print(f"  Total rows: {total_rows:,}")
    print(f"  Skipped - parse errors: {skipped_parse:,}  captive: {skipped_captive:,}  "
          f"community: {skipped_community:,}  positional_accuracy: {skipped_accuracy:,}")
    print(f"  Unique bg taxa: {len(bg_count):,}")

    # --- restrict to top max_taxa bg taxa (guaranteed taxa exempt) ---
    all_bg_taxa = sorted(bg_count.keys())
    if args.max_taxa is not None and len(all_bg_taxa) > args.max_taxa:
        rankable = [t for t in all_bg_taxa if t not in guarantee_taxa_set]
        top_bg = sorted(rankable, key=lambda t: bg_count[t], reverse=True)[:args.max_taxa]
        kept_bg_set = set(top_bg) | (guarantee_taxa_set & set(all_bg_taxa))
        all_bg_taxa = sorted(kept_bg_set)
        print(f"  Kept top {args.max_taxa:,} bg taxa (+ any guaranteed taxa) → {len(all_bg_taxa):,} bg taxa")
    kept_bg_set = set(all_bg_taxa)

    if guarantee_taxa_set:
        missing = guarantee_taxa_set - set(focal_species_locs.keys()) - kept_bg_set
        present = guarantee_taxa_set - missing
        print(f"\nGuaranteed-taxa check: {len(present)}/{len(guarantee_taxa_set)} present in final dataset")
        if missing:
            print(f"  WARNING - {len(missing)} guaranteed taxa have ZERO qualifying observations "
                  f"after captive/accuracy filtering: {sorted(missing)}")

    # --- exact per-taxon final kept count, for pre-allocation ---
    final_bg_n = {}
    for tid in all_bg_taxa:
        final_bg_n[tid] = bg_count[tid] if tid in guarantee_taxa_set else min(bg_count[tid], args.hard_cap)
    bg_upper_bound = sum(final_bg_n.values())
    focal_n = sum(len(v) for v in focal_species_locs.values())
    total_upper_bound = bg_upper_bound + focal_n
    print(f"\nPre-allocating for {total_upper_bound:,} total observations "
          f"({focal_n:,} focal + {bg_upper_bound:,} bg)")

    all_taxa = sorted(set(focal_species_locs.keys()) | kept_bg_set)
    unique_taxa = np.array(all_taxa, dtype=np.int64)
    taxa_to_idx = {t: i for i, t in enumerate(unique_taxa)}

    lons = np.empty(total_upper_bound, dtype=np.float32)
    lats = np.empty(total_upper_bound, dtype=np.float32)
    class_ids = np.empty(total_upper_bound, dtype=np.int32)

    write_idx = 0
    for tid, locs in focal_species_locs.items():
        cls = taxa_to_idx[tid]
        for lon, lat in locs:
            lons[write_idx] = lon; lats[write_idx] = lat; class_ids[write_idx] = cls
            write_idx += 1
    focal_write_end = write_idx
    print(f"  Focal points written: {focal_write_end:,}")

    # ================= Pass 2: fill pre-allocated bg arrays =================
    print(f"\nPass 2: writing bg observations directly into pre-allocated arrays ...")
    bg_running = defaultdict(int)
    total_rows2 = 0
    t1 = time.time()

    with open(args.input_csv, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            total_rows2 += 1
            if total_rows2 % args.chunk_size == 0:
                elapsed = time.time() - t1
                print(f"  {total_rows2:,} rows  |  {write_idx - focal_write_end:,} bg pts written  |  "
                      f"{elapsed:.0f}s", flush=True)

            parsed = row_passes_filters(row, args.exclude_captive, args.max_positional_accuracy,
                                         args.require_community)
            if parsed is None:
                continue
            lon, lat, tid = parsed
            if tid in focal_taxon_ids or tid not in kept_bg_set:
                continue
            if tid not in guarantee_taxa_set and bg_running[tid] >= args.hard_cap:
                continue

            cls = taxa_to_idx[tid]
            lons[write_idx] = lon; lats[write_idx] = lat; class_ids[write_idx] = cls
            write_idx += 1
            bg_running[tid] += 1

    elapsed = time.time() - t1
    print(f"\nPass 2 done in {elapsed:.0f}s: {write_idx - focal_write_end:,} bg observations written")

    # trim any slack from the upper-bound estimate (should be exact, but guard anyway)
    lons = lons[:write_idx]; lats = lats[:write_idx]; class_ids = class_ids[:write_idx]
    locs_arr = np.stack([lons, lats], axis=1)

    n_focal = len(focal_species_locs)
    n_bg = len(all_bg_taxa)
    print(f"\n  Final dataset: {len(class_ids):,} observations, {len(unique_taxa):,} taxa")
    print(f"    focal: {n_focal:,} taxa, {focal_write_end:,} obs (full counts)")
    print(f"    bg:    {n_bg:,} taxa, {len(class_ids) - focal_write_end:,} obs (capped at {args.hard_cap})")
    print(f"  Mean obs/taxon: {len(class_ids)/len(unique_taxa):.0f}")

    locs_path = f"{args.output_prefix}_locs.npy"
    class_ids_path = f"{args.output_prefix}_class_ids.npy"
    taxa_path = f"{args.output_prefix}_unique_taxa.npy"

    np.save(locs_path, locs_arr)
    np.save(class_ids_path, class_ids)
    np.save(taxa_path, unique_taxa)

    KNOWN_7 = [
        ("Pika", 43188), ("Millipede", 47737), ("Lily", 51315), ("Crab", 54952),
        ("Lewisia", 47360), ("Amorph", 57358), ("Ant", 472214),
    ]
    print(f"\n  Known-7 continuity check (post-filter counts, compare to train_5007's pre-filter counts):")
    for name, tid in KNOWN_7:
        n = final_bg_n.get(tid, 0) if tid not in focal_species_locs else len(focal_species_locs[tid])
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
