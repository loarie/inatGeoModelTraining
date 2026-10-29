"""
Generate a log-count effort raster from a large presence-points file.
Streams through the file in chunks — requires only numpy (no pandas).

Output: effort_raster.npy, shape (2160, 4320), float32
  - matches bioclim_coast_scaled.npy spatial resolution (5 arcminutes)
  - values are log1p(count) of all observations per grid cell

Usage:
  python3 make_effort_raster.py spatial_data.csv
"""

import csv
import sys
import numpy as np

HEIGHT, WIDTH = 2160, 4320  # 5 arcminute global grid
CHUNK = 1_000_000


def main():
    input_file = sys.argv[1] if len(sys.argv) > 1 else "spatial_data.csv"
    output_file = sys.argv[2] if len(sys.argv) > 2 else "effort_raster.npy"

    counts = np.zeros((HEIGHT, WIDTH), dtype=np.float32)
    total = 0

    with open(input_file, newline="") as f:
        reader = csv.DictReader(f)
        lats, lons = [], []
        for row in reader:
            try:
                lats.append(float(row["latitude"]))
                lons.append(float(row["longitude"]))
            except (ValueError, KeyError):
                continue

            if len(lats) >= CHUNK:
                _accumulate(np.array(lats), np.array(lons), counts)
                total += len(lats)
                lats, lons = [], []
                print(f"  processed {total:,} rows...", end="\r", flush=True)

        if lats:
            _accumulate(np.array(lats), np.array(lons), counts)
            total += len(lats)

    print(f"\nTotal rows: {total:,}")
    print(f"Cells with at least 1 obs: {(counts > 0).sum():,} ({(counts > 0).mean()*100:.1f}%)")
    print(f"Max count in a cell: {counts.max():.0f}")

    effort = np.log1p(counts).astype(np.float32)
    np.save(output_file, effort)
    print(f"Saved {output_file}  shape={effort.shape}  min={effort.min():.2f}  max={effort.max():.2f}")


def _accumulate(lats, lons, counts):
    valid = (lats >= -90) & (lats <= 90) & (lons >= -180) & (lons <= 180)
    lats, lons = lats[valid], lons[valid]
    row = np.clip(((90 - lats) / 180 * HEIGHT).astype(np.int32), 0, HEIGHT - 1)
    col = np.clip(((lons + 180) / 360 * WIDTH).astype(np.int32), 0, WIDTH - 1)
    np.add.at(counts, (row, col), 1)


if __name__ == "__main__":
    main()
