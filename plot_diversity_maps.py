"""
Plot the stacked-range counts from diversity_maps.py: one PNG per iconic group plus "All species"
and an overview panel, titled with the number of stacked ranges, e.g. "Birds (7,015 ranges)".

  python plot_diversity_maps.py diversity/v47_e003
"""
import os
import sys
import warnings

import geopandas as gpd
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import median_filter

CMAP = mcolors.LinearSegmentedColormap.from_list("richness", [
    (0.00, "#ffffff"), (0.06, "#d4e9f1"), (0.22, "#4f9e3a"), (0.45, "#9cc63b"),
    (0.62, "#e8d830"), (0.80, "#f39a2b"), (1.00, "#d7301f")])


def draw(ax, grid, title, world, extent):
    vmax = max(np.percentile(grid[grid > 0], 99.5), 1) if (grid > 0).any() else 1
    ax.imshow(grid, extent=extent, origin="upper", cmap=CMAP, vmin=0, vmax=vmax,
              interpolation="nearest", aspect="equal")
    world.boundary.plot(ax=ax, linewidth=0.3, color="#555555")
    ax.set_xlim(-180, 180); ax.set_ylim(-90, 90)
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.set_title(title, fontsize=13)
    return vmax


def main(out_dir, despike=False):
    d = np.load(os.path.join(out_dir, "counts.npz"))
    counts, labels, n_ranges = d["counts"], list(d["labels"]), d["n_ranges"]
    suffix = ""
    if despike:
        # display-only cleanup: land cells only, and single-cell spikes (cells where a large share
        # of all ranges flip on at once) replaced by a 3x3 median
        raster = np.load("bioclim_coast_effort_landmask_interact_scaled.npy", mmap_mode="r")
        step = raster.shape[0] // counts.shape[1]
        land = ~np.isnan(np.asarray(raster[step // 2::step, step // 2::step, 1]))[:counts.shape[1], :counts.shape[2]]
        counts = np.stack([np.where(land, median_filter(c, size=3), 0) for c in counts])
        suffix = "_despiked"
    extent = [-180, 180, -90, 90]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        world = gpd.read_file(gpd.datasets.get_path("naturalearth_lowres"))

    order = np.argsort(-n_ranges)
    panels = [("All species", counts.sum(0), int(n_ranges.sum()))] + \
             [(labels[i], counts[i], int(n_ranges[i])) for i in order]
    with open(os.path.join(out_dir, f"range_counts{suffix}.csv"), "w") as f:
        f.write("group,ranges,max_overlap\n")
        for name, grid, n in panels:
            f.write(f"{name},{n},{int(grid.max())}\n")

    for name, grid, n in panels:
        fig, ax = plt.subplots(figsize=(16, 8.4))
        vmax = draw(ax, grid, f"{name} ({n:,} ranges)", world, extent)
        sm = plt.cm.ScalarMappable(cmap=CMAP, norm=mcolors.Normalize(0, vmax))
        fig.colorbar(sm, ax=ax, fraction=0.025, pad=0.01, label="overlapping ranges")
        fn = name.split(" (")[0].lower().replace(" ", "_").replace("-", "_")
        fig.savefig(os.path.join(out_dir, f"diversity_{fn}{suffix}.png"), dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"{name}: {n:,} ranges, max overlap {int(grid.max()):,}")

    cols = 3
    rows = int(np.ceil(len(panels) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 9, rows * 4.8))
    for ax, (name, grid, n) in zip(axes.ravel(), panels):
        draw(ax, grid, f"{name} ({n:,} ranges)", world, extent)
    for ax in axes.ravel()[len(panels):]:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, f"diversity_overview{suffix}.png"), dpi=110)
    print(f"saved {out_dir}/diversity_*.png")


if __name__ == "__main__":
    main(sys.argv[1], despike="--despike" in sys.argv[2:])
