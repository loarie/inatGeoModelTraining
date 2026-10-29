import numpy as np, pandas as pd, tensorflow as tf, yaml
import matplotlib.pyplot as plt, matplotlib.colors as mcolors
import geopandas as gpd, warnings
from lib.models.encoders import CoordEncoder
from lib.models.geo_model_net import FocalScatterAdd, IndependentPerClassHead

import argparse, os, sys
parser = argparse.ArgumentParser()
parser.add_argument("--train_npy", default=None, help="Prefix for npy files (e.g. 'train_10k')")
parser.add_argument("--train_csv", default="combined_train_5k.csv", help="CSV fallback")
parser.add_argument("--model", default="test_pika_model.keras")
args, _ = parser.parse_known_args()

parser.add_argument("--config", default="focal_10k_config.yml")
parser.add_argument("--sensitivity", type=float, default=95.0,
                     help="Target sensitivity pct (e.g. 95 or 97.5). Threshold is the "
                          "(100-sensitivity)th percentile of each species' own training scores.")
parser.add_argument("--out", default=None, help="Output PNG path (default: combined_distribution.png)")
parser.add_argument("--focal_taxa", default=None,
                     help="Comma-separated taxon_ids to plot instead of the classic 7 named species.")
args, _ = parser.parse_known_args()

with open(args.config) as f:
    config = yaml.safe_load(f)

raster = np.load(config["bioclim_data"]).astype(np.float32)
encoder = CoordEncoder(config["inputs"]["loc_feat_encoding"], raster)
model = tf.keras.models.load_model(
    args.model,
    custom_objects={"FocalScatterAdd": FocalScatterAdd, "IndependentPerClassHead": IndependentPerClassHead},
)

if args.train_npy:
    unique_taxa = np.load(f"{args.train_npy}_unique_taxa.npy")
    _locs = np.load(f"{args.train_npy}_locs.npy")
    _cids = np.load(f"{args.train_npy}_class_ids.npy")
    # build a lightweight dataframe-like structure for presence point lookups
    _taxon_ids = unique_taxa[_cids]
    df = pd.DataFrame({"longitude": _locs[:, 0], "latitude": _locs[:, 1], "taxon_id": _taxon_ids})
else:
    df = pd.read_csv(args.train_csv, usecols=["longitude","latitude","taxon_id"])
    unique_taxa = np.sort(df["taxon_id"].unique())

if args.focal_taxa:
    all_focal = [(str(tid), int(tid)) for tid in args.focal_taxa.split(",")]
else:
    all_focal = [
        ("Pika",       43188),
        ("Millipede",  47737),
        ("Lily",       51315),
        ("Crab",       54952),
        ("Lewisia",    47360),
        ("Amorph",     57358),
        ("Ant",       472214),
    ]
# only keep focal species present in this dataset
focal = [(name, tid) for name, tid in all_focal if tid in unique_taxa]
if len(focal) < len(all_focal):
    missing = [name for name, tid in all_focal if tid not in unique_taxa]
    print(f"Note: {missing} not in dataset, skipping")

class_indices = {tid: int(np.where(unique_taxa==tid)[0][0]) for _, tid in focal}

width, height = 2160, 1080
lons = np.linspace(-180, 180, width, dtype=np.float32)
lats = np.linspace(90, -90, height, dtype=np.float32)
lon_grid, lat_grid = np.meshgrid(lons, lats)
locs = np.stack([lon_grid.ravel(), lat_grid.ravel()], axis=1)
encoded = encoder.encode(locs)

focal_indices = list(class_indices.values())
all_preds = []
for start in range(0, len(encoded), 4096):
    batch = model(encoded[start:start+4096, :-1], training=False).numpy()
    all_preds.append(batch[:, focal_indices])
all_preds = np.concatenate(all_preds)  # shape: (pixels, 7)
# remap cidx to position in focal_indices
cidx_to_col = {cidx: col for col, cidx in enumerate(focal_indices)}

PERCENTILE = 100.0 - args.sensitivity
thresholds = {}
for name, tid in focal:
    cidx = class_indices[tid]
    presence = df[df["taxon_id"]==tid][["longitude","latitude"]].values.astype(np.float32)
    enc = encoder.encode(presence)
    scores = []
    for start in range(0, len(enc), 4096):
        scores.append(model(enc[start:start+4096, :-1], training=False).numpy()[:, cidx])
    scores = np.concatenate(scores)
    p = float(np.percentile(scores, PERCENTILE))
    thresholds[tid] = p
    print(f"{name}: mean={scores.mean():.3f}  p{PERCENTILE}={p:.3f}")

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    world = gpd.read_file(gpd.datasets.get_path("naturalearth_lowres"))

cmap_binary = mcolors.ListedColormap(["#f0f0f0", "#d73027"])
n = len(focal)
fig, axes = plt.subplots(n, 3, figsize=(36, 6*n))

for row, (name, tid) in enumerate(focal):
    cidx = class_indices[tid]
    p = thresholds[tid]
    grid = all_preds[:, cidx_to_col[cidx]].reshape(height, width)
    binary = (grid >= p).astype(np.uint8)
    obs = df[df["taxon_id"] == tid]

    ax = axes[row, 0]
    im = ax.imshow(grid, extent=[-180,180,-90,90], origin="upper",
                   cmap="YlOrRd", aspect="equal", vmin=0, vmax=np.percentile(grid, 99.99))
    world.boundary.plot(ax=ax, linewidth=0.4, color="black")
    plt.colorbar(im, ax=ax, label="Intensity")
    ax.set_title(f"{name} — continuous", fontsize=12)
    ax.set_xlim(-180,180); ax.set_ylim(-90,90)

    ax = axes[row, 1]
    ax.imshow(binary, extent=[-180,180,-90,90], origin="upper",
              cmap=cmap_binary, aspect="equal", vmin=0, vmax=1)
    world.boundary.plot(ax=ax, linewidth=0.4, color="black")
    ax.set_title(f"{name} — {100-PERCENTILE}% sensitivity ({p:.3f})", fontsize=12)
    ax.set_xlim(-180,180); ax.set_ylim(-90,90)
    patches = [plt.Rectangle((0,0),1,1,color="#f0f0f0",ec="grey",label="Absent"),
               plt.Rectangle((0,0),1,1,color="#d73027",label="Present")]
    ax.legend(handles=patches, loc="lower left", fontsize=9)

    ax = axes[row, 2]
    # zoom extent from presence points with 10% padding
    lon_min, lon_max = obs["longitude"].min(), obs["longitude"].max()
    lat_min, lat_max = obs["latitude"].min(), obs["latitude"].max()
    pad_lon = max((lon_max - lon_min) * 0.1, 2)
    pad_lat = max((lat_max - lat_min) * 0.1, 2)
    x0, x1 = max(lon_min - pad_lon, -180), min(lon_max + pad_lon, 180)
    y0, y1 = max(lat_min - pad_lat, -90), min(lat_max + pad_lat, 90)
    ax.imshow(binary, extent=[-180,180,-90,90], origin="upper",
              cmap=cmap_binary, aspect="equal", vmin=0, vmax=1)
    world.boundary.plot(ax=ax, linewidth=0.4, color="black")
    ax.scatter(obs["longitude"], obs["latitude"],
               s=1, c="#1565c0", alpha=0.6, linewidths=0)
    ax.set_title(f"{name} — presences on threshold (n={len(obs):,})", fontsize=12)
    ax.set_xlim(x0, x1); ax.set_ylim(y0, y1)
    ax.set_aspect("equal")

plt.tight_layout()
out_path = args.out or "combined_distribution.png"
plt.savefig(out_path, dpi=200)
print(f"saved {out_path}")
