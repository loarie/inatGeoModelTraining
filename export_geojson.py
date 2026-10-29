"""
Export thresholded model predictions as H3-resolution-4 GeoJSON files.

Produces one {taxon_id}.geojson per focal species, matching the format of
43188_sample.geojson: a single Feature with a dissolved MultiPolygon geometry
where each polygon is a group of adjacent H3-4 hexagons above the threshold.

Usage:
  python export_geojson.py                          # defaults
  python export_geojson.py --train_npy train_10k --model test_pika_model.keras --percentile 5
"""

import argparse
import json
import numpy as np
import tensorflow as tf
import yaml
import h3
from lib.models.encoders import CoordEncoder

parser = argparse.ArgumentParser()
parser.add_argument("--train_npy", default="train_10k")
parser.add_argument("--model", default="test_pika_model.keras")
parser.add_argument("--percentile", type=float, default=5,
                    help="Presence score percentile used as threshold (default: 5 = 95% sensitivity)")
parser.add_argument("--h3_resolution", type=int, default=4)
parser.add_argument("--geomodel_version", default="test_10k")
args = parser.parse_args()

with open("test_config.yml") as f:
    config = yaml.safe_load(f)

print("Loading raster and encoder...")
raster = np.load(config["bioclim_data"]).astype(np.float32)
encoder = CoordEncoder(config["inputs"]["loc_feat_encoding"], raster)

print(f"Loading model: {args.model}")
model = tf.keras.models.load_model(args.model)

unique_taxa = np.load(f"{args.train_npy}_unique_taxa.npy")
train_locs = np.load(f"{args.train_npy}_locs.npy")
train_class_ids = np.load(f"{args.train_npy}_class_ids.npy")

all_focal = [
    ("Pika",       43188),
    ("Millipede",  47737),
    ("Lily",       51315),
    ("Crab",       54952),
    ("Lewisia",    47360),
    ("Amorph",     57358),
    ("Ant",       472214),
]
focal = [(name, tid) for name, tid in all_focal if tid in unique_taxa]
missing = [name for name, tid in all_focal if tid not in unique_taxa]
if missing:
    print(f"Note: {missing} not in dataset, skipping")

class_indices = {tid: int(np.where(unique_taxa == tid)[0][0]) for _, tid in focal}
focal_indices = list(class_indices.values())

# Build global prediction grid
width, height = 2160, 1080
lons = np.linspace(-180, 180, width, dtype=np.float32)
lats = np.linspace(90, -90, height, dtype=np.float32)
lon_grid, lat_grid = np.meshgrid(lons, lats)
grid_locs = np.stack([lon_grid.ravel(), lat_grid.ravel()], axis=1)

print("Encoding global grid...")
encoded = encoder.encode(grid_locs)

print("Running model inference on global grid...")
all_preds = []
for start in range(0, len(encoded), 4096):
    batch = model(encoded[start:start+4096], training=False).numpy()
    all_preds.append(batch[:, focal_indices])
all_preds = np.concatenate(all_preds)  # (pixels, n_focal)

# Compute presence scores for all focal species in one pass
print("Computing presence-point scores for threshold...")
# encode all training locs once
all_train_encoded = []
for start in range(0, len(train_locs), 4096):
    batch_enc = encoder.encode(train_locs[start:start+4096].astype(np.float32))
    all_train_encoded.append(batch_enc)
all_train_encoded = np.concatenate(all_train_encoded)

presence_scores = {}  # tid -> array of scores at presence locations
for col, (name, tid) in enumerate(focal):
    cidx = class_indices[tid]
    mask = (unique_taxa[train_class_ids] == tid)
    enc = all_train_encoded[mask]
    scores = []
    for start in range(0, len(enc), 4096):
        scores.append(model(enc[start:start+4096], training=False).numpy()[:, cidx])
    presence_scores[tid] = np.concatenate(scores)

# Export one GeoJSON per species
print(f"\nExporting GeoJSON at H3 resolution {args.h3_resolution}, P{args.percentile} threshold...\n")
for col, (name, tid) in enumerate(focal):
    cidx = class_indices[tid]
    scores = presence_scores[tid]
    threshold = float(np.percentile(scores, args.percentile))
    print(f"{name} ({tid}): threshold={threshold:.4f}, n_presence={len(scores)}")

    grid = all_preds[:, col].reshape(height, width)
    rows, cols = np.where(grid >= threshold)

    h3_cells = set()
    for r, c in zip(rows.tolist(), cols.tolist()):
        h3_cells.add(h3.geo_to_h3(float(lats[r]), float(lons[c]), args.h3_resolution))

    print(f"  {len(h3_cells):,} H3 cells above threshold")

    # dissolve adjacent cells into MultiPolygon
    multi_poly = h3.h3_set_to_multi_polygon(h3_cells, geo_json=True)

    # remove singleton islands (outer ring == single hex == 7 coords)
    # and single-hex holes (inner rings with 7 coords)
    cleaned = []
    for polygon in multi_poly:
        outer, *holes = polygon
        if len(outer) <= 7:
            continue
        holes = [h for h in holes if len(h) > 7]
        cleaned.append([outer] + holes)
    n_removed = len(multi_poly) - len(cleaned)
    multi_poly = cleaned
    print(f"  dissolved into {len(multi_poly)} polygon(s) ({n_removed} singletons/holes removed)")

    feature = {
        "type": "Feature",
        "properties": {
            "taxon_id": tid,
            "geomodel_version": args.geomodel_version,
            "threshold": threshold,
            "percentile": args.percentile,
        },
        "geometry": {
            "type": "MultiPolygon",
            "coordinates": multi_poly,
        },
    }

    out_path = f"{tid}.geojson"
    with open(out_path, "w") as f:
        json.dump(feature, f)
    print(f"  saved {out_path}\n")

print("Done.")
