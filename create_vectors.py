"""
Export thresholded model predictions as GeoJSON range maps for the 7 known focal species.

Builds on export_geojson.py's approach (dissolve H3-resolution cells above a per-species
presence-score threshold into a MultiPolygon, drop single-hex islands/holes as noise) but fixes
a stale bug there: the model must be called on encoded[:, :-1] (all but the last raster-encoded
column), not the full encoded width - see train_tf_sinr_test.py's EFFORT_COL comment for why the
last column can't be trusted as-is since v6 added coast_dist*bioclim interaction channels after
effort in the raster. export_geojson.py predates that fix and would mis-score with a v6+ model.

Usage:
  python3 create_vectors.py --model focal_10k_model_v29.keras --config focal_10k_config_v29.yml \\
      --train_npy train_10k_focal_clean --output_dir vector_exports
"""

import argparse
import os
import numpy as np
import tensorflow as tf
import yaml
import h3
import geopandas as gpd
from shapely.geometry import MultiPolygon, Polygon
from lib.models.encoders import CoordEncoder

ALL_FOCAL = [
    ("Pika",       43188),
    ("Millipede",  47737),
    ("Lily",       51315),
    ("Crab",       54952),
    ("Lewisia",    47360),
    ("Amorph",     57358),
    ("Ant",       472214),
]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="focal_10k_model_v29.keras")
    p.add_argument("--config", default="focal_10k_config_v29.yml")
    p.add_argument("--train_npy", default="train_10k_focal_clean",
                   help="Prefix used to compute each species' presence-score threshold - must "
                        "match whatever dataset --model was actually trained on.")
    p.add_argument("--percentile", type=float, default=5,
                   help="Presence-score percentile used as threshold (default: 5 = 95%% sensitivity)")
    p.add_argument("--h3_resolution", type=int, default=4)
    p.add_argument("--output_dir", default="vector_exports")
    p.add_argument("--geomodel_version", default=None,
                   help="Tag stored in output properties. Default: derived from --model's filename.")
    p.add_argument("--land_mask_channel", type=int, default=21,
                   help="Channel of config's bioclim_data holding the 0/1 land mask (21 in "
                        "bioclim_coast_effort_landmask[_interact]_scaled.npy - verified: 1 exactly where "
                        "the bioclim channels are non-NaN). -1 disables land clipping entirely.")
    p.add_argument("--no_land_clip_taxa", default="54952",
                   help="Comma-separated taxon_ids exempt from land clipping - marine/intertidal "
                        "species whose real range includes water (default: Crab, 54952).")
    return p.parse_args()


def dissolve_to_multipolygon(h3_cells):
    """H3 cells -> cleaned MultiPolygon coordinate structure (drops single-hex islands/holes)."""
    multi_poly = h3.h3_set_to_multi_polygon(h3_cells, geo_json=True)
    cleaned = []
    for polygon in multi_poly:
        outer, *holes = polygon
        if len(outer) <= 7:
            continue
        holes = [h for h in holes if len(h) > 7]
        cleaned.append([outer] + holes)
    return cleaned, len(multi_poly) - len(cleaned)


def coords_to_shapely(multi_poly_coords):
    # h3_set_to_multi_polygon can emit self-intersecting rings at finer resolutions (seen at res 5,
    # never at res 4) - buffer(0) repairs them without moving any real edges
    geom = MultiPolygon([Polygon(poly[0], poly[1:]) for poly in multi_poly_coords])
    if not geom.is_valid:
        geom = geom.buffer(0)
        if geom.geom_type == "Polygon":
            geom = MultiPolygon([geom])
    return geom


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    geomodel_version = args.geomodel_version or os.path.basename(args.model).replace(".keras", "")

    with open(args.config) as f:
        config = yaml.safe_load(f)

    print("Loading raster and encoder...")
    raster = np.load(config["bioclim_data"]).astype(np.float32)
    encoder = CoordEncoder(config["inputs"]["loc_feat_encoding"], raster)

    print(f"Loading model: {args.model}")
    model = tf.keras.models.load_model(args.model)

    unique_taxa = np.load(f"{args.train_npy}_unique_taxa.npy")
    train_locs = np.load(f"{args.train_npy}_locs.npy")
    train_class_ids = np.load(f"{args.train_npy}_class_ids.npy")

    focal = [(name, tid) for name, tid in ALL_FOCAL if tid in unique_taxa]
    missing = [name for name, tid in ALL_FOCAL if tid not in unique_taxa]
    if missing:
        print(f"Note: {missing} not in {args.train_npy}, skipping")
    class_indices = {tid: int(np.where(unique_taxa == tid)[0][0]) for _, tid in focal}
    focal_indices = list(class_indices.values())

    width, height = 2160, 1080
    lons = np.linspace(-180, 180, width, dtype=np.float32)
    lats = np.linspace(90, -90, height, dtype=np.float32)
    half_lon = float(lons[1] - lons[0]) / 2
    half_lat = float(lats[0] - lats[1]) / 2
    lon_grid, lat_grid = np.meshgrid(lons, lats)
    grid_locs = np.stack([lon_grid.ravel(), lat_grid.ravel()], axis=1)

    # Ocean pixels get NaN->0 bioclim values in the encoder, and near-shore ones blend real land
    # climate with those zeros, so a terrestrial species' prediction bleeds up to ~0.75 deg
    # offshore (measured on v42: Pika 3%, Lewisia 4%, Millipede 5% of above-threshold pixels).
    # Clip terrestrial species to land; exempt marine/intertidal ones via --no_land_clip_taxa.
    land = None
    if args.land_mask_channel >= 0:
        mask = raster[:, :, args.land_mask_channel]
        ri = np.clip(((90 - lats) / 180 * mask.shape[0]).astype(int), 0, mask.shape[0] - 1)
        ci = np.clip(((lons + 180) / 360 * mask.shape[1]).astype(int), 0, mask.shape[1] - 1)
        land = mask[ri[:, None], ci[None, :]] > 0.5
    no_clip = {int(t) for t in args.no_land_clip_taxa.split(",") if t.strip()}

    print("Encoding global grid...")
    encoded = encoder.encode(grid_locs)

    print("Running model inference on global grid...")
    all_preds = []
    for start in range(0, len(encoded), 4096):
        batch = model(encoded[start:start + 4096, :-1], training=False).numpy()
        all_preds.append(batch[:, focal_indices])
    all_preds = np.concatenate(all_preds)

    print("Computing presence-point scores for threshold...")
    all_train_encoded = []
    for start in range(0, len(train_locs), 4096):
        batch_enc = encoder.encode(train_locs[start:start + 4096].astype(np.float32))
        all_train_encoded.append(batch_enc)
    all_train_encoded = np.concatenate(all_train_encoded)

    presence_scores = {}
    for name, tid in focal:
        cidx = class_indices[tid]
        mask = (unique_taxa[train_class_ids] == tid)
        enc = all_train_encoded[mask]
        scores = []
        for start in range(0, len(enc), 4096):
            scores.append(model(enc[start:start + 4096, :-1], training=False).numpy()[:, cidx])
        presence_scores[tid] = np.concatenate(scores)

    print(f"\nExporting GeoJSON at H3 resolution {args.h3_resolution}, "
          f"P{args.percentile} threshold, to {args.output_dir}/\n")

    for col, (name, tid) in enumerate(focal):
        scores = presence_scores[tid]
        threshold = float(np.percentile(scores, args.percentile))
        print(f"{name} ({tid}): threshold={threshold:.4f}, n_presence={len(scores)}")

        grid = all_preds[:, col].reshape(height, width)
        present = grid >= threshold
        land_clipped = land is not None and tid not in no_clip
        if land_clipped:
            n_ocean = int((present & ~land).sum())
            present &= land
            print(f"  land clip: dropped {n_ocean} ocean pixels")
        rows, cols = np.where(present)

        h3_cells = set()
        for r, c in zip(rows.tolist(), cols.tolist()):
            lat, lon = float(lats[r]), float(lons[c])
            # the cell containing the pixel's own center point (correct at any resolution)...
            h3_cells.add(h3.geo_to_h3(lat, lon, args.h3_resolution))
            # ...PLUS every cell whose centroid falls inside the pixel's footprint. Needed once
            # h3_resolution cells are smaller than a grid pixel (e.g. res 6 hexes are ~36 km^2,
            # a grid pixel here is ~344 km^2 at the equator) - otherwise only one of several
            # hexes under a "present" pixel gets marked, leaving the dissolved polygon full of
            # holes. No-op in practice at coarse resolutions where hexes are bigger than a pixel.
            pixel_poly = {
                "type": "Polygon",
                "coordinates": [[
                    [lon - half_lon, lat - half_lat], [lon + half_lon, lat - half_lat],
                    [lon + half_lon, lat + half_lat], [lon - half_lon, lat + half_lat],
                    [lon - half_lon, lat - half_lat],
                ]],
            }
            h3_cells.update(h3.polyfill_geojson(pixel_poly, args.h3_resolution))
        print(f"  {len(h3_cells):,} H3 cells above threshold")

        multi_poly_coords, n_removed = dissolve_to_multipolygon(h3_cells)
        print(f"  dissolved into {len(multi_poly_coords)} polygon(s) ({n_removed} singletons/holes removed)")

        geometry = coords_to_shapely(multi_poly_coords)
        properties = {
            "taxon_id": tid,
            "name": name,
            "geomodel_version": geomodel_version,
            "threshold": threshold,
            "percentile": args.percentile,
            "h3_resolution": args.h3_resolution,
            "land_clipped": land_clipped,
        }

        gdf = gpd.GeoDataFrame([properties], geometry=[geometry], crs="EPSG:4326")
        path = os.path.join(args.output_dir, f"{tid}.geojson")
        gdf.to_file(path, driver="GeoJSON")
        print(f"  saved {path}\n")

    print("Done.")


if __name__ == "__main__":
    main()
