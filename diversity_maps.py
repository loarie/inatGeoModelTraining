"""
Stacked-range diversity maps by iNaturalist iconic taxon: threshold every species' prediction at
its 95%-sensitivity point (5th percentile of the model's scores at its own training observations,
the same threshold as visualize_combined.py / check_population.py) and count, per grid cell, how
many species' ranges cover it.

Only the model's backbone runs per point; each class's logits are h @ kernel[:, c] + bias[c] (+ the
focal head for the focal species), compared in logit space, so all 120k classes fit on a laptop.

  python diversity_maps.py --model v47/focal_full_model_v47.epoch003.keras --config v47/config_v47.yml \
      --train_npy train_full_v1 --iconic_csv diversity/iconic_train_full_v1.csv --out_dir diversity/v47_e003
"""
import argparse
import os

import numpy as np
import pandas as pd
import tensorflow as tf
import yaml

from lib.models.encoders import CoordEncoder
from lib.models.geo_model_net import FocalScatterAdd, IndependentPerClassHead

# iconic_taxon_name -> plot label; anything else (or missing) is grouped as "Other"
ICONIC_LABELS = {
    "Plantae": "Plants", "Aves": "Birds", "Insecta": "Insects", "Mammalia": "Mammals",
    "Reptilia": "Reptiles", "Amphibia": "Amphibians", "Actinopterygii": "Ray-finned fishes",
    "Mollusca": "Mollusks", "Arachnida": "Arachnids", "Fungi": "Fungi",
    "Animalia": "Other animals", "Chromista": "Chromista (kelps, diatoms & allies)",
    "Protozoa": "Protozoa",
}


def split_model(model):
    """(features model -> [h, head input], kernel, bias, focal class ids in head order, head).
    The head is scored per class (IndependentPerClassHead.score_rows/score_classes), never in
    full: a head over all 120k classes is (rows, 120k, dim)."""
    scatter = next((l for l in model.layers if isinstance(l, FocalScatterAdd)), None)
    out_dense = [l for l in model.layers if isinstance(l, tf.keras.layers.Dense)
                 and l.units == model.output.shape[-1]]
    shared = out_dense[0]
    bias = shared.bias if shared.use_bias else tf.zeros(shared.units)
    if scatter is None:
        return tf.keras.Model(model.input, shared.input), shared.kernel, bias, None, None
    head = next(l for l in model.layers if isinstance(l, IndependentPerClassHead) and l.output is scatter.input[1])
    features = tf.keras.Model(model.input, [shared.input, head.input])
    return features, shared.kernel, bias, np.array(scatter.indices), head


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--train_npy", required=True)
    ap.add_argument("--iconic_csv", required=True, help="taxon_id,name,rank,iconic_taxon_name")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--sensitivity", type=float, default=95.0)
    ap.add_argument("--max_pts", type=int, default=1000,
                    help="observations per class used for its threshold (random sample if more)")
    ap.add_argument("--res_deg", type=float, default=1 / 6)
    ap.add_argument("--thresholds_only", action="store_true")
    ap.add_argument("--ranks", default="species", help="comma-separated ranks to stack")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    with open(args.config) as f:
        config = yaml.safe_load(f)
    raster = np.load(config["bioclim_data"]).astype(np.float32)
    encoder = CoordEncoder(config["inputs"]["loc_feat_encoding"], raster)
    model = tf.keras.models.load_model(
        args.model, custom_objects={"FocalScatterAdd": FocalScatterAdd, "IndependentPerClassHead": IndependentPerClassHead})
    features, kernel, bias, focal_ids, head = split_model(model)
    num_classes = kernel.shape[1]
    focal_pos = {int(c): i for i, c in enumerate(focal_ids)} if focal_ids is not None else {}
    head_col = np.full(num_classes, -1, np.int64)  # class -> head column, -1 if not in the head
    if focal_ids is not None:
        head_col[focal_ids] = np.arange(len(focal_ids))

    # ---- per-class thresholds (logit space) from a sample of each class's own observations
    thr_path = os.path.join(args.out_dir, "thresholds_logit.npy")
    if os.path.exists(thr_path):
        thr = np.load(thr_path)
        print("loaded thresholds")
    else:
        cids = np.load(f"{args.train_npy}_class_ids.npy")
        locs = np.load(f"{args.train_npy}_locs.npy", mmap_mode="r")
        rng = np.random.default_rng(0)
        order = np.argsort(cids, kind="stable")
        starts = np.searchsorted(cids[order], np.arange(num_classes + 1))
        sel = []
        for c in range(num_classes):
            idx = order[starts[c]:starts[c + 1]]
            if len(idx) > args.max_pts:
                idx = rng.choice(idx, args.max_pts, replace=False)
            sel.append(np.sort(idx))
        sel = np.concatenate(sel)
        del order
        sel_cls = cids[sel]
        del cids
        print(f"threshold sample: {len(sel):,} points", flush=True)
        scores = np.empty(len(sel), np.float32)
        B = 65536
        for s in range(0, len(sel), B):
            x = encoder.encode(np.asarray(locs[sel[s:s + B]]))[:, :-1]
            out = features(x, training=False)
            h, f = out if focal_ids is not None else (out, None)
            c = sel_cls[s:s + B]
            lg = tf.reduce_sum(h * tf.transpose(tf.gather(kernel, c, axis=1)), axis=1) + tf.gather(bias, c)
            lg = lg.numpy()
            if f is not None:
                fp = head_col[c]
                m = fp >= 0
                if m.any():
                    lg[m] += head.score_rows(tf.boolean_mask(f, m), fp[m]).numpy()
            scores[s:s + B] = lg
            if (s // B) % 50 == 0:
                print(f"  {s:,}/{len(sel):,}", flush=True)
        thr = np.full(num_classes, np.inf, np.float32)
        bounds = np.searchsorted(sel_cls, np.arange(num_classes + 1))
        for c in range(num_classes):
            v = scores[bounds[c]:bounds[c + 1]]
            if len(v):
                thr[c] = np.percentile(v, 100.0 - args.sensitivity)
        np.save(thr_path, thr)

    unique_taxa = np.load(f"{args.train_npy}_unique_taxa.npy")
    tax = pd.read_csv(args.iconic_csv).drop_duplicates("taxon_id").set_index("taxon_id")
    tax = tax.reindex(unique_taxa)
    ranks = set(args.ranks.split(","))
    keep = tax["rank"].isin(ranks).to_numpy()
    group = tax["iconic_taxon_name"].map(ICONIC_LABELS).fillna("Other").to_numpy()
    print(f"{num_classes} classes; {keep.sum()} of rank {sorted(ranks)}; "
          f"{tax['rank'].isna().sum()} without taxonomy")

    if args.thresholds_only:
        return

    # ---- stack thresholded ranges per group over a lat/lon grid
    labels = sorted(set(group[keep]))
    cls_by_group = {g: np.flatnonzero(keep & (group == g)) for g in labels}
    order = np.concatenate([cls_by_group[g] for g in labels])
    seg = np.cumsum([0] + [len(cls_by_group[g]) for g in labels])
    k_ord = tf.constant(tf.gather(kernel, order, axis=1))
    b_ord = tf.constant(tf.gather(bias, order) - thr[order])  # logit - threshold >= 0  <=> present
    # focal classes among the kept ones: add their head output column
    fcols = [(j, focal_pos[int(c)]) for j, c in enumerate(order) if int(c) in focal_pos]
    fcol_j = np.array([j for j, _ in fcols], np.int64)
    fcol_h = np.array([h for _, h in fcols], np.int64)

    if len(fcol_h):  # this head's columns only, gathered once
        hw1, hb1 = tf.gather(head.W1, fcol_h), tf.gather(head.b1, fcol_h)[tf.newaxis]
        hw2, hb2 = tf.gather(head.W2, fcol_h), tf.gather(head.b2, fcol_h)[tf.newaxis]

    w, hgt = int(round(360 / args.res_deg)), int(round(180 / args.res_deg))
    lons = -180 + (np.arange(w) + 0.5) * args.res_deg
    lats = 90 - (np.arange(hgt) + 0.5) * args.res_deg
    counts = np.zeros((len(labels), hgt * w), np.int32)
    CH = 1024 if len(fcol_h) <= 5000 else 64  # the head's hidden layer is (CH, len(fcol_h), dim)
    grid_lon, grid_lat = np.meshgrid(lons, lats)
    grid = np.stack([grid_lon.ravel(), grid_lat.ravel()], 1).astype(np.float32)
    seg_ids = np.repeat(np.arange(len(labels)), np.diff(seg))
    onehot = tf.constant(np.eye(len(labels), dtype=np.float32)[seg_ids])  # (K_kept, G)
    print(f"grid {w}x{hgt}, {len(order):,} ranges in {len(labels)} groups", flush=True)
    for s in range(0, len(grid), CH):
        x = encoder.encode(grid[s:s + CH])[:, :-1]
        out = features(x, training=False)
        h, f = out if focal_ids is not None else (out, None)
        lg = tf.matmul(h, k_ord) + b_ord
        if f is not None and len(fcol_j):
            hid = tf.nn.relu(tf.einsum("bi,nih->bnh", f, hw1) + hb1)
            add = tf.einsum("bnh,nh->bn", hid, hw2) + hb2
            lg = tf.transpose(tf.tensor_scatter_nd_add(tf.transpose(lg), fcol_j[:, None], tf.transpose(add)))
        pres = tf.cast(lg >= 0, tf.float32)
        counts[:, s:s + CH] = tf.transpose(tf.matmul(pres, onehot)).numpy().astype(np.int32)
        if (s // CH) % 200 == 0:
            print(f"  grid {s:,}/{len(grid):,}", flush=True)
    counts = counts.reshape(len(labels), hgt, w)
    np.savez_compressed(os.path.join(args.out_dir, "counts.npz"), counts=counts, labels=np.array(labels),
                        n_ranges=np.diff(seg), lons=lons, lats=lats)
    for g, n in zip(labels, np.diff(seg)):
        print(f"{g}: {n:,} ranges")


if __name__ == "__main__":
    main()
