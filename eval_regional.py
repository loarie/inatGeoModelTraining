"""
Score a regional model (make_regional_subset.py) on held-out data, per species:
  auc          held-out presences vs absence cells (well-sampled H3 cells with no record of the
               species anywhere in the source data) - ranking, independent of any threshold
  sens         share of held-out presences at or above the species' threshold (5th percentile of
               its training-observation scores, as in the range maps)
  spec         share of absence cells below that threshold
  local_*      the same against only the absence cells within --local_km of the species' training
               observations - the hard ones (Central Valley next to the foothills), which the far
               ones (deserts) swamp in the plain scores
  wide_*       the same within --wide_km: look-alike habitat a little further off (the Sierra
               foothills for C. amabilis) that --local_km misses

  python eval_regional.py --model ca/ca_base.epoch003.keras --config ca/config_ca_base.yml \
      --data train_ca_v1 --src train_full_v1 --bbox=-124.5,-114,32.5,42 --taxa train_ca_v1_calochortus.npy
"""
import argparse
import os
import pickle
from collections import defaultdict

import h3
import numpy as np
import tensorflow as tf
import yaml
from sklearn.neighbors import BallTree

from check_population import make_focal_scorer
from lib.models.encoders import CoordEncoder
from lib.models.geo_model_net import FocalScatterAdd, IndependentPerClassHead, ZeroInputColumns


def absence_cells(src, bbox, res, min_obs, cache):
    """{cell: set of taxon ids} and cell centres for well-sampled cells in the box (source data)."""
    if os.path.exists(cache):
        with open(cache, "rb") as f:
            return pickle.load(f)
    x0, x1, y0, y1 = bbox
    locs = np.load(f"{src}_locs.npy", mmap_mode="r")
    cids = np.load(f"{src}_class_ids.npy", mmap_mode="r")
    taxa = np.load(f"{src}_unique_taxa.npy")
    lon, lat = np.asarray(locs[:, 0]), np.asarray(locs[:, 1])
    rows = np.flatnonzero((lon >= x0) & (lon <= x1) & (lat >= y0) & (lat <= y1))
    sp = defaultdict(set)
    n = defaultdict(int)
    for (lo, la), c in zip(np.asarray(locs[rows]), np.asarray(cids[rows])):
        cell = h3.geo_to_h3(float(la), float(lo), res)
        sp[cell].add(int(taxa[c]))
        n[cell] += 1
    cells = [c for c in n if n[c] >= min_obs]
    centres = np.array([h3.h3_to_geo(c)[::-1] for c in cells], np.float32)
    out = ([sp[c] for c in cells], centres)
    with open(cache, "wb") as f:
        pickle.dump(out, f)
    return out


def auc(pos, neg):
    """P(score of a random presence > score of a random absence), ties count half."""
    allv = np.concatenate([pos, neg])
    ranks = allv.argsort().argsort() + 1.0
    return (ranks[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--data", default="train_ca_v1")
    ap.add_argument("--src", default="train_full_v1")
    ap.add_argument("--bbox", default="-124.5,-114,32.5,42")
    ap.add_argument("--taxa", required=True, help="npy of taxon ids to score")
    ap.add_argument("--res", type=int, default=5)
    ap.add_argument("--min_obs", type=int, default=500, help="observations for a cell to count as well sampled")
    ap.add_argument("--min_test", type=int, default=10, help="skip species with fewer held-out observations")
    ap.add_argument("--local_km", type=float, default=100)
    ap.add_argument("--wide_km", type=float, default=250)
    ap.add_argument("--out_csv", default=None)
    args = ap.parse_args()
    bbox = [float(v) for v in args.bbox.split(",")]

    cfg = yaml.safe_load(open(args.config))
    encoder = CoordEncoder(cfg["inputs"]["loc_feat_encoding"], np.load(cfg["bioclim_data"]).astype(np.float32))
    model = tf.keras.models.load_model(args.model, custom_objects={
        "FocalScatterAdd": FocalScatterAdd, "IndependentPerClassHead": IndependentPerClassHead,
        "ZeroInputColumns": ZeroInputColumns})
    ut = np.load(f"{args.data}_unique_taxa.npy")
    pos_of = {int(t): i for i, t in enumerate(ut)}
    taxa = [int(t) for t in np.load(args.taxa) if int(t) in pos_of]
    cls = [pos_of[t] for t in taxa]
    score = make_focal_scorer(model, cls)

    def scores(xy):
        x = encoder.encode(xy.astype(np.float32))[:, :-1]
        return np.concatenate([score(x[s:s + 8192]) for s in range(0, len(x), 8192)]) if len(x) else np.zeros((0, len(cls)))

    cell_sp, centres = absence_cells(args.src, bbox, args.res, args.min_obs,
                                     f"h3r{args.res}_cells_{args.data}_min{args.min_obs}.pkl")
    cell_scores = scores(centres)
    tr_locs, tr_c = np.load(f"{args.data}_locs.npy"), np.load(f"{args.data}_class_ids.npy")
    te_locs, te_c = np.load(f"{args.data}_test_locs.npy"), np.load(f"{args.data}_test_class_ids.npy")
    print(f"{len(centres):,} well-sampled cells (>= {args.min_obs} obs, H3-{args.res}); {len(taxa)} species")

    rows = []
    for j, (t, c) in enumerate(zip(taxa, cls)):
        te = te_locs[te_c == c]
        if len(te) < args.min_test:
            continue
        absent = np.array([t not in s for s in cell_sp])
        tr = tr_locs[tr_c == c]
        thr = np.percentile(scores(tr)[:, j], 5)
        p = scores(te)[:, j]
        a = cell_scores[absent, j]
        tree = BallTree(np.radians(tr[:, ::-1]), metric="haversine")
        km = tree.query(np.radians(centres[absent][:, ::-1]))[0][:, 0] * 6371
        near = km <= args.local_km
        la, wa = a[near], a[km <= args.wide_km]
        rows.append((t, len(te), int(near.sum()), auc(p, a), (p >= thr).mean(), (a < thr).mean(),
                     auc(p, la) if len(la) else np.nan, (la < thr).mean() if len(la) else np.nan,
                     auc(p, wa) if len(wa) else np.nan, (wa < thr).mean() if len(wa) else np.nan))
    r = np.array([x[3:] for x in rows], np.float64)
    m = np.nanmean(r, 0)
    print(f"scored {len(rows)} species: auc {m[0]:.3f}  sens {m[1]:.3f}  spec {m[2]:.3f}  |  "
          f"local auc {m[3]:.3f}  local spec {m[4]:.3f}  (within {args.local_km:g} km)  |  "
          f"wide auc {m[5]:.3f}  wide spec {m[6]:.3f}  (within {args.wide_km:g} km)")
    if args.out_csv:
        with open(args.out_csv, "w") as f:
            f.write("taxon_id,n_test,n_local_absent,auc,sens,spec,local_auc,local_spec,wide_auc,wide_spec\n")
            for row in rows:
                f.write(",".join(str(v) if i < 3 else f"{v:.4f}" for i, v in enumerate(row)) + "\n")


if __name__ == "__main__":
    main()
