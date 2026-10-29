"""Synthetic per-step timing of make_train_step at increasing class counts, to extrapolate the cost
of a full-scale (~120k-176k species) run without needing the full dataset.

Uses random inputs with v42's shapes (55 input features, hidden_dim 512, batch 512 positives +
512 background) and synthetic effort / range-penalty tables, so it exercises exactly the same
compiled step as a real run. Prints ms/step for each configuration; anything that runs out of
GPU memory is reported and skipped.

  python cloud/bench_scale.py                # default sweep
  python cloud/bench_scale.py --jit          # same sweep with XLA
"""
import argparse
import os
import sys
import time

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import numpy as np
import tensorflow as tf

from lib.models.geo_model_net import make_geo_model_net_independent_head, IndependentPerClassHead
from train_tf_sinr_test import make_train_step

N_FEATS = 56          # encoded width; the model sees N_FEATS - 1, as in training
EFFORT_COL = 14 + 22  # sinusoidal_hd coord feats + effort raster channel
N_CELLS = 288_122     # H3 res-4 cells worldwide


def bench(num_classes, n_focal, batch, hidden, head_dim, jit, steps, species_per_cell):
    rng = np.random.default_rng(0)
    focal = sorted(rng.choice(num_classes, n_focal, replace=False).tolist())
    model = make_geo_model_net_independent_head(num_classes, focal, N_FEATS - 1, hidden, head_dim)
    logits_model = tf.keras.Model(model.input, model.get_layer("predictions").input)
    head = next(l for l in model.layers if isinstance(l, IndependentPerClassHead))
    refs = {id(w) for w in head.trainable_weights}

    # synthetic effort table: ~species_per_cell classes present in each of 40k occupied cells
    occupied = rng.choice(N_CELLS, 40_000, replace=False)
    lengths = np.zeros(N_CELLS, np.int64)
    lengths[occupied] = rng.poisson(species_per_cell, len(occupied))
    values = rng.integers(0, num_classes, lengths.sum()).astype(np.int32)
    cell_species = tf.RaggedTensor.from_row_lengths(values, lengths)
    cell_range_penalty = tf.constant(rng.uniform(0, 15, (N_CELLS, n_focal)).astype(np.float16))

    step = make_train_step(
        tf.keras.optimizers.Adam(1e-4, clipnorm=1.0), logits_model, bg_base_weight=1.5,
        grad_clip_value=1.0, head_optimizer=tf.keras.optimizers.Adam(1e-4), head_weight_refs=refs,
        effort_col=EFFORT_COL, effort_scale=5.0, cell_species=cell_species, focal_cols=focal,
        cell_range_penalty=cell_range_penalty, jit_compile=jit, bg_batch_size=batch,
    )
    x = tf.random.normal((batch, N_FEATS))
    fake_x = tf.random.normal((batch, N_FEATS))
    ys = tf.constant(rng.integers(0, num_classes, batch))
    pw = tf.constant(np.full(num_classes, 2.25, np.float32))
    bg_cells = tf.constant(occupied[rng.integers(0, len(occupied), batch)].astype(np.int32))

    for _ in range(3):  # trace + warm up
        float(step(x, ys, fake_x, pw, bg_cells))
    t0 = time.perf_counter()
    for _ in range(steps):
        loss = step(x, ys, fake_x, pw, bg_cells)
    float(loss)
    ms = (time.perf_counter() - t0) / steps * 1000
    params = model.count_params() / 1e6
    return ms, params


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jit", action="store_true")
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--max_classes", type=int, default=None, help="skip larger configs (local smoke test)")
    args = ap.parse_args()

    print(f"GPUs: {tf.config.list_physical_devices('GPU')}  jit={args.jit}  batch={args.batch}")
    configs = [
        # (num_classes, n_focal, species_per_cell)
        (5_004, 500, 27),        # v42
        (50_000, 500, 100),
        (176_000, 500, 200),     # full species count, 500 independent heads
        (176_000, 5_000, 200),   # full species count, 5,000 independent heads
    ]
    print(f"{'classes':>9} {'focal':>6} {'params(M)':>10} {'ms/step':>9} {'h/epoch @230M obs':>18}")
    for num_classes, n_focal, spc in configs:
        if args.max_classes and num_classes > args.max_classes:
            continue
        try:
            ms, params = bench(num_classes, n_focal, args.batch, 512, 16, args.jit, args.steps, spc)
            hours = 230e6 / args.batch * ms / 1000 / 3600
            print(f"{num_classes:>9,} {n_focal:>6,} {params:>10.1f} {ms:>9.1f} {hours:>18.1f}", flush=True)
        except (tf.errors.ResourceExhaustedError, tf.errors.InternalError) as e:
            print(f"{num_classes:>9,} {n_focal:>6,} {'':>10} {'OOM/err':>9}  {type(e).__name__}", flush=True)
        tf.keras.backend.clear_session()


if __name__ == "__main__":
    main()
