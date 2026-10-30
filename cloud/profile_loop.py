"""Where does real-run step time go beyond the compiled step itself? Times the same v42-shaped
compiled step (synthetic tables, as in bench_scale.py) three ways:
  1. no per-step host sync (what bench_scale.py measures)
  2. host sync every step via loss.numpy() (what the training loop does)
  3. indices pulled from a tf.data shuffle/batch pipeline + per-step sync (the old real loop)
  4. blocks of --steps_per_call batches through make_multi_step, one sync per block (current loop)
  python cloud/profile_loop.py [--jit]
"""
import argparse
import os
import sys
import time

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import numpy as np
import tensorflow as tf

from lib.models.encoders import CoordEncoder
from lib.models.geo_model_net import make_geo_model_net_independent_head, IndependentPerClassHead
from train_tf_sinr_test import make_train_step, make_multi_step, build_cell_presence_bits
from utils import make_rand_samples_tf

ap = argparse.ArgumentParser()
ap.add_argument("--jit", action="store_true")
ap.add_argument("--steps", type=int, default=300)
ap.add_argument("--steps_per_call", type=int, default=50)
ap.add_argument("--num_classes", type=int, default=5_004)
ap.add_argument("--species_per_cell", type=float, default=27, help="v42 real data: ~121 present classes per sampled bg row")
args = ap.parse_args()

K, F, B, N_CELLS, N_FEATS, N_POINTS = args.num_classes, 500, 512, 288_122, 56, 8_000_000
rng = np.random.default_rng(0)
focal = sorted(rng.choice(K, F, replace=False).tolist())
model = make_geo_model_net_independent_head(K, focal, N_FEATS - 1, 512, 16)
logits_model = tf.keras.Model(model.input, model.get_layer("predictions").input)
head = next(l for l in model.layers if isinstance(l, IndependentPerClassHead))
occupied = rng.choice(N_CELLS, 41_549, replace=False)
lengths = np.zeros(N_CELLS, np.int64)
lengths[occupied] = rng.poisson(args.species_per_cell, len(occupied))
cell_presence = build_cell_presence_bits(np.repeat(np.arange(N_CELLS), lengths), rng.integers(0, K, lengths.sum()), N_CELLS, K)
occupied_t = tf.constant(occupied.astype(np.int32))
train_locs = tf.constant(np.stack([rng.uniform(-180, 180, N_POINTS), rng.uniform(-90, 90, N_POINTS)], 1).astype(np.float32))
train_classes = tf.constant(rng.integers(0, K, N_POINTS))
encoder = CoordEncoder("sinusoidal_hd", rng.normal(size=(2160, 4320, N_FEATS - 14)).astype(np.float32))

def sample_background(batch_size):
    cells = tf.gather(occupied_t, tf.random.uniform([batch_size], 0, len(occupied), dtype=tf.int32))
    return make_rand_samples_tf(batch_size), cells

step = make_train_step(
    tf.keras.optimizers.Adam(1e-4, clipnorm=1.0), logits_model, encoder, train_locs, train_classes,
    sample_background, bg_base_weight=1.5, grad_clip_value=1.0,
    head_optimizer=tf.keras.optimizers.Adam(1e-4), head_weight_refs={id(w) for w in head.trainable_weights},
    effort_col=36, effort_scale=5.0, cell_presence=cell_presence, focal_cols=focal,
    cell_range_penalty=tf.constant(rng.uniform(0, 15, (N_CELLS, F)).astype(np.float16)),
    jit_compile=args.jit,
)
pw = tf.constant(np.full(K, 2.25, np.float32))
idx_fixed = tf.random.uniform([B], 0, N_POINTS, dtype=tf.int64)
ds = tf.data.Dataset.from_tensor_slices(rng.integers(0, N_POINTS, B * (args.steps + 10)).astype(np.int64))
ds = ds.shuffle(500_000).batch(B).prefetch(tf.data.AUTOTUNE)

run_steps = make_multi_step(step)
ds_blocks = tf.data.Dataset.from_tensor_slices(rng.integers(0, N_POINTS, B * (args.steps + 10)).astype(np.int64))
ds_blocks = (ds_blocks.shuffle(500_000).batch(B, drop_remainder=True).batch(args.steps_per_call)
             .prefetch(tf.data.AUTOTUNE))

for _ in range(5):
    float(step(idx_fixed, pw))
run_steps(next(iter(ds_blocks)), pw).numpy()  # trace

def timed(name, fn):
    t0 = time.perf_counter()
    fn()
    print(f"{name:45s} {(time.perf_counter() - t0) / args.steps * 1000:6.2f} ms/step", flush=True)

def no_sync():
    for _ in range(args.steps):
        loss = step(idx_fixed, pw)
    float(loss)

def sync_each():
    for _ in range(args.steps):
        step(idx_fixed, pw).numpy()

def real_loop():
    for i, idx in enumerate(ds):
        if i == args.steps:
            break
        step(idx, pw).numpy()

def block_loop():
    done = 0
    for block in ds_blocks:
        done += len(run_steps(block, pw).numpy())
        if done >= args.steps:
            break

def dataset_only():
    for i, idx in enumerate(ds):
        if i == args.steps:
            break

print(f"GPUs: {tf.config.list_physical_devices('GPU')}  jit={args.jit}")
timed("1. compiled step, no per-step sync", no_sync)
timed("2. compiled step, loss.numpy() every step", sync_each)
timed("3. tf.data indices + loss.numpy() every step", real_loop)
timed(f"4. make_multi_step, {args.steps_per_call} steps per call", block_loop)
timed("   tf.data iteration alone", dataset_only)
