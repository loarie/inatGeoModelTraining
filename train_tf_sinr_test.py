import json
import math
import os
from pathlib import Path

import click
import h3
import numpy as np
import pandas as pd
import tensorflow as tf
import yaml
import tqdm
import wandb
from scipy.spatial import cKDTree

from lib.models.geo_model_net import (
    make_geo_model_net, make_geo_model_net_focal_head, make_geo_model_net_independent_head,
    FocalScatterAdd, IndependentPerClassHead,
)
from utils import make_rand_samples_tf
from lib.models.encoders import CoordEncoder
from sinr_loss import sinr_loss
from lib.data.data_loader import (
    make_subsampled_dataset,
    load_inat_dataset_from_parquet,
    load_inat_dataset_from_parquet_h3,
    load_sinr_dataset_from_parquet,
    load_sinr_dataset_from_parquet_h3,
)
from lib.data.data_cleaner import clean_dataset


def load_sinr_dataset_from_csv(file):
    print(f"loading sinr dataset from csv: {file}")
    df = pd.read_csv(file, usecols=["longitude", "latitude", "taxon_id"])
    df = clean_dataset(df)
    locs = np.vstack((df["longitude"].values, df["latitude"].values)).T.astype(np.float32)
    taxon_ids = df["taxon_id"].values.astype(int)
    unique_taxa, class_ids = np.unique(taxon_ids, return_inverse=True)
    print(f" found {len(unique_taxa)} unique taxa")
    return locs, class_ids, unique_taxa


def load_sinr_dataset_from_npy(prefix):
    """Load preprocessed numpy arrays written by preprocess_training_data.py."""
    locs = np.load(f"{prefix}_locs.npy")
    class_ids = np.load(f"{prefix}_class_ids.npy").astype(np.int64)
    unique_taxa = np.load(f"{prefix}_unique_taxa.npy")
    print(f"loaded npy dataset: {len(unique_taxa):,} taxa, {len(class_ids):,} obs from prefix '{prefix}'")
    return locs, class_ids, unique_taxa


def make_train_step(optimizer, model, encoder, train_locs, train_classes, sample_background,
                    bg_base_weight=1.0, grad_clip_value=None, head_optimizer=None,
                    head_weight_refs=None, effort_col=None, effort_scale=0.0, cell_presence=None,
                    focal_cols=None, cell_range_penalty=None, jit_compile=False):
    """Returns a compiled train_step(idx, pos_weight) -> loss, where idx is a batch of indices
    into the training points.

    `model` here must be the logits-output view (pre-sigmoid) so sinr_loss can compute a
    numerically-stable, non-vanishing gradient - see sinr_loss.py's docstring.

    Everything per-step happens on-device: training points are kept as raw (lon, lat) degrees
    (train_locs, with train_classes) and encoded per batch with encoder.encode_tf, instead of
    holding a full encoded copy of every point in host memory (~15GB at 8M points, ~400GB at
    230M). sample_background(batch_size) is a TF function returning (normalized [-1, 1]
    background locations, their H3 cell ids or None).

    The background weighting is computed on-device from per-run lookup tables instead of dense
    (B_bg, num_classes) matrices built in numpy every step (which cost ~85ms/step at 176k
    classes, plus ~150ms/step for 500 per-step KD-tree queries at v42 scale):
      cell_presence:      (cell_row, bits) from build_cell_presence_bits - which classes were
                          observed in each H3 cell, as one packed bit per class.
      cell_range_penalty: (n_cells, n_focal) range penalty per H3 cell centroid, focal_cols order.
    effort_scale > 0 with cell_presence enables effort weighting; cell_range_penalty enables the
    range penalty. All shapes are static, so the whole step can be XLA-compiled (jit_compile).

    head_optimizer/head_weight_refs: if given, IndependentPerClassHead's own W1/b1/W2/b2 get
    updated through a SEPARATE optimizer with no clipping at all, instead of through `optimizer`
    with grad_clip_value + clipnorm. Both clips were found (v38 postmortem) to badly cripple
    boosted classes: grad_clip_value clips each element to +-1 regardless of pos_weight (so
    boosting pos_weight past ~10 has literally zero effect on the actual step taken), and Adam's
    own clipnorm=1.0 then ALSO rescales the *entire* per-layer tensor (spanning all 500 classes
    at once) by its combined norm - so a few boosted classes dominate that norm and crush the
    effective gradient for every OTHER class sharing the same W1/b1/W2/b2 tensor, including ones
    that were already converging fine. Neither clip's justification applies to the independent
    head: it exists specifically to stop one class's gradient from dominating a SHARED parameter
    (see the comment below, added after v30), and IndependentPerClassHead has no shared
    parameters between classes by construction - there's nothing for one class to crowd out."""
    weights = model.trainable_weights
    head_weight_refs = head_weight_refs or set()
    head_vars = [w for w in weights if id(w) in head_weight_refs]
    other_vars = [w for w in weights if id(w) not in head_weight_refs]
    # build optimizer slots eagerly - a tf.function may only create variables on its first trace
    optimizer.build(other_vars)
    if head_optimizer is not None and head_vars:
        head_optimizer.build(head_vars)
    use_effort = effort_scale > 0 and cell_presence is not None
    if use_effort:
        cell_row, cell_bits = cell_presence
    focal_cols_t = tf.constant(focal_cols, dtype=tf.int32) if cell_range_penalty is not None else None
    bit_shifts = tf.constant(np.arange(8, dtype=np.uint8))

    # copy the raster to the device now - creating it lazily inside the trace would embed a
    # 1.5GB+ constant in the graph
    encoder.encode_tf(tf.zeros([1, 2]))

    @tf.function(reduce_retracing=True)
    def train_step(idx, pos_weight):
        x_raw = tf.gather(train_locs, idx)
        ys = tf.gather(train_classes, idx)
        fake_norm, bg_cells = sample_background(tf.shape(idx)[0])
        return core_step(x_raw, ys, fake_norm, pos_weight, bg_cells)

    @tf.function(reduce_retracing=True, jit_compile=jit_compile)
    def core_step(x_raw, ys, fake_norm, pos_weight, bg_cells):
        x = encoder.encode_tf(x_raw, normalize=True)
        fake_x = encoder.encode_tf(fake_norm, normalize=False)
        with tf.GradientTape() as tape:
            logits = model(x[:, :-1], training=True)
            fake_logits = model(fake_x[:, :-1], training=True)
            bg_effort = bg_present = bg_range_penalty = None
            if use_effort:
                bg_effort = effort_scale * fake_x[:, effort_col]
                # unpack each background row's cell bitmask into a (B_bg, num_classes) 0/1 mask
                packed = tf.gather(cell_bits, tf.gather(cell_row, bg_cells))
                bits = tf.bitwise.bitwise_and(tf.bitwise.right_shift(packed[:, :, None], bit_shifts), 1)
                n_bg, n_classes = tf.shape(fake_logits)[0], tf.shape(fake_logits)[1]
                bg_present = tf.cast(tf.reshape(bits, [n_bg, -1])[:, :n_classes], tf.float32)
            if cell_range_penalty is not None:
                bg_range_penalty = tf.cast(tf.gather(cell_range_penalty, bg_cells), tf.float32)
            loss = sinr_loss(
                ys, logits, fake_logits, pos_weight, bg_base_weight=bg_base_weight,
                bg_effort=bg_effort, bg_present=bg_present,
                focal_cols=focal_cols_t, bg_range_penalty=bg_range_penalty,
            )

        gradients = tape.gradient(loss, weights)
        head_pairs = [(g, w) for g, w in zip(gradients, weights) if id(w) in head_weight_refs]
        other_pairs = [(g, w) for g, w in zip(gradients, weights) if id(w) not in head_weight_refs]

        if grad_clip_value is not None:
            # per-element clip BEFORE the optimizer's own global-norm clip (clipnorm=1.0, set on
            # the Adam optimizer itself) - clipnorm alone can't stop one output column (e.g. one
            # class's weights in a 5,000+-wide shared layer) from absorbing a disproportionate,
            # badly-directed share of the norm budget. This bounds any single gradient element
            # directly, added after v30's 5,007-species run diverged (85% of classes collapsed to
            # near-zero output within a few epochs). Only applied to `other_pairs` (the shared
            # backbone) - see head_optimizer docstring above for why the independent head is exempt.
            other_pairs = [
                (tf.clip_by_value(g, -grad_clip_value, grad_clip_value) if g is not None else None, w)
                for g, w in other_pairs
            ]

        if other_pairs:
            optimizer.apply_gradients(other_pairs)
        if head_pairs:
            head_optimizer.apply_gradients(head_pairs)

        return loss

    return train_step


def make_multi_step(train_step):
    """Wraps a make_train_step step so one compiled call runs a whole block of steps:
    run(idx_block, pos_weight) -> per-step losses, for idx_block of shape (n_steps, batch).
    Calling the step once per batch from Python costs a host sync plus Python/dispatch overhead
    every step - on an H100 that roughly doubled the time of a v42-sized step (10ms vs a
    ~3-5ms compiled step)."""
    @tf.function(reduce_retracing=True)
    def run(idx_block, pos_weight):
        n_steps = tf.shape(idx_block)[0]
        losses = tf.TensorArray(tf.float32, size=n_steps)
        for i in tf.range(n_steps):
            losses = losses.write(i, train_step(idx_block[i], pos_weight))
        return losses.stack()

    return run


def h3_cell_ids(lonlat, cell_to_id, res):
    """Global integer H3 cell id for each (lon, lat) row."""
    return np.fromiter(
        (cell_to_id[h3.geo_to_h3(float(lat), float(lon), res)] for lon, lat in lonlat),
        dtype=np.int32, count=len(lonlat),
    )


def build_cell_presence_bits(train_cell_ids, class_ids, n_cells, num_classes):
    """Which classes were observed in each H3 cell, as a packed bitmask on-device:
      cell_row: (n_cells,) int64, global cell id -> row of `bits` (0 = an all-zero row)
      bits:     (n_occupied + 1, ceil(num_classes / 8)) uint8, bit (k % 8) of byte k // 8 = class k
    ~168MB at 25k classes / 54k occupied cells; ~1.2GB at 120k classes. Replaces a ragged
    cell->classes lookup whose per-step gather/padding ran off the GPU and scaled with the
    busiest cell (3,066 classes in one cell at 25k classes -> 1.57M padded pairs per step)."""
    # unique (cell, class) pairs via a single int64 key - np.unique(axis=0) on an (N, 2) array
    # makes several full copies, which adds up at 148M points
    keys = np.unique(train_cell_ids.astype(np.int64) * num_classes + class_ids)
    pair_cells, pair_classes = keys // num_classes, keys % num_classes
    del keys
    occupied, row_of_pair = np.unique(pair_cells, return_inverse=True)
    bits = np.zeros((len(occupied) + 1, (num_classes + 7) // 8), dtype=np.uint8)
    np.bitwise_or.at(bits, (row_of_pair + 1, pair_classes >> 3), (1 << (pair_classes & 7)).astype(np.uint8))
    cell_row = np.zeros(n_cells, dtype=np.int64)  # int64: int32 tensors stay in host memory
    cell_row[occupied] = np.arange(1, len(occupied) + 1, dtype=np.int64)
    print(f"  {len(occupied):,} occupied H3 cells, {len(pair_cells):,} cell-species pairs, "
          f"presence table {bits.nbytes / 1e6:.0f}MB")
    return tf.constant(cell_row), tf.constant(bits)


def build_cell_range_penalty(cells, focal_class_ids, locs, class_ids, scale, norm_deg, cap):
    """(n_cells, n_focal) float16: for each H3 cell centroid, the range penalty for each focal
    class - min(distance in degrees to that species' nearest real observation / norm_deg, cap)
    * scale. Same formula as the old per-step KD-tree query, evaluated once per cell centroid
    instead of once per background point (cells are ~0.2 degrees across at res 4)."""
    centroids = np.array([h3.h3_to_geo(c)[::-1] for c in cells], dtype=np.float64)  # (lon, lat)
    table = np.empty((len(cells), len(focal_class_ids)), dtype=np.float16)
    for i, fc in enumerate(tqdm.tqdm(focal_class_ids, desc="  range penalty table")):
        dist_deg, _ = cKDTree(locs[class_ids == fc]).query(
            centroids, distance_upper_bound=norm_deg * cap, workers=-1
        )
        table[:, i] = np.minimum(dist_deg / norm_deg, cap) * scale
    return table


PREDICT_CHUNK = 4096  # rows per forward pass in adapt_pos_weight (4096 x 120k classes = 2GB)


def adapt_pos_weight(fcnet, encoder, locs, class_ids, watch_class_ids, pos_weight_np,
                      base_pos_weight, threshold, factor, max_multiple):
    """In-loop closed-loop controller: for each class in watch_class_ids, evaluate its current mean
    predicted probability at its own real observations (using the live fcnet, not a saved checkpoint),
    and boost pos_weight_np in place for any class still below threshold. Returns the count boosted
    this check, for logging. A class that has already crossed threshold is left alone (its weight
    stays wherever it got there) - only still-stuck classes keep compounding."""
    boosted = 0
    for cid in watch_class_ids:
        sp_locs = locs[class_ids == cid]
        if len(sp_locs) == 0:
            continue
        # in chunks, keeping only column cid on-device: one pass over a common focal species'
        # ~100k points used to materialize a (points x num_classes) array - 9.8GB of host RAM
        # at 25k classes, ~47GB (more than fits beside training on the GPU) at 120k
        preds = np.concatenate([
            tf.gather(fcnet(encoder.encode(sp_locs[i:i + PREDICT_CHUNK])[:, :-1], training=False),
                      cid, axis=1).numpy()
            for i in range(0, len(sp_locs), PREDICT_CHUNK)
        ])
        if preds.mean() < threshold:
            cap = base_pos_weight * max_multiple
            pos_weight_np[cid] = min(pos_weight_np[cid] * factor, cap)
            boosted += 1
    return boosted


def subsample_to_n_taxa(locs, class_ids, unique_taxa, n_taxa, h3_idx=None):
    """Keep only the first n_taxa taxa (by their sorted taxon index)."""
    n_taxa = min(n_taxa, len(unique_taxa))
    keep_classes = set(range(n_taxa))
    mask = np.isin(class_ids, list(keep_classes))
    locs = locs[mask]
    class_ids = class_ids[mask]
    unique_taxa = unique_taxa[:n_taxa]
    h3_idx = h3_idx[mask] if h3_idx is not None else None
    print(f" subsampled to {n_taxa} taxa ({mask.sum()} observations)")
    return locs, class_ids, unique_taxa, h3_idx


def _subsample_indices(hard_cap, class_ids, focal_class_ids=None):
    """Returns indices of the subsampled training set. focal_class_ids are never capped."""
    focal_set = set(focal_class_ids) if focal_class_ids else set()
    if hard_cap == -1:
        return np.arange(len(class_ids))
    print(f"  subsampling (up to) {hard_cap} per class for the training set")
    # a uniform random hard_cap points of each class (all points of focal classes): visit points
    # in random order and keep each one whose rank within its class is < hard_cap. Vectorized -
    # the equivalent per-point Python loop took minutes and GBs of Python ints at 148M points.
    rng = np.random.default_rng()
    idx_rand = rng.permutation(len(class_ids))
    cls = class_ids[idx_rand]
    order = np.argsort(cls, kind="stable")  # by class, random order within each class
    sorted_cls = cls[order]
    group_start = np.searchsorted(sorted_cls, sorted_cls, side="left")
    rank = np.empty(len(cls), dtype=np.int64)
    rank[order] = np.arange(len(cls)) - group_start
    keep = rank < hard_cap
    if focal_set:
        keep |= np.isin(cls, list(focal_set))
    return np.sort(idx_rand[keep])


def make_subsampled_dataset_focal(
    hard_cap, class_ids, batch_size, shuffle_buffer_size,
    focal_class_ids=None, focal_oversample=1, focal_oversample_map=None, steps_per_call=1,
    full_shuffle=False,
):
    """Dataset of blocks of shuffled batches of training-point INDICES, shape
    (<= steps_per_call, batch_size) - the step gathers and encodes the points on-device (see
    make_train_step / make_multi_step).

    focal_oversample_map: optional {class_id: factor} for PER-SPECIES oversampling (e.g.
    sparser members of a large focal group get a bigger factor than common ones) - takes
    precedence over the flat focal_oversample int when provided. See --oversample_target_exponent."""
    idx = _subsample_indices(hard_cap, class_ids, focal_class_ids)
    class_ids_ss = class_ids[idx]

    if focal_class_ids and focal_oversample_map:
        extra = [
            np.tile(idx[class_ids_ss == fc], focal_oversample_map.get(fc, 1) - 1)
            for fc in focal_class_ids if focal_oversample_map.get(fc, 1) > 1
        ]
        if extra:
            idx = np.concatenate([idx] + extra)
        print(f"  focal oversampled per-species (map, {len(focal_oversample_map)} classes): "
              f"{len(idx):,} total obs")
    elif focal_class_ids and focal_oversample > 1:
        focal_idx = idx[np.isin(class_ids_ss, focal_class_ids)]
        idx = np.concatenate([idx, np.tile(focal_idx, focal_oversample - 1)])
        print(f"  focal oversampled {focal_oversample}x: {len(idx):,} total obs")
    else:
        print(f"  final training set size: {len(idx)}")

    num_steps = len(idx) // batch_size
    if full_shuffle:
        # a fresh uniform permutation of ALL indices every epoch (the map re-runs on each pass over
        # the dataset). The .npy files keep each species' rows in long runs, so a shuffle buffer
        # only mixes nearby rows: with the default 500k buffer, a batch sees ~300 species at a
        # time (of 5k in v42, of 25k in the pilot) and the epoch sweeps through species in turn.
        ds = (tf.data.Dataset.from_tensors(tf.constant(idx.astype(np.int64)))
              .map(tf.random.shuffle).unbatch())
    else:
        ds = tf.data.Dataset.from_tensor_slices(idx.astype(np.int64))
        buf = ds.cardinality() if shuffle_buffer_size is None else shuffle_buffer_size
        ds = ds.shuffle(buffer_size=buf)
    # full batches only (the < batch_size leftover points of each shuffled epoch are skipped),
    # grouped into blocks of steps_per_call batches for make_multi_step
    ds = ds.batch(batch_size, drop_remainder=True).batch(steps_per_call).prefetch(tf.data.AUTOTUNE)
    return ds, num_steps


@click.command()
@click.option("--config_file", required=True)
@click.option("--train_csv", default=None, type=str,
              help="Path to a CSV with longitude,latitude,taxon_id columns (sinr format). Overrides config train_data.")
@click.option("--train_npy", default=None, type=str,
              help="Prefix for preprocessed numpy files (e.g. 'train_10k' loads train_10k_locs.npy, "
                   "train_10k_class_ids.npy, train_10k_unique_taxa.npy). Written by preprocess_training_data.py. "
                   "Use this instead of --train_csv for large datasets that don't fit in memory as CSV.")
@click.option("--bg_csv", default=None, type=str,
              help="Optional CSV of additional background-only observations (any species). Used only for target-group sampling.")
@click.option("--bg_sampling", default="random", type=click.Choice(["random", "target_group"]),
              help="Background sampling strategy. 'random': uniform random locations (default). "
                   "'target_group': sample background from observation locations of other species, "
                   "so areas with many observations but no species X get low intensity and "
                   "unsampled areas rely on environmental covariates.")
@click.option("--bg_tg_fraction", default=1.0, type=float,
              help="When using target_group sampling, fraction of each background batch drawn from "
                   "the target-group pool (0.0–1.0). The remainder is uniform random. "
                   "E.g. 0.5 mixes half target-group, half random. Default: 1.0 (pure target-group).")
@click.option("--num_taxa", default=None, type=int,
              help="Limit training to this many taxa (for quick troubleshooting).")
@click.option("--num_epochs", default=None, type=int,
              help="Override num_epochs from config.")
@click.option("--batch_size", default=None, type=int,
              help="Override batch_size from config.")
@click.option("--no_wandb", is_flag=True, default=False,
              help="Disable wandb logging.")
@click.option("--focal_taxa", default=None, type=str,
              help="Comma-separated taxon IDs that are never hard-capped during subsampling (e.g. '43188,47737,51315').")
@click.option("--pos_weight", default=None, type=float,
              help="Override sinr_hyperparams.pos_weight from config.")
@click.option("--focal_oversample", default=1, type=int,
              help="Repeat focal species observations this many times per epoch so they appear more "
                   "frequently in batches. E.g. 5 means focal obs appear 5x (requires --focal_taxa). "
                   "Ignored if --oversample_target_exponent > 0 (per-species map takes precedence).")
@click.option("--oversample_target_exponent", default=0.0, type=float,
              help="Per-species oversample factor instead of a flat --focal_oversample: for each focal "
                   "taxon, factor = round(clip((median_group_count/class_count)^exponent, 1, "
                   "--oversample_max)). 0 disables (default: flat --focal_oversample used instead). "
                   "Lets a large, automatically-selected focal group (not just a hand-picked few) get "
                   "oversampling scaled to how sparse each member actually is, rather than one flat "
                   "multiplier for everyone regardless of size. Try 1.0 (direct inverse proportion).")
@click.option("--oversample_max", default=20, type=int,
              help="Cap on the per-species oversample factor from --oversample_target_exponent, so the "
                   "sparsest classes don't get an extreme multiplier. Default 20.")
@click.option("--start_epoch", default=0, type=int,
              help="Resume training from this epoch (0-indexed). Load checkpoint saved after epoch N "
                   "by passing --start_epoch N. Checkpoint files must exist at {model_save_name}.epochNNN.keras.")
@click.option("--range_penalty_scale", default=0.0, type=float,
              help="Opposite of pos_weight: for each focal taxon, extra absence penalty at background "
                   "points scaled by distance (in degrees, capped) from that species' own nearest "
                   "real observation - independent of environmental covariate similarity or other "
                   "species' effort. 0 disables. Requires --focal_taxa.")
@click.option("--bg_base_weight", default=1.0, type=float,
              help="Flat absence-pressure floor applied to every background point's loss, independent "
                   "of effort_scale or range_penalty_scale. The lever for 'lower intensity where there's "
                   "no presence, regardless of sampling effort.' Default 1.0 (no change from baseline).")
@click.option("--focal_head_dim", default=0, type=int,
              help="Give focal taxa a small dedicated head (this many hidden units, 2 layers) that "
                   "takes the raw input directly, in addition to the shared backbone every class "
                   "uses - so learning a subtle covariate signal for one sparse focal species doesn't "
                   "have to compete for capacity/gradient budget with the other ~10,000 classes in "
                   "the shared backbone. 0 disables (default: shared backbone only, as before). "
                   "Requires --focal_taxa. See lib/models/geo_model_net.py:make_geo_model_net_focal_head.")
@click.option("--focal_head_random_init", is_flag=True, default=False,
              help="Use normal (he_normal) init for the focal head's final layer instead of the "
                   "default zero-init. Zero-init (the default, NEVER change for the proven "
                   "7-species v28 recipe) starts every focal class at an identical point, forcing "
                   "gradient descent to break symmetry between them - fine for a handful of "
                   "classes, but a likely cause of the ~50-85% dead-class collapse seen scaling "
                   "to 500+ classes (v30, v33), since it's a race from a shared degenerate start "
                   "rather than the normal-init pattern the plain shared backbone already uses "
                   "successfully at a similar classes-per-hidden-unit ratio. Test at scale only.")
@click.option("--independent_head_dim", default=0, type=int,
              help="Like --focal_head_dim, but each focal class gets a fully INDEPENDENT small "
                   "2-layer MLP (this many hidden units) with ZERO weight-sharing between classes, "
                   "instead of all focal classes reading out through one shared hidden bottleneck. "
                   "Added after --focal_head_dim was confirmed (v30 at N=5,007, v33/v34 at N=500, "
                   "both zero-init and random-init) to let a large fraction of a big focal group's "
                   "classes go permanently dead - a symmetric-race/capacity-competition failure "
                   "that a shared bottleneck is exposed to regardless of width or init scheme. "
                   "0 disables (default). Mutually exclusive with --focal_head_dim. Requires "
                   "--focal_taxa. See lib/models/geo_model_net.py:make_geo_model_net_independent_head.")
@click.option("--pos_weight_balance_exponent", default=0.0, type=float,
              help="Scale pos_weight per class by (median_obs_count / class_obs_count)^exponent, "
                   "capped at 1.0 (classes at/below the median are unaffected). 0 disables (default: "
                   "flat pos_weight for every class, as before). Added after a 5,007-species run with "
                   "no curation diverged - a flat pos_weight let classes with 100,000 real "
                   "observations contribute ~1000x the aggregate positive-loss gradient of classes "
                   "with 100, destabilizing the shared output layer. Try 0.5 (sqrt scaling) first.")
@click.option("--grad_clip_value", default=None, type=float,
              help="Per-element gradient clip applied before the optimizer's own global-norm clip "
                   "(clipnorm=1.0, always on). None disables (default). clipnorm alone can't stop one "
                   "output column from absorbing a disproportionate, badly-directed share of the norm "
                   "budget - this bounds any single gradient element directly. Try 1.0 first.")
@click.option("--stability_check", is_flag=True, default=False,
              help="Stop training (keeping all checkpoints saved so far) if epoch mean loss increases "
                   "by >5% for 2 consecutive epochs, rather than silently continuing. Off by default "
                   "since a legitimate run could have a rare benign bump; turn on for unattended runs "
                   "on new/untested data where you won't be watching every epoch.")
@click.option("--pos_weight_boost_taxa", default=None, type=str,
              help="Comma-separated taxon_ids to multiply pos_weight by --pos_weight_boost_factor "
                   "(on top of the flat --pos_weight / --pos_weight_balance_exponent value). Diagnostic "
                   "lever for classes stuck at a stable sub-threshold logit despite non-vanishing "
                   "gradient (v36 finding: 161/500 classes plateaued for 6+ epochs with no positional, "
                   "count, or oversampling explanation) - tests whether their own positive-vs-background "
                   "loss balance, not something more fundamental, is what's holding them down.")
@click.option("--pos_weight_boost_factor", default=1.0, type=float,
              help="Multiplier applied to pos_weight for classes in --pos_weight_boost_taxa. Default 1.0 "
                   "(no-op).")
@click.option("--pos_weight_adapt", is_flag=True, default=False,
              help="Automatically, repeatedly boost pos_weight in-loop for any focal class still stuck "
                   "below --pos_weight_adapt_threshold, re-checked every --pos_weight_adapt_every epochs "
                   "for the rest of training. Generalizes --pos_weight_boost_taxa (a one-time, manually- "
                   "curated, flat boost - v37 found this fixed ~27% of 161 stuck classes in 5 epochs, "
                   "left the rest flat, including some with literally zero response) into a closed-loop "
                   "controller: no manual stuck-list needed (defaults to monitoring every class in "
                   "--focal_taxa), and classes that don't respond to one boost keep getting boosted "
                   "further rather than being left at a single fixed multiplier. Classes that cross the "
                   "threshold stop being boosted (their weight is left wherever it got them there).")
@click.option("--pos_weight_adapt_every", default=3, type=int,
              help="Epochs between adaptation checks (default: 3).")
@click.option("--pos_weight_adapt_threshold", default=0.05, type=float,
              help="Mean predicted probability (at a class's own real observations) below which a class "
                   "is considered still stuck and gets boosted again at the next check (default: 0.05).")
@click.option("--pos_weight_adapt_factor", default=2.0, type=float,
              help="Multiplier applied to a still-stuck class's pos_weight at each adaptation check "
                   "(default: 2.0, compounds every check a class remains stuck).")
@click.option("--pos_weight_adapt_max", default=50.0, type=float,
              help="Cap on a class's pos_weight as a multiple of the base pos_weight, so a "
                   "never-responding class can't be boosted without bound (default: 50.0).")
@click.option("--pos_weight_adapt_taxa", default=None, type=str,
              help="Comma-separated taxon_ids to monitor/adapt. None (default) monitors every class in "
                   "--focal_taxa - the point of this mechanism is not needing a manually-curated "
                   "stuck-list the way --pos_weight_boost_taxa did.")
@click.option("--pos_weight_init_file", default=None, type=str,
              help="Load the full per-class pos_weight vector directly from a .npy file (shape "
                   "(num_classes,)), bypassing --pos_weight/--pos_weight_balance_exponent/"
                   "--pos_weight_boost_taxa entirely. Use this to resume --pos_weight_adapt runs: "
                   "the adapted weights live only in memory during training and are NOT part of the "
                   "saved .keras checkpoint, so an interrupted adaptive run needs this to pick up "
                   "where it left off instead of silently resetting every class back to its un-adapted "
                   "starting weight. Saved automatically alongside every checkpoint whenever "
                   "--pos_weight_adapt is on (see {model_save_name}.epoch{N}.pos_weight.npy) - pass that "
                   "file back in here when resuming.")
@click.option("--full_shuffle", is_flag=True, default=False,
              help="Shuffle the whole training set every epoch instead of through the config's "
                   "shuffle_buffer_size window. The .npy datasets keep each species' rows in long runs, "
                   "so a 500k-row buffer only mixes ~300 species at a time. Off by default so earlier "
                   "runs stay reproducible; recommended for anything new.")
@click.option("--steps_per_call", default=50, type=int,
              help="Training steps run per compiled call (make_multi_step). The host syncs, logs and "
                   "updates the progress bar once per block instead of every step. 1 = per-step.")
@click.option("--jit_compile", is_flag=True, default=False,
              help="XLA-compile the training step (tf.function(jit_compile=True)). Can be much faster "
                   "on GPU; off by default until verified on the target hardware.")
def train_model(config_file, train_csv, train_npy, bg_csv, bg_sampling, bg_tg_fraction, num_taxa, num_epochs,
                batch_size, no_wandb, focal_taxa, pos_weight, focal_oversample, start_epoch,
                range_penalty_scale, bg_base_weight, focal_head_dim, focal_head_random_init,
                independent_head_dim,
                pos_weight_balance_exponent, grad_clip_value, stability_check,
                oversample_target_exponent, oversample_max,
                pos_weight_boost_taxa, pos_weight_boost_factor,
                pos_weight_adapt, pos_weight_adapt_every, pos_weight_adapt_threshold,
                pos_weight_adapt_factor, pos_weight_adapt_max, pos_weight_adapt_taxa,
                pos_weight_init_file, jit_compile, steps_per_call, full_shuffle):
    if focal_head_dim > 0 and independent_head_dim > 0:
        raise click.UsageError("--focal_head_dim and --independent_head_dim are mutually exclusive")
    with open(config_file, "r") as f:
        config = yaml.safe_load(f)

    if num_epochs is not None:
        config["num_epochs"] = num_epochs
    if batch_size is not None:
        config["batch_size"] = batch_size
    if pos_weight is not None:
        config["sinr_hyperparams"]["pos_weight"] = pos_weight

    wandb.init(
        project="geomodel_tf_sinr",
        config=config,
        mode="disabled" if no_wandb else "online",
    )

    if train_npy is not None:
        (locs, class_ids, unique_taxa) = load_sinr_dataset_from_npy(train_npy)
        h3_idx = None
    elif train_csv is not None:
        (locs, class_ids, unique_taxa) = load_sinr_dataset_from_csv(train_csv)
        h3_idx = None
    elif config["sampling"]["discretized_sampling"]:
        if config["dataset_type"] == "sinr":
            (locs, class_ids, unique_taxa, h3_idx) = load_sinr_dataset_from_parquet_h3(
                config["sinr_dataset"]["train_data"],
                h3_resolution=config["sampling"]["h3_resolution"]
            )
        else:
            (locs, class_ids, unique_taxa, h3_idx) = load_inat_dataset_from_parquet_h3(
                config["inat_dataset"]["spatial_data"],
                h3_resolution=config["sampling"]["h3_resolution"]
            )
    else:
        if config["dataset_type"] == "sinr":
            (locs, class_ids, unique_taxa) = load_sinr_dataset_from_parquet(
                config["sinr_dataset"]["train_data"]
            )
        elif config["dataset_type"] == "inat":
            (locs, class_ids, unique_taxa) = load_inat_dataset_from_parquet(
                config["inat_dataset"]["spatial_data"]
            )
        h3_idx = None

    if num_taxa is not None:
        locs, class_ids, unique_taxa, h3_idx = subsample_to_n_taxa(
            locs, class_ids, unique_taxa, num_taxa, h3_idx=h3_idx
        )

    if config["inputs"]["covariates"] == "env":
        raster = np.load(config["bioclim_data"]).astype(np.float32)
    elif config["inputs"]["covariates"] == "elev":
        raster = np.load(config["elev_data"]).astype(np.float32)
    else:
        raster = None

    encoder = CoordEncoder(config["inputs"]["loc_feat_encoding"], raster)

    # fixed raster channel layout (see focal_10k_config_v* comments): elev(0), bio1-19(1-19),
    # coast_dist(20), land_mask(21), effort(22), [coast_dist*bio1-19 interactions(23-41) if
    # present]. Encoded column = coord_feat_dim + raster_channel. Do NOT use fake_x[:, -1] for
    # effort - that only worked before v6 added channels after effort; see note at effort_vals.
    EFFORT_RASTER_CHANNEL = 22
    coord_feat_dim = encoder.num_input_feats() - raster.shape[-1] if raster is not None else encoder.num_input_feats()
    EFFORT_COL = coord_feat_dim + EFFORT_RASTER_CHANNEL

    num_classes = len(unique_taxa)
    print(f" training with {num_classes} classes")

    # training points stay as raw (lon, lat) degrees, on-device; each batch is encoded there
    # (see make_train_step). Background pools are likewise raw locations + H3 cell ids.
    locs = np.asarray(locs, dtype=np.float32)
    class_ids = np.asarray(class_ids, dtype=np.int64)
    train_locs_tf = tf.constant(locs)
    train_classes_tf = tf.constant(class_ids)

    bg_pool_raw = None   # target-group background locations, (lon, lat) degrees
    rand_pool_raw = None  # pre-drawn uniform random background locations, normalized [-1, 1]
    if bg_sampling == "target_group":
        if bg_csv is not None:
            print(f"loading additional background observations from {bg_csv}")
            bg_cols = pd.read_csv(bg_csv, nrows=0).columns.tolist()
            usecols = [c for c in ["longitude", "latitude", "taxon_id"] if c in bg_cols]
            bg_df = pd.read_csv(bg_csv, usecols=usecols)
            bg_df = clean_dataset(bg_df)
            bg_locs_extra = np.vstack(
                (bg_df["longitude"].values, bg_df["latitude"].values)
            ).T.astype(np.float32)
            bg_pool_raw = np.concatenate([locs, bg_locs_extra], axis=0)
            print(f" background pool: {len(bg_pool_raw):,} points "
                  f"({len(locs):,} training + {len(bg_locs_extra):,} extra)")
        else:
            bg_pool_raw = locs
            print(f" background pool: {len(bg_pool_raw):,} points (training observations only)")

    # resolve focal taxon IDs → class indices
    focal_class_ids = None
    if focal_taxa is not None:
        focal_taxon_ids = set(int(t.strip()) for t in focal_taxa.split(","))
        focal_class_ids = [
            int(np.where(unique_taxa == tid)[0][0])
            for tid in focal_taxon_ids
            if tid in unique_taxa
        ]
        print(f" focal class ids (uncapped): {focal_class_ids}")

    focal_oversample_map = None
    if focal_class_ids and oversample_target_exponent > 0:
        focal_counts = np.array([np.sum(class_ids == fc) for fc in focal_class_ids], dtype=np.float64)
        median_group_count = np.median(focal_counts)
        raw_factor = (median_group_count / np.maximum(focal_counts, 1.0)) ** oversample_target_exponent
        factor = np.clip(np.round(raw_factor), 1, oversample_max).astype(int)
        focal_oversample_map = {fc: int(f) for fc, f in zip(focal_class_ids, factor)}
        print(f" oversample_target_exponent={oversample_target_exponent}: "
              f"factor range [{factor.min()}, {factor.max()}] across {len(focal_class_ids)} focal classes")

    # per-run lookup tables keyed by global H3 res-4 cell id, so the per-step background
    # weighting runs on-device (see make_train_step):
    #  - effort weighting: which classes were observed in each cell (their effort weight is 0)
    #  - range penalty (opposite of pos_weight - punishes confident predictions far from any
    #    known observation of THIS species, independent of environmental covariate similarity
    #    or other species' effort/presence): penalty per cell centroid for each focal class
    H3_RES = 4
    RANGE_PENALTY_NORM_DEG = 10.0   # ~1000km at the equator
    RANGE_PENALTY_CAP = 5.0
    effort_scale = config.get("effort_scale", 0.0)
    use_effort = effort_scale > 0 and bg_sampling == "target_group"
    use_range_penalty = range_penalty_scale > 0 and bool(focal_class_ids)
    cell_to_id = cell_presence = cell_range_penalty = None
    bg_pool_cells = rand_pool_cells = None

    # a fixed pool of uniform random background points: for target-group sampling's random
    # fraction, and for pure random sampling when the range penalty needs each point's H3 cell
    # (cells can't be computed on-device; otherwise random sampling draws fresh points each step)
    if (bg_sampling == "target_group" and bg_tg_fraction < 1.0) or (
        bg_sampling == "random" and use_range_penalty
    ):
        n_rand_pool = len(locs) if bg_pool_raw is None else len(bg_pool_raw)
        print(f" pre-computing random background pool ({n_rand_pool:,} points)...")
        rand_pool_raw = np.array(make_rand_samples_tf(n_rand_pool), dtype=np.float32)

    if use_effort or use_range_penalty:
        cells = [c for r0 in sorted(h3.get_res0_indexes()) for c in sorted(h3.h3_to_children(r0, H3_RES))]
        cell_to_id = {c: i for i, c in enumerate(cells)}
        print(f" pre-computing H3-{H3_RES} cell ids ({len(cells):,} cells) for training points...")
        train_cells = h3_cell_ids(locs, cell_to_id, H3_RES)
        if bg_sampling == "target_group":
            bg_pool_cells = train_cells if bg_csv is None else h3_cell_ids(bg_pool_raw, cell_to_id, H3_RES)
        if rand_pool_raw is not None:
            # rand_pool_raw is normalized [-1, 1]; H3 needs degrees. (Before this fix every random
            # point was looked up as if it sat within ~1 degree of 0N 0E, so its effort "present"
            # set and range penalty came from the Gulf of Guinea instead of its real location.)
            rand_pool_cells = h3_cell_ids(rand_pool_raw * np.array([180.0, 90.0], np.float32), cell_to_id, H3_RES)
        if use_effort:
            print(f" building cell→species lookup for effort weighting (effort_scale={effort_scale})...")
            cell_presence = build_cell_presence_bits(train_cells, class_ids, len(cells), num_classes)
        if use_range_penalty:
            print(f" building range penalty table for {len(focal_class_ids)} focal classes "
                  f"(scale={range_penalty_scale})...")
            cell_range_penalty = tf.constant(build_cell_range_penalty(
                cells, focal_class_ids, locs, class_ids, range_penalty_scale,
                RANGE_PENALTY_NORM_DEG, RANGE_PENALTY_CAP,
            ))

    # background pools on-device, sampled inside the train step
    bg_locs_tf = None if bg_pool_raw is None else (
        train_locs_tf if bg_pool_raw is locs else tf.constant(bg_pool_raw)
    )
    # int64, not int32: TF keeps int32 tensors in host memory, so each per-step gather from a
    # 34M-entry int32 table copied the whole table (26.7ms vs 0.56ms per 512-gather on an H100)
    bg_cells_tf = None if bg_pool_cells is None else tf.constant(bg_pool_cells.astype(np.int64))
    rand_locs_tf = None if rand_pool_raw is None else tf.constant(rand_pool_raw)
    rand_cells_tf = None if rand_pool_cells is None else tf.constant(rand_pool_cells.astype(np.int64))

    def sample_background(batch_size):
        """(normalized background locations, their H3 cell ids or None) for one batch."""
        if bg_sampling == "target_group":
            n_tg = tf.cast(tf.cast(batch_size, tf.float64) * bg_tg_fraction, tf.int32)
            tg_idx = tf.random.uniform([n_tg], 0, tf.shape(bg_locs_tf, out_type=tf.int64)[0], dtype=tf.int64)
            loc_parts = [encoder._normalize_coords(tf.gather(bg_locs_tf, tg_idx))]
            cell_parts = [tf.gather(bg_cells_tf, tg_idx)] if bg_cells_tf is not None else None
            if rand_locs_tf is not None:
                rand_idx = tf.random.uniform(
                    [batch_size - n_tg], 0, tf.shape(rand_locs_tf, out_type=tf.int64)[0], dtype=tf.int64
                )
                loc_parts.append(tf.gather(rand_locs_tf, rand_idx))
                if cell_parts is not None:
                    cell_parts.append(tf.gather(rand_cells_tf, rand_idx))
            bg_cells = tf.concat(cell_parts, axis=0) if cell_parts is not None else None
            return tf.concat(loc_parts, axis=0), bg_cells
        if rand_locs_tf is not None:
            rand_idx = tf.random.uniform([batch_size], 0, tf.shape(rand_locs_tf, out_type=tf.int64)[0], dtype=tf.int64)
            return tf.gather(rand_locs_tf, rand_idx), tf.gather(rand_cells_tf, rand_idx)
        return make_rand_samples_tf(batch_size), None

    if focal_head_dim > 0:
        if not focal_class_ids:
            raise click.UsageError("--focal_head_dim requires --focal_taxa")
        print(f" building dedicated focal head (dim={focal_head_dim}, "
              f"random_init={focal_head_random_init}) for classes {focal_class_ids}")
        fcnet = make_geo_model_net_focal_head(
            num_classes=num_classes, focal_class_indices=focal_class_ids,
            num_input_feats=encoder.num_input_feats() - 1,
            hidden_dim=config.get("hidden_dim", 256), focal_hidden_dim=focal_head_dim,
            zero_init_final_layer=not focal_head_random_init,
        )
    elif independent_head_dim > 0:
        if not focal_class_ids:
            raise click.UsageError("--independent_head_dim requires --focal_taxa")
        print(f" building independent per-class focal head (dim={independent_head_dim}) "
              f"for {len(focal_class_ids)} classes {focal_class_ids}")
        fcnet = make_geo_model_net_independent_head(
            num_classes=num_classes, focal_class_indices=focal_class_ids,
            num_input_feats=encoder.num_input_feats() - 1,
            hidden_dim=config.get("hidden_dim", 256), independent_head_dim=independent_head_dim,
        )
    else:
        fcnet = make_geo_model_net(
            num_classes=num_classes, num_input_feats=encoder.num_input_feats() - 1,
            hidden_dim=config.get("hidden_dim", 256)
        )

    save_stem = config["model_save_name"].replace(".keras", "")
    if start_epoch > 0:
        ckpt_path = f"{save_stem}.epoch{start_epoch:03d}.keras"
        print(f"Resuming from checkpoint: {ckpt_path}")
        custom_objects = None
        if focal_head_dim > 0:
            custom_objects = {"FocalScatterAdd": FocalScatterAdd}
        elif independent_head_dim > 0:
            custom_objects = {"FocalScatterAdd": FocalScatterAdd, "IndependentPerClassHead": IndependentPerClassHead}
        fcnet = tf.keras.models.load_model(ckpt_path, custom_objects=custom_objects)

    # view into fcnet exposing the pre-sigmoid logit instead of the final "predictions"
    # Activation("sigmoid") output - shares all of fcnet's weights (same underlying layers),
    # used only so the training loop can compute a numerically-stable loss (see sinr_loss.py).
    # fcnet itself, and every saved checkpoint, is unchanged - still a single sigmoid-output model.
    logits_model = tf.keras.Model(
        inputs=fcnet.input, outputs=fcnet.get_layer("predictions").input, name="logits_view"
    )

    losses = []

    ds, num_train_steps_per_epoch = make_subsampled_dataset_focal(
        config["hard_cap"],
        class_ids,
        config["batch_size"],
        config["shuffle_buffer_size"],
        focal_class_ids=focal_class_ids,
        focal_oversample=focal_oversample,
        focal_oversample_map=focal_oversample_map,
        steps_per_call=steps_per_call,
        full_shuffle=full_shuffle,
    )

    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=config["initial_lr"],
        decay_rate=config["lr_decay"],
        decay_steps=num_train_steps_per_epoch,
        staircase=True,
    )
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule, clipnorm=1.0)

    # IndependentPerClassHead's own W1/b1/W2/b2 get a separate, unclipped optimizer - see
    # make_train_step's docstring for why: grad_clip_value AND the main optimizer's own clipnorm
    # both operate on these tensors as if they were a single shared parameter (which they are,
    # as a matter of storage - one batched tensor per layer, shape (num_classes, ...)) even
    # though by construction there is zero cross-class coupling in what they represent. Both
    # clips end up capping the effective per-class update to the same tiny, pos_weight-independent
    # size regardless of how large pos_weight_boost/pos_weight_adapt push it, and worse, a few
    # heavily-boosted classes dominate the shared tensor's norm and crush every OTHER class
    # sharing that tensor too - including ones that were already converging fine unboosted.
    head_optimizer = None
    head_weight_refs = None
    independent_head_layer = next(
        (l for l in fcnet.layers if isinstance(l, IndependentPerClassHead)), None
    )
    if independent_head_layer is not None:
        head_optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
        head_weight_refs = {id(w) for w in independent_head_layer.trainable_weights}

    if start_epoch > 0:
        # fast-forward optimizer step counter(s) so LR schedule is at the right epoch
        optimizer.iterations.assign(start_epoch * num_train_steps_per_epoch)
        if head_optimizer is not None:
            head_optimizer.iterations.assign(start_epoch * num_train_steps_per_epoch)

    base_pos_weight = config["sinr_hyperparams"]["pos_weight"]
    if pos_weight_init_file:
        pos_weight_np = np.load(pos_weight_init_file).astype(np.float32)
        if pos_weight_np.shape != (num_classes,):
            raise click.UsageError(
                f"--pos_weight_init_file has shape {pos_weight_np.shape}, expected ({num_classes},)"
            )
        print(f" pos_weight_init_file={pos_weight_init_file}: loaded directly, "
              f"pos_weight range [{pos_weight_np.min():.4f}, {pos_weight_np.max():.4f}] "
              "(--pos_weight/--pos_weight_balance_exponent/--pos_weight_boost_taxa ignored)")
    elif pos_weight_balance_exponent > 0:
        class_counts = np.bincount(class_ids, minlength=num_classes).astype(np.float64)
        class_counts = np.maximum(class_counts, 1.0)
        median_count = np.median(class_counts)
        scale = np.minimum(1.0, (median_count / class_counts) ** pos_weight_balance_exponent)
        pos_weight_np = (base_pos_weight * scale).astype(np.float32)
        print(f" pos_weight_balance_exponent={pos_weight_balance_exponent}: "
              f"scale range [{scale.min():.4f}, {scale.max():.4f}], "
              f"pos_weight range [{pos_weight_np.min():.4f}, {pos_weight_np.max():.4f}]")
    else:
        pos_weight_np = np.full(num_classes, base_pos_weight, dtype=np.float32)

    if not pos_weight_init_file and pos_weight_boost_taxa:
        boost_taxon_ids = [int(t.strip()) for t in pos_weight_boost_taxa.split(",")]
        boost_class_ids = [
            int(np.where(unique_taxa == tid)[0][0]) for tid in boost_taxon_ids if tid in unique_taxa
        ]
        pos_weight_np[boost_class_ids] *= pos_weight_boost_factor
        print(f" pos_weight_boost_factor={pos_weight_boost_factor}: boosted {len(boost_class_ids)} classes, "
              f"pos_weight range [{pos_weight_np.min():.4f}, {pos_weight_np.max():.4f}]")

    adapt_watch_class_ids = None
    if pos_weight_adapt:
        if pos_weight_adapt_taxa:
            watch_taxon_ids = [int(t.strip()) for t in pos_weight_adapt_taxa.split(",")]
            adapt_watch_class_ids = [
                int(np.where(unique_taxa == tid)[0][0]) for tid in watch_taxon_ids if tid in unique_taxa
            ]
        else:
            if not focal_class_ids:
                raise click.UsageError("--pos_weight_adapt requires --focal_taxa or --pos_weight_adapt_taxa")
            adapt_watch_class_ids = focal_class_ids
        print(f" pos_weight_adapt enabled: monitoring {len(adapt_watch_class_ids)} classes every "
              f"{pos_weight_adapt_every} epochs, threshold={pos_weight_adapt_threshold}, "
              f"factor={pos_weight_adapt_factor}, max={pos_weight_adapt_max}x base")

    pos_weight = tf.constant(pos_weight_np, dtype=tf.float32)

    train_step = make_train_step(
        optimizer, logits_model, encoder, train_locs_tf, train_classes_tf, sample_background,
        bg_base_weight=bg_base_weight, grad_clip_value=grad_clip_value,
        head_optimizer=head_optimizer, head_weight_refs=head_weight_refs, effort_col=EFFORT_COL,
        effort_scale=effort_scale if use_effort else 0.0, cell_presence=cell_presence,
        focal_cols=focal_class_ids, cell_range_penalty=cell_range_penalty, jit_compile=jit_compile,
    )
    run_steps = make_multi_step(train_step)

    prev_epoch_loss = None
    prev_prev_epoch_loss = None

    for epoch in range(start_epoch, config["num_epochs"]):
        print(f"Epoch {epoch+1}/{config['num_epochs']}")
        epoch_losses = []

        print(f" optimizer lr is {optimizer.learning_rate.numpy()}")

        if config["subsample_each_epoch"] and epoch != 0:
            print(" re subsampling dataset")
            ds, _ = make_subsampled_dataset_focal(
                config["hard_cap"],
                class_ids,
                config["batch_size"],
                config["shuffle_buffer_size"],
                focal_class_ids=focal_class_ids,
                focal_oversample=focal_oversample,
                focal_oversample_map=focal_oversample_map,
                steps_per_call=steps_per_call,
                full_shuffle=full_shuffle,
            )

        pbar = tqdm.tqdm(total=num_train_steps_per_epoch, dynamic_ncols=True)
        step = 0
        for idx_block in ds:
            block_losses = run_steps(idx_block, pos_weight).numpy()
            pbar.update(len(block_losses))
            for i in range(-(-step // 10) * 10 - step, len(block_losses), 10):
                # same cadence as before: batch_loss logged every 10th step
                wandb.log(
                    {"batch_loss": block_losses[i], "learning_rate": optimizer.learning_rate.numpy()},
                    step=epoch * num_train_steps_per_epoch + step + i, commit=True,
                )
            pbar.set_description(f" Loss {block_losses[-1]:.4f}")
            epoch_losses.extend(block_losses.tolist())
            step += len(block_losses)

        pbar.close()
        epoch_loss = np.mean(epoch_losses)
        losses.append(epoch_loss)
        print(f" Epoch mean loss: {epoch_loss:.4f}")

        ckpt_path = f"{save_stem}.epoch{epoch+1:03d}.keras"
        fcnet.save(ckpt_path)
        print(f" Saved checkpoint: {ckpt_path}")

        if pos_weight_adapt:
            # pos_weight_np lives only in memory - NOT part of fcnet's saved weights - so an
            # interrupted run has no way to resume its adapted state without this. Save every
            # epoch (not just on adapt-check epochs) so resuming from ANY checkpoint is exact.
            pw_path = f"{save_stem}.epoch{epoch+1:03d}.pos_weight.npy"
            np.save(pw_path, pos_weight_np)

        if pos_weight_adapt and (epoch + 1) % pos_weight_adapt_every == 0:
            watch_class_ids = adapt_watch_class_ids
            n_boosted = adapt_pos_weight(
                fcnet, encoder, locs, class_ids, watch_class_ids, pos_weight_np,
                base_pos_weight, pos_weight_adapt_threshold, pos_weight_adapt_factor,
                pos_weight_adapt_max,
            )
            pos_weight = tf.constant(pos_weight_np, dtype=tf.float32)
            print(f" pos_weight_adapt check @ epoch {epoch+1}: {n_boosted}/{len(watch_class_ids)} classes "
                  f"still below {pos_weight_adapt_threshold}, boosted "
                  f"(pos_weight range [{pos_weight_np.min():.4f}, {pos_weight_np.max():.4f}])")

        if stability_check and prev_epoch_loss is not None and prev_prev_epoch_loss is not None:
            increased_this_epoch = epoch_loss > prev_epoch_loss * 1.05
            increased_last_epoch = prev_epoch_loss > prev_prev_epoch_loss * 1.05
            if increased_this_epoch and increased_last_epoch:
                print(f" STABILITY CHECK FAILED: epoch loss increased >5% for 2 consecutive "
                      f"epochs ({prev_prev_epoch_loss:.4f} -> {prev_epoch_loss:.4f} -> "
                      f"{epoch_loss:.4f}). Stopping training - all checkpoints saved so far "
                      f"(through epoch {epoch+1:03d}) are kept. This usually means diverging "
                      f"training (e.g. see v30's history: a flat pos_weight across a huge "
                      f"class-size range collapsed most classes' output within a few epochs).")
                break
        prev_prev_epoch_loss = prev_epoch_loss
        prev_epoch_loss = epoch_loss

    fcnet.save(config["model_save_name"])
    wandb.finish()


if __name__ == "__main__":
    train_model()
