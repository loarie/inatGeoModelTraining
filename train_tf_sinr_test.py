import json
import math
import os
from collections import defaultdict
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


def apply_gradient(optimizer, model, x, ys, fake_x, pos_weight, effort_bg_matrix=None, bg_base_weight=1.0,
                    grad_clip_value=None, head_optimizer=None, head_weight_refs=None):
    """`model` here must be the logits-output view (pre-sigmoid) so sinr_loss can compute a
    numerically-stable, non-vanishing gradient - see sinr_loss.py's docstring.

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
    with tf.GradientTape() as tape:
        logits = model(x[:, :-1], training=True)
        fake_logits = model(fake_x[:, :-1], training=True)
        loss = sinr_loss(ys, logits, fake_logits, pos_weight, effort_bg_matrix=effort_bg_matrix, bg_base_weight=bg_base_weight)

    weights = model.trainable_weights
    gradients = tape.gradient(loss, weights)

    if head_optimizer is not None:
        head_pairs = [(g, w) for g, w in zip(gradients, weights) if id(w) in head_weight_refs]
        other_pairs = [(g, w) for g, w in zip(gradients, weights) if id(w) not in head_weight_refs]
    else:
        head_pairs = []
        other_pairs = list(zip(gradients, weights))

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
        enc = encoder.encode(sp_locs)
        preds = fcnet(enc[:, :-1], training=False).numpy()[:, cid]
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


def _subsample_arrays(hard_cap, encoded_locs, class_ids, focal_class_ids=None):
    """Returns subsampled (locs, class_ids) arrays. focal_class_ids are never capped."""
    focal_set = set(focal_class_ids) if focal_class_ids else set()
    if hard_cap == -1:
        idx_ss = np.arange(len(class_ids))
    else:
        print(f"  subsampling (up to) {hard_cap} per class for the training set")
        class_counts = {c: 0 for c in np.unique(class_ids)}
        rng = np.random.default_rng()
        idx_rand = rng.permutation(len(class_ids))
        idx_ss = []
        for i in idx_rand:
            c = class_ids[i]
            if c in focal_set or class_counts[c] < hard_cap:
                idx_ss.append(i)
                class_counts[c] += 1
        idx_ss = np.sort(idx_ss)
    locs_ss = np.array(encoded_locs)[idx_ss]
    class_ids_ss = np.array(class_ids)[idx_ss]
    return locs_ss, class_ids_ss


def _arrays_to_dataset(locs, class_ids, batch_size, shuffle_buffer_size):
    num_steps = len(locs) // batch_size
    ds = tf.data.Dataset.from_tensor_slices((locs, class_ids))
    buf = ds.cardinality() if shuffle_buffer_size is None else shuffle_buffer_size
    ds = ds.shuffle(buffer_size=buf).batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds, num_steps


def make_subsampled_dataset_focal(
    hard_cap, encoded_locs, class_ids, batch_size, shuffle_buffer_size,
    focal_class_ids=None, focal_oversample=1, focal_oversample_map=None
):
    """focal_oversample_map: optional {class_id: factor} for PER-SPECIES oversampling (e.g.
    sparser members of a large focal group get a bigger factor than common ones) - takes
    precedence over the flat focal_oversample int when provided. See --oversample_target_exponent."""
    locs_ss, class_ids_ss = _subsample_arrays(
        hard_cap, encoded_locs, class_ids, focal_class_ids
    )

    if focal_class_ids and focal_oversample_map:
        extra_locs_list, extra_cls_list = [], []
        for fc in focal_class_ids:
            factor = focal_oversample_map.get(fc, 1)
            if factor > 1:
                mask = class_ids_ss == fc
                extra_locs_list.append(np.tile(locs_ss[mask], (factor - 1, 1)))
                extra_cls_list.append(np.tile(class_ids_ss[mask], factor - 1))
        if extra_locs_list:
            locs_ss = np.concatenate([locs_ss] + extra_locs_list)
            class_ids_ss = np.concatenate([class_ids_ss] + extra_cls_list)
        print(f"  focal oversampled per-species (map, {len(focal_oversample_map)} classes): "
              f"{len(locs_ss):,} total obs")
    elif focal_class_ids and focal_oversample > 1:
        focal_mask = np.isin(class_ids_ss, focal_class_ids)
        extra_locs = np.tile(locs_ss[focal_mask], (focal_oversample - 1, 1))
        extra_cls = np.tile(class_ids_ss[focal_mask], focal_oversample - 1)
        locs_ss = np.concatenate([locs_ss, extra_locs])
        class_ids_ss = np.concatenate([class_ids_ss, extra_cls])
        print(f"  focal oversampled {focal_oversample}x: {len(locs_ss):,} total obs")
    else:
        print(f"  final training set size: {len(locs_ss)}")

    return _arrays_to_dataset(locs_ss, class_ids_ss, batch_size, shuffle_buffer_size)


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
def train_model(config_file, train_csv, train_npy, bg_csv, bg_sampling, bg_tg_fraction, num_taxa, num_epochs,
                batch_size, no_wandb, focal_taxa, pos_weight, focal_oversample, start_epoch,
                range_penalty_scale, bg_base_weight, focal_head_dim, focal_head_random_init,
                independent_head_dim,
                pos_weight_balance_exponent, grad_clip_value, stability_check,
                oversample_target_exponent, oversample_max,
                pos_weight_boost_taxa, pos_weight_boost_factor,
                pos_weight_adapt, pos_weight_adapt_every, pos_weight_adapt_threshold,
                pos_weight_adapt_factor, pos_weight_adapt_max, pos_weight_adapt_taxa,
                pos_weight_init_file):
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

    encoded_locs = encoder.encode(locs)
    num_classes = len(unique_taxa)
    print(f" training with {num_classes} classes")

    # build the background pool for target-group sampling
    rand_pool = None
    rand_pool_raw = None  # raw (lon,lat) parallel to rand_pool rows
    bg_pool_h3 = None   # H3 cell index parallel to bg_pool rows
    rand_pool_h3 = None  # H3 cell index parallel to rand_pool rows
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
            bg_encoded_extra = encoder.encode(bg_locs_extra)
            bg_pool = np.concatenate([np.array(encoded_locs), np.array(bg_encoded_extra)], axis=0)
            bg_pool_raw = np.concatenate([locs, bg_locs_extra], axis=0)
            print(f" background pool: {len(bg_pool):,} points "
                  f"({len(encoded_locs):,} training + {len(bg_locs_extra):,} extra)")
        else:
            bg_pool = np.array(encoded_locs)
            bg_pool_raw = locs
            print(f" background pool: {len(bg_pool):,} points (training observations only)")
        bg_pool = bg_pool.astype(np.float32)

        if bg_tg_fraction < 1.0:
            # pre-compute random background pool to avoid per-step raster lookups
            n_rand_pool = len(bg_pool)
            print(f" pre-computing random background pool ({n_rand_pool:,} points)...")
            rand_locs = make_rand_samples_tf(n_rand_pool)
            rand_pool = np.array(encoder.encode(rand_locs, normalize=False)).astype(np.float32)
            rand_pool_raw = np.array(rand_locs).astype(np.float32)
            print(f" random pool ready")

        effort_scale = config.get("effort_scale", 0.0)
        if effort_scale > 0:
            H3_RES = 4
            print(f" pre-computing H3-{H3_RES} cell→species lookup from training data...")
            cell_to_species_sets = defaultdict(set)
            for i in range(len(locs)):
                cell = h3.geo_to_h3(float(locs[i, 1]), float(locs[i, 0]), H3_RES)
                cell_to_species_sets[cell].add(int(class_ids[i]))
            # store as numpy arrays for vectorized fancy indexing per step
            cell_to_species = {
                cell: np.array(list(species), dtype=np.int32)
                for cell, species in cell_to_species_sets.items()
            }
            print(f"  {len(cell_to_species):,} occupied H3 cells")

            print(f" pre-computing H3 cells for bg pool ({len(bg_pool_raw):,} points)...")
            bg_pool_h3 = np.array([
                h3.geo_to_h3(float(bg_pool_raw[i, 1]), float(bg_pool_raw[i, 0]), H3_RES)
                for i in range(len(bg_pool_raw))
            ])
            if rand_pool is not None:
                print(f" pre-computing H3 cells for rand pool ({len(rand_locs):,} points)...")
                rand_locs_np = np.array(rand_locs)
                rand_pool_h3 = np.array([
                    h3.geo_to_h3(float(rand_locs_np[i, 1]), float(rand_locs_np[i, 0]), H3_RES)
                    for i in range(len(rand_locs_np))
                ])
            print(f" effort weighting pre-computation done")

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

    # per-focal-species KDTree of that species' own real observations, for the
    # distance-based range penalty (opposite of pos_weight - punishes confident
    # predictions far from any known observation of THIS species, independent of
    # environmental covariate similarity or other species' effort/presence).
    focal_kdtrees = {}
    RANGE_PENALTY_NORM_DEG = 10.0   # ~1000km at the equator
    RANGE_PENALTY_CAP = 5.0
    if range_penalty_scale > 0 and focal_class_ids:
        print(f" building per-focal-species KDTrees for range penalty (scale={range_penalty_scale})...")
        for fc in focal_class_ids:
            fc_locs = locs[class_ids == fc]
            focal_kdtrees[fc] = cKDTree(fc_locs)
            print(f"  class {fc}: {len(fc_locs)} points")

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
        encoded_locs,
        class_ids,
        config["batch_size"],
        config["shuffle_buffer_size"],
        focal_class_ids=focal_class_ids,
        focal_oversample=focal_oversample,
        focal_oversample_map=focal_oversample_map,
    )

    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=config["initial_lr"],
        decay_rate=config["lr_decay"],
        decay_steps=num_train_steps_per_epoch,
        staircase=True,
    )
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule, clipnorm=1.0)

    # IndependentPerClassHead's own W1/b1/W2/b2 get a separate, unclipped optimizer - see
    # apply_gradient's docstring for why: grad_clip_value AND the main optimizer's own clipnorm
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
                encoded_locs,
                class_ids,
                config["batch_size"],
                config["shuffle_buffer_size"],
                focal_class_ids=focal_class_ids,
                focal_oversample=focal_oversample,
                focal_oversample_map=focal_oversample_map,
            )

        pbar = tqdm.tqdm(total=num_train_steps_per_epoch, dynamic_ncols=True)
        for step, (x_batch_train, y_batch_train) in enumerate(ds):
            pbar.update()
            batch_size_actual = len(x_batch_train)
            if bg_sampling == "target_group":
                n_tg = int(batch_size_actual * bg_tg_fraction)
                n_rand = batch_size_actual - n_tg
                tg_idx = np.random.randint(0, len(bg_pool), size=n_tg)
                parts = [bg_pool[tg_idx]]
                raw_parts = [bg_pool_raw[tg_idx]] if focal_kdtrees else None
                batch_h3 = list(bg_pool_h3[tg_idx]) if bg_pool_h3 is not None else None
                if n_rand > 0 and rand_pool is not None:
                    rand_idx = np.random.randint(0, len(rand_pool), size=n_rand)
                    parts.append(rand_pool[rand_idx])
                    if raw_parts is not None:
                        raw_parts.append(rand_pool_raw[rand_idx])
                    if rand_pool_h3 is not None:
                        batch_h3.extend(rand_pool_h3[rand_idx])
                fake_x = tf.constant(np.concatenate(parts, axis=0))
                fake_raw = np.concatenate(raw_parts, axis=0) if raw_parts is not None else None
            else:
                rand_loc = make_rand_samples_tf(batch_size_actual)
                fake_x = encoder.encode(rand_loc, normalize=False)
                fake_raw = np.array(rand_loc) if focal_kdtrees else None
                batch_h3 = None

            effort_scale = config.get("effort_scale", 0.0)
            if effort_scale > 0 and batch_h3 is not None:
                # build (B_bg, K) weight matrix: effort * (1 - species_present)
                # species present in the H3 cell → weight 0, absent → weight = effort
                # NOTE: effort is raster channel EFFORT_RASTER_CHANNEL, NOT the last encoded
                # column - that assumption broke in v6 when 19 coast_dist*bio interaction
                # channels were appended AFTER effort in the raster, making channel 22 no
                # longer last. fake_x[:, -1] silently grabbed the last interaction channel
                # instead (range ~[-0.6, 3.4], unlike effort's true [0,1]), corrupting
                # effort_bg_matrix with negative "effort" values since v6. EFFORT_COL is the
                # correct fixed index.
                effort_vals = fake_x[:, EFFORT_COL].numpy()  # normalized effort [0,1]
                species_mask = np.ones((batch_size_actual, num_classes), dtype=np.float32)
                for j, cell in enumerate(batch_h3):
                    idxs = cell_to_species.get(cell)
                    if idxs is not None:
                        species_mask[j, idxs] = 0.0  # vectorized: zero all present species at once
                effort_bg_matrix = effort_scale * effort_vals[:, None] * species_mask
            else:
                effort_bg_matrix = None

            if focal_kdtrees and fake_raw is not None:
                # opposite of pos_weight: extra absence penalty scaled by distance from
                # each focal species' own nearest real observation, independent of
                # covariate similarity or other species' effort
                range_penalty_matrix = np.zeros((batch_size_actual, num_classes), dtype=np.float32)
                for fc, tree in focal_kdtrees.items():
                    dist_deg, _ = tree.query(fake_raw)
                    range_penalty_matrix[:, fc] = (
                        np.minimum(dist_deg / RANGE_PENALTY_NORM_DEG, RANGE_PENALTY_CAP)
                        * range_penalty_scale
                    )
                effort_bg_matrix = (
                    range_penalty_matrix if effort_bg_matrix is None
                    else effort_bg_matrix + range_penalty_matrix
                )

            loss = apply_gradient(
                optimizer,
                logits_model,
                x_batch_train,
                y_batch_train,
                fake_x,
                pos_weight,
                effort_bg_matrix=effort_bg_matrix,
                bg_base_weight=bg_base_weight,
                grad_clip_value=grad_clip_value,
                head_optimizer=head_optimizer,
                head_weight_refs=head_weight_refs,
            )

            global_step = (epoch * num_train_steps_per_epoch) + step
            if step % 10 == 0:
                log_entry = {
                    "batch_loss": loss,
                    "learning_rate": optimizer.learning_rate.numpy(),
                }
                wandb.log(log_entry, step=global_step, commit=True)

            pbar.set_description(f" Loss {loss:.4f}")
            epoch_losses.append(loss.numpy())

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
