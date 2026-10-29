# Geo Model Training

## Data Prep

For people outside of iNaturalist, your best bet will be to train using the dataset that was included with the sinr paper. You can download it here: https://github.com/elijahcole/sinr/tree/main/data

For iNat internal folks who want to use our internal exports, look for the spatial_data.csv file in the export. You'll want to convert it to h5 for the training, try duckdb. You'll also need the taxonomy.csv file.

If you want to train with just elevation, generate the elevation data. Intructions TBD.
If you want to train with all bioclim covariates, the instructions are in the sinr dataset download instructions above.
If you want to train without covariates, you're all set.

## Config File

Copy config.yml.example to config.yml and fill in stuff. Keep the training config vars the same if you want to reproduce the sinr paper results.

## Train

$ `python train.py <path_to_config.yml>`

## Eval

These evals are designed to mostly match the evaluations done for the SINR paper.

$ `python sinr_paper_evals.py <path_to_config.yml>`

## Export

`export_coreml_geomodel.py` converts to coreml for iOS deployment, `export_tflite_geomodel.py` converts to tflite for android deployment.

### Tests

$ `pytest` runs 'em.

## Focal-species SINR pipeline (experimental, loarie fork)

An extended SINR training path aimed at scaling well-resolved range maps from a handful of
hand-picked species to thousands, without per-species curation. It lives alongside the original
pipeline above; `train_tf_sinr.py` is unchanged in behavior (it uses `sinr_loss_legacy`).

### What's different from the original SINR code

- **Loss** (`sinr_loss.py`): computed on raw logits via `softplus`, which avoids the vanishing
  gradient the old `-log(p + 1e-5)` form has on confidently-wrong predictions. The positive term
  no longer dilutes signal across all K species. Per-class `pos_weight`. Optional effort-weighted
  absence penalty, applied only where a species is absent from a well-sampled H3 cell.
- **Covariates**: bioclim plus distance-to-coast, a land mask, an observer-effort raster
  (`make_effort_raster.py`) and coast × bioclim interactions.
- **Background sampling**: optional target-group sampling (`--bg_sampling target_group`).
- **Architecture** (`lib/models/geo_model_net.py`): configurable backbone width; a dedicated
  focal-species head (`make_geo_model_net_focal_head`); and `IndependentPerClassHead`, a
  vectorized, fully independent 2-layer MLP per focal class. Shared-bottleneck heads collapsed
  into permanently dead classes at 500-5,000 species; independent heads don't.
- **Training stability** (`train_tf_sinr_test.py`): count-balanced and adaptive per-class
  `pos_weight` (`--pos_weight_adapt`), a separate unclipped optimizer for the independent head
  (global gradient clipping was silently discarding its boosted gradients), `--stability_check`
  auto-stop, per-epoch checkpoints and `--start_epoch` resume.

### Workflow

1. **Data**: build `<prefix>_{locs,class_ids,unique_taxa}.npy` from a production
   `spatial_data.csv` with `preprocess_training_data_v3.py` (two-pass, scales to hundreds of
   millions of rows; use `--exclude_captive --require_community`). `extract_random_species_dataset.py`
   draws a random N-species subset with no curation; `trim_to_exact_taxa.py` and
   `stitch_focal_species.py` align datasets for comparisons. `run_clean_preprocess.sh` is an example.
2. **Focal group**: `select_stratified_group.py` picks species stratified by log observation
   count (`group500_taxa_v41.npy` is the current 500-species group).
3. **Train**: `python train_tf_sinr_test.py --config_file focal_10k_config_v42.yml ...`. The full
   command is in that config's header. `focal_10k_config_v2.yml`-`v42.yml` are the experiment
   history; each header records what changed, why, and what happened.
4. **Evaluate**: `check_population.py --model ... --config ... --group ...` (specificity at
   well-surveyed true-absence H3 cells across the focal group: the metric to trust, since mean
   prediction measures calibration, not discrimination), `check_specificity.py`, and
   `visualize_combined.py` for maps.
5. **Export**: `create_vectors.py` writes land-clipped, thresholded H3 GeoJSON range maps.

### Status (v42, Sep 2026)

5,007 randomly sampled species (230M-observation clean export trimmed to the matched taxon set),
500 with independent heads, 30 epochs. 0/500 focal species broken (v39: 13/500); median
specificity 0.991. Known issues: some cross-continent overprediction, offshore bleed (handled at
export by land-clipping, not yet fixed in the model), and a few persistently weak taxa. Not yet
run at the full ~120k-species scale. Checkpoints, datasets and exports are not in git.

## Todos

- [ ] write script to convert sinr dataset to an inat-style dataset so we don't need separate dataloaders
- [ ] write instructions to create the elevation npy file


