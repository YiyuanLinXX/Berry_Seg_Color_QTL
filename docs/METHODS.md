# Reproduction workflow

Run commands from the repository root. Paths below can be replaced with local
or archive-downloaded data paths.

## 1. External model source and weights

`scripts/setup_external_dependencies.sh` checks out the exact recorded source
versions. Install each model's own environment by following its upstream
README. Provide these large files locally:

- berry SAM-CLIP checkpoint directory containing `args.json` and
  `checkpoint_best.pth`;
- FoundationStereo checkpoint directory containing `cfg.yaml` and
  `model_best_bp2.pth`;
- two-line camera intrinsics file: flattened 3 x 3 matrix, then stereo baseline
  in metres.

## 2. Berry segmentation

```bash
python external/SAM-CLIP/inference_sam_clip.py \
  --checkpoint_dir /path/to/berry_checkpoint \
  --image_dir /path/to/right_camera_rgb \
  --output_dir results/generated/masks \
  --text_prompt "grape berry cluster"
```

## 3. Stereo depth

```bash
python scripts/depth/run_foundation_stereo_pairs.py \
  --foundation_stereo_root external/FoundationStereo \
  --left_dir /path/to/left_camera_rgb \
  --right_dir /path/to/right_camera_rgb \
  --intrinsic_file /path/to/K_and_baseline.txt \
  --ckpt_dir /path/to/model_best_bp2.pth \
  --out_dir results/generated/depth \
  --output_view right \
  --get_pc 0
```

The right-view implementation flips the swapped pair before inference. This
adapter is derived from the NVIDIA demo and therefore uses the FoundationStereo
license, not this repository's MIT license.

## 4. Depth filtering and color features

For the paper's retention thresholds (`20`, `40`, `60`, `80`; `100` is the
unfiltered reference), run:

```bash
python scripts/vision/filter_masks_by_depth.py \
  --mask_dir results/generated/masks \
  --depth_dir results/generated/depth \
  --csv_file data/processed/spatial/frame_gps.csv \
  --out_dir results/generated/phenotyping/berry_depth_filtered_60 \
  --keep_percentile 60

python scripts/vision/export_mask_instances.py \
  --mask_dir results/generated/phenotyping/berry_depth_filtered_60 \
  --out_dir results/generated/phenotyping/berry_instance_depth_filtered_60 \
  --min_component_area 200 \
  --output_layout flat

python scripts/vision/extract_instance_color_features.py \
  --instance_dir results/generated/phenotyping/berry_instance_depth_filtered_60 \
  --image_dir /path/to/right_camera_rgb \
  --out_csv results/generated/features/berry_cluster_instance_color_features_depth_filter_60.csv \
  --instance_layout flat \
  --save_masked_crop 0
```

Repeat for each threshold. Saving masked crops is useful for QA but produces a
large derived dataset and is disabled in the compact example.

## 5. Frame-to-vine mapping

The study configuration was: camera forward offset `-50 cm`, lateral offset
`57 cm`, horizontal FOV `60.5 degrees`, and maximum viewing distance `1.5 m`.

```bash
bash scripts/run_spatial_mapping.sh
```

This produces a per-frame coverage table in `results/generated/spatial/`.

## 6. Instance assignment and vine aggregation

```bash
python scripts/spatial/assign_berry_instances_to_vines.py \
  --feature-dir results/generated/features \
  --strategy-root results/generated/phenotyping \
  --aligned-csv data/processed/spatial/vine_reference.csv \
  --coverage-csv results/generated/spatial/frame_gps_camera_coverage.csv

python scripts/analysis/analyze_berry_vine_level.py \
  --strategy-root results/generated/phenotyping
```

The first script converts each instance bounding box into an interval along
the visible row and assigns it by interval overlap, falling back to the nearest
vine centre when needed. The second script averages accepted instance features
within each vine, with north/south counts retained for QA.

Depth-filter sensitivity summaries and the paper-style multi-panel figure can
then be rebuilt with:

```bash
python scripts/analysis/analyze_depth_filter_coverage.py
python scripts/analysis/analyze_depth_filter_feature_stability.py
python scripts/analysis/make_final_depth_filter_multi_panel_figure.py
```

## 7. QTL mapping

The included phenotype CSVs correspond to all five depth-retention settings.
To reproduce the main continuous scan:

```bash
N_PERM=1000 TRAITS=lab_b_mean \
PHENOTYPE_FILE=data/processed/qtl/phenotypes/depth_60.csv \
OUT_DIR=results/generated/qtl/depth_60_lab_b_mean \
Rscript scripts/qtl/run_continuous_qtl.R
```

To reproduce the categorical comparison:

```bash
N_PERM=1000 Rscript scripts/qtl/run_binary_qtl.R
```

Random seeds default to `20260501`. Continuous scans use the normal
Haley-Knott model; categorical scans use the binary Haley-Knott model.
