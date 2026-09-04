#!/usr/bin/env bash
set -euo pipefail

python3 scripts/spatial/map_camera_frames_to_vines.py \
  data/processed/spatial/frame_gps.csv \
  --aligned-csv data/processed/spatial/vine_reference.csv \
  --x-cm -50 \
  --y-cm 57 \
  --fov-deg 60.5 \
  --max-distance-m 1.5 \
  --neighbor-threshold-m 2.0 \
  --edge-extension-m 1.0 \
  --row-end-margin-m 2.0 \
  --parallel-angle-threshold-deg 45.0 \
  --min-vine-coverage-percent 0 \
  --output-dir results/generated/spatial
