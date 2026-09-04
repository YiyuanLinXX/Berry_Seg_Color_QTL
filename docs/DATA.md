# Data inventory and publication boundary

## Included processed data

| Path | Purpose | Approximate size |
| --- | --- | ---: |
| `data/processed/spatial/frame_gps.csv` | Per-frame RTK-GPS log | 0.5 MB |
| `data/processed/spatial/vine_reference.csv` | Aligned vine coordinates | 0.05 MB |
| `data/processed/qtl/genotype_map.csv` | Four-way cross genotype map | 2.8 MB |
| `data/processed/qtl/phenotypes/depth_*.csv` | Vine color traits at five depth thresholds | 6.4 MB |
| `data/processed/qtl/categorical_color.txt` | Prior categorical color phenotype | 0.02 MB |

Checksums are stored in `data/manifest.sha256`. These files are small enough
for ordinary Git; Git LFS is not needed.

## Intentionally excluded

- raw stereo RGB images (about 6.8 GB in the working copy);
- all predicted masks, instance masks, masked crops, and depth arrays;
- FoundationStereo and SAM/SAM-CLIP weights;
- multi-gigabyte random-forest model files;
- local R package libraries, caches, temporary plots, and old result trees;
- the original repository's 13 GB Git object database.

For a public release, place the raw imagery, checkpoints that you are allowed
to redistribute, and any desired intermediate products in Zenodo, Dryad, Box,
or another archival store. Record a DOI or stable URL and SHA-256 values here.
Do not use GitHub releases as the primary store for the full image dataset.

## Suggested external archive layout

```text
berry-color-data-v1/
├── raw/right_camera_rgb/
├── raw/left_camera_rgb/
├── metadata/K_and_baseline.txt
├── checkpoints/berry_sam_clip/README.txt
├── derived/depth/                 # optional
├── derived/masks/                 # optional
└── SHA256SUMS
```

## Data permission note

The software MIT license does not license the data. Before publishing this
repository, confirm with the University of Minnesota breeding-data owner and
all relevant coauthors that the genotype map, phenotype tables, vineyard
coordinates, timestamps, and categorical labels may be made public. Then add
an explicit data license (for example CC BY 4.0 or CC0) and a data-availability
statement. If precise coordinates or timestamps should not be public, replace
the spatial files with documented de-identified versions and host controlled
data separately.
