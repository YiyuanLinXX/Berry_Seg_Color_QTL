# Data dictionary

## Spatial inputs

`frame_gps.csv` contains one row per acquired frame. Required fields are
`Frame ID`, `Latitude`, and `Longitude`; acquisition and ROS time fields are
retained for traceability.

`vine_reference.csv` contains one row per vine. `Name` follows
`row_<row number>_<vine number>`. `Longitude` and `Latitude` are the aligned
RTK coordinates used to construct row axes and vine intervals.

## Genotype map

`genotype_map.csv` uses the R/qtl four-way-cross CSV layout:

- row 1: marker names after the phenotype identifier column;
- row 2: chromosome number for each marker;
- row 3: genetic position in centimorgans;
- remaining rows: individual ID and marker genotype calls.

The four genotype states map to `AC`, `BC`, `AD`, and `BD` in the analysis.

## Continuous phenotype tables

Each `depth_<tau>.csv` row represents one vine after retaining the nearest
`tau` percent of scene depth. Columns are:

- `Genotype_Vine_ID`: ID used to join the R/qtl map;
- `assigned_vine_id`: spatial vine identifier;
- `<space>_<channel>_<statistic>`: instance-level color summaries averaged
  within the vine;
- `instance_count`, `n_side_instance_count`, `s_side_instance_count`: QA
  counts.

Color spaces are RGB, OpenCV HSV, and OpenCV Lab. The QTL scan uses stored
values exactly. Plotting converts RGB to 0–1, hue to degrees, saturation/value
to percent, L to CIE L*, and offsets Lab a/b location statistics by 128.

## Categorical phenotype

`categorical_color.txt` has `TaxaID` and `Color`. The binary comparison script
defines Noir as `Color == 2` and White as `Color == 0`; other records form the
respective negative classes.
