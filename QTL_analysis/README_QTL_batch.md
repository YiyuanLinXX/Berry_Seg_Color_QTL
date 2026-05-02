# Batch QTL Analysis README

This directory contains scripts for running QTL analysis on one or more phenotype CSV files that share the same structure.

## Files

- `QTL_grapecolor_preliminary.r`: main R script. It reads one genotype file and one phenotype file, runs QTL analysis trait by trait, and writes per-trait results.
- `run_qtl_batch.sh`: bash wrapper for running the R script across multiple phenotype files.
- `genotypes/GE1783genotypeForRqtl_fromLu.csv`: genotype file for `R/qtl`.
- `phenotypes/*.csv`: phenotype files. These can be different filtered versions of the same phenotype table.

## Expected Phenotype Format

Each phenotype CSV must have the same structure:

- Column 1: genotype/vine ID used for matching to the genotype map. In the current file this is `Genotype_Vine_ID`.
- Column 2: alignment-only vine ID. In the current file this is `assigned_vine_id`.
- Columns 3 through `ncol - 3`: numeric phenotype traits to analyze.
- Last 3 columns: instance-count metadata, excluded from QTL analysis.

The script automatically excludes the first two columns and the last three columns. Only numeric trait columns in the middle are analyzed.

## Output Structure

The output path is:

```text
qtl_results/{phenotype_filename}/{trait}/
```

For example, for:

```text
phenotypes/vine_level_all_sides_means_genotype_aligned_60.csv
```

and trait:

```text
lab_b_mean
```

the output folder is:

```text
qtl_results/vine_level_all_sides_means_genotype_aligned_60/lab_b_mean/
```

Each phenotype file also gets a run-level summary:

```text
qtl_results/{phenotype_filename}/run_summary.csv
```

## Per-Trait Output Files

Each trait folder contains:

- `scanone_results.csv`: genome-wide `scanone()` result, including marker, chromosome, position, and LOD score.
- `permutation_results.csv`: permutation results from `scanone(..., n.perm = N_PERM)`.
- `threshold_5pct.csv`: 5% genome-wide significance threshold from the permutation result.
- `peaks_summary.csv`: QTL peak summary from `summary(scanone_result, perms = permutation_result, alpha = 0.05, pvalues = TRUE)`.
- `lod_plot.png`: genome-wide LOD curve with the 5% threshold line.
- `effect_size_summary.csv`: one-row effect-size summary for the peak QTL.
- `genotype_group_means.csv`: phenotype mean, SD, SE, min, max, and sample size for each genotype class at the peak marker.
- `genotype_effect_boxplot.png`: phenotype distribution by peak-marker genotype class.
- `fitqtl_full_model.csv`: full model table from `fitqtl()`.
- `fitqtl_effect_estimates.csv`: estimated QTL effects from `fitqtl()`.

## How QTL Analysis Is Calculated

The script uses the `qtl` R package.

The genotype map is read as a 4-way cross:

```r
read.cross(
  format = "csv",
  file = genotype_file,
  estimate.map = FALSE,
  crosstype = "4way",
  genotypes = NULL,
  alleles = c("A", "B", "C", "D")
)
```

Co-located markers are adjusted with:

```r
jittermap()
```

Genotype probabilities are calculated once per phenotype file:

```r
calc.genoprob(
  map,
  step = 10,
  off.end = 0.0,
  error.prob = 1.0e-4,
  stepwidth = "fixed",
  map.function = "kosambi"
)
```

For each trait, the phenotype values are attached to the genotype map as `trait_value`, then the genome scan is run with:

```r
scanone(map, pheno.col = "trait_value", model = "normal")
```

Permutation thresholds are calculated with:

```r
scanone(map, pheno.col = "trait_value", model = "normal", n.perm = N_PERM)
summary(permutation_result, alpha = 0.05)
```

The main peak is the marker/position with the highest LOD score:

```r
which.max(scan_result$lod)
```

`significant_5pct` is `TRUE` when:

```text
max_lod >= 5% permutation threshold
```

## How Effect Size Is Calculated

The script reports several effect-size measures because each answers a slightly different question.

### `fitqtl_pve_percent`

This is the main QTL effect-size estimate. The script creates a QTL object at the peak chromosome and position:

```r
qtl_obj <- makeqtl(map, chr = peak_chr, pos = peak_pos, what = "prob")
```

Then it fits a one-QTL model:

```r
fitqtl(
  map,
  qtl = qtl_obj,
  pheno.col = "trait_value",
  formula = y ~ Q1,
  method = "hk",
  get.ests = TRUE
)
```

The `%var` value from the `fitqtl()` full model table is saved as:

```text
fitqtl_pve_percent
```

This is the percent of phenotypic variance explained by the peak QTL under the Haley-Knott model.

### `marker_pve_percent`

This is calculated using the nearest peak marker genotype classes. The script finds the nearest marker to the peak position:

```r
peak_marker <- find.marker(map, chr = peak_chr, pos = peak_pos)
```

Then it fits a simple marker-level linear model:

```r
lm(phenotype ~ genotype)
```

The model R-squared is multiplied by 100:

```text
marker_pve_percent = R^2 * 100
```

This is useful as an intuitive marker-level effect size, but `fitqtl_pve_percent` is generally the more QTL-model-based estimate.

### `raw_effect_max_minus_min`

For each genotype class at the peak marker, the script calculates the phenotype mean.

Then:

```text
raw_effect_max_minus_min = highest genotype mean - lowest genotype mean
```

This is in the original trait units.

### `standardized_effect`

The raw effect is standardized by the phenotype standard deviation:

```text
standardized_effect = raw_effect_max_minus_min / sd(phenotype)
```

This makes effects easier to compare across traits with different units or scales.

### Genotype Labels

The genotype codes are labeled as:

```text
1 = AC
2 = BC
3 = AD
4 = BD
```

These labels are used in `genotype_group_means.csv` and `genotype_effect_boxplot.png`.

## Running One Phenotype File

Run with the default phenotype file and 1000 permutations:

```bash
Rscript QTL_grapecolor_preliminary.r
```

Run a specific phenotype file:

```bash
PHENOTYPE_FILE=phenotypes/vine_level_all_sides_means_genotype_aligned_60.csv \
N_PERM=1000 \
Rscript QTL_grapecolor_preliminary.r
```

The default output will be:

```text
qtl_results/vine_level_all_sides_means_genotype_aligned_60/
```

Run one trait only for testing:

```bash
PHENOTYPE_FILE=phenotypes/vine_level_all_sides_means_genotype_aligned_60.csv \
TRAITS=lab_b_mean \
N_PERM=1000 \
Rscript QTL_grapecolor_preliminary.r
```

Run several selected traits:

```bash
PHENOTYPE_FILE=phenotypes/vine_level_all_sides_means_genotype_aligned_60.csv \
TRAITS=lab_b_mean,rgb_r_mean,hsv_s_mean \
N_PERM=1000 \
Rscript QTL_grapecolor_preliminary.r
```

Use a custom output folder:

```bash
PHENOTYPE_FILE=phenotypes/vine_level_all_sides_means_genotype_aligned_60.csv \
OUT_DIR=qtl_results/custom_test_run \
N_PERM=1000 \
Rscript QTL_grapecolor_preliminary.r
```

## Running Multiple Phenotype Files

Edit `run_qtl_batch.sh` and add all phenotype files to this list:

```bash
PHENOTYPE_FILES=(
  "phenotypes/vine_level_all_sides_means_genotype_aligned_20.csv"
  "phenotypes/vine_level_all_sides_means_genotype_aligned_40.csv"
  "phenotypes/vine_level_all_sides_means_genotype_aligned_60.csv"
  "phenotypes/vine_level_all_sides_means_genotype_aligned_80.csv"
  "phenotypes/vine_level_all_sides_means_genotype_aligned_100.csv"
)
```

Then run:

```bash
N_PERM=1000 bash run_qtl_batch.sh
```

The output will be:

```text
qtl_results/vine_level_all_sides_means_genotype_aligned_20/
qtl_results/vine_level_all_sides_means_genotype_aligned_40/
qtl_results/vine_level_all_sides_means_genotype_aligned_60/
qtl_results/vine_level_all_sides_means_genotype_aligned_80/
qtl_results/vine_level_all_sides_means_genotype_aligned_100/
```

Each phenotype output folder will contain one subfolder per trait.

## Quick Test Before a Long Run

Before running all traits with 1000 permutations, it is useful to test one trait with a small number of permutations:

```bash
PHENOTYPE_FILE=phenotypes/vine_level_all_sides_means_genotype_aligned_60.csv \
TRAITS=lab_b_mean \
N_PERM=2 \
OUT_DIR=qtl_results/test_lab_b_mean \
Rscript QTL_grapecolor_preliminary.r
```

If that succeeds, run the full batch:

```bash
N_PERM=1000 bash run_qtl_batch.sh
```

## Notes

- Missing phenotype values are allowed. `R/qtl` will drop individuals with missing phenotype values for that trait.
- The genotype probabilities are calculated once per phenotype file, then reused for all traits in that phenotype file.
- The script stops if the phenotype ID column contains duplicate IDs.
- The script skips non-numeric trait columns.
- The `run_summary.csv` file is the best starting point for comparing traits and filters.
