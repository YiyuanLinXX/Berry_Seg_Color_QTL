#!/usr/bin/env bash
set -euo pipefail

# Run all continuous color traits for every depth-retention threshold.
# Execute from the repository root. Override N_PERM for a quick smoke test.

GENOTYPE_FILE="${GENOTYPE_FILE:-data/processed/qtl/genotype_map.csv}"
N_PERM="${N_PERM:-1000}"

for threshold in 20 40 60 80 100; do
  phenotype_file="data/processed/qtl/phenotypes/depth_${threshold}.csv"
  output_dir="results/generated/qtl/depth_${threshold}"

  echo "QTL: depth=${threshold}, permutations=${N_PERM}"
  GENOTYPE_FILE="$GENOTYPE_FILE" \
  PHENOTYPE_FILE="$phenotype_file" \
  OUT_DIR="$output_dir" \
  N_PERM="$N_PERM" \
  Rscript scripts/qtl/run_continuous_qtl.R
done
