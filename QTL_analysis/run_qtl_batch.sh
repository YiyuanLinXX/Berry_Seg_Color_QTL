#!/usr/bin/env bash
set -euo pipefail

# Run all traits in one or more phenotype files.
#
# Usage:
#   bash run_qtl_batch.sh
#
# Optional:
#   N_PERM=100 bash run_qtl_batch.sh
#   N_PERM=1000 bash run_qtl_batch.sh

GENOTYPE_FILE="genotypes/GE1783genotypeForRqtl_fromLu.csv"
N_PERM="${N_PERM:-1000}"

# Add more phenotype CSV files here.
PHENOTYPE_FILES=(
  "phenotypes/vine_level_all_sides_means_genotype_aligned_20.csv"
  "phenotypes/vine_level_all_sides_means_genotype_aligned_40.csv"
  "phenotypes/vine_level_all_sides_means_genotype_aligned_60.csv"
  "phenotypes/vine_level_all_sides_means_genotype_aligned_80.csv"
  "phenotypes/vine_level_all_sides_means_genotype_aligned_100.csv"

)

for PHENOTYPE_FILE in "${PHENOTYPE_FILES[@]}"; do
  base_name="$(basename "$PHENOTYPE_FILE" .csv)"
  OUT_DIR="qtl_results/${base_name}"

  echo "Running QTL analysis"
  echo "  phenotype: ${PHENOTYPE_FILE}"
  echo "  output:    ${OUT_DIR}"
  echo "  n.perm:    ${N_PERM}"

  GENOTYPE_FILE="$GENOTYPE_FILE" \
  PHENOTYPE_FILE="$PHENOTYPE_FILE" \
  OUT_DIR="$OUT_DIR" \
  N_PERM="$N_PERM" \
  Rscript QTL_grapecolor_preliminary.r
done
