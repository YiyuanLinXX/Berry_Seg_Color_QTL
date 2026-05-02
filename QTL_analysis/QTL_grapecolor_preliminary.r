# Batch QTL analysis for grape color phenotype traits.
#
# Phenotype columns used as traits:
#   - exclude the first two ID/alignment columns
#   - exclude the last three instance-count columns
#
# Optional environment variables:
#   GENOTYPE_FILE=genotypes/GE1783genotypeForRqtl_fromLu.csv
#   PHENOTYPE_FILE=phenotypes/vine_level_all_sides_means_genotype_aligned_60.csv
#   N_PERM=1000        number of permutations per trait
#   OUT_DIR=qtl_results/<phenotype_filename>
#   TRAITS=lab_b_mean,rgb_r_mean
#   SEED=20260501

library(qtl)

sanitize_name <- function(x) {
  x <- gsub("[^A-Za-z0-9._-]+", "_", x)
  x <- gsub("^_+|_+$", "", x)
  ifelse(nchar(x) == 0, "trait", x)
}

genotype_file <- Sys.getenv(
  "GENOTYPE_FILE",
  unset = "genotypes/GE1783genotypeForRqtl_fromLu.csv"
)
phenotype_file <- Sys.getenv(
  "PHENOTYPE_FILE",
  unset = "phenotypes/vine_level_all_sides_means_genotype_aligned_60.csv"
)
phenotype_base_name <- tools::file_path_sans_ext(basename(phenotype_file))
out_dir <- Sys.getenv(
  "OUT_DIR",
  unset = file.path("qtl_results", sanitize_name(phenotype_base_name))
)
n_perm <- as.integer(Sys.getenv("N_PERM", unset = "1000"))
seed <- as.integer(Sys.getenv("SEED", unset = "20260501"))
requested_traits <- Sys.getenv("TRAITS", unset = "")

if (is.na(n_perm) || n_perm < 0) {
  stop("N_PERM must be a non-negative integer.")
}

set.seed(seed)
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

message("Reading genotype map: ", genotype_file)
map <- read.cross(
  format = "csv",
  file = genotype_file,
  estimate.map = FALSE,
  crosstype = "4way",
  genotypes = NULL,
  alleles = c("A", "B", "C", "D")
)

map <- jittermap(map)

if (!"ID." %in% names(map$pheno)) {
  stop("Expected genotype map phenotype ID column named 'ID.'. Found: ",
       paste(names(map$pheno), collapse = ", "))
}

map$pheno$ID. <- as.character(map$pheno$ID.)

message("Reading phenotype table: ", phenotype_file)
pheno <- read.csv(phenotype_file, header = TRUE, check.names = FALSE)

if (ncol(pheno) < 6) {
  stop("Phenotype file must have at least 6 columns: 2 ID columns, traits, and 3 count columns.")
}

id_col <- names(pheno)[1]
alignment_col <- names(pheno)[2]
count_cols <- tail(names(pheno), 3)
trait_cols <- names(pheno)[3:(ncol(pheno) - 3)]

if (requested_traits != "") {
  selected_traits <- trimws(strsplit(requested_traits, ",", fixed = TRUE)[[1]])
  missing_traits <- setdiff(selected_traits, trait_cols)
  if (length(missing_traits) > 0) {
    stop("Requested traits are not valid phenotype trait columns: ",
         paste(missing_traits, collapse = ", "))
  }
  trait_cols <- selected_traits
}

non_numeric_traits <- trait_cols[!vapply(pheno[trait_cols], is.numeric, logical(1))]
if (length(non_numeric_traits) > 0) {
  warning("Skipping non-numeric trait columns: ", paste(non_numeric_traits, collapse = ", "))
  trait_cols <- setdiff(trait_cols, non_numeric_traits)
}

if (length(trait_cols) == 0) {
  stop("No numeric trait columns available for QTL analysis.")
}

duplicated_pheno_ids <- unique(pheno[[id_col]][duplicated(pheno[[id_col]])])
if (length(duplicated_pheno_ids) > 0) {
  stop("Phenotype file contains duplicated genotype IDs in ", id_col, ": ",
       paste(head(duplicated_pheno_ids, 10), collapse = ", "))
}

message("ID column used for matching: ", id_col)
message("Alignment-only column excluded: ", alignment_col)
message("Count columns excluded: ", paste(count_cols, collapse = ", "))
message("Number of traits to analyze: ", length(trait_cols))
message("Permutations per trait: ", n_perm)

missing_in_map <- setdiff(pheno[[id_col]], map$pheno$ID.)
if (length(missing_in_map) > 0) {
  message("Dropping phenotype rows not present in genotype map: ", length(missing_in_map))
  pheno <- pheno[pheno[[id_col]] %in% map$pheno$ID., , drop = FALSE]
}

missing_in_pheno <- setdiff(map$pheno$ID., pheno[[id_col]])
if (length(missing_in_pheno) > 0) {
  message("Genotype map IDs without phenotype rows: ", length(missing_in_pheno))
}

pheno <- pheno[match(map$pheno$ID., pheno[[id_col]]), , drop = FALSE]
pheno[[id_col]] <- map$pheno$ID.

message("Calculating genotype probabilities once for all traits.")
map <- calc.genoprob(
  map,
  step = 10,
  off.end = 0.0,
  error.prob = 1.0e-4,
  stepwidth = "fixed",
  map.function = "kosambi"
)

write_scan_csv <- function(scan_result, file) {
  scan_df <- data.frame(
    marker = rownames(scan_result),
    scan_result,
    row.names = NULL,
    check.names = FALSE
  )
  write.csv(scan_df, file = file, row.names = FALSE)
}

write_threshold_csv <- function(threshold, file) {
  threshold_df <- data.frame(
    alpha = 0.05,
    threshold = as.numeric(threshold[1])
  )
  write.csv(threshold_df, file = file, row.names = FALSE)
}

plot_scan <- function(scan_result, threshold, trait, file) {
  png(file, width = 1200, height = 700, res = 120)
  on.exit(dev.off(), add = TRUE)

  plot(
    scan_result,
    col = "darkblue",
    main = paste("Normal Model QTL:", trait),
    ylab = "LOD Score"
  )

  if (!is.null(threshold)) {
    abline(h = threshold[1], col = "red", lwd = 2, lty = 2)
    legend(
      "topright",
      legend = c("LOD Score", "5% Threshold"),
      col = c("darkblue", "red"),
      lty = c(1, 2),
      lwd = c(1, 2)
    )
  }
}

genotype_label <- function(genotype_code) {
  labels <- c("1" = "AC", "2" = "BC", "3" = "AD", "4" = "BD")
  genotype_code <- as.character(genotype_code)
  ifelse(genotype_code %in% names(labels), labels[genotype_code], genotype_code)
}

estimate_effect_size <- function(trait_map, scan_result, threshold, trait, trait_dir) {
  scan_df <- as.data.frame(scan_result)
  max_idx <- which.max(scan_df$lod)
  peak_chr <- as.character(scan_df$chr[max_idx])
  peak_pos <- scan_df$pos[max_idx]
  peak_lod <- scan_df$lod[max_idx]
  threshold_value <- if (!is.null(threshold)) as.numeric(threshold[1]) else NA_real_
  significant_5pct <- if (!is.na(threshold_value)) peak_lod >= threshold_value else NA

  peak_marker <- find.marker(trait_map, chr = peak_chr, pos = peak_pos)
  marker_genotypes <- pull.geno(trait_map)[, peak_marker]
  effect_data <- data.frame(
    ID. = trait_map$pheno$ID.,
    genotype_code = marker_genotypes,
    genotype = genotype_label(marker_genotypes),
    phenotype = trait_map$pheno$trait_value,
    check.names = FALSE
  )
  effect_data <- effect_data[!is.na(effect_data$genotype_code) & !is.na(effect_data$phenotype), ]
  effect_data$genotype <- factor(effect_data$genotype)

  group_split <- split(effect_data$phenotype, effect_data$genotype)
  group_means <- data.frame(
    genotype = names(group_split),
    n = vapply(group_split, length, integer(1)),
    mean = vapply(group_split, mean, numeric(1)),
    sd = vapply(group_split, sd, numeric(1)),
    se = vapply(group_split, function(x) sd(x) / sqrt(length(x)), numeric(1)),
    min = vapply(group_split, min, numeric(1)),
    max = vapply(group_split, max, numeric(1)),
    row.names = NULL,
    check.names = FALSE
  )
  group_means <- group_means[order(group_means$mean), , drop = FALSE]
  write.csv(group_means, file.path(trait_dir, "genotype_group_means.csv"), row.names = FALSE)

  lm_r2 <- NA_real_
  lm_adj_r2 <- NA_real_
  lm_pvalue <- NA_real_
  raw_effect <- NA_real_
  standardized_effect <- NA_real_

  if (nrow(group_means) >= 2) {
    phenotype_sd <- sd(effect_data$phenotype)
    raw_effect <- max(group_means$mean) - min(group_means$mean)
    standardized_effect <- raw_effect / phenotype_sd

    marker_lm <- lm(phenotype ~ genotype, data = effect_data)
    lm_summary <- summary(marker_lm)
    lm_r2 <- lm_summary$r.squared
    lm_adj_r2 <- lm_summary$adj.r.squared
    lm_anova <- anova(marker_lm)
    if ("Pr(>F)" %in% names(lm_anova) && nrow(lm_anova) >= 1) {
      lm_pvalue <- lm_anova[1, "Pr(>F)"]
    }
  }

  fitqtl_lod <- NA_real_
  fitqtl_pve <- NA_real_
  fitqtl_pvalue_f <- NA_real_
  fitqtl_error <- NA_character_

  tryCatch({
    qtl_obj <- makeqtl(trait_map, chr = peak_chr, pos = peak_pos, what = "prob")
    qtl_fit <- fitqtl(
      trait_map,
      qtl = qtl_obj,
      pheno.col = "trait_value",
      formula = y ~ Q1,
      method = "hk",
      get.ests = TRUE
    )
    qtl_summary <- summary(qtl_fit)

    full_model <- as.data.frame(qtl_summary$result.full, check.names = FALSE)
    full_model$term <- rownames(full_model)
    full_model <- full_model[, c("term", setdiff(names(full_model), "term"))]
    write.csv(full_model, file.path(trait_dir, "fitqtl_full_model.csv"), row.names = FALSE)

    if (!is.null(qtl_summary$ests)) {
      estimates <- as.data.frame(qtl_summary$ests, check.names = FALSE)
      estimates$term <- rownames(estimates)
      estimates <- estimates[, c("term", setdiff(names(estimates), "term"))]
      write.csv(estimates, file.path(trait_dir, "fitqtl_effect_estimates.csv"), row.names = FALSE)
    }

    if ("Model" %in% rownames(qtl_summary$result.full)) {
      fitqtl_lod <- qtl_summary$result.full["Model", "LOD"]
      fitqtl_pve <- qtl_summary$result.full["Model", "%var"]
      if ("Pvalue(F)" %in% colnames(qtl_summary$result.full)) {
        fitqtl_pvalue_f <- qtl_summary$result.full["Model", "Pvalue(F)"]
      }
    }
  }, error = function(e) {
    fitqtl_error <<- conditionMessage(e)
  })

  lod_pve_approx <- 100 * (1 - 10^((-2 * peak_lod) / nrow(effect_data)))

  effect_summary <- data.frame(
    trait = trait,
    peak_chr = peak_chr,
    peak_pos = peak_pos,
    peak_marker = peak_marker,
    max_lod = peak_lod,
    threshold_5pct = threshold_value,
    significant_5pct = significant_5pct,
    non_missing_n = sum(!is.na(trait_map$pheno$trait_value)),
    marker_effect_n = nrow(effect_data),
    marker_group_count = nrow(group_means),
    marker_lm_r2 = lm_r2,
    marker_lm_adj_r2 = lm_adj_r2,
    marker_lm_pvalue = lm_pvalue,
    marker_pve_percent = lm_r2 * 100,
    fitqtl_lod = fitqtl_lod,
    fitqtl_pve_percent = fitqtl_pve,
    fitqtl_pvalue_f = fitqtl_pvalue_f,
    lod_pve_approx_percent = lod_pve_approx,
    raw_effect_max_minus_min = raw_effect,
    standardized_effect = standardized_effect,
    lowest_mean_genotype = if (nrow(group_means) > 0) group_means$genotype[1] else NA,
    highest_mean_genotype = if (nrow(group_means) > 0) group_means$genotype[nrow(group_means)] else NA,
    fitqtl_error = fitqtl_error,
    check.names = FALSE
  )
  write.csv(effect_summary, file.path(trait_dir, "effect_size_summary.csv"), row.names = FALSE)

  png(file.path(trait_dir, "genotype_effect_boxplot.png"), width = 900, height = 700, res = 120)
  on.exit(dev.off(), add = TRUE)
  boxplot(
    phenotype ~ genotype,
    data = effect_data,
    col = "gray90",
    border = "gray35",
    main = paste("Peak Marker Effect:", trait),
    xlab = paste("Peak marker", peak_marker),
    ylab = trait
  )
  stripchart(
    phenotype ~ genotype,
    data = effect_data,
    method = "jitter",
    pch = 16,
    cex = 0.45,
    col = rgb(0.1, 0.25, 0.55, 0.35),
    vertical = TRUE,
    add = TRUE
  )

  effect_summary
}

summary_rows <- list()

for (trait in trait_cols) {
  trait_dir <- file.path(out_dir, sanitize_name(trait))
  dir.create(trait_dir, recursive = TRUE, showWarnings = FALSE)

  message("Analyzing trait: ", trait)

  values <- pheno[[trait]]
  non_missing_n <- sum(!is.na(values))

  if (non_missing_n < 3 || length(unique(values[!is.na(values)])) < 2) {
    warning("Skipping trait with too few usable or variable values: ", trait)
    summary_rows[[trait]] <- data.frame(
      trait = trait,
      status = "skipped",
      non_missing_n = non_missing_n,
      max_lod = NA_real_,
      max_lod_chr = NA,
      max_lod_pos = NA_real_,
      threshold_5pct = NA_real_,
      significant_5pct = NA,
      peak_marker = NA,
      marker_pve_percent = NA_real_,
      fitqtl_pve_percent = NA_real_,
      raw_effect_max_minus_min = NA_real_,
      standardized_effect = NA_real_,
      error = "Too few usable or variable values"
    )
    next
  }

  trait_map <- map
  trait_map$pheno <- data.frame(
    ID. = map$pheno$ID.,
    trait_value = values,
    check.names = FALSE
  )

  result <- tryCatch({
    scan_norm <- scanone(trait_map, pheno.col = "trait_value", model = "normal")

    perm_norm <- NULL
    threshold_5pct <- NULL
    if (n_perm > 0) {
      perm_norm <- scanone(
        trait_map,
        pheno.col = "trait_value",
        model = "normal",
        n.perm = n_perm
      )
      threshold_5pct <- summary(perm_norm, alpha = 0.05)
    }

    write_scan_csv(scan_norm, file.path(trait_dir, "scanone_results.csv"))

    if (!is.null(perm_norm)) {
      write.csv(
        data.frame(permutation = seq_len(nrow(perm_norm)), perm_norm, check.names = FALSE),
        file = file.path(trait_dir, "permutation_results.csv"),
        row.names = FALSE
      )
      write_threshold_csv(threshold_5pct, file.path(trait_dir, "threshold_5pct.csv"))
    }

    peaks <- if (!is.null(perm_norm)) {
      summary(scan_norm, perms = perm_norm, alpha = 0.05, pvalues = TRUE)
    } else {
      summary(scan_norm)
    }

    write.csv(
      data.frame(marker = rownames(peaks), peaks, row.names = NULL, check.names = FALSE),
      file = file.path(trait_dir, "peaks_summary.csv"),
      row.names = FALSE
    )

    plot_scan(
      scan_result = scan_norm,
      threshold = threshold_5pct,
      trait = trait,
      file = file.path(trait_dir, "lod_plot.png")
    )

    scan_df <- as.data.frame(scan_norm)
    max_idx <- which.max(scan_df$lod)
    effect_summary <- estimate_effect_size(
      trait_map = trait_map,
      scan_result = scan_norm,
      threshold = threshold_5pct,
      trait = trait,
      trait_dir = trait_dir
    )

    data.frame(
      trait = trait,
      status = "ok",
      non_missing_n = non_missing_n,
      max_lod = scan_df$lod[max_idx],
      max_lod_chr = as.character(scan_df$chr[max_idx]),
      max_lod_pos = scan_df$pos[max_idx],
      threshold_5pct = if (!is.null(threshold_5pct)) threshold_5pct[1] else NA_real_,
      significant_5pct = effect_summary$significant_5pct,
      peak_marker = effect_summary$peak_marker,
      marker_pve_percent = effect_summary$marker_pve_percent,
      fitqtl_pve_percent = effect_summary$fitqtl_pve_percent,
      raw_effect_max_minus_min = effect_summary$raw_effect_max_minus_min,
      standardized_effect = effect_summary$standardized_effect,
      error = NA_character_
    )
  }, error = function(e) {
    warning("Trait failed: ", trait, " | ", conditionMessage(e))
    data.frame(
      trait = trait,
      status = "failed",
      non_missing_n = non_missing_n,
      max_lod = NA_real_,
      max_lod_chr = NA,
      max_lod_pos = NA_real_,
      threshold_5pct = NA_real_,
      significant_5pct = NA,
      peak_marker = NA,
      marker_pve_percent = NA_real_,
      fitqtl_pve_percent = NA_real_,
      raw_effect_max_minus_min = NA_real_,
      standardized_effect = NA_real_,
      error = conditionMessage(e)
    )
  })

  summary_rows[[trait]] <- result
}

run_summary <- do.call(rbind, summary_rows)
write.csv(run_summary, file = file.path(out_dir, "run_summary.csv"), row.names = FALSE)

message("Done. Results written to: ", normalizePath(out_dir))
