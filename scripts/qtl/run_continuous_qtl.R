# Batch QTL analysis for grape color phenotype traits.
#
# Phenotype columns used as traits:
#   - exclude the first two ID/alignment columns
#   - exclude the last three instance-count columns
#
# Optional environment variables:
#   GENOTYPE_FILE=data/processed/qtl/genotype_map.csv
#   PHENOTYPE_FILE=data/processed/qtl/phenotypes/depth_60.csv
#   N_PERM=1000        number of permutations per trait
#   OUT_DIR=qtl_results/<phenotype_filename>
#   TRAITS=lab_b_mean,rgb_r_mean
#   SEED=20260501
#
# Plotting note:
#   QTL statistics are calculated on the extracted phenotype values.
#   Plot y-axes for OpenCV-derived color traits are converted to physical
#   color-space units, e.g. CIE L*a*b*, hue degrees, and percentages.

library(qtl)

use_ggplot <- all(vapply(
  c("ggplot2", "ggrepel", "scales"),
  requireNamespace,
  quietly = TRUE,
  FUN.VALUE = logical(1)
))

sanitize_name <- function(x) {
  x <- gsub("[^A-Za-z0-9._-]+", "_", x)
  x <- gsub("^_+|_+$", "", x)
  ifelse(nchar(x) == 0, "trait", x)
}

genotype_file <- Sys.getenv(
  "GENOTYPE_FILE",
  unset = "data/processed/qtl/genotype_map.csv"
)
phenotype_file <- Sys.getenv(
  "PHENOTYPE_FILE",
  unset = "data/processed/qtl/phenotypes/depth_60.csv"
)
phenotype_base_name <- tools::file_path_sans_ext(basename(phenotype_file))
out_dir <- Sys.getenv(
  "OUT_DIR",
  unset = file.path("results/generated/qtl", sanitize_name(phenotype_base_name))
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

trait_display_info <- function(trait) {
  parts <- strsplit(trait, "_", fixed = TRUE)[[1]]
  color_key <- if (length(parts) >= 2) paste(parts[1], parts[2], sep = "_") else trait
  stat_name <- if (length(parts) >= 3) parts[length(parts)] else ""
  offset_stats <- c("mean", "min", "max", "median", "p5", "p25", "p75", "p95")

  info <- list(
    scale = 1,
    offset = 0,
    raw_effect_scale = 1,
    y_label = trait,
    transform_note = "No display transform applied."
  )

  if (grepl("^rgb_[rgb]$", color_key)) {
    channel <- toupper(sub("^rgb_", "", color_key))
    info$scale <- 1 / 255
    info$raw_effect_scale <- 1 / 255
    info$y_label <- paste0("sRGB ", channel, " (0-1)")
    info$transform_note <- "OpenCV 8-bit RGB value divided by 255."
  } else if (color_key == "hsv_h") {
    info$scale <- 2
    info$raw_effect_scale <- 2
    info$y_label <- "Hue (degrees)"
    info$transform_note <- "OpenCV hue value multiplied by 2 to convert 0-179 to 0-358 degrees."
  } else if (color_key == "hsv_s") {
    info$scale <- 100 / 255
    info$raw_effect_scale <- 100 / 255
    info$y_label <- "Saturation (%)"
    info$transform_note <- "OpenCV saturation value divided by 255 and multiplied by 100."
  } else if (color_key == "hsv_v") {
    info$scale <- 100 / 255
    info$raw_effect_scale <- 100 / 255
    info$y_label <- "Value (%)"
    info$transform_note <- "OpenCV value channel divided by 255 and multiplied by 100."
  } else if (color_key == "lab_l") {
    info$scale <- 100 / 255
    info$raw_effect_scale <- 100 / 255
    info$y_label <- "CIE L*"
    info$transform_note <- "OpenCV L channel divided by 255 and multiplied by 100."
  } else if (color_key == "lab_a") {
    info$scale <- 1
    info$offset <- if (stat_name %in% offset_stats) -128 else 0
    info$raw_effect_scale <- 1
    info$y_label <- "CIE a*"
    info$transform_note <- "OpenCV a channel converted to CIE a* by subtracting 128 for location statistics."
  } else if (color_key == "lab_b") {
    info$scale <- 1
    info$offset <- if (stat_name %in% offset_stats) -128 else 0
    info$raw_effect_scale <- 1
    info$y_label <- "CIE b*"
    info$transform_note <- "OpenCV b channel converted to CIE b* by subtracting 128 for location statistics."
  }

  info
}

to_display_value <- function(x, display_info) {
  x * display_info$scale + display_info$offset
}

to_display_spread <- function(x, display_info) {
  x * display_info$raw_effect_scale
}

theme_qtl_publication <- function(base_size = 11) {
  ggplot2::theme_classic(base_size = base_size) +
    ggplot2::theme(
      text = ggplot2::element_text(color = "#263238"),
      plot.title = ggplot2::element_text(face = "bold", size = base_size + 3),
      axis.title = ggplot2::element_text(face = "bold"),
      axis.text = ggplot2::element_text(color = "#263238"),
      axis.line = ggplot2::element_line(color = "#263238", linewidth = 0.45),
      axis.ticks = ggplot2::element_line(color = "#263238", linewidth = 0.4),
      panel.border = ggplot2::element_rect(color = "#263238", fill = NA, linewidth = 0.55),
      panel.grid.major.y = ggplot2::element_line(color = "#E7ECEF", linewidth = 0.28),
      panel.grid.major.x = ggplot2::element_blank(),
      panel.grid.minor = ggplot2::element_blank(),
      plot.background = ggplot2::element_rect(fill = "white", color = NA),
      panel.background = ggplot2::element_rect(fill = "white", color = NA),
      plot.margin = ggplot2::margin(10, 12, 8, 10)
    )
}

prepare_scan_positions <- function(scan_result) {
  scan_df <- data.frame(
    marker = rownames(scan_result),
    as.data.frame(scan_result),
    row.names = NULL,
    check.names = FALSE
  )
  scan_df$chr <- as.character(scan_df$chr)
  scan_df$chr_num <- suppressWarnings(as.numeric(scan_df$chr))
  chr_levels <- unique(scan_df$chr[order(scan_df$chr_num, scan_df$chr)])
  scan_df$chr <- factor(scan_df$chr, levels = chr_levels)
  scan_df$chr_rainbow <- as.character(scan_df$chr)

  gap <- 7
  offset <- 0
  chr_ranges <- data.frame(
    chr = chr_levels,
    midpoint = NA_real_,
    start = NA_real_,
    end = NA_real_
  )

  for (i in seq_along(chr_levels)) {
    chr_i <- chr_levels[i]
    idx <- scan_df$chr == chr_i
    chr_min <- min(scan_df$pos[idx], na.rm = TRUE)
    chr_max <- max(scan_df$pos[idx], na.rm = TRUE)
    scan_df$cum_pos[idx] <- scan_df$pos[idx] - chr_min + offset
    chr_ranges$start[i] <- offset
    chr_ranges$end[i] <- offset + chr_max - chr_min
    chr_ranges$midpoint[i] <- (chr_ranges$start[i] + chr_ranges$end[i]) / 2
    offset <- offset + chr_max - chr_min + gap
  }

  list(scan = scan_df, chr_ranges = chr_ranges, chr_levels = chr_levels)
}

get_significant_lod_peaks <- function(scan_df, threshold_value) {
  scan_df$.row_id <- seq_len(nrow(scan_df))

  if (is.na(threshold_value)) {
    peak_df <- scan_df[which.max(scan_df$lod), , drop = FALSE]
  } else {
    sig_df <- scan_df[scan_df$lod >= threshold_value, , drop = FALSE]

    if (nrow(sig_df) == 0) {
      peak_df <- scan_df[which.max(scan_df$lod), , drop = FALSE]
    } else {
      peak_rows <- list()
      for (chr_i in unique(as.character(sig_df$chr))) {
        chr_sig <- sig_df[as.character(sig_df$chr) == chr_i, , drop = FALSE]
        chr_sig <- chr_sig[order(chr_sig$.row_id), , drop = FALSE]
        run_id <- cumsum(c(TRUE, diff(chr_sig$.row_id) > 1))

        for (run in unique(run_id)) {
          run_df <- chr_sig[run_id == run, , drop = FALSE]
          peak_rows[[length(peak_rows) + 1]] <- run_df[which.max(run_df$lod), , drop = FALSE]
        }
      }
      peak_df <- do.call(rbind, peak_rows)
    }
  }

  peak_df <- peak_df[order(-peak_df$lod), , drop = FALSE]
  peak_df$peak_rank <- seq_len(nrow(peak_df))
  peak_df
}

plot_scan <- function(scan_result, threshold, trait, file, effect_summary = NULL) {
  if (use_ggplot) {
    scan_parts <- prepare_scan_positions(scan_result)
    scan_df <- scan_parts$scan
    chr_ranges <- scan_parts$chr_ranges
    chr_levels <- scan_parts$chr_levels
    threshold_value <- if (!is.null(threshold)) as.numeric(threshold[1]) else NA_real_
    peak <- scan_df[which.max(scan_df$lod), , drop = FALSE]
    significant_peaks <- get_significant_lod_peaks(scan_df, threshold_value)
    peak_count_label <- if (!is.na(threshold_value)) {
      paste("Significant peaks:", nrow(significant_peaks))
    } else {
      "Top peak shown"
    }
    rainbow_palette <- scales::hue_pal(h = c(15, 375), c = 95, l = 55)(length(chr_levels))
    names(rainbow_palette) <- chr_levels

    significant_peaks$label <- sprintf(
      "Peak: %s, Chr %s, %.2f cM\nLOD %.2f",
      significant_peaks$marker,
      as.character(significant_peaks$chr),
      significant_peaks$pos,
      significant_peaks$lod
    )
    if (!is.null(effect_summary) && nrow(significant_peaks) > 0) {
      top_idx <- which(significant_peaks$marker == effect_summary$peak_marker[1])
      if (length(top_idx) == 1) {
        significant_peaks$label[top_idx] <- sprintf(
          "Peak: %s, Chr %s, %.2f cM\nLOD %.2f; PVE %.1f%%",
          effect_summary$peak_marker[1],
          effect_summary$peak_chr[1],
          effect_summary$peak_pos[1],
          effect_summary$max_lod[1],
          effect_summary$fitqtl_pve_percent[1]
        )
      }
    }

    peak_text <- if (!is.null(effect_summary)) {
      sprintf(
        "%s\nTop: %s, Chr %s, %.2f cM\nLOD %.2f; PVE %.1f%%",
        peak_count_label,
        effect_summary$peak_marker[1],
        effect_summary$peak_chr[1],
        effect_summary$peak_pos[1],
        effect_summary$max_lod[1],
        effect_summary$fitqtl_pve_percent[1]
      )
    } else {
      sprintf("Peak: Chr %s, %.2f cM\nLOD %.2f", peak$chr, peak$pos, peak$lod)
    }

    lod_plot <- ggplot2::ggplot(scan_df, ggplot2::aes(x = cum_pos, y = lod)) +
      ggplot2::geom_rect(
        data = chr_ranges[seq(1, nrow(chr_ranges), by = 2), ],
        ggplot2::aes(xmin = start, xmax = end, ymin = -Inf, ymax = Inf),
        inherit.aes = FALSE,
        fill = "#F6F8FA",
        color = NA
      ) +
      ggplot2::geom_line(ggplot2::aes(group = chr), color = "#B0BEC5", linewidth = 0.22, alpha = 0.7) +
      ggplot2::geom_point(ggplot2::aes(color = chr_rainbow), size = 0.82, alpha = 0.92, show.legend = FALSE) +
      ggplot2::scale_color_manual(values = rainbow_palette) +
      ggplot2::geom_hline(
        ggplot2::aes(yintercept = threshold_value, linetype = "5% threshold"),
        color = "#D1495B",
        linewidth = 0.75,
        na.rm = TRUE
      ) +
      ggplot2::geom_point(
        data = significant_peaks,
        ggplot2::aes(x = cum_pos, y = lod, shape = "Significant peak"),
        inherit.aes = FALSE,
        size = 3.2,
        stroke = 0.8,
        fill = "#F2A541",
        color = "#263238"
      ) +
      ggrepel::geom_label_repel(
        data = significant_peaks,
        ggplot2::aes(x = cum_pos, y = lod, label = label),
        inherit.aes = FALSE,
        size = 3.05,
        label.size = 0.25,
        label.padding = grid::unit(0.16, "lines"),
        box.padding = 0.38,
        point.padding = 0.28,
        min.segment.length = 0,
        segment.color = "#607D8B",
        fill = scales::alpha("white", 0.92),
        color = "#263238",
        max.overlaps = Inf
      ) +
      ggplot2::annotate(
        "label",
        x = max(scan_df$cum_pos, na.rm = TRUE) * 0.02,
        y = max(scan_df$lod, threshold_value, na.rm = TRUE) * 0.98,
        label = peak_text,
        hjust = 0,
        vjust = 1,
        fill = scales::alpha("white", 0.88),
        color = "#263238",
        size = 3.2
      ) +
      ggplot2::scale_linetype_manual(name = NULL, values = c("5% threshold" = "longdash")) +
      ggplot2::scale_shape_manual(name = NULL, values = c("Significant peak" = 21)) +
      ggplot2::scale_x_continuous(
        breaks = chr_ranges$midpoint,
        labels = chr_ranges$chr,
        expand = ggplot2::expansion(mult = c(0.006, 0.02))
      ) +
      ggplot2::scale_y_continuous(expand = ggplot2::expansion(mult = c(0, 0.08))) +
      ggplot2::labs(
        title = paste("QTL scan:", trait),
        x = "Chromosome",
        y = "LOD score"
      ) +
      ggplot2::coord_cartesian(clip = "off") +
      theme_qtl_publication(base_size = 12) +
      ggplot2::theme(
        legend.position = c(0.86, 0.86),
        legend.background = ggplot2::element_rect(fill = scales::alpha("white", 0.88), color = "#B0BEC5"),
        legend.key = ggplot2::element_blank()
      )

    ggplot2::ggsave(file, lod_plot, width = 12, height = 6.8, dpi = 300)
    ggplot2::ggsave(sub("\\.png$", ".svg", file), lod_plot, width = 12, height = 6.8, device = svg)
    return(invisible(TRUE))
  }

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
    display_y_label = trait_display_info(trait)$y_label,
    display_raw_effect_max_minus_min = to_display_spread(raw_effect, trait_display_info(trait)),
    display_transform_note = trait_display_info(trait)$transform_note,
    lowest_mean_genotype = if (nrow(group_means) > 0) group_means$genotype[1] else NA,
    highest_mean_genotype = if (nrow(group_means) > 0) group_means$genotype[nrow(group_means)] else NA,
    fitqtl_error = fitqtl_error,
    check.names = FALSE
  )
  write.csv(effect_summary, file.path(trait_dir, "effect_size_summary.csv"), row.names = FALSE)

  display_info <- trait_display_info(trait)
  group_means_display <- group_means
  group_means_display$mean <- to_display_value(group_means_display$mean, display_info)
  group_means_display$sd <- to_display_spread(group_means_display$sd, display_info)
  group_means_display$se <- to_display_spread(group_means_display$se, display_info)
  group_means_display$min <- to_display_value(group_means_display$min, display_info)
  group_means_display$max <- to_display_value(group_means_display$max, display_info)
  write.csv(
    group_means_display,
    file.path(trait_dir, "genotype_group_means_physical_units.csv"),
    row.names = FALSE
  )

  if (use_ggplot) {
    plot_data <- effect_data
    plot_data$phenotype_display <- to_display_value(plot_data$phenotype, display_info)
    plot_data$genotype <- factor(plot_data$genotype, levels = group_means$genotype)
    group_means_display$genotype <- factor(group_means_display$genotype, levels = group_means$genotype)
    group_means_display$mean_label <- sprintf("%.2f", group_means_display$mean)

    plot_range <- range(plot_data$phenotype_display, na.rm = TRUE)
    info_label <- sprintf(
      "Marker: %s\nLOD: %.2f\nPVE: %.1f%%\nEffect: %.2f %s",
      peak_marker,
      peak_lod,
      fitqtl_pve,
      to_display_spread(raw_effect, display_info),
      display_info$y_label
    )

    genotype_palette <- c("AC" = "#2F5597", "BC" = "#009E9A", "AD" = "#F2A541", "BD" = "#D1495B")
    missing_cols <- setdiff(levels(plot_data$genotype), names(genotype_palette))
    if (length(missing_cols) > 0) {
      extra_cols <- scales::hue_pal(h = c(15, 375), c = 80, l = 55)(length(missing_cols))
      names(extra_cols) <- missing_cols
      genotype_palette <- c(genotype_palette, extra_cols)
    }

    effect_plot <- ggplot2::ggplot(plot_data, ggplot2::aes(x = genotype, y = phenotype_display, fill = genotype)) +
      ggplot2::stat_boxplot(geom = "errorbar", width = 0.28, color = "#263238", linewidth = 0.55) +
      ggplot2::geom_boxplot(
        width = 0.56,
        alpha = 0.65,
        outlier.shape = NA,
        color = "#263238",
        linewidth = 0.5
      ) +
      ggplot2::geom_jitter(
        ggplot2::aes(color = genotype),
        width = 0.16,
        height = 0,
        size = 0.65,
        alpha = 0.26,
        show.legend = FALSE
      ) +
      ggplot2::geom_errorbar(
        data = group_means_display,
        ggplot2::aes(x = genotype, ymin = mean - se, ymax = mean + se),
        inherit.aes = FALSE,
        width = 0.15,
        linewidth = 0.78,
        color = "#263238"
      ) +
      ggplot2::geom_point(
        data = group_means_display,
        ggplot2::aes(x = genotype, y = mean, shape = "Mean ± SE"),
        inherit.aes = FALSE,
        size = 3.1,
        fill = "#FFFFFF",
        color = "#263238",
        stroke = 0.85
      ) +
      ggplot2::geom_text(
        data = group_means_display,
        ggplot2::aes(x = genotype, y = mean, label = mean_label),
        inherit.aes = FALSE,
        vjust = -1.25,
        size = 3.1,
        color = "#263238"
      ) +
      ggplot2::geom_text(
        data = group_means_display,
        ggplot2::aes(x = genotype, y = plot_range[1] - 0.035 * diff(plot_range), label = paste0("n=", n)),
        inherit.aes = FALSE,
        size = 3.0,
        color = "#607D8B"
      ) +
      ggplot2::annotate(
        "label",
        x = Inf,
        y = Inf,
        label = info_label,
        hjust = 1.04,
        vjust = 1.08,
        fill = scales::alpha("white", 0.88),
        color = "#263238",
        size = 3.1
      ) +
      ggplot2::scale_fill_manual(name = "Genotype", values = genotype_palette, drop = FALSE) +
      ggplot2::scale_color_manual(values = genotype_palette, drop = FALSE) +
      ggplot2::scale_shape_manual(name = NULL, values = c("Mean ± SE" = 23)) +
      ggplot2::scale_y_continuous(expand = ggplot2::expansion(mult = c(0.12, 0.12))) +
      ggplot2::labs(
        title = paste("Peak marker effect:", trait),
        x = "Peak-marker genotype",
        y = display_info$y_label
      ) +
      theme_qtl_publication(base_size = 12) +
      ggplot2::theme(
        legend.position = c(0.16, 0.84),
        legend.background = ggplot2::element_rect(fill = scales::alpha("white", 0.88), color = "#B0BEC5"),
        legend.key = ggplot2::element_blank(),
        panel.grid.major.x = ggplot2::element_blank()
      )

    ggplot2::ggsave(file.path(trait_dir, "genotype_effect_boxplot.png"), effect_plot, width = 8.2, height = 6.2, dpi = 300)
    ggplot2::ggsave(file.path(trait_dir, "genotype_effect_boxplot.svg"), effect_plot, width = 8.2, height = 6.2, device = svg)

    return(effect_summary)
  }

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

    scan_df <- as.data.frame(scan_norm)
    max_idx <- which.max(scan_df$lod)
    effect_summary <- estimate_effect_size(
      trait_map = trait_map,
      scan_result = scan_norm,
      threshold = threshold_5pct,
      trait = trait,
      trait_dir = trait_dir
    )

    plot_scan(
      scan_result = scan_norm,
      threshold = threshold_5pct,
      trait = trait,
      file = file.path(trait_dir, "lod_plot.png"),
      effect_summary = effect_summary
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
