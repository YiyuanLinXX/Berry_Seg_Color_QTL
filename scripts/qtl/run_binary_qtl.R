library(qtl)

use_ggplot <- all(vapply(
  c("ggplot2", "ggrepel", "scales"),
  requireNamespace,
  quietly = TRUE,
  FUN.VALUE = logical(1)
))

if (!use_ggplot) {
  stop("ggplot2, ggrepel, and scales are required for the publication-style plots.")
}

genotype_file <- Sys.getenv(
  "GENOTYPE_FILE",
  unset = "data/processed/qtl/genotype_map.csv"
)
phenotype_file <- Sys.getenv(
  "BINARY_PHENOTYPE_FILE",
  unset = "data/processed/qtl/categorical_color.txt"
)
out_dir <- Sys.getenv("BINARY_OUT_DIR", unset = "results/generated/qtl_binary")
n_perm <- as.integer(Sys.getenv("N_PERM", unset = "1000"))
seed <- as.integer(Sys.getenv("SEED", unset = "20260501"))

if (is.na(n_perm) || n_perm < 0) {
  stop("N_PERM must be a non-negative integer.")
}

set.seed(seed)
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

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

write_scan_csv <- function(scan_result, file) {
  scan_df <- data.frame(
    marker = rownames(scan_result),
    scan_result,
    row.names = NULL,
    check.names = FALSE
  )
  write.csv(scan_df, file = file, row.names = FALSE)
}

plot_lod <- function(scan_result, threshold_value, trait_label, trait_dir) {
  scan_parts <- prepare_scan_positions(scan_result)
  scan_df <- scan_parts$scan
  chr_ranges <- scan_parts$chr_ranges
  chr_levels <- scan_parts$chr_levels
  peaks <- get_significant_lod_peaks(scan_df, threshold_value)
  rainbow_palette <- scales::hue_pal(h = c(15, 375), c = 95, l = 55)(length(chr_levels))
  names(rainbow_palette) <- chr_levels

  peaks$label <- sprintf(
    "Peak: %s, Chr %s, %.2f cM\nLOD %.2f",
    peaks$marker,
    as.character(peaks$chr),
    peaks$pos,
    peaks$lod
  )

  info_label <- sprintf(
    "Model: binary HK\n5%% threshold: %.2f\nSignificant peaks: %s",
    threshold_value,
    sum(peaks$lod >= threshold_value)
  )

  p <- ggplot2::ggplot(scan_df, ggplot2::aes(x = cum_pos, y = lod)) +
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
      linewidth = 0.75
    ) +
    ggplot2::geom_point(
      data = peaks,
      ggplot2::aes(x = cum_pos, y = lod, shape = "Significant peak"),
      inherit.aes = FALSE,
      size = 3.2,
      stroke = 0.8,
      fill = "#F2A541",
      color = "#263238"
    ) +
    ggrepel::geom_label_repel(
      data = peaks,
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
      label = info_label,
      hjust = 0,
      vjust = 1,
      fill = scales::alpha("white", 0.88),
      color = "#263238",
      size = 3.15
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
      title = paste("Binary QTL scan:", trait_label),
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

  ggplot2::ggsave(file.path(trait_dir, "lod_plot.png"), p, width = 12, height = 6.8, dpi = 300)
  ggplot2::ggsave(file.path(trait_dir, "lod_plot.svg"), p, width = 12, height = 6.8, device = svg)

  peaks_out <- data.frame(
    marker = peaks$marker,
    chr = as.character(peaks$chr),
    pos = peaks$pos,
    lod = peaks$lod,
    above_threshold = peaks$lod >= threshold_value,
    row.names = NULL
  )
  write.csv(peaks_out, file.path(trait_dir, "significant_peak_labels.csv"), row.names = FALSE)
}

message("Reading genotype map: ", genotype_file)
map <- read.cross(
  "csv",
  file = genotype_file,
  estimate.map = FALSE,
  crosstype = "4way",
  genotypes = NULL,
  alleles = c("A", "B", "C", "D")
)
map <- jittermap(map)
map$pheno$ID. <- as.character(map$pheno$ID.)

message("Reading binary phenotype source: ", phenotype_file)
pheno <- read.table(phenotype_file, header = TRUE)
pheno$TaxaID <- as.character(pheno$TaxaID)
pheno$Color <- as.numeric(as.character(pheno$Color))

pheno <- pheno[pheno$TaxaID %in% map$pheno$ID., , drop = FALSE]
pheno <- pheno[match(map$pheno$ID., pheno$TaxaID), , drop = FALSE]

map$pheno <- data.frame(
  ID. = map$pheno$ID.,
  Color = pheno$Color,
  check.names = FALSE
)

message("Calculating genotype probabilities.")
map <- calc.genoprob(
  map,
  step = 10,
  off.end = 0.0,
  error.prob = 1e-4,
  stepwidth = "fixed",
  map.function = "kosambi"
)

binary_traits <- list(
  noir_vs_no_noir = list(
    label = "Noir vs no noir",
    values = ifelse(map$pheno$Color == 2, 1, 0),
    positive_class = "Noir",
    rule = "Color == 2"
  ),
  white_vs_no_white = list(
    label = "White vs no white",
    values = ifelse(map$pheno$Color == 0, 1, 0),
    positive_class = "White",
    rule = "Color == 0"
  )
)

summary_rows <- list()

for (trait_name in names(binary_traits)) {
  trait <- binary_traits[[trait_name]]
  trait_dir <- file.path(out_dir, trait_name)
  dir.create(trait_dir, recursive = TRUE, showWarnings = FALSE)

  message("Running binary QTL: ", trait$label)
  trait_map <- map
  trait_map$pheno$Color_binary <- trait$values

  scan_bin <- scanone(
    trait_map,
    pheno.col = "Color_binary",
    model = "binary",
    method = "hk"
  )

  perm_bin <- scanone(
    trait_map,
    pheno.col = "Color_binary",
    model = "binary",
    method = "hk",
    n.perm = n_perm
  )

  threshold_5pct <- as.numeric(summary(perm_bin, alpha = 0.05)[1])
  scan_df <- as.data.frame(scan_bin)
  max_idx <- which.max(scan_df$lod)

  write_scan_csv(scan_bin, file.path(trait_dir, "scanone_results.csv"))
  write.csv(
    data.frame(permutation = seq_len(nrow(perm_bin)), perm_bin, check.names = FALSE),
    file.path(trait_dir, "permutation_results.csv"),
    row.names = FALSE
  )
  write.csv(
    data.frame(alpha = 0.05, threshold = threshold_5pct),
    file.path(trait_dir, "threshold_5pct.csv"),
    row.names = FALSE
  )
  write.csv(
    data.frame(
      phenotype = trait_name,
      label = trait$label,
      positive_class = trait$positive_class,
      binary_rule = trait$rule,
      n_0 = sum(trait$values == 0, na.rm = TRUE),
      n_1 = sum(trait$values == 1, na.rm = TRUE)
    ),
    file.path(trait_dir, "binary_trait_definition.csv"),
    row.names = FALSE
  )

  peak_summary <- summary(scan_bin, threshold = threshold_5pct)
  write.csv(
    data.frame(marker = rownames(peak_summary), peak_summary, row.names = NULL, check.names = FALSE),
    file.path(trait_dir, "peaks_summary.csv"),
    row.names = FALSE
  )

  plot_lod(scan_bin, threshold_5pct, trait$label, trait_dir)

  summary_rows[[trait_name]] <- data.frame(
    phenotype = trait_name,
    label = trait$label,
    positive_class = trait$positive_class,
    binary_rule = trait$rule,
    n_0 = sum(trait$values == 0, na.rm = TRUE),
    n_1 = sum(trait$values == 1, na.rm = TRUE),
    max_lod = scan_df$lod[max_idx],
    max_lod_chr = as.character(scan_df$chr[max_idx]),
    max_lod_pos = scan_df$pos[max_idx],
    threshold_5pct = threshold_5pct,
    significant = scan_df$lod[max_idx] >= threshold_5pct
  )
}

run_summary <- do.call(rbind, summary_rows)
write.csv(run_summary, file.path(out_dir, "binary_run_summary.csv"), row.names = FALSE)

message("Done. Binary QTL results written to: ", normalizePath(out_dir))
