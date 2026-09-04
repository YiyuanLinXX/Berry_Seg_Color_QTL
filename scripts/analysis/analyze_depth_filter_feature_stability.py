import argparse
import os
import tempfile
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "matplotlib"))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pearsonr


THRESHOLDS = [20, 40, 60, 80, 100]
COMPARISON_THRESHOLDS = [80, 60, 40, 20]
BASELINE_THRESHOLD = 100
CSV_TEMPLATE = (
    "berry_instance_depth_filtered_{threshold}/"
    "vine_level_analysis/all_sides/vine_level_all_sides_means.csv"
)
OUTPUT_DIRNAME = "depth_filter_feature_stability_plots"


def find_column(columns, candidates, required=True):
    lowered_to_original = {column.lower(): column for column in columns}
    for candidate in candidates:
        if candidate.lower() in lowered_to_original:
            return lowered_to_original[candidate.lower()]

    if not required:
        return None

    raise ValueError(
        f"Could not infer a matching column from candidates {candidates}. "
        f"Available columns: {list(columns)}"
    )


def load_single_threshold(base_dir, threshold):
    csv_path = base_dir / CSV_TEMPLATE.format(threshold=threshold)
    df = pd.read_csv(csv_path)

    vine_id_col = find_column(df.columns, ["assigned_vine_id", "vine_id", "vine"])
    feature_columns = {
        "lab_l": find_column(df.columns, ["lab_l_mean", "lab_L_mean", "lab_l"]),
        "lab_a": find_column(df.columns, ["lab_a_mean", "lab_A_mean", "lab_a"]),
        "lab_b": find_column(df.columns, ["lab_b_mean", "lab_B_mean", "lab_b"]),
        "hsv_h": find_column(df.columns, ["hsv_h_mean", "hsv_H_mean", "hsv_h"], required=False),
    }

    print(f"\nThreshold {threshold}")
    print(f"  File: {csv_path}")
    print(f"  Detected vine ID column: {vine_id_col}")
    print(f"  Detected L* column: {feature_columns['lab_l']}")
    print(f"  Detected a* column: {feature_columns['lab_a']}")
    print(f"  Detected b* column: {feature_columns['lab_b']}")
    print(f"  Detected hue column: {feature_columns['hsv_h']}")
    print(f"  Available columns: {list(df.columns)}")

    keep_columns = [vine_id_col] + [column for column in feature_columns.values() if column is not None]
    renamed = {vine_id_col: "vine_id"}
    for feature_name, column_name in feature_columns.items():
        if column_name is not None:
            renamed[column_name] = f"{feature_name}_tau_{threshold}"

    return df[keep_columns].rename(columns=renamed)


def load_all_thresholds(base_dir):
    tables = {}
    for threshold in THRESHOLDS:
        tables[threshold] = load_single_threshold(base_dir, threshold)
    return tables


def build_aligned_feature_table(tables, feature_prefix):
    aligned = None
    required_columns = []

    for threshold in THRESHOLDS:
        column_name = f"{feature_prefix}_tau_{threshold}"
        if column_name not in tables[threshold].columns:
            raise ValueError(f"Missing expected column {column_name} for threshold {threshold}.")

        current = tables[threshold][["vine_id", column_name]].dropna()
        required_columns.append(column_name)
        if aligned is None:
            aligned = current
        else:
            aligned = aligned.merge(current, on="vine_id", how="inner")

    aligned = aligned.dropna(subset=required_columns).sort_values("vine_id").reset_index(drop=True)
    return aligned


def compute_correlations(aligned, feature_label):
    baseline_column = f"{feature_label}_tau_{BASELINE_THRESHOLD}"
    results = []
    for threshold in COMPARISON_THRESHOLDS:
        current_column = f"{feature_label}_tau_{threshold}"
        correlation, p_value = pearsonr(aligned[current_column], aligned[baseline_column])
        results.append(
            {
                "threshold": threshold,
                "correlation": correlation,
                "p_value": p_value,
            }
        )
    return pd.DataFrame(results).sort_values("threshold")


def build_absolute_difference_table(aligned, feature_label):
    baseline_column = f"{feature_label}_tau_{BASELINE_THRESHOLD}"
    absolute_differences = {}
    for threshold in COMPARISON_THRESHOLDS:
        current_column = f"{feature_label}_tau_{threshold}"
        absolute_differences[str(threshold)] = (
            aligned[current_column] - aligned[baseline_column]
        ).abs()
    return pd.DataFrame(absolute_differences)


def save_figure(fig, output_dir, stem):
    fig.tight_layout()
    fig.savefig(output_dir / f"{stem}.png", dpi=300)
    fig.savefig(output_dir / f"{stem}.svg")
    plt.close(fig)


def plot_lab_correlation(correlation_tables, output_dir):
    fig, ax = plt.subplots(figsize=(7, 5))
    for label, df in correlation_tables.items():
        ax.plot(df["threshold"], df["correlation"], marker="o", label=label)

    ax.set_xlabel("Depth Filtering Threshold (τ)", fontsize=11)
    ax.set_ylabel("Pearson Correlation with τ=100", fontsize=11)
    ax.set_title("Correlation of Vine-Level Lab Features with No-Filtering Baseline", fontsize=12)
    ax.set_xticks(sorted(COMPARISON_THRESHOLDS))
    ax.legend(fontsize=10)
    save_figure(fig, output_dir, "lab_feature_correlation_vs_threshold")


def plot_lab_absolute_difference(abs_diff_tables, output_dir):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharex=True)
    panel_labels = [("L*", "lab_l"), ("a*", "lab_a"), ("b*", "lab_b")]

    for ax, (display_name, feature_key) in zip(axes, panel_labels):
        data = [abs_diff_tables[feature_key][str(threshold)].dropna() for threshold in COMPARISON_THRESHOLDS]
        ax.boxplot(data, tick_labels=[str(threshold) for threshold in COMPARISON_THRESHOLDS])
        ax.set_title(display_name, fontsize=12)
        ax.set_xlabel("Depth Filtering Threshold (τ)", fontsize=11)
        ax.set_ylabel(f"|{display_name}(τ) - {display_name}(100)|", fontsize=11)
        ax.tick_params(axis="both", labelsize=10)

    fig.suptitle("Absolute Change in Vine-Level Lab Features Relative to No Filtering", fontsize=13)
    save_figure(fig, output_dir, "lab_feature_absolute_difference_boxplots")


def plot_lab_l_distribution(aligned_lab_l, output_dir):
    fig, ax = plt.subplots(figsize=(7, 5))
    data = [aligned_lab_l[f"lab_l_tau_{threshold}"].dropna() for threshold in THRESHOLDS]
    ax.boxplot(data, tick_labels=[str(threshold) for threshold in THRESHOLDS])
    ax.set_xlabel("Depth Filtering Threshold (τ)", fontsize=11)
    ax.set_ylabel("Vine-Level L*", fontsize=11)
    ax.set_title("Distribution of Vine-Level L* Across Depth Filtering Thresholds", fontsize=12)
    ax.tick_params(axis="both", labelsize=10)
    save_figure(fig, output_dir, "lab_l_distribution_across_thresholds")


def hue_crosses_circular_boundary(aligned_hue):
    baseline_values = aligned_hue[f"hsv_h_tau_{BASELINE_THRESHOLD}"].dropna().to_numpy()
    if baseline_values.size == 0:
        return False

    scale_max = 360.0 if baseline_values.max() > 180 else 180.0
    low_threshold = 0.10 * scale_max
    high_threshold = 0.90 * scale_max
    return np.any(baseline_values <= low_threshold) and np.any(baseline_values >= high_threshold)


def plot_hue_correlation(correlation_df, output_dir):
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(correlation_df["threshold"], correlation_df["correlation"], marker="o")
    ax.set_xlabel("Depth Filtering Threshold (τ)", fontsize=11)
    ax.set_ylabel("Pearson Correlation with τ=100", fontsize=11)
    ax.set_title("Correlation of Vine-Level Hue with No-Filtering Baseline", fontsize=12)
    ax.set_xticks(sorted(COMPARISON_THRESHOLDS))
    save_figure(fig, output_dir, "optional_hue_correlation_vs_threshold")


def print_correlation_summary(feature_name, correlation_df):
    print(f"\nCorrelation summary for {feature_name}")
    print(correlation_df.to_string(index=False))


def main():
    parser = argparse.ArgumentParser(description="Compare vine-level color features across depth thresholds.")
    parser.add_argument("--strategy-root", type=Path, default=Path("results/generated/phenotyping"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/generated/depth_filter_stability"))
    args = parser.parse_args()
    project_root = args.strategy_root
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    tables = load_all_thresholds(project_root)

    aligned_lab = {
        "lab_l": build_aligned_feature_table(tables, "lab_l"),
        "lab_a": build_aligned_feature_table(tables, "lab_a"),
        "lab_b": build_aligned_feature_table(tables, "lab_b"),
    }

    print("\nVines retained after inner join")
    print(f"  Lab L*: {len(aligned_lab['lab_l'])}")
    print(f"  Lab a*: {len(aligned_lab['lab_a'])}")
    print(f"  Lab b*: {len(aligned_lab['lab_b'])}")

    lab_correlations = {
        "L*": compute_correlations(aligned_lab["lab_l"], "lab_l"),
        "a*": compute_correlations(aligned_lab["lab_a"], "lab_a"),
        "b*": compute_correlations(aligned_lab["lab_b"], "lab_b"),
    }
    for feature_name, correlation_df in lab_correlations.items():
        print_correlation_summary(feature_name, correlation_df)

    abs_diff_tables = {
        "lab_l": build_absolute_difference_table(aligned_lab["lab_l"], "lab_l"),
        "lab_a": build_absolute_difference_table(aligned_lab["lab_a"], "lab_a"),
        "lab_b": build_absolute_difference_table(aligned_lab["lab_b"], "lab_b"),
    }

    plot_lab_correlation(lab_correlations, output_dir)
    plot_lab_absolute_difference(abs_diff_tables, output_dir)
    plot_lab_l_distribution(aligned_lab["lab_l"], output_dir)

    hue_present_in_all_tables = all(
        f"hsv_h_tau_{threshold}" in tables[threshold].columns for threshold in THRESHOLDS
    )
    if hue_present_in_all_tables:
        aligned_hue = build_aligned_feature_table(tables, "hsv_h")
        print(f"\nVines retained after inner join for hue: {len(aligned_hue)}")
        print("Detected hue column name pattern: hsv_h_tau_<threshold>")
        if hue_crosses_circular_boundary(aligned_hue):
            print(
                "WARNING: Hue values appear to span the circular boundary. "
                "Pearson correlation may be inappropriate for hue."
            )
        hue_correlation = compute_correlations(aligned_hue, "hsv_h")
        print_correlation_summary("Hue", hue_correlation)
        plot_hue_correlation(hue_correlation, output_dir)
    else:
        print("\nHue column not found across all thresholds. Skipping optional hue analysis.")

    print(f"\nFigures written to: {output_dir}")


if __name__ == "__main__":
    main()
