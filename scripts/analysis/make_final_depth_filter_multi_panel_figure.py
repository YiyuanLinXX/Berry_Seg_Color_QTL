import argparse
import os
import tempfile
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "matplotlib"))

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Patch
import numpy as np
import pandas as pd
from scipy.stats import pearsonr


THRESHOLDS = [20, 40, 60, 80, 100]
COMPARISON_THRESHOLDS = [20, 40, 60, 80]
BASELINE_THRESHOLD = 100
FIGURE_WIDTH_PT = 492.5
FIGURE_WIDTH_IN = FIGURE_WIDTH_PT / 72.0
FIGURE_HEIGHT_IN = 6.55
OUTPUT_DIRNAME = "final_depth_filter_multi_panel_figure"
VINE_CSV_TEMPLATE = (
    "berry_instance_depth_filtered_{threshold}/"
    "vine_level_analysis/all_sides/vine_level_all_sides_means.csv"
)

THRESHOLD_COLORS = {
    20: "#C65D5D",
    40: "#D8924A",
    60: "#8E9A4D",
    80: "#4C8C7A",
    100: "#4F6FAE",
}
FEATURE_COLORS = {
    "L*": "#1F1F1F",
    "a*": "#B35C1E",
    "b*": "#2C7C91",
    "Hue": "#7A4DA3",
}


def configure_matplotlib():
    mpl.rcParams.update(
        {
            "font.family": "Arial",
            "font.size": 10,
            "axes.labelsize": 10,
            "axes.titlesize": 10,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 10,
            "axes.linewidth": 0.8,
            "xtick.major.width": 0.8,
            "ytick.major.width": 0.8,
            "xtick.major.size": 3.0,
            "ytick.major.size": 3.0,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )


def style_axis(ax, grid_axis="y"):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(0.8)
    ax.spines["bottom"].set_linewidth(0.8)
    ax.grid(True, axis=grid_axis, color="#D9D9D9", linewidth=0.5, alpha=0.7)
    ax.set_axisbelow(True)


def add_panel_label(ax, label):
    ax.text(
        -0.15,
        1.04,
        label,
        transform=ax.transAxes,
        fontsize=10,
        fontweight="bold",
        va="top",
        ha="left",
    )


def find_column(columns, candidates, required=True):
    lowered = {column.lower(): column for column in columns}
    for candidate in candidates:
        if candidate.lower() in lowered:
            return lowered[candidate.lower()]
    if not required:
        return None
    raise ValueError(
        f"Could not infer a matching column from candidates {candidates}. "
        f"Available columns: {list(columns)}"
    )


def find_instance_file(threshold_dir):
    candidates = sorted(
        threshold_dir.rglob("berry_instances_assigned_to_vines_sorted_by_instance.csv")
    )
    if candidates:
        return candidates[0]

    for candidate in sorted(threshold_dir.rglob("*.csv")):
        try:
            sample = pd.read_csv(candidate, nrows=5)
        except Exception:
            continue
        if any("vine" in column.lower() for column in sample.columns):
            return candidate

    raise FileNotFoundError(f"Could not find an instance-level CSV in {threshold_dir}")


def load_vine_tables(project_root):
    tables = {}
    detected_columns = None

    for threshold in THRESHOLDS:
        csv_path = project_root / VINE_CSV_TEMPLATE.format(threshold=threshold)
        df = pd.read_csv(csv_path)

        columns = {
            "vine_id": find_column(df.columns, ["assigned_vine_id", "vine_id"]),
            "lab_l": find_column(df.columns, ["lab_l_mean", "lab_L_mean", "lab_l"]),
            "lab_a": find_column(df.columns, ["lab_a_mean", "lab_A_mean", "lab_a"]),
            "lab_b": find_column(df.columns, ["lab_b_mean", "lab_B_mean", "lab_b"]),
            "hsv_h": find_column(df.columns, ["hsv_h_mean", "hsv_H_mean", "hsv_h"], required=False),
        }
        if detected_columns is None:
            detected_columns = columns.copy()

        print(f"\nThreshold {threshold} vine-level file: {csv_path}")
        print(f"  vine ID column: {columns['vine_id']}")
        print(f"  Lab L* column: {columns['lab_l']}")
        print(f"  Lab a* column: {columns['lab_a']}")
        print(f"  Lab b* column: {columns['lab_b']}")
        print(f"  HSV hue column: {columns['hsv_h']}")

        keep = [columns["vine_id"], columns["lab_l"], columns["lab_a"], columns["lab_b"]]
        if columns["hsv_h"] is not None:
            keep.append(columns["hsv_h"])

        renamed = {
            columns["vine_id"]: "vine_id",
            columns["lab_l"]: f"lab_l_tau_{threshold}",
            columns["lab_a"]: f"lab_a_tau_{threshold}",
            columns["lab_b"]: f"lab_b_tau_{threshold}",
        }
        if columns["hsv_h"] is not None:
            renamed[columns["hsv_h"]] = f"hsv_h_tau_{threshold}"

        tables[threshold] = df[keep].rename(columns=renamed)

    return tables, detected_columns


def load_instance_tables(project_root):
    instance_tables = {}
    detected_columns = None

    for threshold in THRESHOLDS:
        threshold_dir = project_root / f"berry_instance_depth_filtered_{threshold}"
        csv_path = find_instance_file(threshold_dir)
        df = pd.read_csv(csv_path, low_memory=False)

        vine_id_col = find_column(df.columns, ["assigned_vine_id", "vine_id", "vine"])
        assigned = df[df[vine_id_col].notna()].copy()
        assigned[vine_id_col] = assigned[vine_id_col].astype(str)

        if detected_columns is None:
            detected_columns = {"vine_id": vine_id_col, "file": str(csv_path)}

        print(f"\nThreshold {threshold} instance-level file: {csv_path}")
        print(f"  instance vine ID column: {vine_id_col}")
        print(f"  assigned instances retained: {len(assigned)}")
        print(f"  vines with >=1 instance: {assigned[vine_id_col].nunique()}")

        instance_tables[threshold] = assigned[[vine_id_col]].rename(columns={vine_id_col: "vine_id"})

    return instance_tables, detected_columns


def align_feature_tables(vine_tables, feature_prefix):
    aligned = None
    required_columns = []
    for threshold in THRESHOLDS:
        column_name = f"{feature_prefix}_tau_{threshold}"
        current = vine_tables[threshold][["vine_id", column_name]].dropna()
        required_columns.append(column_name)
        aligned = current if aligned is None else aligned.merge(current, on="vine_id", how="inner")
    return aligned.dropna(subset=required_columns).sort_values("vine_id").reset_index(drop=True)


def compute_feature_correlations(aligned, feature_prefix):
    baseline = f"{feature_prefix}_tau_{BASELINE_THRESHOLD}"
    rows = []
    for threshold in COMPARISON_THRESHOLDS:
        current = f"{feature_prefix}_tau_{threshold}"
        correlation, p_value = pearsonr(aligned[current], aligned[baseline])
        rows.append({"threshold": threshold, "correlation": correlation, "p_value": p_value})
    return pd.DataFrame(rows)


def build_absolute_difference_series(aligned, feature_prefix):
    baseline = f"{feature_prefix}_tau_{BASELINE_THRESHOLD}"
    return {
        threshold: (aligned[f"{feature_prefix}_tau_{threshold}"] - aligned[baseline]).abs()
        for threshold in COMPARISON_THRESHOLDS
    }


def build_instance_summary(instance_tables):
    rows = []
    counts_per_threshold = {}
    for threshold in THRESHOLDS:
        counts = instance_tables[threshold].groupby("vine_id").size().rename("instance_count")
        counts_per_threshold[threshold] = counts
        rows.append(
            {
                "threshold": threshold,
                "total_instances": int(len(instance_tables[threshold])),
                "vines_ge_1": int((counts >= 1).sum()),
                "vines_ge_3": int((counts >= 3).sum()),
                "vines_ge_5": int((counts >= 5).sum()),
            }
        )
    summary = pd.DataFrame(rows).sort_values("threshold").reset_index(drop=True)
    baseline_total = float(
        summary.loc[summary["threshold"] == BASELINE_THRESHOLD, "total_instances"].iloc[0]
    )
    summary["retained_ratio"] = summary["total_instances"] / baseline_total
    return summary, counts_per_threshold


def hue_circularity_warning(aligned_hue):
    baseline_values = aligned_hue[f"hsv_h_tau_{BASELINE_THRESHOLD}"].dropna().to_numpy()
    if baseline_values.size == 0:
        return False, "Hue unavailable."
    scale_max = 360.0 if baseline_values.max() > 180 else 180.0
    lower_cut = 0.10 * scale_max
    upper_cut = 0.90 * scale_max
    warning = bool(np.any(baseline_values <= lower_cut) and np.any(baseline_values >= upper_cut))
    if warning:
        return True, "WARNING: Hue values appear to span the circular boundary; Pearson correlation may be inappropriate."
    return False, "No obvious hue circular-boundary issue detected."


def get_outlier_mask(values):
    q1, q3 = np.percentile(values, [25, 75])
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    return (values < lower) | (values > upper)


def draw_half_violin_scatter_box(ax, data_by_threshold, positions, widths=0.72):
    ordered_thresholds = list(data_by_threshold.keys())
    data = [pd.Series(data_by_threshold[threshold]).dropna().to_numpy() for threshold in ordered_thresholds]

    violin = ax.violinplot(
        data,
        positions=positions,
        widths=widths,
        showmeans=False,
        showmedians=False,
        showextrema=False,
    )
    for body, threshold, position in zip(violin["bodies"], ordered_thresholds, positions):
        vertices = body.get_paths()[0].vertices
        vertices[:, 0] = np.maximum(vertices[:, 0], position)
        body.set_facecolor(THRESHOLD_COLORS[threshold])
        body.set_edgecolor(THRESHOLD_COLORS[threshold])
        body.set_alpha(0.38)
        body.set_linewidth(0.8)

    rng = np.random.default_rng(20260427)
    for threshold, values, position in zip(ordered_thresholds, data, positions):
        outlier_mask = get_outlier_mask(values)
        regular = values[~outlier_mask]
        outliers = values[outlier_mask]

        regular_x = position - rng.uniform(0.05, widths * 0.48, size=regular.size)
        ax.scatter(
            regular_x,
            regular,
            s=5,
            c=THRESHOLD_COLORS[threshold],
            alpha=0.22,
            linewidths=0,
            zorder=2.2,
        )

        if outliers.size > 0:
            outlier_x = position - rng.uniform(0.05, widths * 0.48, size=outliers.size)
            ax.scatter(
                outlier_x,
                outliers,
                s=14,
                facecolors="white",
                edgecolors=THRESHOLD_COLORS[threshold],
                linewidths=0.8,
                alpha=0.95,
                zorder=3.0,
            )

    box = ax.boxplot(
        data,
        positions=positions + widths * 0.12,
        widths=widths * 0.18,
        patch_artist=True,
        showfliers=False,
        medianprops={"color": "#222222", "linewidth": 1.0},
        whiskerprops={"color": "#444444", "linewidth": 0.8},
        capprops={"color": "#444444", "linewidth": 0.8},
        boxprops={"edgecolor": "#444444", "linewidth": 0.8},
    )
    for patch in box["boxes"]:
        patch.set_facecolor("white")
        patch.set_alpha(0.96)

    style_axis(ax, grid_axis="y")


def save_figure(fig, output_dir, stem):
    svg_path = output_dir / f"{stem}.svg"
    png_path = output_dir / f"{stem}.png"
    pdf_path = output_dir / f"{stem}.pdf"
    fig.savefig(svg_path, format="svg")
    fig.savefig(png_path, format="png", dpi=600)
    try:
        fig.savefig(pdf_path, format="pdf")
    except Exception as exc:
        print(f"PDF save failed: {exc}")
        pdf_path = None
    return svg_path, png_path, pdf_path


def write_summary_file(output_path, instance_summary, lab_corr_tables, hue_corr_df, hue_note):
    with output_path.open("w", encoding="utf-8") as handle:
        handle.write("Depth filter multi-panel figure summary\n")
        handle.write("=====================================\n\n")
        handle.write("Total instances and vine coverage by threshold\n")
        handle.write(instance_summary.to_string(index=False))
        handle.write("\n\nPearson correlations with tau=100 baseline\n")
        for feature_name, df in lab_corr_tables.items():
            handle.write(f"\n{feature_name}\n")
            handle.write(df.to_string(index=False))
            handle.write("\n")
        if hue_corr_df is not None:
            handle.write("\nHue\n")
            handle.write(hue_corr_df.to_string(index=False))
            handle.write("\n")
        handle.write("\nHue circularity note\n")
        handle.write(hue_note)
        handle.write("\n")


def main():
    parser = argparse.ArgumentParser(description="Build the multi-panel depth-filter sensitivity figure.")
    parser.add_argument("--strategy-root", type=Path, default=Path("results/generated/phenotyping"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/generated/depth_filter_figure"))
    args = parser.parse_args()
    configure_matplotlib()
    project_root = args.strategy_root
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    vine_tables, vine_columns = load_vine_tables(project_root)
    instance_tables, instance_columns = load_instance_tables(project_root)

    aligned_lab_l = align_feature_tables(vine_tables, "lab_l")
    aligned_lab_a = align_feature_tables(vine_tables, "lab_a")
    aligned_lab_b = align_feature_tables(vine_tables, "lab_b")
    aligned_hue = None
    if all(f"hsv_h_tau_{threshold}" in vine_tables[threshold].columns for threshold in THRESHOLDS):
        aligned_hue = align_feature_tables(vine_tables, "hsv_h")

    print("\nAligned vine counts")
    print(f"  Lab L*: {len(aligned_lab_l)}")
    print(f"  Lab a*: {len(aligned_lab_a)}")
    print(f"  Lab b*: {len(aligned_lab_b)}")
    if aligned_hue is not None:
        print(f"  Hue: {len(aligned_hue)}")

    lab_corr_tables = {
        "L*": compute_feature_correlations(aligned_lab_l, "lab_l"),
        "a*": compute_feature_correlations(aligned_lab_a, "lab_a"),
        "b*": compute_feature_correlations(aligned_lab_b, "lab_b"),
    }
    hue_corr_df = compute_feature_correlations(aligned_hue, "hsv_h") if aligned_hue is not None else None
    hue_warning, hue_note = hue_circularity_warning(aligned_hue) if aligned_hue is not None else (False, "Hue unavailable.")
    print(f"\nHue note: {hue_note}")

    abs_diff_lab = {
        "L*": build_absolute_difference_series(aligned_lab_l, "lab_l"),
        "a*": build_absolute_difference_series(aligned_lab_a, "lab_a"),
        "b*": build_absolute_difference_series(aligned_lab_b, "lab_b"),
    }
    instance_summary, instance_counts = build_instance_summary(instance_tables)

    all_corr_values = []
    for df in lab_corr_tables.values():
        all_corr_values.extend(df["correlation"].tolist())
    if hue_corr_df is not None:
        all_corr_values.extend(hue_corr_df["correlation"].tolist())
    corr_ymin = max(0.65, np.floor((min(all_corr_values) - 0.02) * 20) / 20)
    corr_ymax = 1.01

    fig = plt.figure(figsize=(FIGURE_WIDTH_IN, FIGURE_HEIGHT_IN), constrained_layout=False)
    gs = GridSpec(
        2,
        4,
        figure=fig,
        width_ratios=[0.94, 0.94, 1.04, 1.04],
        height_ratios=[1.28, 1.0],
        wspace=0.28,
        hspace=0.35,
    )

    ax_a = fig.add_subplot(gs[0, 0:2])
    panel_c_gs = gs[0, 2:4].subgridspec(3, 1, hspace=0.24)
    ax_c1 = fig.add_subplot(panel_c_gs[0, 0])
    ax_c2 = fig.add_subplot(panel_c_gs[1, 0], sharex=ax_c1)
    ax_c3 = fig.add_subplot(panel_c_gs[2, 0], sharex=ax_c1)
    ax_d = fig.add_subplot(gs[1, 0])
    ax_e = fig.add_subplot(gs[1, 1])
    ax_f = fig.add_subplot(gs[1, 2])
    ax_g = fig.add_subplot(gs[1, 3])

    add_panel_label(ax_a, "(A)")
    for feature_name, df in lab_corr_tables.items():
        ax_a.plot(
            df["threshold"],
            df["correlation"],
            marker="o",
            markersize=4.5,
            linewidth=1.5,
            color=FEATURE_COLORS[feature_name],
            label=feature_name,
        )
    if hue_corr_df is not None:
        ax_a.plot(
            hue_corr_df["threshold"],
            hue_corr_df["correlation"],
            marker="o",
            markersize=4.5,
            linewidth=1.5,
            color=FEATURE_COLORS["Hue"],
            label="Hue",
        )
    ax_a.set_title("Correlation with no-filtering baseline")
    ax_a.set_xlabel("Threshold (τ)")
    ax_a.set_ylabel("Pearson r vs τ=100")
    ax_a.set_xticks(COMPARISON_THRESHOLDS)
    ax_a.set_ylim(corr_ymin, corr_ymax)
    style_axis(ax_a, grid_axis="y")
    ax_a.legend(frameon=False, loc="lower right", ncol=2, columnspacing=0.9, handlelength=1.8)
    if hue_warning:
        ax_a.text(
            0.02,
            0.05,
            "Hue circularity warning",
            transform=ax_a.transAxes,
            fontsize=8,
            color="#7A1F1F",
            ha="left",
            va="bottom",
        )

    add_panel_label(ax_c1, "(B)")
    for axis, feature_name in zip([ax_c1, ax_c2, ax_c3], ["L*", "a*", "b*"]):
        draw_half_violin_scatter_box(
            axis,
            abs_diff_lab[feature_name],
            positions=np.arange(1, len(COMPARISON_THRESHOLDS) + 1),
            widths=0.82,
        )
        axis.set_xlim(0.4, len(COMPARISON_THRESHOLDS) + 0.6)
        axis.set_xticks(np.arange(1, len(COMPARISON_THRESHOLDS) + 1))
        axis.set_xticklabels([str(t) for t in COMPARISON_THRESHOLDS])
        axis.set_title(feature_name, loc="left", pad=2)
        if feature_name != "b*":
            axis.tick_params(axis="x", labelbottom=False)
        if feature_name == "a*":
            axis.set_ylabel("|feature(τ) - feature(100)|")
    ax_c1.set_title("Absolute change from no filtering", pad=4)
    ax_c1.title.set_position((0.52, 1.0))
    ax_c3.set_xlabel("Threshold (τ)")
    threshold_handles = [
        Patch(facecolor=THRESHOLD_COLORS[t], edgecolor=THRESHOLD_COLORS[t], alpha=0.38, label=str(t))
        for t in COMPARISON_THRESHOLDS
    ]
    ax_c1.legend(
        handles=threshold_handles,
        title="τ",
        frameon=False,
        ncol=2,
        loc="upper right",
        bbox_to_anchor=(1.0, 1.12),
        handlelength=1.0,
        columnspacing=0.8,
    )

    add_panel_label(ax_d, "(C)")
    lab_l_distribution = {threshold: aligned_lab_l[f"lab_l_tau_{threshold}"] for threshold in THRESHOLDS}
    draw_half_violin_scatter_box(
        ax_d,
        lab_l_distribution,
        positions=np.arange(1, len(THRESHOLDS) + 1),
        widths=0.74,
    )
    ax_d.set_title("Vine-level L*")
    ax_d.set_xlabel("Threshold (τ)")
    ax_d.set_ylabel("L*")
    ax_d.set_xticks(np.arange(1, len(THRESHOLDS) + 1))
    ax_d.set_xticklabels([str(t) for t in THRESHOLDS])

    add_panel_label(ax_e, "(D)")
    count_data = {threshold: instance_counts[threshold] for threshold in THRESHOLDS}
    draw_half_violin_scatter_box(
        ax_e,
        count_data,
        positions=np.arange(1, len(THRESHOLDS) + 1),
        widths=0.74,
    )
    ax_e.set_title("Instances per vine")
    ax_e.set_xlabel("Threshold (τ)")
    ax_e.set_ylabel("Count")
    ax_e.set_xticks(np.arange(1, len(THRESHOLDS) + 1))
    ax_e.set_xticklabels([str(t) for t in THRESHOLDS])

    add_panel_label(ax_f, "(E)")
    ax_f.plot(
        instance_summary["threshold"],
        instance_summary["retained_ratio"],
        color="#333333",
        marker="o",
        markersize=4.5,
        linewidth=1.5,
    )
    for _, row in instance_summary.iterrows():
        ax_f.text(
            row["threshold"],
            row["retained_ratio"] + 0.03,
            f"{row['retained_ratio']:.3f}",
            fontsize=8,
            ha="center",
            va="bottom",
            color="#333333",
        )
    ax_f.set_title("Retained ratio")
    ax_f.set_xlabel("Threshold (τ)")
    ax_f.set_ylabel("Ratio")
    ax_f.set_xticks(THRESHOLDS)
    ax_f.set_ylim(0, 1.05)
    style_axis(ax_f, grid_axis="y")

    add_panel_label(ax_g, "(F)")
    coverage_lines = [
        ("vines_ge_1", ">=1", "#2E2E2E"),
        ("vines_ge_3", ">=3", "#4C8C7A"),
        ("vines_ge_5", ">=5", "#C65D5D"),
    ]
    for column, label, color in coverage_lines:
        ax_g.plot(
            instance_summary["threshold"],
            instance_summary[column],
            marker="o",
            markersize=4.5,
            linewidth=1.5,
            label=label,
            color=color,
        )
    ax_g.set_title("Vine coverage")
    ax_g.set_xlabel("Threshold (τ)")
    ax_g.set_ylabel("Vines")
    ax_g.set_xticks(THRESHOLDS)
    ax_g.set_ylim(
        int(min(instance_summary["vines_ge_5"].min(), instance_summary["vines_ge_3"].min()) - 25),
        int(instance_summary["vines_ge_1"].max() + 10),
    )
    style_axis(ax_g, grid_axis="y")
    ax_g.legend(frameon=False, loc="lower left", ncol=1, handlelength=1.5)

    fig.subplots_adjust(left=0.07, right=0.985, top=0.965, bottom=0.085)

    svg_path, png_path, pdf_path = save_figure(fig, output_dir, "final_depth_filter_multi_panel")
    plt.close(fig)

    summary_path = output_dir / "figure_summary.txt"
    write_summary_file(summary_path, instance_summary, lab_corr_tables, hue_corr_df, hue_note)

    print("\nFinal checks")
    print(f"  Configured font family: {mpl.rcParams['font.family']}")
    print(f"  Configured base font size: {mpl.rcParams['font.size']} pt")
    print(f"  SVG fonttype: {mpl.rcParams['svg.fonttype']}")
    print(f"  Figure width: {FIGURE_WIDTH_IN * 72:.1f} pt")
    print(f"  Instance-level vine column: {instance_columns['vine_id']}")
    print(f"  Vine-level vine column: {vine_columns['vine_id']}")
    print(f"  Saved SVG: {svg_path}")
    print(f"  Saved PNG: {png_path}")
    if pdf_path is not None:
        print(f"  Saved PDF: {pdf_path}")
    print(f"  Saved summary: {summary_path}")


if __name__ == "__main__":
    main()
