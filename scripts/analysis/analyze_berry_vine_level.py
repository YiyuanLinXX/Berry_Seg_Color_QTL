#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ.setdefault("NUMBA_CACHE_DIR", "/tmp/numba_cache")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from matplotlib.lines import Line2D

import umap


PRIMARY_PLOT_FEATURES = [
    "rgb_r_mean",
    "rgb_g_mean",
    "rgb_b_mean",
    "lab_l_mean",
    "lab_a_mean",
    "lab_b_mean",
]

EXCLUDED_NUMERIC_COLUMNS = {
    "reason",
    "frame_id",
    "frame_id_key",
    "frame_id_numeric",
    "pixel_count",
    "bbox_x_min",
    "bbox_y_min",
    "bbox_x_max",
    "bbox_y_max",
    "bbox_w",
    "bbox_h",
    "matched_row_id",
    "coverage_start_s",
    "coverage_end_s",
    "frame_row_length_s",
    "image_width_px_inferred",
    "best_overlap_length_m",
    "best_overlap_fraction_of_instance",
    "best_overlap_fraction_of_vine",
    "instance_row_start_s",
    "instance_row_end_s",
    "instance_row_center_s",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Aggregate assigned berry instances to vine-level N/S summaries and compare "
            "same-vine north vs south color differences."
        )
    )
    parser.add_argument(
        "--strategy-root",
        type=Path,
        default=Path("results/generated/phenotyping"),
        help="Root directory containing berry_instance_depth_filtered_* folders.",
    )
    parser.add_argument(
        "--input-name",
        default="berry_instances_assigned_to_vines_sorted_by_instance.csv",
        help="Assigned instance CSV filename expected inside each strategy folder.",
    )
    parser.add_argument(
        "--output-subdir",
        default="vine_level_analysis",
        help="Subdirectory to create inside each strategy folder for vine-level outputs.",
    )
    return parser.parse_args()


def benjamini_hochberg(p_values: pd.Series) -> pd.Series:
    p_values = pd.Series(p_values, dtype=float)
    adjusted = pd.Series(np.nan, index=p_values.index, dtype=float)
    valid = p_values.dropna().sort_values()
    if valid.empty:
        return adjusted

    ranks = np.arange(1, len(valid) + 1, dtype=float)
    raw = valid.to_numpy() * len(valid) / ranks
    monotonic = np.minimum.accumulate(raw[::-1])[::-1]
    adjusted.loc[valid.index] = np.clip(monotonic, 0.0, 1.0)
    return adjusted


def format_p_value(value: float) -> str:
    if pd.isna(value):
        return "NA"
    if value < 1e-4:
        return f"{value:.2e}"
    return f"{value:.4f}"


def hedges_g(sample_a: pd.Series, sample_b: pd.Series) -> float:
    n_a = len(sample_a)
    n_b = len(sample_b)
    if n_a < 2 or n_b < 2:
        return np.nan

    var_a = sample_a.var(ddof=1)
    var_b = sample_b.var(ddof=1)
    pooled_denom = ((n_a - 1) * var_a + (n_b - 1) * var_b) / (n_a + n_b - 2)
    if pooled_denom <= 0:
        return np.nan

    pooled_sd = float(np.sqrt(pooled_denom))
    correction = 1 - 3 / (4 * (n_a + n_b) - 9)
    return correction * (sample_a.mean() - sample_b.mean()) / pooled_sd


def render_markdown_table(
    df: pd.DataFrame,
    columns: list[str],
    max_rows: int | None = None,
) -> str:
    if df.empty:
        return "_No rows._"

    subset = df.loc[:, columns].copy()
    if max_rows is not None:
        subset = subset.head(max_rows)

    def format_value(value: object) -> str:
        if pd.isna(value):
            return "NA"
        if isinstance(value, (float, np.floating)):
            return f"{value:.4f}"
        return str(value)

    headers = list(subset.columns)
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for _, row in subset.iterrows():
        lines.append("| " + " | ".join(format_value(row[column]) for column in headers) + " |")
    return "\n".join(lines)


def get_feature_columns(df: pd.DataFrame) -> list[str]:
    feature_columns: list[str] = []
    for column in df.select_dtypes(include=[np.number]).columns:
        if column in EXCLUDED_NUMERIC_COLUMNS:
            continue
        if column.startswith("coverage_"):
            continue
        feature_columns.append(column)
    return feature_columns


def load_assigned_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, low_memory=False)
    df.columns = [str(column).strip() for column in df.columns]
    required = ["assigned_vine_id", "assigned_canopy_side", "assignment_status", "status"]
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError(f"{path} is missing required columns: {missing}")

    df = df.copy()
    df["assigned_canopy_side"] = df["assigned_canopy_side"].astype(str).str.strip().str.upper()
    return df


def split_subsets(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    assigned_df = df[
        df["status"].fillna("").eq("ok")
        & df["assignment_status"].isin({"assigned_by_overlap", "assigned_by_nearest_center"})
        & df["assigned_vine_id"].astype(str).str.strip().ne("")
        & df["assigned_canopy_side"].isin(["N", "S"])
    ].copy()

    vine_side_counts = (
        assigned_df.groupby(["assigned_vine_id", "assigned_canopy_side"])
        .size()
        .unstack(fill_value=0)
        .reindex(columns=["N", "S"], fill_value=0)
    )
    paired_vines = vine_side_counts[(vine_side_counts["N"] > 0) & (vine_side_counts["S"] > 0)].index
    paired_df = assigned_df[assigned_df["assigned_vine_id"].isin(paired_vines)].copy()
    return assigned_df, paired_df


def build_vine_level_means(
    paired_df: pd.DataFrame,
    feature_columns: list[str],
) -> pd.DataFrame:
    counts = (
        paired_df.groupby(["assigned_vine_id", "assigned_canopy_side"])
        .size()
        .rename("instance_count")
        .reset_index()
    )
    means = (
        paired_df.groupby(["assigned_vine_id", "assigned_canopy_side"], as_index=False)[feature_columns]
        .mean(numeric_only=True)
    )
    vine_level = means.merge(
        counts,
        on=["assigned_vine_id", "assigned_canopy_side"],
        how="left",
    )
    return vine_level.sort_values(["assigned_vine_id", "assigned_canopy_side"], kind="stable").reset_index(drop=True)


def build_vine_level_all_sides_means(
    assigned_df: pd.DataFrame,
    feature_columns: list[str],
) -> pd.DataFrame:
    counts = (
        assigned_df.groupby("assigned_vine_id")
        .size()
        .rename("instance_count")
        .reset_index()
    )
    side_breakdown = (
        assigned_df.groupby(["assigned_vine_id", "assigned_canopy_side"])
        .size()
        .unstack(fill_value=0)
        .reindex(columns=["N", "S"], fill_value=0)
        .reset_index()
        .rename(columns={"N": "n_side_instance_count", "S": "s_side_instance_count"})
    )
    means = (
        assigned_df.groupby("assigned_vine_id", as_index=False)[feature_columns]
        .mean(numeric_only=True)
    )
    vine_level = means.merge(counts, on="assigned_vine_id", how="left").merge(
        side_breakdown,
        on="assigned_vine_id",
        how="left",
    )
    return vine_level.sort_values("assigned_vine_id", kind="stable").reset_index(drop=True)


def build_all_sides_feature_summary(
    vine_level_all_sides: pd.DataFrame,
    feature_columns: list[str],
) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    for feature in feature_columns:
        values = vine_level_all_sides[feature].dropna()
        if values.empty:
            continue
        records.append(
            {
                "feature": feature,
                "n_vines": len(values),
                "mean": values.mean(),
                "std": values.std(ddof=1),
                "median": values.median(),
                "min": values.min(),
                "max": values.max(),
                "p5": values.quantile(0.05),
                "p25": values.quantile(0.25),
                "p75": values.quantile(0.75),
                "p95": values.quantile(0.95),
            }
        )
    return pd.DataFrame(records).sort_values("feature", kind="stable").reset_index(drop=True)


def build_quantile_groups(
    vine_level_all_sides: pd.DataFrame,
    feature_column: str,
    level_count: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    group_prefix = feature_column.replace("_mean", "")
    cluster_df = vine_level_all_sides.copy()
    if feature_column not in cluster_df.columns or cluster_df[feature_column].dropna().empty:
        empty_summary = pd.DataFrame(
            columns=[f"{group_prefix}_{level_count}level_group", "vine_count", f"{feature_column}_min", f"{feature_column}_max", feature_column]
        )
        return cluster_df, empty_summary

    valid = cluster_df[feature_column].dropna()
    quantile_points = np.linspace(0, 1, level_count + 1)[1:-1]
    cut_values = valid.quantile(quantile_points).to_numpy(dtype=float)
    unique_cut_values = np.unique(cut_values)

    label_map = {
        3: ["Low", "Medium", "High"],
        5: ["Very Low", "Low", "Medium", "High", "Very High"],
    }
    labels = label_map.get(level_count, [f"Q{i + 1}" for i in range(level_count)])
    label_to_id = {label: idx + 1 for idx, label in enumerate(labels)}

    group_col = f"{group_prefix}_{level_count}level_group"
    group_id_col = f"{group_prefix}_{level_count}level_group_id"

    if len(unique_cut_values) < level_count - 1:
        rank_bins = min(level_count, max(2, int(valid.nunique())))
        ranked = pd.qcut(cluster_df[feature_column], q=rank_bins, duplicates="drop")
        effective_labels = labels[:rank_bins]
        cluster_df[group_col] = pd.qcut(
            cluster_df[feature_column],
            q=rank_bins,
            labels=effective_labels,
            duplicates="drop",
        ).astype(str)
        cluster_df[group_id_col] = cluster_df[group_col].map({label: idx + 1 for idx, label in enumerate(effective_labels)}).astype("Int64")
        cut_value_list = list(np.unique(valid.quantile(np.linspace(0, 1, rank_bins + 1)[1:-1]).to_numpy(dtype=float)))
    else:
        bins = [-np.inf, *unique_cut_values.tolist(), np.inf]
        cluster_df[group_col] = pd.cut(
            cluster_df[feature_column],
            bins=bins,
            labels=labels,
            include_lowest=True,
        ).astype(str)
        cluster_df[group_id_col] = cluster_df[group_col].map(label_to_id).astype("Int64")
        cut_value_list = unique_cut_values.tolist()

    for idx in range(level_count - 1):
        value = cut_value_list[idx] if idx < len(cut_value_list) else np.nan
        cluster_df[f"{group_prefix}_{level_count}level_cut_{idx + 1}"] = value

    summary = (
        cluster_df.groupby(group_col, dropna=False)[feature_column]
        .agg(vine_count="size", **{f"{feature_column}_min": "min", f"{feature_column}_max": "max", feature_column: "mean"})
        .reset_index()
    )
    for idx in range(level_count - 1):
        value = cut_value_list[idx] if idx < len(cut_value_list) else np.nan
        summary[f"{group_prefix}_{level_count}level_cut_{idx + 1}"] = value

    summary["sort_key"] = summary[group_col].map(label_to_id).fillna(99)
    summary = summary.sort_values("sort_key", kind="stable").drop(columns="sort_key").reset_index(drop=True)
    cluster_df = cluster_df.sort_values(
        [group_id_col, feature_column, "assigned_vine_id"],
        kind="stable",
    ).reset_index(drop=True)
    return cluster_df, summary


def build_lab_b_genotype_ready(
    lab_b_clusters: pd.DataFrame,
) -> pd.DataFrame:
    preferred_columns = [
        "assigned_vine_id",
        "lab_b_mean",
        "lab_b_3level_group",
        "lab_b_3level_group_id",
        "lab_b_3level_cut_1",
        "lab_b_3level_cut_2",
    ]
    available_columns = [column for column in preferred_columns if column in lab_b_clusters.columns]
    genotype_ready = lab_b_clusters[available_columns].copy()
    return genotype_ready.sort_values(["lab_b_3level_group_id", "lab_b_mean", "assigned_vine_id"], kind="stable").reset_index(drop=True)


def prepare_embedding_input(
    vine_level_all_sides: pd.DataFrame,
    feature_columns: list[str],
) -> tuple[pd.DataFrame, np.ndarray]:
    matrix = vine_level_all_sides[feature_columns].replace([np.inf, -np.inf], np.nan).dropna()
    if len(matrix) < 10:
        return pd.DataFrame(), np.empty((0, 0))

    working = vine_level_all_sides.loc[matrix.index].copy().reset_index(drop=True)
    scaled = StandardScaler().fit_transform(matrix.to_numpy(dtype=float))
    return working, scaled


def compute_embedding_coordinates(
    working: pd.DataFrame,
    scaled: np.ndarray,
) -> dict[str, pd.DataFrame]:
    if working.empty or scaled.size == 0:
        return {}

    tsne = TSNE(
        n_components=2,
        random_state=42,
        init="pca",
        learning_rate="auto",
        perplexity=min(30.0, max(5.0, len(working) / 20.0)),
    ).fit_transform(scaled)
    umap_coords = umap.UMAP(n_components=2, random_state=42).fit_transform(scaled)

    return {
        "tsne": pd.DataFrame({"x": tsne[:, 0], "y": tsne[:, 1]}),
        "umap": pd.DataFrame({"x": umap_coords[:, 0], "y": umap_coords[:, 1]}),
    }


def assign_kmeans_clusters(
    matrix: np.ndarray,
    reference_values: pd.Series,
    n_clusters: int,
) -> pd.DataFrame:
    if len(matrix) < n_clusters:
        return pd.DataFrame(columns=["cluster_id", "cluster_label"])

    raw_labels = KMeans(n_clusters=n_clusters, random_state=42, n_init=20).fit_predict(matrix)
    reference_values = pd.Series(reference_values, dtype=float).reset_index(drop=True)
    ordering = (
        pd.DataFrame({"raw_label": raw_labels, "reference": reference_values})
        .groupby("raw_label", as_index=False)["reference"]
        .mean()
        .sort_values("reference", kind="stable")
        ["raw_label"]
        .tolist()
    )
    label_map = {raw_label: idx + 1 for idx, raw_label in enumerate(ordering)}
    cluster_ids = pd.Series(raw_labels).map(label_map).astype(int)
    cluster_labels = cluster_ids.map(lambda value: f"Cluster {value}")
    return pd.DataFrame({"cluster_id": cluster_ids, "cluster_label": cluster_labels})


def build_vine_level_wide(vine_level_means: pd.DataFrame, feature_columns: list[str]) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    for vine_id, group in vine_level_means.groupby("assigned_vine_id"):
        sides = {row["assigned_canopy_side"]: row for _, row in group.iterrows()}
        if "N" not in sides or "S" not in sides:
            continue
        row_n = sides["N"]
        row_s = sides["S"]
        record: dict[str, object] = {
            "assigned_vine_id": vine_id,
            "n_instance_count": row_n["instance_count"],
            "s_instance_count": row_s["instance_count"],
        }
        for feature in feature_columns:
            record[f"{feature}_N"] = row_n[feature]
            record[f"{feature}_S"] = row_s[feature]
            record[f"{feature}_S_minus_N"] = row_s[feature] - row_n[feature]
        records.append(record)
    return pd.DataFrame(records)


def paired_feature_scan(vine_level_means: pd.DataFrame, feature_columns: list[str]) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    for feature in feature_columns:
        wide = (
            vine_level_means.pivot(index="assigned_vine_id", columns="assigned_canopy_side", values=feature)
            .reindex(columns=["N", "S"])
            .dropna(subset=["N", "S"])
        )
        if wide.empty:
            continue

        diff = wide["S"] - wide["N"]
        diff_nonzero = diff[diff != 0]
        diff_std = diff.std(ddof=1)
        north_mean = wide["N"].mean()
        south_mean = wide["S"].mean()

        ttest_p = np.nan
        if len(wide) >= 2:
            ttest_p = stats.ttest_rel(wide["S"], wide["N"], nan_policy="omit").pvalue

        wilcoxon_p = np.nan
        if len(diff_nonzero) > 0:
            try:
                wilcoxon_p = stats.wilcoxon(diff).pvalue
            except ValueError:
                wilcoxon_p = np.nan

        records.append(
            {
                "feature": feature,
                "n_pairs": len(wide),
                "north_mean": north_mean,
                "south_mean": south_mean,
                "mean_diff_s_minus_n": diff.mean(),
                "median_diff_s_minus_n": diff.median(),
                "std_diff": diff_std,
                "abs_mean_diff": abs(diff.mean()),
                "south_gt_north_pairs": int((diff > 0).sum()),
                "north_gt_south_pairs": int((diff < 0).sum()),
                "equal_pairs": int((diff == 0).sum()),
                "paired_t_p_value": ttest_p,
                "wilcoxon_p_value": wilcoxon_p,
                "cohens_dz": diff.mean() / diff_std if pd.notna(diff_std) and diff_std > 0 else np.nan,
            }
        )

    result = pd.DataFrame(records)
    if result.empty:
        return result
    result["paired_t_q_value"] = benjamini_hochberg(result["paired_t_p_value"])
    result["wilcoxon_q_value"] = benjamini_hochberg(result["wilcoxon_p_value"])
    return result.sort_values(
        ["wilcoxon_q_value", "wilcoxon_p_value", "abs_mean_diff", "feature"],
        na_position="last",
    ).reset_index(drop=True)


def assignment_level_scan(assigned_df: pd.DataFrame, feature_columns: list[str]) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    north_df = assigned_df[assigned_df["assigned_canopy_side"] == "N"]
    south_df = assigned_df[assigned_df["assigned_canopy_side"] == "S"]

    for feature in feature_columns:
        north_values = north_df[feature].dropna()
        south_values = south_df[feature].dropna()
        if north_values.empty or south_values.empty:
            continue
        welch_p = stats.ttest_ind(south_values, north_values, equal_var=False, nan_policy="omit").pvalue
        mannwhitney_p = stats.mannwhitneyu(south_values, north_values, alternative="two-sided").pvalue
        records.append(
            {
                "feature": feature,
                "north_n": len(north_values),
                "south_n": len(south_values),
                "north_mean": north_values.mean(),
                "south_mean": south_values.mean(),
                "mean_diff_s_minus_n": south_values.mean() - north_values.mean(),
                "mannwhitney_p_value": mannwhitney_p,
                "welch_t_p_value": welch_p,
                "hedges_g_south_vs_north": hedges_g(south_values, north_values),
            }
        )

    result = pd.DataFrame(records)
    if result.empty:
        return result
    result["welch_t_q_value"] = benjamini_hochberg(result["welch_t_p_value"])
    result["mannwhitney_q_value"] = benjamini_hochberg(result["mannwhitney_p_value"])
    return result.sort_values(
        ["mannwhitney_q_value", "mannwhitney_p_value", "feature"],
        na_position="last",
    ).reset_index(drop=True)


def plot_paired_rgb_lab(vine_level_means: pd.DataFrame, paired_stats: pd.DataFrame, output_path: Path) -> None:
    colors = {"N": "#4C78A8", "S": "#F58518"}
    fig, axes = plt.subplots(2, 3, figsize=(16, 10), constrained_layout=True)
    axes = axes.flatten()

    for ax, feature in zip(axes, PRIMARY_PLOT_FEATURES):
        wide = (
            vine_level_means.pivot(index="assigned_vine_id", columns="assigned_canopy_side", values=feature)
            .reindex(columns=["N", "S"])
            .dropna(subset=["N", "S"])
        )
        if wide.empty:
            ax.set_axis_off()
            continue

        diff = wide["S"] - wide["N"]
        line_colors = np.where(diff >= 0, "#C44E52", "#55A868")
        for idx, (_, row) in enumerate(wide.iterrows()):
            ax.plot([0, 1], [row["N"], row["S"]], color=line_colors[idx], alpha=0.5, linewidth=1.2)
            ax.scatter([0, 1], [row["N"], row["S"]], color=[colors["N"], colors["S"]], s=26, zorder=3)

        north_mean = wide["N"].mean()
        south_mean = wide["S"].mean()
        north_sem = wide["N"].sem()
        south_sem = wide["S"].sem()
        ax.errorbar(
            [0, 1],
            [north_mean, south_mean],
            yerr=[north_sem, south_sem],
            color="black",
            linewidth=2.0,
            marker="o",
            markersize=7,
            capsize=4,
            zorder=4,
        )

        stats_row = paired_stats[paired_stats["feature"] == feature]
        if not stats_row.empty:
            stats_row = stats_row.iloc[0]
            title = (
                f"{feature}\nS-N={stats_row['mean_diff_s_minus_n']:.2f}, "
                f"q={format_p_value(stats_row['wilcoxon_q_value'])}"
            )
        else:
            title = feature
        ax.set_title(title)
        ax.set_xticks([0, 1], ["N", "S"])
        ax.grid(axis="y", alpha=0.25)

    fig.suptitle("Paired Same-Vine Berry RGB/Lab Means", fontsize=16)
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_assignment_rgb_lab_distributions(
    assigned_df: pd.DataFrame,
    assignment_scan_df: pd.DataFrame,
    output_path: Path,
) -> None:
    colors = {"N": "#4C78A8", "S": "#F58518"}
    rng = np.random.default_rng(7)
    fig, axes = plt.subplots(2, 3, figsize=(16, 10), constrained_layout=True)
    axes = axes.flatten()

    for ax, feature in zip(axes, PRIMARY_PLOT_FEATURES):
        north_values = assigned_df.loc[assigned_df["assigned_canopy_side"] == "N", feature].dropna().to_numpy()
        south_values = assigned_df.loc[assigned_df["assigned_canopy_side"] == "S", feature].dropna().to_numpy()
        for position, values, facecolor, edgecolor in [
            (0, north_values, "#D9E4F0", colors["N"]),
            (1, south_values, "#FDE2C2", colors["S"]),
        ]:
            ax.boxplot(
                [values],
                positions=[position],
                widths=0.45,
                patch_artist=True,
                boxprops={"facecolor": facecolor, "edgecolor": edgecolor, "alpha": 0.65},
                medianprops={"color": "black", "linewidth": 1.5},
                whiskerprops={"color": "#666666"},
                capprops={"color": "#666666"},
            )
        for position, values, color in [(0, north_values, colors["N"]), (1, south_values, colors["S"])]:
            if len(values) == 0:
                continue
            sample = values if len(values) <= 400 else rng.choice(values, size=400, replace=False)
            jitter = rng.uniform(-0.1, 0.1, size=len(sample))
            ax.scatter(
                np.full(len(sample), position) + jitter,
                sample,
                color=color,
                alpha=0.18,
                s=14,
                linewidth=0,
            )
        stats_row = assignment_scan_df[assignment_scan_df["feature"] == feature]
        if not stats_row.empty:
            stats_row = stats_row.iloc[0]
            title = (
                f"{feature}\nS-N={stats_row['mean_diff_s_minus_n']:.2f}, "
                f"q={format_p_value(stats_row['mannwhitney_q_value'])}"
            )
        else:
            title = feature
        ax.set_title(title)
        ax.set_xticks([0, 1], ["N", "S"])
        ax.grid(axis="y", alpha=0.25)

    fig.suptitle("All Assigned Berry RGB/Lab Distributions", fontsize=16)
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_all_sides_rgb_lab_distributions(
    vine_level_all_sides: pd.DataFrame,
    output_path: Path,
) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(16, 10), constrained_layout=True)
    axes = axes.flatten()

    for ax, feature in zip(axes, PRIMARY_PLOT_FEATURES):
        values = vine_level_all_sides[feature].dropna().to_numpy()
        if len(values) == 0:
            ax.set_axis_off()
            continue
        ax.hist(values, bins=30, color="#4C78A8", alpha=0.7, edgecolor="white")
        ax.axvline(values.mean(), color="black", linestyle="--", linewidth=1.5, label="mean")
        ax.axvline(np.median(values), color="#F58518", linestyle="-", linewidth=1.5, label="median")
        ax.set_title(feature)
        ax.grid(axis="y", alpha=0.25)
        ax.legend()

    fig.suptitle("Vine-Level RGB/Lab Distributions (N/S Combined)", fontsize=16)
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_embedding_groups(
    embedding_df: pd.DataFrame,
    group_labels: pd.Series,
    output_path: Path,
    title: str,
    xlabel: str,
    ylabel: str,
    legend_title: str,
) -> None:
    if embedding_df.empty or len(group_labels) != len(embedding_df):
        return

    unique_labels = list(pd.Series(group_labels, dtype=str).dropna().unique())
    color_pool = ["#1f77b4", "#17becf", "#f1c40f", "#ff7f0e", "#d62728", "#8c564b", "#2ca02c"]
    palette = {label: color_pool[idx % len(color_pool)] for idx, label in enumerate(unique_labels)}

    fig, ax = plt.subplots(figsize=(7, 6))
    point_colors = [palette[str(group)] for group in group_labels]
    ax.scatter(
        embedding_df["x"],
        embedding_df["y"],
        c=point_colors,
        s=42,
        alpha=0.9,
        edgecolors="black",
        linewidths=0.45,
    )
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.2)
    legend_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="",
            markerfacecolor=palette[label],
            markeredgecolor="black",
            markeredgewidth=0.45,
            markersize=7,
            label=str(label),
        )
        for label in palette
    ]
    ax.legend(handles=legend_handles, title=legend_title, loc="best", frameon=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def write_report(
    output_dir: Path,
    strategy_name: str,
    assigned_df: pd.DataFrame,
    paired_df: pd.DataFrame,
    vine_level_means: pd.DataFrame,
    paired_stats: pd.DataFrame,
    assignment_scan_df: pd.DataFrame,
) -> Path:
    report_path = output_dir / "berry_vine_level_report.md"
    paired_vines = vine_level_means["assigned_vine_id"].nunique() if not vine_level_means.empty else 0
    primary_rgb_lab = paired_stats[paired_stats["feature"].isin(PRIMARY_PLOT_FEATURES)].copy()

    lines = [
        "# Berry Vine-Level Analysis",
        "",
        f"- Strategy: `{strategy_name}`",
        f"- Assigned instance-vine rows: {len(assigned_df)}",
        f"- Paired same-vine rows: {len(paired_df)}",
        f"- Paired same-vine unique vines: {paired_vines}",
        "",
        "## Primary RGB/Lab Table",
        "",
        render_markdown_table(
            primary_rgb_lab,
            [
                "feature",
                "n_pairs",
                "north_mean",
                "south_mean",
                "mean_diff_s_minus_n",
                "wilcoxon_p_value",
                "wilcoxon_q_value",
                "cohens_dz",
            ],
        ),
        "",
        "## Top Paired Features",
        "",
        render_markdown_table(
            paired_stats,
            [
                "feature",
                "n_pairs",
                "north_mean",
                "south_mean",
                "mean_diff_s_minus_n",
                "wilcoxon_p_value",
                "wilcoxon_q_value",
            ],
            max_rows=12,
        ),
        "",
        "## Top Assignment-Level Features",
        "",
        render_markdown_table(
            assignment_scan_df,
            [
                "feature",
                "north_n",
                "south_n",
                "north_mean",
                "south_mean",
                "mean_diff_s_minus_n",
                "mannwhitney_p_value",
                "mannwhitney_q_value",
            ],
            max_rows=12,
        ),
        "",
        "## Output Files",
        "",
        "- `all_assigned_berry_rows.csv`",
        "- `paired_same_vine_berry_rows.csv`",
        "- `vine_level_side_means.csv`",
        "- `vine_level_side_means_wide.csv`",
        "- `paired_feature_scan.csv`",
        "- `all_assignment_orientation_scan.csv`",
        "- `paired_vine_rgb_lab_means.png`",
        "- `all_assignment_rgb_lab_distributions.png`",
    ]
    report_path.write_text("\n".join(lines), encoding="utf-8")
    return report_path


def write_all_sides_report(
    output_dir: Path,
    strategy_name: str,
    assigned_df: pd.DataFrame,
    vine_level_all_sides: pd.DataFrame,
    all_sides_summary: pd.DataFrame,
) -> Path:
    report_path = output_dir / "berry_vine_level_all_sides_report.md"
    primary_rgb_lab = all_sides_summary[all_sides_summary["feature"].isin(PRIMARY_PLOT_FEATURES)].copy()
    lines = [
        "# Berry Vine-Level Analysis (N/S Combined)",
        "",
        f"- Strategy: `{strategy_name}`",
        f"- Assigned instance-vine rows used: {len(assigned_df)}",
        f"- Unique vines: {vine_level_all_sides['assigned_vine_id'].nunique() if not vine_level_all_sides.empty else 0}",
        "",
        "## RGB/Lab Vine-Level Summary",
        "",
        render_markdown_table(
            primary_rgb_lab,
            ["feature", "n_vines", "mean", "std", "median", "p5", "p95"],
        ),
        "",
        "## Output Files",
        "",
        "- `vine_level_all_sides_means.csv`",
        "- `vine_level_all_sides_feature_summary.csv`",
        "- `vine_level_all_sides_lab_l_3level_groups.csv` / `summary.csv`",
        "- `vine_level_all_sides_lab_l_5level_groups.csv` / `summary.csv`",
        "- `vine_level_all_sides_lab_a_3level_groups.csv` / `summary.csv`",
        "- `vine_level_all_sides_lab_a_5level_groups.csv` / `summary.csv`",
        "- `vine_level_all_sides_lab_b_3level_groups.csv`",
        "- `vine_level_all_sides_lab_b_3level_summary.csv`",
        "- `vine_level_all_sides_lab_b_5level_groups.csv` / `summary.csv`",
        "- `vine_level_all_sides_lab_b_3level_genotype_ready.csv`",
        "- `vine_level_all_sides_embedding_coordinates.csv`",
        "- `vine_level_all_sides_rgb_lab_distributions.png`",
        "- `vine_level_all_sides_tsne_lab_l_3level.png`",
        "- `vine_level_all_sides_tsne_lab_l_5level.png`",
        "- `vine_level_all_sides_umap_lab_l_3level.png`",
        "- `vine_level_all_sides_umap_lab_l_5level.png`",
        "- `vine_level_all_sides_tsne_lab_a_3level.png`",
        "- `vine_level_all_sides_tsne_lab_a_5level.png`",
        "- `vine_level_all_sides_umap_lab_a_3level.png`",
        "- `vine_level_all_sides_umap_lab_a_5level.png`",
        "- `vine_level_all_sides_tsne_lab_b_3level.png`",
        "- `vine_level_all_sides_tsne_lab_b_5level.png`",
        "- `vine_level_all_sides_umap_lab_b_3level.png`",
        "- `vine_level_all_sides_umap_lab_b_5level.png`",
        "- `vine_level_all_sides_feature_space_kmeans_3clusters.csv` / `5clusters.csv`",
        "- `vine_level_all_sides_tsne_space_kmeans_3clusters.csv` / `5clusters.csv`",
        "- `vine_level_all_sides_umap_space_kmeans_3clusters.csv` / `5clusters.csv`",
        "- `vine_level_all_sides_tsne_feature_space_kmeans_3clusters.png` / `5clusters.png`",
        "- `vine_level_all_sides_umap_feature_space_kmeans_3clusters.png` / `5clusters.png`",
        "- `vine_level_all_sides_tsne_space_kmeans_3clusters.png` / `5clusters.png`",
        "- `vine_level_all_sides_umap_space_kmeans_3clusters.png` / `5clusters.png`",
        "",
        "## Notes",
        "",
        "- This summary ignores canopy side and aggregates all assigned berry instances for each vine.",
        "- It is useful when you want one vine-level color profile per vine without separating N and S.",
        "- The Lab L / a / b group files record vine membership under 3-level and 5-level quantile bins.",
        "- KMeans clustering is exported both from the original feature space and directly from the 2D t-SNE/UMAP spaces.",
    ]
    report_path.write_text("\n".join(lines), encoding="utf-8")
    return report_path


def process_strategy(strategy_dir: Path, input_name: str, output_subdir: str) -> None:
    input_csv = strategy_dir / input_name
    if not input_csv.exists():
        return

    df = load_assigned_csv(input_csv)
    assigned_df, paired_df = split_subsets(df)
    feature_columns = get_feature_columns(df)
    if not feature_columns:
        raise ValueError(f"No numeric feature columns found in {input_csv}")

    vine_level_means = build_vine_level_means(paired_df, feature_columns)
    vine_level_wide = build_vine_level_wide(vine_level_means, feature_columns)
    vine_level_all_sides = build_vine_level_all_sides_means(assigned_df, feature_columns)
    all_sides_summary = build_all_sides_feature_summary(vine_level_all_sides, feature_columns)
    working_embedding_df, scaled_embedding_matrix = prepare_embedding_input(vine_level_all_sides, feature_columns)
    embedding_outputs = compute_embedding_coordinates(working_embedding_df, scaled_embedding_matrix)
    quantile_cluster_outputs: dict[tuple[str, int], tuple[pd.DataFrame, pd.DataFrame]] = {}
    for feature_column in ("lab_l_mean", "lab_a_mean", "lab_b_mean"):
        for level_count in (3, 5):
            quantile_cluster_outputs[(feature_column, level_count)] = build_quantile_groups(
                vine_level_all_sides,
                feature_column=feature_column,
                level_count=level_count,
            )

    lab_b_clusters, lab_b_cluster_summary = quantile_cluster_outputs[("lab_b_mean", 3)]
    lab_b_genotype_ready = build_lab_b_genotype_ready(lab_b_clusters)
    paired_stats = paired_feature_scan(vine_level_means, feature_columns)
    assignment_scan_df = assignment_level_scan(assigned_df, feature_columns)

    output_dir = strategy_dir / output_subdir
    output_dir.mkdir(parents=True, exist_ok=True)
    reports_dir = output_dir / "reports"
    paired_ns_dir = output_dir / "paired_ns"
    all_sides_dir = output_dir / "all_sides"
    quantile_groups_dir = output_dir / "quantile_groups"
    embeddings_dir = output_dir / "embeddings"
    clustering_dir = output_dir / "clustering"
    for directory in (
        reports_dir,
        paired_ns_dir,
        all_sides_dir,
        quantile_groups_dir,
        embeddings_dir,
        clustering_dir,
    ):
        directory.mkdir(parents=True, exist_ok=True)

    assigned_df.to_csv(paired_ns_dir / "all_assigned_berry_rows.csv", index=False)
    paired_df.to_csv(paired_ns_dir / "paired_same_vine_berry_rows.csv", index=False)
    vine_level_means.to_csv(paired_ns_dir / "vine_level_side_means.csv", index=False)
    vine_level_wide.to_csv(paired_ns_dir / "vine_level_side_means_wide.csv", index=False)
    vine_level_all_sides.to_csv(all_sides_dir / "vine_level_all_sides_means.csv", index=False)
    all_sides_summary.to_csv(all_sides_dir / "vine_level_all_sides_feature_summary.csv", index=False)
    if embedding_outputs:
        embedding_csv = working_embedding_df[["assigned_vine_id"]].copy()
        embedding_csv["tsne_x"] = embedding_outputs["tsne"]["x"]
        embedding_csv["tsne_y"] = embedding_outputs["tsne"]["y"]
        embedding_csv["umap_x"] = embedding_outputs["umap"]["x"]
        embedding_csv["umap_y"] = embedding_outputs["umap"]["y"]
        embedding_csv.to_csv(embeddings_dir / "vine_level_all_sides_embedding_coordinates.csv", index=False)
    for (feature_column, level_count), (cluster_df, summary_df) in quantile_cluster_outputs.items():
        feature_prefix = feature_column.replace("_mean", "")
        cluster_df.to_csv(quantile_groups_dir / f"vine_level_all_sides_{feature_prefix}_{level_count}level_groups.csv", index=False)
        summary_df.to_csv(quantile_groups_dir / f"vine_level_all_sides_{feature_prefix}_{level_count}level_summary.csv", index=False)
    lab_b_genotype_ready.to_csv(quantile_groups_dir / "vine_level_all_sides_lab_b_3level_genotype_ready.csv", index=False)
    paired_stats.to_csv(paired_ns_dir / "paired_feature_scan.csv", index=False)
    assignment_scan_df.to_csv(paired_ns_dir / "all_assignment_orientation_scan.csv", index=False)
    plot_paired_rgb_lab(vine_level_means, paired_stats, paired_ns_dir / "paired_vine_rgb_lab_means.png")
    plot_assignment_rgb_lab_distributions(
        assigned_df,
        assignment_scan_df,
        paired_ns_dir / "all_assignment_rgb_lab_distributions.png",
    )
    plot_all_sides_rgb_lab_distributions(
        vine_level_all_sides,
        output_path=all_sides_dir / "vine_level_all_sides_rgb_lab_distributions.png",
    )
    axis_labels = {"tsne": ("t-SNE-1", "t-SNE-2"), "umap": ("UMAP-1", "UMAP-2")}
    if embedding_outputs:
        for method in ("tsne", "umap"):
            for color_column, color_label, feature_prefix in (
                ("lab_l_mean", "Lab L", "lab_l"),
                ("lab_a_mean", "Lab a", "lab_a"),
                ("lab_b_mean", "Lab b", "lab_b"),
            ):
                for quantile_bins in (3, 5):
                    cluster_df, _ = quantile_cluster_outputs[(color_column, quantile_bins)]
                    merged = working_embedding_df[["assigned_vine_id"]].merge(
                        cluster_df[["assigned_vine_id", f"{feature_prefix}_{quantile_bins}level_group"]],
                        on="assigned_vine_id",
                        how="left",
                    )
                    plot_embedding_groups(
                        embedding_df=embedding_outputs[method],
                        group_labels=merged[f"{feature_prefix}_{quantile_bins}level_group"],
                        output_path=embeddings_dir / f"vine_level_all_sides_{method}_{feature_prefix}_{quantile_bins}level.png",
                        title=f"Vine-Level {method.upper()} ({color_label}, {quantile_bins}-Level)",
                        xlabel=axis_labels[method][0],
                        ylabel=axis_labels[method][1],
                        legend_title=f"{quantile_bins}-Level Group",
                    )

        for n_clusters in (3, 5):
            feature_clusters = assign_kmeans_clusters(
                scaled_embedding_matrix,
                reference_values=working_embedding_df["lab_b_mean"],
                n_clusters=n_clusters,
            )
            if not feature_clusters.empty:
                feature_cluster_csv = pd.concat(
                    [
                        working_embedding_df[["assigned_vine_id", "lab_l_mean", "lab_a_mean", "lab_b_mean"]].reset_index(drop=True),
                        feature_clusters.reset_index(drop=True),
                        embedding_csv[["tsne_x", "tsne_y", "umap_x", "umap_y"]].reset_index(drop=True),
                    ],
                    axis=1,
                )
                feature_cluster_csv.to_csv(
                    clustering_dir / f"vine_level_all_sides_feature_space_kmeans_{n_clusters}clusters.csv",
                    index=False,
                )
                for method in ("tsne", "umap"):
                    plot_embedding_groups(
                        embedding_df=embedding_outputs[method],
                        group_labels=feature_clusters["cluster_label"],
                        output_path=clustering_dir / f"vine_level_all_sides_{method}_feature_space_kmeans_{n_clusters}clusters.png",
                        title=f"Vine-Level {method.upper()} with Feature-Space KMeans ({n_clusters} clusters)",
                        xlabel=axis_labels[method][0],
                        ylabel=axis_labels[method][1],
                        legend_title="Feature KMeans",
                    )

            for method in ("tsne", "umap"):
                embedding_clusters = assign_kmeans_clusters(
                    embedding_outputs[method].to_numpy(dtype=float),
                    reference_values=working_embedding_df["lab_b_mean"],
                    n_clusters=n_clusters,
                )
                if embedding_clusters.empty:
                    continue
                embedding_cluster_csv = pd.concat(
                    [
                        working_embedding_df[["assigned_vine_id", "lab_l_mean", "lab_a_mean", "lab_b_mean"]].reset_index(drop=True),
                        embedding_outputs[method].rename(columns={"x": f"{method}_x", "y": f"{method}_y"}).reset_index(drop=True),
                        embedding_clusters.reset_index(drop=True),
                    ],
                    axis=1,
                )
                embedding_cluster_csv.to_csv(
                    clustering_dir / f"vine_level_all_sides_{method}_space_kmeans_{n_clusters}clusters.csv",
                    index=False,
                )
                plot_embedding_groups(
                    embedding_df=embedding_outputs[method],
                    group_labels=embedding_clusters["cluster_label"],
                    output_path=clustering_dir / f"vine_level_all_sides_{method}_space_kmeans_{n_clusters}clusters.png",
                    title=f"Vine-Level {method.upper()}-Space KMeans ({n_clusters} clusters)",
                    xlabel=axis_labels[method][0],
                    ylabel=axis_labels[method][1],
                    legend_title=f"{method.upper()} KMeans",
                )
    report_path = write_report(
        output_dir=reports_dir,
        strategy_name=strategy_dir.name,
        assigned_df=assigned_df,
        paired_df=paired_df,
        vine_level_means=vine_level_means,
        paired_stats=paired_stats,
        assignment_scan_df=assignment_scan_df,
    )
    all_sides_report_path = write_all_sides_report(
        output_dir=reports_dir,
        strategy_name=strategy_dir.name,
        assigned_df=assigned_df,
        vine_level_all_sides=vine_level_all_sides,
        all_sides_summary=all_sides_summary,
    )
    print(f"Analysis written to: {output_dir}")
    print(f"Report: {report_path}")
    print(f"All-sides report: {all_sides_report_path}")


def main() -> None:
    args = parse_args()
    strategy_dirs = sorted(args.strategy_root.glob("berry_instance_depth_filtered_*"))
    if not strategy_dirs:
        raise FileNotFoundError(f"No berry_instance_depth_filtered_* folders found in {args.strategy_root}")
    for strategy_dir in strategy_dirs:
        process_strategy(strategy_dir, args.input_name, args.output_subdir)


if __name__ == "__main__":
    main()
