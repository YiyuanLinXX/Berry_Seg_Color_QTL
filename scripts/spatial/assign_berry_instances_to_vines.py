#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

try:
    import umap  # type: ignore
except Exception:  # pragma: no cover - UMAP is optional for QA plots
    umap = None

from map_camera_frames_to_vines import (
    CoordinateTransformer,
    build_row_models,
    parse_vine_name,
)


FEATURE_PATTERN = re.compile(r"berry_cluster_instance_color_features_depth_filter_(\d+)\.csv$")
DEFAULT_FEATURE_DIR = Path("results/generated/features")
DEFAULT_STRATEGY_ROOT = Path("results/generated/phenotyping")
DEFAULT_ALIGNED_CSV = Path("data/processed/spatial/vine_reference.csv")
DEFAULT_COVERAGE_CSV = Path("results/generated/spatial/frame_gps_camera_coverage.csv")
RANDOM_SEED = 42
EPSILON = 1e-9
NUMERIC_EXCLUDE_PREFIXES = ("bbox_",)
NUMERIC_EXCLUDE_COLUMNS = {
    "pixel_count",
    "frame_id_numeric",
    "matched_row_id",
    "coverage_start_s",
    "coverage_end_s",
    "frame_row_length_s",
    "instance_row_start_s",
    "instance_row_end_s",
    "instance_row_center_s",
    "best_overlap_length_m",
    "best_overlap_fraction_of_instance",
    "best_overlap_fraction_of_vine",
    "image_width_px_inferred",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Assign berry cluster instances to vine IDs and canopy side, then generate "
            "sorted CSVs and color-feature visualizations for each depth-filter strategy."
        )
    )
    parser.add_argument(
        "--feature-dir",
        type=Path,
        default=DEFAULT_FEATURE_DIR,
        help="Directory containing berry_cluster_instance_color_features_depth_filter_*.csv files.",
    )
    parser.add_argument(
        "--feature-csv",
        type=Path,
        default=None,
        help="If provided, process only this single feature CSV instead of scanning feature-dir.",
    )
    parser.add_argument(
        "--strategy-root",
        type=Path,
        default=DEFAULT_STRATEGY_ROOT,
        help="Root directory that contains berry_instance_depth_filtered_* folders.",
    )
    parser.add_argument(
        "--aligned-csv",
        type=Path,
        default=DEFAULT_ALIGNED_CSV,
        help="Aligned vine survey CSV used to rebuild vine coverage intervals.",
    )
    parser.add_argument(
        "--coverage-csv",
        type=Path,
        default=DEFAULT_COVERAGE_CSV,
        help="Per-frame vine coverage CSV.",
    )
    parser.add_argument(
        "--neighbor-threshold-m",
        type=float,
        default=2.0,
        help="Neighbor threshold used when building vine coverage intervals.",
    )
    parser.add_argument(
        "--edge-extension-m",
        type=float,
        default=1.0,
        help="Edge extension used when building vine coverage intervals.",
    )
    parser.add_argument(
        "--max-embedding-points",
        type=int,
        default=5000,
        help="Maximum number of assigned instances to use for each UMAP/t-SNE plot.",
    )
    return parser.parse_args()


def normalize_frame_id(value: object) -> str | None:
    if pd.isna(value):
        return None
    text = str(value).strip()
    if not text:
        return None
    match = re.search(r"(\d+)", text)
    if match is None:
        return None
    return f"{int(match.group(1)):05d}"


def strategy_suffix_from_path(path: Path) -> str:
    match = FEATURE_PATTERN.search(path.name)
    if match is None:
        raise ValueError(f"Unexpected feature CSV name: {path.name}")
    return match.group(1)


def load_row_models(
    aligned_csv: Path,
    neighbor_threshold_m: float,
    edge_extension_m: float,
) -> tuple[list, CoordinateTransformer]:
    vines = pd.read_csv(aligned_csv)
    parsed_names = vines["Name"].map(parse_vine_name)
    vines["row_id"] = parsed_names.map(lambda item: item[0])
    vines["vine_id"] = parsed_names.map(lambda item: item[1])
    transformer = CoordinateTransformer.from_points(vines["Longitude"], vines["Latitude"])
    vines["x"], vines["y"] = transformer.lonlat_to_xy(vines["Longitude"], vines["Latitude"])
    row_models = build_row_models(
        vines=vines,
        neighbor_threshold_m=neighbor_threshold_m,
        edge_extension_m=edge_extension_m,
    )
    return row_models, transformer


def build_vine_interval_lookup(row_models: list) -> dict[int, pd.DataFrame]:
    lookup: dict[int, pd.DataFrame] = {}
    for row_model in row_models:
        intervals = row_model.vine_intervals.copy()
        intervals["coverage_length_m"] = (
            intervals["coverage_s_max"].astype(float) - intervals["coverage_s_min"].astype(float)
        )
        lookup[row_model.row_id] = intervals
    return lookup


def load_coverage_dataframe(coverage_csv: Path) -> pd.DataFrame:
    coverage = pd.read_csv(coverage_csv, low_memory=False)
    coverage.columns = [str(column).strip() for column in coverage.columns]
    required = [
        "Frame ID",
        "Matched Row",
        "Matched Row ID",
        "Row Coverage Start (m)",
        "Row Coverage End (m)",
        "Canopy Facing To",
        "Outlier",
        "Status",
    ]
    missing = [column for column in required if column not in coverage.columns]
    if missing:
        raise ValueError(f"Coverage CSV is missing required columns: {missing}")

    coverage = coverage.copy()
    coverage["frame_id_key"] = coverage["Frame ID"].map(normalize_frame_id)
    coverage["matched_row_id"] = pd.to_numeric(coverage["Matched Row ID"], errors="coerce")
    coverage["coverage_start_s"] = pd.to_numeric(coverage["Row Coverage Start (m)"], errors="coerce")
    coverage["coverage_end_s"] = pd.to_numeric(coverage["Row Coverage End (m)"], errors="coerce")
    coverage["frame_row_length_s"] = coverage["coverage_end_s"] - coverage["coverage_start_s"]
    coverage["coverage_canopy_side"] = (
        coverage["Canopy Facing To"].astype(str).str.strip().str.upper().replace({"": np.nan})
    )
    coverage["coverage_outlier_bool"] = (
        coverage["Outlier"].astype(str).str.strip().str.lower().isin({"true", "1", "yes", "y"})
    )
    return coverage


def infer_image_width_px(df: pd.DataFrame) -> int:
    bbox_limit = pd.to_numeric(df["bbox_x_max"], errors="coerce").dropna()
    if bbox_limit.empty:
        raise ValueError("Could not infer image width because bbox_x_max is empty for all rows.")
    return int(bbox_limit.max()) + 1


def overlap_length(interval_a: tuple[float, float], interval_b: tuple[float, float]) -> float:
    return max(0.0, min(interval_a[1], interval_b[1]) - max(interval_a[0], interval_b[0]))


def assign_instances(
    features: pd.DataFrame,
    coverage: pd.DataFrame,
    vine_lookup: dict[int, pd.DataFrame],
) -> pd.DataFrame:
    merged = features.merge(
        coverage[
            [
                "frame_id_key",
                "Matched Row",
                "matched_row_id",
                "coverage_start_s",
                "coverage_end_s",
                "frame_row_length_s",
                "coverage_canopy_side",
                "coverage_outlier_bool",
                "Status",
            ]
        ],
        on="frame_id_key",
        how="left",
    )

    image_width_px = infer_image_width_px(merged)
    merged["image_width_px_inferred"] = image_width_px
    merged["assignment_status"] = "unassigned"
    merged["assigned_vine_id"] = ""
    merged["assigned_canopy_side"] = ""
    merged["best_overlap_length_m"] = np.nan
    merged["best_overlap_fraction_of_instance"] = np.nan
    merged["best_overlap_fraction_of_vine"] = np.nan
    merged["instance_row_start_s"] = np.nan
    merged["instance_row_end_s"] = np.nan
    merged["instance_row_center_s"] = np.nan

    numeric_columns = ["bbox_x_min", "bbox_w", "bbox_x_max"]
    for column in numeric_columns:
        merged[column] = pd.to_numeric(merged[column], errors="coerce")

    ok_feature_mask = merged["status"].fillna("").eq("ok")
    quality_mask = (
        ok_feature_mask
        & merged["matched_row_id"].notna()
        & merged["coverage_start_s"].notna()
        & merged["coverage_end_s"].notna()
        & merged["frame_row_length_s"].abs().gt(EPSILON)
        & ~merged["coverage_outlier_bool"].fillna(True)
        & merged["coverage_canopy_side"].isin(["N", "S"])
        & merged["bbox_x_min"].notna()
        & merged["bbox_w"].notna()
    )

    merged.loc[~ok_feature_mask, "assignment_status"] = "feature_status_not_ok"
    merged.loc[ok_feature_mask & merged["matched_row_id"].isna(), "assignment_status"] = "no_frame_match"
    merged.loc[ok_feature_mask & merged["coverage_outlier_bool"].fillna(False), "assignment_status"] = "coverage_outlier"
    merged.loc[
        ok_feature_mask & merged["matched_row_id"].notna() & ~merged["coverage_canopy_side"].isin(["N", "S"]),
        "assignment_status",
    ] = "missing_canopy_side"
    merged.loc[
        ok_feature_mask & merged["matched_row_id"].notna() & merged["bbox_x_min"].isna(),
        "assignment_status",
    ] = "missing_bbox"

    quality_indices = merged.index[quality_mask]
    for index in quality_indices:
        row = merged.loc[index]
        frame_start_s = float(row["coverage_start_s"])
        frame_length_s = float(row["frame_row_length_s"])
        instance_start_ratio = float(row["bbox_x_min"]) / image_width_px
        instance_end_ratio = float(row["bbox_x_min"] + row["bbox_w"]) / image_width_px
        instance_start_s = frame_start_s + instance_start_ratio * frame_length_s
        instance_end_s = frame_start_s + instance_end_ratio * frame_length_s
        instance_interval = (
            min(instance_start_s, instance_end_s),
            max(instance_start_s, instance_end_s),
        )
        instance_length_s = instance_interval[1] - instance_interval[0]
        instance_center_s = (instance_interval[0] + instance_interval[1]) / 2.0

        row_id = int(row["matched_row_id"])
        vine_intervals = vine_lookup.get(row_id)
        if vine_intervals is None or vine_intervals.empty:
            merged.at[index, "assignment_status"] = "row_model_missing"
            continue

        candidates = vine_intervals.copy()
        candidates["overlap_length_m"] = candidates.apply(
            lambda vine: overlap_length(
                (float(vine["coverage_s_min"]), float(vine["coverage_s_max"])),
                instance_interval,
            ),
            axis=1,
        )
        candidates = candidates[candidates["overlap_length_m"] > EPSILON].copy()
        if candidates.empty:
            candidates = vine_intervals.copy()
            candidates["center_distance"] = (
                candidates["s"].astype(float) - instance_center_s
            ).abs()
            best = candidates.sort_values(["center_distance", "vine_id"], kind="stable").iloc[0]
            merged.at[index, "assignment_status"] = "assigned_by_nearest_center"
            merged.at[index, "assigned_vine_id"] = best["Name"]
            merged.at[index, "assigned_canopy_side"] = row["coverage_canopy_side"]
            merged.at[index, "instance_row_start_s"] = instance_interval[0]
            merged.at[index, "instance_row_end_s"] = instance_interval[1]
            merged.at[index, "instance_row_center_s"] = instance_center_s
            continue

        candidates["overlap_fraction_of_instance"] = np.where(
            instance_length_s > EPSILON,
            candidates["overlap_length_m"] / instance_length_s,
            0.0,
        )
        candidates["overlap_fraction_of_vine"] = np.where(
            candidates["coverage_length_m"].astype(float) > EPSILON,
            candidates["overlap_length_m"] / candidates["coverage_length_m"].astype(float),
            0.0,
        )
        candidates["center_distance"] = (candidates["s"].astype(float) - instance_center_s).abs()
        best = candidates.sort_values(
            ["overlap_length_m", "overlap_fraction_of_instance", "overlap_fraction_of_vine", "center_distance"],
            ascending=[False, False, False, True],
            kind="stable",
        ).iloc[0]

        merged.at[index, "assignment_status"] = "assigned_by_overlap"
        merged.at[index, "assigned_vine_id"] = best["Name"]
        merged.at[index, "assigned_canopy_side"] = row["coverage_canopy_side"]
        merged.at[index, "best_overlap_length_m"] = float(best["overlap_length_m"])
        merged.at[index, "best_overlap_fraction_of_instance"] = float(best["overlap_fraction_of_instance"])
        merged.at[index, "best_overlap_fraction_of_vine"] = float(best["overlap_fraction_of_vine"])
        merged.at[index, "instance_row_start_s"] = instance_interval[0]
        merged.at[index, "instance_row_end_s"] = instance_interval[1]
        merged.at[index, "instance_row_center_s"] = instance_center_s

    return merged


def vine_sort_key(series: pd.Series) -> pd.DataFrame:
    extracted = series.astype(str).str.extract(r"^row_(\d+)_(\d+)$")
    return pd.DataFrame(
        {
            "row_num": pd.to_numeric(extracted[0], errors="coerce"),
            "vine_num": pd.to_numeric(extracted[1], errors="coerce"),
        }
    )


def color_feature_columns(df: pd.DataFrame) -> list[str]:
    columns: list[str] = []
    for column in df.columns:
        if not pd.api.types.is_numeric_dtype(df[column]):
            continue
        if column in NUMERIC_EXCLUDE_COLUMNS:
            continue
        if any(column.startswith(prefix) for prefix in NUMERIC_EXCLUDE_PREFIXES):
            continue
        if column.startswith("rgb_") or column.startswith("hsv_") or column.startswith("lab_"):
            columns.append(column)
    return columns


def sample_embedding_dataframe(df: pd.DataFrame, max_points: int) -> pd.DataFrame:
    if len(df) <= max_points:
        return df.copy()

    samples: list[pd.DataFrame] = []
    for side, side_df in df.groupby("assigned_canopy_side"):
        n_side = max(1, round(max_points * len(side_df) / len(df)))
        samples.append(side_df.sample(n=min(n_side, len(side_df)), random_state=RANDOM_SEED))
    sampled = pd.concat(samples, ignore_index=False).drop_duplicates()
    if len(sampled) > max_points:
        sampled = sampled.sample(n=max_points, random_state=RANDOM_SEED)
    return sampled.copy()


def plot_lab_2d(df: pd.DataFrame, output_path: Path) -> None:
    if df.empty:
        return
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    color_map = {"N": "#1f77b4", "S": "#d62728"}
    pairs = [
        ("lab_a_mean", "lab_b_mean"),
        ("lab_l_mean", "lab_a_mean"),
        ("lab_l_mean", "lab_b_mean"),
    ]
    for ax, (x_col, y_col) in zip(axes, pairs):
        for side, side_df in df.groupby("assigned_canopy_side"):
            ax.scatter(
                side_df[x_col],
                side_df[y_col],
                s=8,
                alpha=0.35,
                label=side,
                color=color_map.get(side, "0.5"),
            )
        ax.set_xlabel(x_col)
        ax.set_ylabel(y_col)
        ax.grid(True, alpha=0.2)
    axes[0].legend(title="Side")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_lab_3d(df: pd.DataFrame, output_path: Path) -> None:
    if df.empty:
        return
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")
    color_map = {"N": "#1f77b4", "S": "#d62728"}
    for side, side_df in df.groupby("assigned_canopy_side"):
        ax.scatter(
            side_df["lab_l_mean"],
            side_df["lab_a_mean"],
            side_df["lab_b_mean"],
            s=8,
            alpha=0.35,
            label=side,
            color=color_map.get(side, "0.5"),
        )
    ax.set_xlabel("lab_l_mean")
    ax.set_ylabel("lab_a_mean")
    ax.set_zlabel("lab_b_mean")
    ax.legend(title="Side")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_embedding(
    embedded: np.ndarray,
    labels: pd.Series,
    output_path: Path,
    title: str,
    axis_labels: tuple[str, str],
) -> None:
    fig, ax = plt.subplots(figsize=(7, 6))
    color_map = {"N": "#1f77b4", "S": "#d62728"}
    for side in sorted(labels.unique()):
        mask = labels == side
        ax.scatter(
            embedded[mask, 0],
            embedded[mask, 1],
            s=10,
            alpha=0.45,
            label=side,
            color=color_map.get(side, "0.5"),
        )
    ax.set_title(title)
    ax.set_xlabel(axis_labels[0])
    ax.set_ylabel(axis_labels[1])
    ax.grid(True, alpha=0.2)
    ax.legend(title="Side")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def run_embeddings(
    df: pd.DataFrame,
    feature_columns: list[str],
    output_dir: Path,
    max_points: int,
) -> None:
    embedding_df = sample_embedding_dataframe(df, max_points)
    if len(embedding_df) < 10:
        return
    matrix = embedding_df[feature_columns].replace([np.inf, -np.inf], np.nan).dropna()
    embedding_df = embedding_df.loc[matrix.index].copy()
    if len(embedding_df) < 10:
        return

    scaled = StandardScaler().fit_transform(matrix.to_numpy(dtype=float))
    labels = embedding_df["assigned_canopy_side"].astype(str)

    if umap is not None:
        reducer = umap.UMAP(n_components=2, random_state=RANDOM_SEED)
        embedded_umap = reducer.fit_transform(scaled)
        plot_embedding(
            embedded_umap,
            labels,
            output_dir / "umap_color_features_by_side.png",
            "UMAP Of Color Features",
            ("UMAP-1", "UMAP-2"),
        )

    tsne = TSNE(
        n_components=2,
        random_state=RANDOM_SEED,
        init="pca",
        learning_rate="auto",
        perplexity=min(30.0, max(5.0, len(embedding_df) / 20.0)),
    )
    embedded_tsne = tsne.fit_transform(scaled)
    plot_embedding(
        embedded_tsne,
        labels,
        output_dir / "tsne_color_features_by_side.png",
        "t-SNE Of Color Features",
        ("t-SNE-1", "t-SNE-2"),
    )


def write_outputs(df: pd.DataFrame, strategy_dir: Path) -> None:
    strategy_dir.mkdir(parents=True, exist_ok=True)
    sortable = df.copy()
    vine_keys = vine_sort_key(sortable["assigned_vine_id"])
    sortable["sort_row_num"] = vine_keys["row_num"]
    sortable["sort_vine_num"] = vine_keys["vine_num"]
    sortable["frame_id_numeric"] = pd.to_numeric(sortable["frame_id"], errors="coerce")
    sortable["instance_sort_key"] = sortable["instance_id"].astype(str)

    by_vine = sortable.sort_values(
        [
            "sort_row_num",
            "sort_vine_num",
            "assigned_canopy_side",
            "frame_id_numeric",
            "instance_sort_key",
        ],
        kind="stable",
        na_position="last",
    ).drop(columns=["sort_row_num", "sort_vine_num", "frame_id_numeric", "instance_sort_key"])
    by_instance = sortable.sort_values(
        ["frame_id_numeric", "instance_sort_key"],
        kind="stable",
        na_position="last",
    ).drop(columns=["sort_row_num", "sort_vine_num", "frame_id_numeric", "instance_sort_key"])

    by_vine.to_csv(strategy_dir / "berry_instances_assigned_to_vines_sorted_by_vine.csv", index=False)
    by_instance.to_csv(strategy_dir / "berry_instances_assigned_to_vines_sorted_by_instance.csv", index=False)


def process_feature_file(
    feature_csv: Path,
    strategy_root: Path,
    coverage: pd.DataFrame,
    vine_lookup: dict[int, pd.DataFrame],
    max_embedding_points: int,
) -> None:
    suffix = strategy_suffix_from_path(feature_csv)
    strategy_dir = strategy_root / f"berry_instance_depth_filtered_{suffix}"
    print(f"[{suffix}] Loading feature CSV: {feature_csv}", flush=True)
    features = pd.read_csv(feature_csv, low_memory=False)
    features.columns = [str(column).strip() for column in features.columns]
    features["frame_id_key"] = features["frame_id"].map(normalize_frame_id)

    assigned = assign_instances(features, coverage, vine_lookup)
    write_outputs(assigned, strategy_dir)
    print(f"[{suffix}] Wrote sorted CSVs to: {strategy_dir}", flush=True)

    assigned_ok = assigned[
        assigned["assignment_status"].isin({"assigned_by_overlap", "assigned_by_nearest_center"})
        & assigned["assigned_canopy_side"].isin(["N", "S"])
        & assigned["status"].fillna("").eq("ok")
    ].copy()
    feature_columns = color_feature_columns(assigned_ok)
    if {"lab_l_mean", "lab_a_mean", "lab_b_mean"}.issubset(assigned_ok.columns):
        plot_lab_2d(assigned_ok, strategy_dir / "lab_space_2d.png")
        plot_lab_3d(assigned_ok, strategy_dir / "lab_space_3d.png")
    if feature_columns:
        run_embeddings(
            assigned_ok,
            feature_columns=feature_columns,
            output_dir=strategy_dir,
            max_points=max_embedding_points,
        )
    print(f"[{suffix}] Wrote visualizations to: {strategy_dir}", flush=True)

    summary = {
        "strategy_suffix": suffix,
        "feature_csv": str(feature_csv),
        "rows_total": len(assigned),
        "rows_feature_status_ok": int(assigned["status"].fillna("").eq("ok").sum()),
        "rows_assigned_by_overlap": int(assigned["assignment_status"].eq("assigned_by_overlap").sum()),
        "rows_assigned_by_nearest_center": int(
            assigned["assignment_status"].eq("assigned_by_nearest_center").sum()
        ),
        "rows_with_final_assignment": int(
            assigned["assignment_status"].isin({"assigned_by_overlap", "assigned_by_nearest_center"}).sum()
        ),
        "rows_n_side": int(assigned["assigned_canopy_side"].eq("N").sum()),
        "rows_s_side": int(assigned["assigned_canopy_side"].eq("S").sum()),
    }
    pd.DataFrame([summary]).to_csv(strategy_dir / "assignment_summary.csv", index=False)
    print(f"[{suffix}] Summary saved.", flush=True)


def main() -> None:
    args = parse_args()
    row_models, _ = load_row_models(
        aligned_csv=args.aligned_csv,
        neighbor_threshold_m=args.neighbor_threshold_m,
        edge_extension_m=args.edge_extension_m,
    )
    vine_lookup = build_vine_interval_lookup(row_models)
    coverage = load_coverage_dataframe(args.coverage_csv)

    if args.feature_csv is not None:
        feature_files = [args.feature_csv]
    else:
        feature_files = sorted(args.feature_dir.glob("berry_cluster_instance_color_features_depth_filter_*.csv"))
    if not feature_files:
        raise FileNotFoundError(f"No feature CSVs found in {args.feature_dir}")

    for feature_csv in feature_files:
        process_feature_file(
            feature_csv=feature_csv,
            strategy_root=args.strategy_root,
            coverage=coverage,
            vine_lookup=vine_lookup,
            max_embedding_points=args.max_embedding_points,
        )
        print(f"Finished: {feature_csv.name}")


if __name__ == "__main__":
    main()
