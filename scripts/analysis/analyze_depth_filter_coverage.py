import argparse
import os
import tempfile
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "matplotlib"))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


THRESHOLDS = [20, 40, 60, 80, 100]
BASELINE_THRESHOLD = 100
OUTPUT_DIRNAME = "depth_filter_coverage_analysis"


def find_instance_file(threshold_dir):
    candidates = sorted(
        threshold_dir.rglob("berry_instances_assigned_to_vines_sorted_by_instance.csv")
    )
    if candidates:
        return candidates[0]

    fallback_candidates = sorted(threshold_dir.rglob("*.csv"))
    for candidate in fallback_candidates:
        try:
            sample = pd.read_csv(candidate, nrows=5)
        except Exception:
            continue
        if any("vine" in column.lower() for column in sample.columns):
            return candidate

    raise FileNotFoundError(f"Could not find an instance-level CSV in {threshold_dir}")


def find_column(columns, candidates):
    lowered_to_original = {column.lower(): column for column in columns}
    for candidate in candidates:
        if candidate.lower() in lowered_to_original:
            return lowered_to_original[candidate.lower()]
    raise ValueError(
        f"Could not infer a matching column from candidates {candidates}. "
        f"Available columns: {list(columns)}"
    )


def load_threshold_instances(project_root, threshold):
    threshold_dir = project_root / f"berry_instance_depth_filtered_{threshold}"
    csv_path = find_instance_file(threshold_dir)
    df = pd.read_csv(csv_path, low_memory=False)

    vine_id_col = find_column(df.columns, ["assigned_vine_id", "vine_id", "assigned_vine", "vine"])
    assignment_status_col = None
    for candidate in ["assignment_status", "status"]:
        lowered = {column.lower(): column for column in df.columns}
        if candidate.lower() in lowered:
            assignment_status_col = lowered[candidate.lower()]
            break

    print(f"\nThreshold {threshold}")
    print(f"  Instance file: {csv_path}")
    print(f"  Detected vine ID column: {vine_id_col}")
    print(f"  Detected assignment/status column: {assignment_status_col}")

    assigned_df = df[df[vine_id_col].notna()].copy()
    assigned_df[vine_id_col] = assigned_df[vine_id_col].astype(str)

    print(f"  Total raw rows: {len(df)}")
    print(f"  Assigned instance rows retained: {len(assigned_df)}")
    print(f"  Unique vines with >=1 assigned instance: {assigned_df[vine_id_col].nunique()}")

    if assignment_status_col is not None:
        print("  Assignment/status counts:")
        print(assigned_df[assignment_status_col].value_counts(dropna=False).to_string())

    return assigned_df[[vine_id_col]].rename(columns={vine_id_col: "vine_id"})


def summarize_threshold(assigned_df, threshold):
    counts = (
        assigned_df.groupby("vine_id")
        .size()
        .rename("instance_count")
        .sort_values()
    )

    return {
        "threshold": threshold,
        "total_instances": int(len(assigned_df)),
        "vines_ge_1": int((counts >= 1).sum()),
        "vines_ge_3": int((counts >= 3).sum()),
        "vines_ge_5": int((counts >= 5).sum()),
        "counts_per_vine": counts,
    }


def save_figure(fig, output_dir, stem):
    fig.tight_layout()
    fig.savefig(output_dir / f"{stem}.png", dpi=300)
    fig.savefig(output_dir / f"{stem}.svg")
    plt.close(fig)


def plot_instances_per_vine_boxplot(summary_df, counts_by_threshold, output_dir):
    fig, ax = plt.subplots(figsize=(7, 5))
    data = [counts_by_threshold[threshold].to_numpy() for threshold in THRESHOLDS]
    ax.boxplot(data, tick_labels=[str(threshold) for threshold in THRESHOLDS])
    ax.set_xlabel("Depth Filtering Threshold (τ)", fontsize=11)
    ax.set_ylabel("Number of Instances per Vine", fontsize=11)
    ax.set_title("Number of Instances per Vine Across Depth Filtering Thresholds", fontsize=12)
    ax.tick_params(axis="both", labelsize=10)
    save_figure(fig, output_dir, "instances_per_vine_boxplot")


def plot_vine_coverage(summary_df, output_dir):
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(summary_df["threshold"], summary_df["vines_ge_1"], marker="o", label=">=1 instance")
    ax.plot(summary_df["threshold"], summary_df["vines_ge_3"], marker="o", label=">=3 instances")
    ax.plot(summary_df["threshold"], summary_df["vines_ge_5"], marker="o", label=">=5 instances")
    ax.set_xlabel("Depth Filtering Threshold (τ)", fontsize=11)
    ax.set_ylabel("Number of Vines", fontsize=11)
    ax.set_title("Vine Coverage Across Depth Filtering Thresholds", fontsize=12)
    ax.set_xticks(THRESHOLDS)
    ax.legend(fontsize=10)
    ax.tick_params(axis="both", labelsize=10)
    save_figure(fig, output_dir, "vine_coverage_vs_threshold")


def plot_retained_instance_ratio(summary_df, output_dir):
    baseline_total = float(
        summary_df.loc[summary_df["threshold"] == BASELINE_THRESHOLD, "total_instances"].iloc[0]
    )
    retained_ratio = summary_df["total_instances"] / baseline_total

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(summary_df["threshold"], retained_ratio, marker="o")
    ax.set_xlabel("Depth Filtering Threshold (τ)", fontsize=11)
    ax.set_ylabel("Retained Instance Ratio", fontsize=11)
    ax.set_title("Retained Instance Ratio Across Depth Filtering Thresholds", fontsize=12)
    ax.set_xticks(THRESHOLDS)
    ax.tick_params(axis="both", labelsize=10)
    save_figure(fig, output_dir, "retained_instance_ratio")

    return retained_ratio


def main():
    parser = argparse.ArgumentParser(description="Summarize retained instances and vine coverage by depth threshold.")
    parser.add_argument("--strategy-root", type=Path, default=Path("results/generated/phenotyping"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/generated/depth_filter_coverage"))
    args = parser.parse_args()
    project_root = args.strategy_root
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    summaries = []
    counts_by_threshold = {}

    for threshold in THRESHOLDS:
        assigned_df = load_threshold_instances(project_root, threshold)
        summary = summarize_threshold(assigned_df, threshold)
        summaries.append(
            {
                "threshold": summary["threshold"],
                "total_instances": summary["total_instances"],
                "vines_ge_1": summary["vines_ge_1"],
                "vines_ge_3": summary["vines_ge_3"],
                "vines_ge_5": summary["vines_ge_5"],
            }
        )
        counts_by_threshold[threshold] = summary["counts_per_vine"]

    summary_df = pd.DataFrame(summaries).sort_values("threshold").reset_index(drop=True)
    retained_ratio = plot_retained_instance_ratio(summary_df, output_dir)
    plot_instances_per_vine_boxplot(summary_df, counts_by_threshold, output_dir)
    plot_vine_coverage(summary_df, output_dir)

    summary_df["retained_ratio_vs_tau_100"] = retained_ratio.to_numpy()
    summary_df.to_csv(output_dir / "coverage_summary_by_threshold.csv", index=False)

    print("\nSummary by threshold")
    print(summary_df.to_string(index=False))
    print(f"\nPlots written to: {output_dir}")


if __name__ == "__main__":
    main()
