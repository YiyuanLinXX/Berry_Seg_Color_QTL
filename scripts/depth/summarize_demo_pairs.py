import os
import csv
import argparse
from pathlib import Path

import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(description="Summarize run_demo_pairs outputs into one CSV.")
    parser.add_argument("--out_dir", type=str, required=True, help="root output directory from run_demo_pairs.py")
    parser.add_argument(
        "--csv_file",
        type=str,
        default=None,
        help="output csv path, default is <out_dir>/summary.csv",
    )
    parser.add_argument(
        "--failure_log",
        type=str,
        default=None,
        help="failure log csv path, default is <out_dir>/failures.csv",
    )
    return parser.parse_args()


def collect_failures(failure_log: Path):
    failures = {}
    if not failure_log.exists():
        return failures
    with open(failure_log, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            failures[row["sample_name"]] = row
    return failures


def find_sample_dirs(out_dir: Path):
    sample_dirs = []
    for vis_file in out_dir.rglob("vis.png"):
        sample_dirs.append(vis_file.parent)
    for depth_file in out_dir.rglob("depth_meter.npy"):
        sample_dirs.append(depth_file.parent)
    for depth_png_file in out_dir.rglob("depth_mm.png"):
        sample_dirs.append(depth_png_file.parent)
    for cloud_file in out_dir.rglob("cloud.ply"):
        sample_dirs.append(cloud_file.parent)
    return sorted(set(sample_dirs))


def depth_shape(depth_file: Path):
    if not depth_file.exists():
        return "", ""
    depth = np.load(depth_file, mmap_mode="r")
    if depth.ndim < 2:
        return "", ""
    return depth.shape[0], depth.shape[1]


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    csv_file = Path(args.csv_file) if args.csv_file else out_dir / "summary.csv"
    failure_log = Path(args.failure_log) if args.failure_log else out_dir / "failures.csv"

    sample_dirs = find_sample_dirs(out_dir)
    failures = collect_failures(failure_log)
    success_names = {str(sample_dir.relative_to(out_dir)).replace("\\", "/") for sample_dir in sample_dirs}
    all_names = sorted(success_names | set(failures.keys()))

    csv_file.parent.mkdir(parents=True, exist_ok=True)
    with open(csv_file, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "sample_name",
                "status",
                "left_file",
                "right_file",
                "reason",
                "sample_dir",
                "has_vis",
                "has_depth_npy",
                "has_depth_png",
                "has_cloud",
                "has_cloud_denoise",
                "depth_height",
                "depth_width",
            ],
        )
        writer.writeheader()

        for sample_name in all_names:
            sample_dir = out_dir / sample_name
            vis_file = sample_dir / "vis.png"
            depth_file = sample_dir / "depth_meter.npy"
            depth_png_file = sample_dir / "depth_mm.png"
            cloud_file = sample_dir / "cloud.ply"
            cloud_denoise_file = sample_dir / "cloud_denoise.ply"
            failure = failures.get(sample_name, {})
            height, width = depth_shape(depth_file)

            if sample_name in success_names and failure.get("status") not in {"failed", "missing_file"}:
                status = "ok" if sample_name not in failures else failure.get("status", "ok")
            else:
                status = failure.get("status", "unknown")

            writer.writerow(
                {
                    "sample_name": sample_name,
                    "status": status,
                    "left_file": failure.get("left_file", ""),
                    "right_file": failure.get("right_file", ""),
                    "reason": failure.get("reason", ""),
                    "sample_dir": str(sample_dir),
                    "has_vis": int(vis_file.exists()),
                    "has_depth_npy": int(depth_file.exists()),
                    "has_depth_png": int(depth_png_file.exists()),
                    "has_cloud": int(cloud_file.exists()),
                    "has_cloud_denoise": int(cloud_denoise_file.exists()),
                    "depth_height": height,
                    "depth_width": width,
                }
            )

    print(f"Summary saved to {csv_file}")
    print(f"Samples found: {len(all_names)}")


if __name__ == "__main__":
    main()
