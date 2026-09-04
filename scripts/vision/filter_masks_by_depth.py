import csv
import argparse
from pathlib import Path

import cv2
import imageio.v2 as imageio
import numpy as np
from tqdm.auto import tqdm


def parse_args():
    parser = argparse.ArgumentParser(
        description="Filter mask connected components by comparing their nearest depth against the image-wide depth percentile."
    )
    parser.add_argument("--mask_dir", type=str, required=True, help="directory containing mask png files")
    parser.add_argument(
        "--depth_dir",
        type=str,
        required=True,
        help="directory containing per-frame depth subfolders, each with depth_meter.npy",
    )
    parser.add_argument("--csv_file", type=str, required=True, help="csv file listing valid frames")
    parser.add_argument("--out_dir", type=str, required=True, help="directory to save filtered masks")
    parser.add_argument(
        "--frame_id_column",
        type=str,
        default="Frame ID",
        help="column name in csv used to locate image ids",
    )
    parser.add_argument(
        "--frame_id_width",
        type=int,
        default=5,
        help="zero-pad width for frame ids, set 0 to disable padding",
    )
    parser.add_argument(
        "--mask_ext",
        type=str,
        default=".png",
        help="extension for input and output mask files",
    )
    parser.add_argument(
        "--depth_filename",
        type=str,
        default="depth_meter.npy",
        help="depth filename inside each frame subfolder",
    )
    parser.add_argument(
        "--keep_percentile",
        type=float,
        default=50.0,
        help="keep component if its nearest depth is closer than this image percentile, e.g. 50 means median depth",
    )
    parser.add_argument(
        "--connectivity",
        type=int,
        default=8,
        choices=[4, 8],
        help="connected-components connectivity",
    )
    parser.add_argument(
        "--min_component_area",
        type=int,
        default=1,
        help="discard connected components smaller than this area before depth filtering",
    )
    parser.add_argument(
        "--positive_depth_only",
        type=int,
        default=1,
        help="when enabled, only positive finite depths are treated as valid",
    )
    parser.add_argument(
        "--overwrite",
        type=int,
        default=0,
        help="overwrite existing filtered mask outputs",
    )
    parser.add_argument(
        "--summary_csv",
        type=str,
        default=None,
        help="optional summary csv path, default is <out_dir>/filter_summary.csv",
    )
    return parser.parse_args()


def normalize_frame_id(frame_id_raw: str, width: int):
    frame_id = str(frame_id_raw).strip()
    if frame_id.endswith(".0"):
        frame_id = frame_id[:-2]
    if width > 0:
        frame_id = frame_id.zfill(width)
    return frame_id


def load_frame_ids(csv_file: Path, frame_id_column: str, frame_id_width: int):
    frame_ids = []
    with open(csv_file, newline="") as f:
        reader = csv.DictReader(f)
        if frame_id_column not in reader.fieldnames:
            raise ValueError(
                f"Column '{frame_id_column}' not found in {csv_file}. Available columns: {reader.fieldnames}"
            )
        for row in reader:
            frame_id = normalize_frame_id(row[frame_id_column], frame_id_width)
            frame_ids.append(frame_id)
    return frame_ids


def valid_depth_mask(depth: np.ndarray, positive_only: bool):
    mask = np.isfinite(depth)
    if positive_only:
        mask &= depth > 0
    return mask


def nearest_component_depth(component_mask: np.ndarray, depth: np.ndarray, positive_only: bool):
    local_depth = depth[component_mask]
    valid = np.isfinite(local_depth)
    if positive_only:
        valid &= local_depth > 0
    if not np.any(valid):
        return None
    return float(local_depth[valid].min())


def filter_single_mask(mask: np.ndarray, depth: np.ndarray, keep_percentile: float, connectivity: int, min_component_area: int, positive_only: bool):
    if mask.shape[:2] != depth.shape[:2]:
        raise ValueError(f"Mask shape {mask.shape} does not match depth shape {depth.shape}")

    mask_binary = (mask > 0).astype(np.uint8)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask_binary, connectivity=connectivity)

    image_valid = valid_depth_mask(depth, positive_only)
    if not np.any(image_valid):
        return np.zeros_like(mask), {
            "image_depth_threshold": "",
            "num_components_total": int(max(num_labels - 1, 0)),
            "num_components_kept": 0,
            "num_components_removed": int(max(num_labels - 1, 0)),
            "num_components_missing_depth": int(max(num_labels - 1, 0)),
        }, []

    threshold_depth = float(np.percentile(depth[image_valid], keep_percentile))
    filtered_mask = np.zeros_like(mask)
    kept = 0
    removed = 0
    missing_depth = 0
    component_rows = []

    for label_id in range(1, num_labels):
        area = int(stats[label_id, cv2.CC_STAT_AREA])
        component_mask = labels == label_id

        if area < min_component_area:
            removed += 1
            component_rows.append(
                {
                    "component_id": label_id,
                    "area": area,
                    "nearest_depth_m": "",
                    "threshold_depth_m": threshold_depth,
                    "status": "removed_small_area",
                }
            )
            continue

        nearest_depth = nearest_component_depth(component_mask, depth, positive_only)
        if nearest_depth is None:
            missing_depth += 1
            removed += 1
            component_rows.append(
                {
                    "component_id": label_id,
                    "area": area,
                    "nearest_depth_m": "",
                    "threshold_depth_m": threshold_depth,
                    "status": "removed_missing_depth",
                }
            )
            continue

        if nearest_depth <= threshold_depth:
            filtered_mask[component_mask] = mask[component_mask]
            kept += 1
            status = "kept"
        else:
            removed += 1
            status = "removed_far"

        component_rows.append(
            {
                "component_id": label_id,
                "area": area,
                "nearest_depth_m": nearest_depth,
                "threshold_depth_m": threshold_depth,
                "status": status,
            }
        )

    frame_summary = {
        "image_depth_threshold": threshold_depth,
        "num_components_total": int(max(num_labels - 1, 0)),
        "num_components_kept": kept,
        "num_components_removed": removed,
        "num_components_missing_depth": missing_depth,
    }
    return filtered_mask, frame_summary, component_rows


def main():
    args = parse_args()
    mask_dir = Path(args.mask_dir)
    depth_dir = Path(args.depth_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_csv = Path(args.summary_csv) if args.summary_csv else out_dir / "filter_summary.csv"

    frame_ids = load_frame_ids(Path(args.csv_file), args.frame_id_column, args.frame_id_width)

    with open(summary_csv, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "frame_id",
                "mask_file",
                "depth_file",
                "output_mask_file",
                "status",
                "reason",
                "image_depth_threshold_m",
                "num_components_total",
                "num_components_kept",
                "num_components_removed",
                "num_components_missing_depth",
            ],
        )
        writer.writeheader()

        progress = tqdm(frame_ids, desc="Filtering masks by depth", unit="frame")
        for frame_id in progress:
            progress.set_postfix_str(frame_id)
            mask_file = mask_dir / f"{frame_id}{args.mask_ext}"
            depth_file = depth_dir / frame_id / args.depth_filename
            output_mask_file = out_dir / f"{frame_id}{args.mask_ext}"

            if output_mask_file.exists() and not args.overwrite:
                writer.writerow(
                    {
                        "frame_id": frame_id,
                        "mask_file": str(mask_file),
                        "depth_file": str(depth_file),
                        "output_mask_file": str(output_mask_file),
                        "status": "skipped_existing",
                        "reason": "output exists",
                        "image_depth_threshold_m": "",
                        "num_components_total": "",
                        "num_components_kept": "",
                        "num_components_removed": "",
                        "num_components_missing_depth": "",
                    }
                )
                continue

            if not mask_file.exists():
                writer.writerow(
                    {
                        "frame_id": frame_id,
                        "mask_file": str(mask_file),
                        "depth_file": str(depth_file),
                        "output_mask_file": str(output_mask_file),
                        "status": "missing_mask",
                        "reason": "mask file not found",
                        "image_depth_threshold_m": "",
                        "num_components_total": "",
                        "num_components_kept": "",
                        "num_components_removed": "",
                        "num_components_missing_depth": "",
                    }
                )
                continue

            if not depth_file.exists():
                writer.writerow(
                    {
                        "frame_id": frame_id,
                        "mask_file": str(mask_file),
                        "depth_file": str(depth_file),
                        "output_mask_file": str(output_mask_file),
                        "status": "missing_depth",
                        "reason": "depth file not found",
                        "image_depth_threshold_m": "",
                        "num_components_total": "",
                        "num_components_kept": "",
                        "num_components_removed": "",
                        "num_components_missing_depth": "",
                    }
                )
                continue

            try:
                mask = imageio.imread(mask_file)
                depth = np.load(depth_file)
                filtered_mask, frame_summary, _ = filter_single_mask(
                    mask=mask,
                    depth=depth,
                    keep_percentile=args.keep_percentile,
                    connectivity=args.connectivity,
                    min_component_area=args.min_component_area,
                    positive_only=bool(args.positive_depth_only),
                )
                imageio.imwrite(output_mask_file, filtered_mask)
                writer.writerow(
                    {
                        "frame_id": frame_id,
                        "mask_file": str(mask_file),
                        "depth_file": str(depth_file),
                        "output_mask_file": str(output_mask_file),
                        "status": "ok",
                        "reason": "",
                        "image_depth_threshold_m": frame_summary["image_depth_threshold"],
                        "num_components_total": frame_summary["num_components_total"],
                        "num_components_kept": frame_summary["num_components_kept"],
                        "num_components_removed": frame_summary["num_components_removed"],
                        "num_components_missing_depth": frame_summary["num_components_missing_depth"],
                    }
                )
            except Exception as exc:
                writer.writerow(
                    {
                        "frame_id": frame_id,
                        "mask_file": str(mask_file),
                        "depth_file": str(depth_file),
                        "output_mask_file": str(output_mask_file),
                        "status": "failed",
                        "reason": str(exc),
                        "image_depth_threshold_m": "",
                        "num_components_total": "",
                        "num_components_kept": "",
                        "num_components_removed": "",
                        "num_components_missing_depth": "",
                    }
                )

    print(f"Saved filtered masks to {out_dir}")
    print(f"Saved summary csv to {summary_csv}")


if __name__ == "__main__":
    main()
