import csv
import argparse
from pathlib import Path

import cv2
import imageio.v2 as imageio
import numpy as np
from tqdm.auto import tqdm


def parse_args():
    parser = argparse.ArgumentParser(
        description="Split binary masks into per-instance files using connected components."
    )
    parser.add_argument("--mask_dir", type=str, required=True, help="directory containing filtered mask images")
    parser.add_argument("--out_dir", type=str, required=True, help="directory to save per-instance masks")
    parser.add_argument("--mask_ext", type=str, default=".png", help="input mask extension")
    parser.add_argument(
        "--pattern",
        type=str,
        default="*.png",
        help="glob pattern used to find mask files under mask_dir",
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
        help="discard connected components smaller than this area",
    )
    parser.add_argument(
        "--output_layout",
        type=str,
        default="per_frame_dir",
        choices=["per_frame_dir", "flat"],
        help="how to organize exported instance files",
    )
    parser.add_argument(
        "--instance_prefix",
        type=str,
        default="instance",
        help="prefix used for exported instance filenames",
    )
    parser.add_argument(
        "--summary_csv",
        type=str,
        default=None,
        help="optional summary csv path, default is <out_dir>/instance_summary.csv",
    )
    parser.add_argument(
        "--overwrite",
        type=int,
        default=0,
        help="overwrite existing instance files",
    )
    return parser.parse_args()


def list_mask_files(mask_dir: Path, pattern: str, mask_ext: str):
    files = []
    for path in mask_dir.glob(pattern):
        if path.is_file() and path.suffix.lower() == mask_ext.lower():
            files.append(path)
    return sorted(files)


def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def export_single_mask(mask_file: Path, out_dir: Path, args, writer):
    mask = imageio.imread(mask_file)
    mask_binary = (mask > 0).astype(np.uint8)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask_binary, connectivity=args.connectivity
    )

    frame_id = mask_file.stem
    frame_out_dir = out_dir / frame_id if args.output_layout == "per_frame_dir" else out_dir
    ensure_dir(frame_out_dir)

    exported = 0
    skipped_small = 0
    for label_id in range(1, num_labels):
        area = int(stats[label_id, cv2.CC_STAT_AREA])
        if area < args.min_component_area:
            skipped_small += 1
            writer.writerow(
                {
                    "frame_id": frame_id,
                    "source_mask_file": str(mask_file),
                    "instance_id": label_id,
                    "instance_file": "",
                    "area": area,
                    "bbox_x": int(stats[label_id, cv2.CC_STAT_LEFT]),
                    "bbox_y": int(stats[label_id, cv2.CC_STAT_TOP]),
                    "bbox_w": int(stats[label_id, cv2.CC_STAT_WIDTH]),
                    "bbox_h": int(stats[label_id, cv2.CC_STAT_HEIGHT]),
                    "status": "skipped_small_area",
                }
            )
            continue

        instance_mask = np.zeros_like(mask, dtype=np.uint8)
        instance_mask[labels == label_id] = 255

        if args.output_layout == "per_frame_dir":
            instance_file = frame_out_dir / f"{args.instance_prefix}_{label_id:03d}{args.mask_ext}"
        else:
            instance_file = frame_out_dir / f"{frame_id}_{args.instance_prefix}_{label_id:03d}{args.mask_ext}"

        if instance_file.exists() and not args.overwrite:
            status = "skipped_existing"
        else:
            imageio.imwrite(instance_file, instance_mask)
            status = "ok"
            exported += 1

        writer.writerow(
            {
                "frame_id": frame_id,
                "source_mask_file": str(mask_file),
                "instance_id": label_id,
                "instance_file": str(instance_file),
                "area": area,
                "bbox_x": int(stats[label_id, cv2.CC_STAT_LEFT]),
                "bbox_y": int(stats[label_id, cv2.CC_STAT_TOP]),
                "bbox_w": int(stats[label_id, cv2.CC_STAT_WIDTH]),
                "bbox_h": int(stats[label_id, cv2.CC_STAT_HEIGHT]),
                "status": status,
            }
        )

    if num_labels == 1:
        writer.writerow(
            {
                "frame_id": frame_id,
                "source_mask_file": str(mask_file),
                "instance_id": "",
                "instance_file": "",
                "area": "",
                "bbox_x": "",
                "bbox_y": "",
                "bbox_w": "",
                "bbox_h": "",
                "status": "no_instance",
            }
        )

    return exported, skipped_small, max(num_labels - 1, 0)


def main():
    args = parse_args()
    mask_dir = Path(args.mask_dir)
    out_dir = Path(args.out_dir)
    ensure_dir(out_dir)
    summary_csv = Path(args.summary_csv) if args.summary_csv else out_dir / "instance_summary.csv"

    mask_files = list_mask_files(mask_dir, args.pattern, args.mask_ext)
    if not mask_files:
        raise ValueError(f"No mask files found in {mask_dir} with pattern {args.pattern}")

    total_instances = 0
    total_exported = 0
    total_skipped_small = 0

    with open(summary_csv, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "frame_id",
                "source_mask_file",
                "instance_id",
                "instance_file",
                "area",
                "bbox_x",
                "bbox_y",
                "bbox_w",
                "bbox_h",
                "status",
            ],
        )
        writer.writeheader()

        progress = tqdm(mask_files, desc="Exporting mask instances", unit="mask")
        for mask_file in progress:
            progress.set_postfix_str(mask_file.stem)
            exported, skipped_small, total_for_mask = export_single_mask(mask_file, out_dir, args, writer)
            total_instances += total_for_mask
            total_exported += exported
            total_skipped_small += skipped_small

    print(f"Processed {len(mask_files)} masks")
    print(f"Total instances found: {total_instances}")
    print(f"Exported instances: {total_exported}")
    print(f"Skipped small instances: {total_skipped_small}")
    print(f"Summary saved to {summary_csv}")


if __name__ == "__main__":
    main()
