import csv
import argparse
from pathlib import Path

import cv2
import imageio.v2 as imageio
import numpy as np
from tqdm.auto import tqdm


IMAGE_EXTS = [".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Extract RGB/HSV/LAB color features for each mask instance."
    )
    parser.add_argument(
        "--instance_dir",
        type=str,
        required=True,
        help="directory containing instance masks exported by export_mask_instances.py",
    )
    parser.add_argument(
        "--image_dir",
        type=str,
        required=True,
        help="directory containing original images named by frame id",
    )
    parser.add_argument(
        "--out_csv",
        type=str,
        required=True,
        help="output csv file path",
    )
    parser.add_argument(
        "--instance_layout",
        type=str,
        default="per_frame_dir",
        choices=["per_frame_dir", "flat"],
        help="instance mask layout under instance_dir",
    )
    parser.add_argument(
        "--instance_pattern",
        type=str,
        default="*.png",
        help="glob pattern used to locate instance masks",
    )
    parser.add_argument(
        "--image_ext",
        type=str,
        default="",
        help="optional fixed original image extension, e.g. .png",
    )
    parser.add_argument(
        "--mask_threshold",
        type=int,
        default=0,
        help="pixels greater than this threshold are treated as foreground",
    )
    parser.add_argument(
        "--include_bbox",
        type=int,
        default=1,
        help="include instance bounding box columns",
    )
    parser.add_argument(
        "--save_masked_crop",
        type=int,
        default=1,
        help="save a masked crop visualization for each instance",
    )
    parser.add_argument(
        "--masked_crop_dir",
        type=str,
        default=None,
        help="output directory for masked crops, default is <out_csv_dir>/masked_crops",
    )
    parser.add_argument(
        "--masked_crop_bg",
        type=str,
        default="black",
        choices=["black", "white", "transparent"],
        help="background style for saved masked crops",
    )
    return parser.parse_args()


def find_instance_files(instance_dir: Path, layout: str, pattern: str):
    if layout == "per_frame_dir":
        return sorted([p for p in instance_dir.glob(f"*/*") if p.is_file() and p.match(f"*/{pattern}")])
    return sorted([p for p in instance_dir.glob(pattern) if p.is_file()])


def infer_frame_id(instance_file: Path, layout: str):
    if layout == "per_frame_dir":
        return instance_file.parent.name
    stem = instance_file.stem
    if "_instance_" in stem:
        return stem.split("_instance_")[0]
    parts = stem.split("_")
    if len(parts) < 2:
        raise ValueError(f"Cannot infer frame id from flat instance filename: {instance_file.name}")
    return "_".join(parts[:-2]) if len(parts) > 2 else parts[0]


def resolve_image_file(image_dir: Path, frame_id: str, image_ext: str):
    if image_ext:
        candidate = image_dir / f"{frame_id}{image_ext}"
        if candidate.exists():
            return candidate
        raise FileNotFoundError(f"Image not found: {candidate}")

    for ext in IMAGE_EXTS:
        candidate = image_dir / f"{frame_id}{ext}"
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Image not found for frame_id={frame_id} under {image_dir}")


def channel_stats(values: np.ndarray, prefix: str):
    values = values.astype(np.float32)
    return {
        f"{prefix}_mean": float(values.mean()),
        f"{prefix}_std": float(values.std()),
        f"{prefix}_min": float(values.min()),
        f"{prefix}_max": float(values.max()),
        f"{prefix}_median": float(np.median(values)),
        f"{prefix}_p5": float(np.percentile(values, 5)),
        f"{prefix}_p25": float(np.percentile(values, 25)),
        f"{prefix}_p75": float(np.percentile(values, 75)),
        f"{prefix}_p95": float(np.percentile(values, 95)),
    }


def compute_bbox(mask_binary: np.ndarray):
    ys, xs = np.where(mask_binary)
    if len(xs) == 0:
        return None
    x_min = int(xs.min())
    x_max = int(xs.max())
    y_min = int(ys.min())
    y_max = int(ys.max())
    return {
        "bbox_x_min": x_min,
        "bbox_y_min": y_min,
        "bbox_x_max": x_max,
        "bbox_y_max": y_max,
        "bbox_w": x_max - x_min + 1,
        "bbox_h": y_max - y_min + 1,
    }


def masked_crop(image_rgb: np.ndarray, mask_binary: np.ndarray, bg_mode: str):
    bbox = compute_bbox(mask_binary)
    if bbox is None:
        return None

    y0 = bbox["bbox_y_min"]
    y1 = bbox["bbox_y_max"] + 1
    x0 = bbox["bbox_x_min"]
    x1 = bbox["bbox_x_max"] + 1

    crop_img = image_rgb[y0:y1, x0:x1].copy()
    crop_mask = mask_binary[y0:y1, x0:x1]

    if bg_mode == "transparent":
        rgba = np.zeros((crop_img.shape[0], crop_img.shape[1], 4), dtype=np.uint8)
        rgba[..., :3] = crop_img
        rgba[..., 3] = crop_mask.astype(np.uint8) * 255
        rgba[..., :3][~crop_mask] = 0
        return rgba

    bg_value = 0 if bg_mode == "black" else 255
    crop_img[~crop_mask] = bg_value
    return crop_img


def masked_crop_path(masked_crop_dir: Path, frame_id: str, instance_stem: str, bg_mode: str):
    suffix = ".png"
    frame_dir = masked_crop_dir / frame_id
    frame_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{instance_stem}_masked"
    if bg_mode == "transparent":
        stem += "_rgba"
    return frame_dir / f"{stem}{suffix}"


def extract_features(image_rgb: np.ndarray, mask_binary: np.ndarray):
    rgb_pixels = image_rgb[mask_binary]
    hsv_image = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2HSV)
    hsv_pixels = hsv_image[mask_binary]
    lab_image = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2LAB)
    lab_pixels = lab_image[mask_binary]

    features = {"pixel_count": int(mask_binary.sum())}

    for idx, channel in enumerate(["r", "g", "b"]):
        features.update(channel_stats(rgb_pixels[:, idx], f"rgb_{channel}"))
    for idx, channel in enumerate(["h", "s", "v"]):
        features.update(channel_stats(hsv_pixels[:, idx], f"hsv_{channel}"))
    for idx, channel in enumerate(["l", "a", "b"]):
        features.update(channel_stats(lab_pixels[:, idx], f"lab_{channel}"))

    return features


def build_fieldnames(include_bbox: bool):
    fields = [
        "frame_id",
        "instance_id",
        "instance_file",
        "image_file",
        "pixel_count",
    ]
    if include_bbox:
        fields.extend(["bbox_x_min", "bbox_y_min", "bbox_x_max", "bbox_y_max", "bbox_w", "bbox_h"])
    for space, channels in [
        ("rgb", ["r", "g", "b"]),
        ("hsv", ["h", "s", "v"]),
        ("lab", ["l", "a", "b"]),
    ]:
        for channel in channels:
            for stat in ["mean", "std", "min", "max", "median", "p5", "p25", "p75", "p95"]:
                fields.append(f"{space}_{channel}_{stat}")
    fields.extend(["masked_crop_file", "status", "reason"])
    return fields


def main():
    args = parse_args()
    instance_dir = Path(args.instance_dir)
    image_dir = Path(args.image_dir)
    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    masked_crop_dir = (
        Path(args.masked_crop_dir) if args.masked_crop_dir else out_csv.parent / "masked_crops"
    )
    if args.save_masked_crop:
        masked_crop_dir.mkdir(parents=True, exist_ok=True)

    instance_files = find_instance_files(instance_dir, args.instance_layout, args.instance_pattern)
    if not instance_files:
        raise ValueError(f"No instance masks found in {instance_dir}")

    fieldnames = build_fieldnames(bool(args.include_bbox))
    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        progress = tqdm(instance_files, desc="Extracting instance colors", unit="instance")
        for instance_file in progress:
            progress.set_postfix_str(instance_file.stem)
            frame_id = infer_frame_id(instance_file, args.instance_layout)
            row = {
                "frame_id": frame_id,
                "instance_id": instance_file.stem,
                "instance_file": str(instance_file),
                "image_file": "",
                "masked_crop_file": "",
                "status": "",
                "reason": "",
            }

            try:
                image_file = resolve_image_file(image_dir, frame_id, args.image_ext)
                row["image_file"] = str(image_file)

                mask = imageio.imread(instance_file)
                image_rgb = imageio.imread(image_file)

                if mask.ndim == 3:
                    mask = mask[..., 0]
                if image_rgb.ndim == 2:
                    image_rgb = np.stack([image_rgb] * 3, axis=-1)
                if image_rgb.shape[:2] != mask.shape[:2]:
                    raise ValueError(
                        f"Shape mismatch: image {image_rgb.shape[:2]} vs mask {mask.shape[:2]}"
                    )

                mask_binary = mask > args.mask_threshold
                if not np.any(mask_binary):
                    row["status"] = "empty_mask"
                    row["reason"] = "mask has no foreground pixels"
                    writer.writerow(row)
                    continue

                row.update(extract_features(image_rgb, mask_binary))
                if args.include_bbox:
                    row.update(compute_bbox(mask_binary))
                if args.save_masked_crop:
                    crop = masked_crop(image_rgb, mask_binary, args.masked_crop_bg)
                    crop_file = masked_crop_path(masked_crop_dir, frame_id, instance_file.stem, args.masked_crop_bg)
                    imageio.imwrite(crop_file, crop)
                    row["masked_crop_file"] = str(crop_file)
                row["status"] = "ok"
            except Exception as exc:
                row["status"] = "failed"
                row["reason"] = str(exc)

            writer.writerow(row)

    print(f"Saved color feature CSV to {out_csv}")
    print(f"Processed {len(instance_files)} instance masks")


if __name__ == "__main__":
    main()
