#!/usr/bin/env python3
# SPDX-License-Identifier: LicenseRef-NVIDIA-FoundationStereo

import os
import sys
import csv
import argparse
import logging
from pathlib import Path

import cv2
import imageio
import numpy as np
import torch
from tqdm.auto import tqdm

# This adapter calls the external FoundationStereo checkout. It is derived from
# NVIDIA's demo and therefore follows the FoundationStereo license; see
# LICENSES/FoundationStereo-LICENSE.


IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Batch inference for multiple stereo image pairs."
    )
    parser.add_argument(
        "--foundation_stereo_root",
        type=str,
        default="external/FoundationStereo",
        help="path to the external FoundationStereo checkout",
    )
    parser.add_argument("--left_dir", type=str, default=None, help="left image root directory")
    parser.add_argument("--right_dir", type=str, default=None, help="right image root directory")
    parser.add_argument(
        "--pair_list",
        type=str,
        default=None,
        help="csv/txt file listing image pairs: left_path,right_path[,sample_name]",
    )
    parser.add_argument(
        "--intrinsic_file",
        type=str,
        default=None,
        help="camera intrinsic matrix and baseline file",
    )
    parser.add_argument(
        "--ckpt_dir",
        type=str,
        default=None,
        help="pretrained model path",
    )
    parser.add_argument("--out_dir", type=str, required=True, help="directory to save outputs")
    parser.add_argument(
        "--pattern",
        type=str,
        default="**/*",
        help="glob pattern under left_dir, default recursively scans all files",
    )
    parser.add_argument(
        "--match_mode",
        type=str,
        default="relative",
        choices=["relative", "basename"],
        help="how to match left/right images when using directories",
    )
    parser.add_argument(
        "--output_view",
        type=str,
        default="left",
        choices=["left", "right", "both"],
        help="which camera-view result to save",
    )
    parser.add_argument("--scale", default=1.0, type=float, help="downsize the image by scale, must be <=1")
    parser.add_argument("--hiera", default=0, type=int, help="hierarchical inference for high-res images")
    parser.add_argument("--z_far", default=10.0, type=float, help="max depth to clip in point cloud")
    parser.add_argument("--valid_iters", type=int, default=32, help="number of flow-field updates")
    parser.add_argument("--save_vis", type=int, default=1, help="save vis.png")
    parser.add_argument("--save_depth", type=int, default=1, help="save depth_meter.npy")
    parser.add_argument("--save_depth_png", type=int, default=1, help="save 16-bit depth_mm.png")
    parser.add_argument("--get_pc", type=int, default=0, help="save point cloud output")
    parser.add_argument(
        "--remove_invisible",
        default=1,
        type=int,
        help="remove non-overlapping observations before generating point cloud",
    )
    parser.add_argument("--denoise_cloud", type=int, default=1, help="whether to denoise point cloud")
    parser.add_argument("--denoise_nb_points", type=int, default=30, help="radius outlier removal nb_points")
    parser.add_argument("--denoise_radius", type=float, default=0.03, help="radius outlier removal radius")
    parser.add_argument("--skip_existing", type=int, default=1, help="skip pair if vis.png already exists")
    parser.add_argument(
        "--failure_log",
        type=str,
        default=None,
        help="csv file to record failed or skipped samples, default is <out_dir>/failures.csv",
    )
    return parser.parse_args()


def list_image_files(root: Path, pattern: str):
    files = []
    for path in root.glob(pattern):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTS:
            files.append(path)
    return sorted(files)


def sanitize_relpath(relpath: str):
    return relpath.replace("\\", "/")


def build_pairs_from_dirs(left_dir: Path, right_dir: Path, pattern: str, match_mode: str):
    left_files = list_image_files(left_dir, pattern)
    if not left_files:
        raise ValueError(f"No images found under left_dir: {left_dir}")

    pairs = []
    if match_mode == "relative":
        for left_path in left_files:
            rel = left_path.relative_to(left_dir)
            right_path = right_dir / rel
            if right_path.exists():
                left_sample_name = sanitize_relpath(str(rel.with_suffix("")))
                right_sample_name = sanitize_relpath(str(right_path.relative_to(right_dir).with_suffix("")))
                pairs.append((left_path, right_path, left_sample_name, right_sample_name))
    else:
        right_files = list_image_files(right_dir, pattern)
        right_map = {}
        for right_path in right_files:
            right_map.setdefault(right_path.name, []).append(right_path)
        for left_path in left_files:
            matches = right_map.get(left_path.name, [])
            if len(matches) == 1:
                rel = left_path.relative_to(left_dir)
                left_sample_name = sanitize_relpath(str(rel.with_suffix("")))
                right_sample_name = sanitize_relpath(str(matches[0].relative_to(right_dir).with_suffix("")))
                pairs.append((left_path, matches[0], left_sample_name, right_sample_name))
            elif len(matches) > 1:
                logging.warning("Skip %s because basename match is ambiguous", left_path)

    if not pairs:
        raise ValueError("No matched left/right image pairs found.")
    return pairs


def build_pairs_from_list(pair_list_file: Path):
    pairs = []
    with open(pair_list_file, "r", newline="") as f:
        if pair_list_file.suffix.lower() == ".csv":
            reader = csv.reader(f)
            rows = reader
        else:
            rows = []
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                rows.append([item.strip() for item in line.split(",")])

        for idx, row in enumerate(rows):
            if len(row) < 2:
                continue
            left_path = Path(row[0]).expanduser()
            right_path = Path(row[1]).expanduser()
            sample_name = row[2] if len(row) >= 3 and row[2] else f"pair_{idx:06d}"
            left_sample_name = sanitize_relpath(sample_name)
            right_sample_name = sanitize_relpath(row[3]) if len(row) >= 4 and row[3] else left_sample_name
            pairs.append((left_path, right_path, left_sample_name, right_sample_name))

    if not pairs:
        raise ValueError(f"No valid pairs found in pair_list: {pair_list_file}")
    return pairs


def load_intrinsics(intrinsic_file: str, scale: float):
    with open(intrinsic_file, "r") as f:
        lines = f.readlines()
    K = np.array(list(map(float, lines[0].rstrip().split())), dtype=np.float32).reshape(3, 3)
    baseline = float(lines[1])
    K[:2] *= scale
    return K, baseline


def ensure_parent(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)


def pair_output_dir(out_dir: Path, sample_name: str):
    return out_dir / sample_name


def should_skip_existing(args, sample_out_dir: Path):
    expected_outputs = []
    if args.save_vis:
        expected_outputs.append(sample_out_dir / "vis.png")
    if args.save_depth:
        expected_outputs.append(sample_out_dir / "depth_meter.npy")
    if args.save_depth_png:
        expected_outputs.append(sample_out_dir / "depth_mm.png")
    if args.get_pc:
        expected_outputs.append(sample_out_dir / "cloud.ply")
    return bool(expected_outputs) and all(path.exists() for path in expected_outputs)


def maybe_write_failure_row(writer, sample_name: str, left_file: Path, right_file: Path, status: str, reason: str):
    writer.writerow(
        {
            "sample_name": sample_name,
            "left_file": str(left_file),
            "right_file": str(right_file),
            "status": status,
            "reason": reason,
        }
    )


def save_outputs(args, sample_name: str, image_ref, disp, K_scaled, baseline):
    sample_out_dir = pair_output_dir(Path(args.out_dir), sample_name)
    if args.skip_existing and should_skip_existing(args, sample_out_dir):
        return "skipped"

    ensure_parent(sample_out_dir / "dummy")
    H, W = image_ref.shape[:2]

    if args.save_vis:
        vis = vis_disparity(disp)
        vis = np.concatenate([image_ref, vis], axis=1)
        imageio.imwrite(sample_out_dir / "vis.png", vis)

    depth = K_scaled[0, 0] * baseline / disp
    if args.save_depth:
        np.save(sample_out_dir / "depth_meter.npy", depth)
    if args.save_depth_png:
        depth_mm = depth * 1000.0
        depth_mm = np.nan_to_num(depth_mm, nan=0.0, posinf=0.0, neginf=0.0)
        depth_mm = np.clip(depth_mm, 0, 65535).astype(np.uint16)
        imageio.imwrite(sample_out_dir / "depth_mm.png", depth_mm)

    disp_for_pc = disp.copy()
    if args.remove_invisible:
        yy, xx = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")
        us_right = xx - disp_for_pc
        invalid = us_right < 0
        disp_for_pc[invalid] = np.inf

    if args.get_pc:
        import open3d as o3d

        depth_for_pc = K_scaled[0, 0] * baseline / disp_for_pc
        xyz_map = depth2xyzmap(depth_for_pc, K_scaled)
        pcd = toOpen3dCloud(xyz_map.reshape(-1, 3), image_ref.reshape(-1, 3))
        keep_mask = (np.asarray(pcd.points)[:, 2] > 0) & (np.asarray(pcd.points)[:, 2] <= args.z_far)
        keep_ids = np.arange(len(np.asarray(pcd.points)))[keep_mask]
        pcd = pcd.select_by_index(keep_ids)
        o3d.io.write_point_cloud(str(sample_out_dir / "cloud.ply"), pcd)

        if args.denoise_cloud:
            cl, ind = pcd.remove_radius_outlier(
                nb_points=args.denoise_nb_points, radius=args.denoise_radius
            )
            del cl
            inlier_cloud = pcd.select_by_index(ind)
            o3d.io.write_point_cloud(str(sample_out_dir / "cloud_denoise.ply"), inlier_cloud)

    return "ok"


def run_inference(
    model, args, left_file: Path, right_file: Path, left_sample_name: str, right_sample_name: str, K_scaled, baseline
):
    statuses = []

    img0 = imageio.imread(left_file)
    img1 = imageio.imread(right_file)
    scale = args.scale
    assert scale <= 1, "scale must be <=1"

    img0 = cv2.resize(img0, fx=scale, fy=scale, dsize=None)
    img1 = cv2.resize(img1, fx=scale, fy=scale, dsize=None)
    H, W = img0.shape[:2]
    img0_ori = img0.copy()
    img1_ori = img1.copy()

    img0_tensor = torch.as_tensor(img0).cuda().float()[None].permute(0, 3, 1, 2)
    img1_tensor = torch.as_tensor(img1).cuda().float()[None].permute(0, 3, 1, 2)
    padder = InputPadder(img0_tensor.shape, divis_by=32, force_square=False)
    img0_tensor, img1_tensor = padder.pad(img0_tensor, img1_tensor)

    with torch.cuda.amp.autocast(True):
        disp_left = None
        disp_right = None
        if args.output_view in {"left", "both"}:
            if not args.hiera:
                disp_left = model.forward(img0_tensor, img1_tensor, iters=args.valid_iters, test_mode=True)
            else:
                disp_left = model.run_hierachical(
                    img0_tensor, img1_tensor, iters=args.valid_iters, test_mode=True, small_ratio=0.5
                )
            disp_left = padder.unpad(disp_left.float())

        if args.output_view in {"right", "both"}:
            img0_flipped_tensor = torch.flip(img0_tensor, dims=[3])
            img1_flipped_tensor = torch.flip(img1_tensor, dims=[3])
            if args.hiera:
                raise ValueError("output_view=right/both currently requires --hiera 0")
            disp_right = model.forward(
                img1_flipped_tensor, img0_flipped_tensor, iters=args.valid_iters, test_mode=True
            )
            disp_right = torch.flip(disp_right, dims=[3])
            disp_right = padder.unpad(disp_right.float())

    if disp_left is not None:
        disp_left = disp_left.data.cpu().numpy().reshape(H, W)
        statuses.append(save_outputs(args, left_sample_name, img0_ori, disp_left, K_scaled, baseline))
    if disp_right is not None:
        disp_right = disp_right.data.cpu().numpy().reshape(H, W)
        statuses.append(save_outputs(args, right_sample_name, img1_ori, disp_right, K_scaled, baseline))

    return "skipped" if statuses and all(status == "skipped" for status in statuses) else "ok"


def main():
    global InputPadder, FoundationStereo, OmegaConf, set_logging_format, set_seed
    global vis_disparity, depth2xyzmap, toOpen3dCloud

    args = parse_args()
    from omegaconf import OmegaConf
    foundation_root = Path(args.foundation_stereo_root).resolve()
    if not (foundation_root / "core" / "foundation_stereo.py").exists():
        raise FileNotFoundError(
            f"FoundationStereo checkout not found at {foundation_root}. "
            "Run scripts/setup_external_dependencies.sh first."
        )
    sys.path.insert(0, str(foundation_root))
    from core.utils.utils import InputPadder
    from Utils import set_logging_format, set_seed, vis_disparity, depth2xyzmap, toOpen3dCloud
    from core.foundation_stereo import FoundationStereo

    if args.intrinsic_file is None:
        args.intrinsic_file = str(foundation_root / "assets" / "K.txt")
    if args.ckpt_dir is None:
        args.ckpt_dir = str(
            foundation_root / "pretrained_models" / "23-51-11" / "model_best_bp2.pth"
        )
    if args.pair_list is None and (args.left_dir is None or args.right_dir is None):
        raise ValueError("Provide either --pair_list or both --left_dir and --right_dir.")

    set_logging_format()
    set_seed(0)
    torch.autograd.set_grad_enabled(False)
    os.makedirs(args.out_dir, exist_ok=True)

    ckpt_dir = args.ckpt_dir
    cfg = OmegaConf.load(f"{os.path.dirname(ckpt_dir)}/cfg.yaml")
    if "vit_size" not in cfg:
        cfg["vit_size"] = "vitl"
    for key, value in vars(args).items():
        cfg[key] = value
    args = OmegaConf.create(cfg)

    logging.info("Using pretrained model from %s", ckpt_dir)
    model = FoundationStereo(args)
    ckpt = torch.load(ckpt_dir)
    model.load_state_dict(ckpt["model"])
    model.cuda()
    model.eval()

    if args.pair_list is not None:
        pairs = build_pairs_from_list(Path(args.pair_list))
    else:
        pairs = build_pairs_from_dirs(
            Path(args.left_dir), Path(args.right_dir), args.pattern, args.match_mode
        )

    K_scaled, baseline = load_intrinsics(args.intrinsic_file, args.scale)
    logging.info("Found %d pairs", len(pairs))

    failure_log = Path(args.failure_log) if args.failure_log else Path(args.out_dir) / "failures.csv"
    ensure_parent(failure_log)
    total_count = len(pairs)
    missing_count = 0
    ok_count = 0
    skipped_count = 0
    failed_count = 0
    with open(failure_log, "w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["sample_name", "left_file", "right_file", "status", "reason"]
        )
        writer.writeheader()

        progress = tqdm(pairs, desc="Processing stereo pairs", unit="pair")
        for left_file, right_file, left_sample_name, right_sample_name in progress:
            sample_name = right_sample_name if args.output_view == "right" else left_sample_name
            progress.set_postfix_str(sample_name)
            if not left_file.exists() or not right_file.exists():
                missing_count += 1
                failed_count += 1
                maybe_write_failure_row(
                    writer, sample_name, left_file, right_file, "missing_file", "left or right image does not exist"
                )
                logging.warning("Missing image(s), skip: %s | %s", left_file, right_file)
                continue

            try:
                status = run_inference(
                    model, args, left_file, right_file, left_sample_name, right_sample_name, K_scaled, baseline
                )
                if status == "skipped":
                    skipped_count += 1
                    maybe_write_failure_row(
                        writer, sample_name, left_file, right_file, "skipped_existing", "expected outputs already exist"
                    )
                else:
                    ok_count += 1
            except Exception as exc:
                failed_count += 1
                maybe_write_failure_row(writer, sample_name, left_file, right_file, "failed", str(exc))
                logging.exception("Failed on sample %s", sample_name)
                continue

    logging.info(
        "Done. total=%d ok=%d skipped=%d failed=%d missing=%d failure_log=%s",
        total_count,
        ok_count,
        skipped_count,
        failed_count,
        missing_count,
        failure_log,
    )


if __name__ == "__main__":
    main()
