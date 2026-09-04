#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover
    tqdm = None


NAME_PATTERN = re.compile(r"^row_(\d+)_(\d+)$")
VINE_REQUIRED_COLUMNS = ["Name", "Longitude", "Latitude"]
GPS_REQUIRED_COLUMNS = ["Frame ID", "Latitude", "Longitude"]
EPSILON = 1e-9
CARDINAL_VECTORS = {
    "E": np.array([1.0, 0.0]),
    "N": np.array([0.0, 1.0]),
    "W": np.array([-1.0, 0.0]),
    "S": np.array([0.0, -1.0]),
}
OPPOSITE_CARDINAL = {"E": "W", "W": "E", "N": "S", "S": "N"}


@dataclass
class CoordinateTransformer:
    lon0: float
    lat0: float
    meters_per_deg_lon: float
    meters_per_deg_lat: float

    @classmethod
    def from_points(cls, longitudes: pd.Series, latitudes: pd.Series) -> "CoordinateTransformer":
        lon0 = float(longitudes.mean())
        lat0 = float(latitudes.mean())
        lat0_rad = math.radians(lat0)
        return cls(
            lon0=lon0,
            lat0=lat0,
            meters_per_deg_lon=111_320.0 * math.cos(lat0_rad),
            meters_per_deg_lat=111_320.0,
        )

    def lonlat_to_xy(
        self, longitudes: pd.Series | np.ndarray, latitudes: pd.Series | np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        x = (np.asarray(longitudes) - self.lon0) * self.meters_per_deg_lon
        y = (np.asarray(latitudes) - self.lat0) * self.meters_per_deg_lat
        return x, y

    def xy_to_lonlat(
        self, x: pd.Series | np.ndarray, y: pd.Series | np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        lon = self.lon0 + (np.asarray(x) / self.meters_per_deg_lon)
        lat = self.lat0 + (np.asarray(y) / self.meters_per_deg_lat)
        return lon, lat


@dataclass
class RowModel:
    row_id: int
    anchor: np.ndarray
    axis: np.ndarray
    s_min: float
    s_max: float
    vine_intervals: pd.DataFrame


def progress_bar(iterable, total: int, desc: str, unit: str):
    if tqdm is None:
        return iterable
    return tqdm(
        iterable,
        total=total,
        desc=desc,
        unit=unit,
        dynamic_ncols=True,
        file=sys.stderr,
    )


def is_finite_point(point: np.ndarray) -> bool:
    return bool(np.isfinite(point).all())


def normalize_vector(vector: np.ndarray) -> np.ndarray | None:
    vector = np.asarray(vector, dtype=float)
    if not np.isfinite(vector).all():
        return None
    length = float(np.linalg.norm(vector))
    if length <= EPSILON:
        return None
    return vector / length


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Project per-frame GPS into camera positions and determine which vines are "
            "visible on the nearest row in the camera viewing direction."
        )
    )
    parser.add_argument("gps_csv", type=Path, help="Per-frame GPS log CSV.")
    parser.add_argument(
        "--aligned-csv",
        type=Path,
        default=Path("data/processed/spatial/vine_reference.csv"),
        help="Aligned vine georeference CSV.",
    )
    parser.add_argument("--x-cm", type=float, required=True, help="Camera offset in forward direction.")
    parser.add_argument("--y-cm", type=float, required=True, help="Camera offset in left direction.")
    parser.add_argument("--fov-deg", type=float, required=True, help="Camera horizontal FOV in degrees.")
    parser.add_argument(
        "--max-distance-m",
        type=float,
        default=1.0,
        help="Maximum horizontal viewing distance in meters.",
    )
    parser.add_argument(
        "--neighbor-threshold-m",
        type=float,
        default=2.0,
        help="If adjacent vines are closer than this threshold, split coverage at the midpoint.",
    )
    parser.add_argument(
        "--edge-extension-m",
        type=float,
        default=1.0,
        help="If no neighboring vine is found within the threshold on one side, extend by this distance.",
    )
    parser.add_argument(
        "--row-end-margin-m",
        type=float,
        default=2.0,
        help="Allowed extra distance beyond the row ends, measured along the row direction.",
    )
    parser.add_argument(
        "--parallel-angle-threshold-deg",
        type=float,
        default=45.0,
        help="Frames whose motion differs from the row direction by more than this angle are marked as outliers.",
    )
    parser.add_argument(
        "--min-vine-coverage-percent",
        type=float,
        default=0.0,
        help=(
            "A vine is counted as visible only if the frame coverage overlaps at least "
            "this percent of that vine's row coverage interval."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/generated/spatial"),
        help="Directory for the per-frame vine coverage CSV.",
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Generate one PNG per frame showing camera position, FOV, and matched vine coverage.",
    )
    parser.add_argument(
        "--visualization-dir",
        type=Path,
        default=None,
        help="Directory for visualization PNGs. Defaults to <output-dir>/<gps_stem>_visualizations.",
    )
    parser.add_argument(
        "--visualize-every-n",
        type=int,
        default=1,
        help="When visualization is enabled, generate one PNG for every N frames in input order.",
    )
    return parser.parse_args()


def validate_columns(df: pd.DataFrame, required: list[str], label: str) -> None:
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError(f"{label} is missing required columns: {missing}")


def parse_vine_name(name: str) -> tuple[int, int]:
    match = NAME_PATTERN.match(str(name))
    if not match:
        raise ValueError(
            f"Name '{name}' does not match expected format 'row_<row_number>_<vine_number>'."
        )
    return int(match.group(1)), int(match.group(2))


def build_row_models(
    vines: pd.DataFrame, neighbor_threshold_m: float, edge_extension_m: float
) -> list[RowModel]:
    row_models: list[RowModel] = []

    for row_id, group in vines.groupby("row_id"):
        group = group.sort_values("vine_id").copy()
        points = group[["x", "y"]].to_numpy(dtype=float)
        anchor = points[0]
        direction = points[-1] - points[0]
        length = float(np.linalg.norm(direction))
        if length <= EPSILON:
            continue

        axis = direction / length
        s = (points - anchor) @ axis
        group["s"] = s

        left_bounds: list[float] = []
        right_bounds: list[float] = []
        for i, s_value in enumerate(s):
            if i == 0:
                left_bound = s_value - edge_extension_m
            else:
                left_gap = s_value - s[i - 1]
                if left_gap < neighbor_threshold_m:
                    left_bound = (s_value + s[i - 1]) / 2.0
                else:
                    left_bound = s_value - edge_extension_m

            if i == len(s) - 1:
                right_bound = s_value + edge_extension_m
            else:
                right_gap = s[i + 1] - s_value
                if right_gap < neighbor_threshold_m:
                    right_bound = (s_value + s[i + 1]) / 2.0
                else:
                    right_bound = s_value + edge_extension_m

            left_bounds.append(left_bound)
            right_bounds.append(right_bound)

        group["coverage_s_min"] = left_bounds
        group["coverage_s_max"] = right_bounds

        row_models.append(
            RowModel(
                row_id=int(row_id),
                anchor=anchor,
                axis=axis,
                s_min=float(min(left_bounds)),
                s_max=float(max(right_bounds)),
                vine_intervals=group[
                    ["Name", "vine_id", "x", "y", "s", "coverage_s_min", "coverage_s_max"]
                ].copy(),
            )
        )

    return row_models


def find_nonzero_neighbor(points: np.ndarray, index: int, direction: int) -> int | None:
    cursor = index + direction
    while 0 <= cursor < len(points):
        if np.linalg.norm(points[cursor] - points[index]) > EPSILON:
            return cursor
        cursor += direction
    return None


def estimate_headings(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    headings = np.full((len(points), 2), np.nan, dtype=float)
    valid = np.zeros(len(points), dtype=bool)

    for i in range(len(points)):
        prev_idx = find_nonzero_neighbor(points, i, -1)
        next_idx = find_nonzero_neighbor(points, i, 1)

        if prev_idx is not None and next_idx is not None:
            delta = points[next_idx] - points[prev_idx]
        elif next_idx is not None:
            delta = points[next_idx] - points[i]
        elif prev_idx is not None:
            delta = points[i] - points[prev_idx]
        else:
            continue

        length = float(np.linalg.norm(delta))
        if length <= EPSILON:
            continue

        headings[i] = delta / length
        valid[i] = True

    return headings, valid


def clip_interval(lo: float, hi: float, a: float, b: float, relation: str) -> tuple[float, float] | None:
    if relation not in {"le", "ge"}:
        raise ValueError(f"Unsupported relation: {relation}")

    if abs(b) <= EPSILON:
        if relation == "le":
            return (lo, hi) if a <= EPSILON else None
        return (lo, hi) if a >= -EPSILON else None

    boundary = -a / b
    if relation == "le":
        if b > 0:
            hi = min(hi, boundary)
        else:
            lo = max(lo, boundary)
    else:
        if b > 0:
            lo = max(lo, boundary)
        else:
            hi = min(hi, boundary)

    if lo <= hi + EPSILON:
        return lo, hi
    return None


def visible_interval_on_row(
    camera_point: np.ndarray,
    camera_direction: np.ndarray,
    row: RowModel,
    max_distance_m: float,
    half_fov_rad: float,
) -> tuple[float, float] | None:
    rel0 = row.anchor - camera_point
    side_axis = np.array([-camera_direction[1], camera_direction[0]])
    tangent = math.tan(half_fov_rad)

    u0 = float(rel0 @ camera_direction)
    u1 = float(row.axis @ camera_direction)
    v0 = float(rel0 @ side_axis)
    v1 = float(row.axis @ side_axis)

    interval: tuple[float, float] | None = (row.s_min, row.s_max)
    constraints = [
        (u0, u1, "ge"),
        (u0 - max_distance_m, u1, "le"),
        (v0 - u0 * tangent, v1 - u1 * tangent, "le"),
        (-v0 - u0 * tangent, -v1 - u1 * tangent, "le"),
    ]

    for a, b, relation in constraints:
        if interval is None:
            break
        interval = clip_interval(interval[0], interval[1], a, b, relation)

    return interval


def nearest_distance_to_interval(camera_point: np.ndarray, row: RowModel, interval: tuple[float, float]) -> float:
    rel0 = row.anchor - camera_point
    s_star = float(-(rel0 @ row.axis))
    s_clamped = min(max(s_star, interval[0]), interval[1])
    closest_point = row.anchor + s_clamped * row.axis
    return float(np.linalg.norm(closest_point - camera_point))


def overlap(interval_a: tuple[float, float], interval_b: tuple[float, float]) -> bool:
    return min(interval_a[1], interval_b[1]) >= max(interval_a[0], interval_b[0]) - EPSILON


def overlap_length(interval_a: tuple[float, float], interval_b: tuple[float, float]) -> float:
    return max(0.0, min(interval_a[1], interval_b[1]) - max(interval_a[0], interval_b[0]))


def format_percent_suffix(percent: float) -> str:
    formatted = f"{percent:.4f}".rstrip("0").rstrip(".")
    if not formatted:
        formatted = "0"
    return formatted.replace(".", "p")


def format_heading_deg(heading: np.ndarray) -> float:
    angle = math.degrees(math.atan2(heading[1], heading[0]))
    if angle < 0:
        angle += 360.0
    return angle


def nearest_cardinal_label(vector: np.ndarray, allowed_labels: tuple[str, ...] | None = None) -> str:
    normalized = normalize_vector(vector)
    if normalized is None:
        return ""

    labels = allowed_labels if allowed_labels is not None else tuple(CARDINAL_VECTORS.keys())
    return max(labels, key=lambda label: float(normalized @ CARDINAL_VECTORS[label]))


def row_side_labels_from_axis(row_axis: np.ndarray) -> tuple[str, str]:
    row_normal = np.array([-row_axis[1], row_axis[0]], dtype=float)
    positive_label = nearest_cardinal_label(row_normal)
    return positive_label, OPPOSITE_CARDINAL[positive_label]


def build_global_row_axis(row_models: list[RowModel]) -> np.ndarray:
    if not row_models:
        raise ValueError("Cannot compute a global row axis without row models.")

    reference = row_models[0].axis.copy()
    aligned_axes = []
    for row_model in row_models:
        axis = row_model.axis.copy()
        if float(axis @ reference) < 0:
            axis = -axis
        aligned_axes.append(axis)

    mean_axis = np.mean(np.vstack(aligned_axes), axis=0)
    length = float(np.linalg.norm(mean_axis))
    if length <= EPSILON:
        raise ValueError("Failed to compute a stable global row axis.")
    return mean_axis / length


def point_on_row(row: RowModel, s_value: float) -> np.ndarray:
    return row.anchor + s_value * row.axis


def visualize_frames(
    gps: pd.DataFrame,
    row_models: dict[int, RowModel],
    max_distance_m: float,
    fov_deg: float,
    output_dir: Path,
    visualize_every_n: int,
) -> int:
    output_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str((output_dir / ".mplconfig").resolve()))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Wedge

    half_fov = fov_deg / 2.0
    saved_count = 0
    sampled_gps = gps.iloc[::visualize_every_n]

    for frame in progress_bar(
        sampled_gps.itertuples(index=False),
        total=len(sampled_gps),
        desc="Rendering PNGs",
        unit="frame",
    ):
        fig, ax = plt.subplots(figsize=(8, 8))
        camera_point = np.array([frame.camera_x, frame.camera_y], dtype=float)
        gps_point = np.array([frame.x, frame.y], dtype=float)
        camera_point_valid = is_finite_point(camera_point)
        gps_point_valid = is_finite_point(gps_point)
        reference_point = camera_point if camera_point_valid else gps_point

        if not is_finite_point(reference_point):
            plt.close(fig)
            continue

        local_radius = max(max_distance_m + 2.0, 3.0)

        for row_model in row_models.values():
            distance = nearest_distance_to_interval(
                reference_point, row_model, (row_model.s_min, row_model.s_max)
            )
            if distance > local_radius * 1.5:
                continue
            start = point_on_row(row_model, row_model.s_min)
            end = point_on_row(row_model, row_model.s_max)
            ax.plot(
                [start[0], end[0]],
                [start[1], end[1]],
                color="0.8",
                linewidth=1.2,
                zorder=1,
            )

        matched_row = None
        if not pd.isna(frame.matched_row_id):
            matched_row = row_models.get(int(frame.matched_row_id))

        if matched_row is not None:
            start = point_on_row(matched_row, matched_row.s_min)
            end = point_on_row(matched_row, matched_row.s_max)
            ax.plot(
                [start[0], end[0]],
                [start[1], end[1]],
                color="black",
                linewidth=2.0,
                zorder=2,
            )

            vine_points = matched_row.vine_intervals[["x", "y"]].to_numpy(dtype=float)
            ax.scatter(vine_points[:, 0], vine_points[:, 1], s=18, color="0.45", zorder=3)

            if not pd.isna(frame.coverage_start_s) and not pd.isna(frame.coverage_end_s):
                coverage_start = point_on_row(matched_row, float(frame.coverage_start_s))
                coverage_end = point_on_row(matched_row, float(frame.coverage_end_s))
                ax.plot(
                    [coverage_start[0], coverage_end[0]],
                    [coverage_start[1], coverage_end[1]],
                    color="tab:red",
                    linewidth=4.0,
                    solid_capstyle="round",
                    zorder=4,
                )

            visible_names = set()
            if isinstance(frame.visible_vines, str) and frame.visible_vines:
                visible_names = set(frame.visible_vines.split(";"))
            if visible_names:
                visible = matched_row.vine_intervals[
                    matched_row.vine_intervals["Name"].isin(visible_names)
                ]
                ax.scatter(
                    visible["x"],
                    visible["y"],
                    s=42,
                    color="tab:orange",
                    edgecolors="black",
                    linewidths=0.4,
                    zorder=5,
                )

        if gps_point_valid:
            ax.scatter([gps_point[0]], [gps_point[1]], color="tab:blue", marker="x", s=56, zorder=6)
        if camera_point_valid:
            ax.scatter([camera_point[0]], [camera_point[1]], color="tab:green", s=42, zorder=7)
        if gps_point_valid and camera_point_valid:
            ax.plot(
                [gps_point[0], camera_point[0]],
                [gps_point[1], camera_point[1]],
                color="tab:green",
                linestyle=":",
                linewidth=1.0,
                zorder=6,
            )

        if frame.has_valid_heading and gps_point_valid and camera_point_valid:
            heading = np.array([frame.heading_x, frame.heading_y], dtype=float)
            camera_direction = np.array([frame.camera_dir_x, frame.camera_dir_y], dtype=float)
            ax.arrow(
                gps_point[0],
                gps_point[1],
                heading[0] * 0.6,
                heading[1] * 0.6,
                width=0.01,
                head_width=0.14,
                head_length=0.2,
                color="tab:blue",
                length_includes_head=True,
                zorder=6,
            )
            ax.arrow(
                camera_point[0],
                camera_point[1],
                camera_direction[0] * 0.8,
                camera_direction[1] * 0.8,
                width=0.01,
                head_width=0.14,
                head_length=0.2,
                color="tab:purple",
                length_includes_head=True,
                zorder=7,
            )

            camera_angle_deg = math.degrees(math.atan2(camera_direction[1], camera_direction[0]))
            wedge = Wedge(
                center=(camera_point[0], camera_point[1]),
                r=max_distance_m,
                theta1=camera_angle_deg - half_fov,
                theta2=camera_angle_deg + half_fov,
                facecolor="tab:purple",
                edgecolor="tab:purple",
                alpha=0.12,
                linewidth=1.2,
                zorder=3,
            )
            ax.add_patch(wedge)

        focus_points = []
        if camera_point_valid:
            focus_points.append(camera_point)
        if gps_point_valid:
            focus_points.append(gps_point)
        if matched_row is not None and not pd.isna(frame.coverage_start_s) and not pd.isna(frame.coverage_end_s):
            focus_points.append(point_on_row(matched_row, float(frame.coverage_start_s)))
            focus_points.append(point_on_row(matched_row, float(frame.coverage_end_s)))
        focus_points = [point for point in focus_points if is_finite_point(point)]
        if not focus_points:
            focus_points = [reference_point]
        focus_array = np.vstack(focus_points)
        x_min, y_min = focus_array.min(axis=0) - (max_distance_m + 1.5)
        x_max, y_max = focus_array.max(axis=0) + (max_distance_m + 1.5)

        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.set_aspect("equal", adjustable="box")
        ax.grid(True, alpha=0.25)
        ax.set_xlabel("Local X (m)")
        ax.set_ylabel("Local Y (m)")
        title_row = f"row_{int(frame.matched_row_id)}" if matched_row is not None else "None"
        canopy_label = frame.canopy_facing_to if isinstance(frame.canopy_facing_to, str) and frame.canopy_facing_to else "NA"
        ax.set_title(
            f"Frame {frame.frame_id} | status={frame.status_note} | matched_row={title_row} | canopy={canopy_label}"
        )

        legend_items = [
            "blue x: GPS",
            "green dot: camera",
            "purple wedge: FOV",
            "red segment: visible coverage",
            "orange dots: visible vines",
        ]
        ax.text(
            0.01,
            0.01,
            "\n".join(legend_items),
            transform=ax.transAxes,
            fontsize=9,
            verticalalignment="bottom",
            bbox={"facecolor": "white", "alpha": 0.85, "edgecolor": "0.75"},
        )

        frame_id = int(frame.frame_id)
        fig.tight_layout()
        fig.savefig(output_dir / f"frame_{frame_id:06d}.png", dpi=140)
        plt.close(fig)
        saved_count += 1

    return saved_count


def main() -> None:
    args = parse_args()

    if args.fov_deg <= 0 or args.fov_deg >= 180:
        raise ValueError("FOV must be between 0 and 180 degrees.")
    if args.max_distance_m <= 0:
        raise ValueError("max-distance-m must be positive.")
    if args.neighbor_threshold_m <= 0:
        raise ValueError("neighbor-threshold-m must be positive.")
    if args.edge_extension_m <= 0:
        raise ValueError("edge-extension-m must be positive.")
    if args.row_end_margin_m < 0:
        raise ValueError("row-end-margin-m must be non-negative.")
    if not 0 < args.parallel_angle_threshold_deg <= 90:
        raise ValueError("parallel-angle-threshold-deg must be in (0, 90].")
    if not 0 <= args.min_vine_coverage_percent <= 100:
        raise ValueError("min-vine-coverage-percent must be between 0 and 100.")
    if args.visualize_every_n <= 0:
        raise ValueError("visualize-every-n must be positive.")

    vines = pd.read_csv(args.aligned_csv)
    gps = pd.read_csv(args.gps_csv)
    validate_columns(vines, VINE_REQUIRED_COLUMNS, "Aligned vine CSV")
    validate_columns(gps, GPS_REQUIRED_COLUMNS, "GPS log CSV")
    gps_original_columns = gps.columns.tolist()

    parsed_names = vines["Name"].map(parse_vine_name)
    vines["row_id"] = parsed_names.map(lambda item: item[0])
    vines["vine_id"] = parsed_names.map(lambda item: item[1])

    transformer = CoordinateTransformer.from_points(
        pd.concat([vines["Longitude"], gps["Longitude"]], ignore_index=True),
        pd.concat([vines["Latitude"], gps["Latitude"]], ignore_index=True),
    )
    vines["x"], vines["y"] = transformer.lonlat_to_xy(vines["Longitude"], vines["Latitude"])
    gps["x"], gps["y"] = transformer.lonlat_to_xy(gps["Longitude"], gps["Latitude"])

    row_models = build_row_models(
        vines=vines,
        neighbor_threshold_m=args.neighbor_threshold_m,
        edge_extension_m=args.edge_extension_m,
    )
    if not row_models:
        raise ValueError("No valid row models could be built from the aligned vine data.")
    global_row_axis = build_global_row_axis(row_models)
    global_cross_axis = np.array([-global_row_axis[1], global_row_axis[0]])
    row_side_labels = row_side_labels_from_axis(global_row_axis)
    vine_points = vines[["x", "y"]].to_numpy(dtype=float)
    vine_u = vine_points @ global_row_axis
    vine_v = vine_points @ global_cross_axis
    global_u_min = float(vine_u.min())
    global_u_max = float(vine_u.max())
    global_v_min = float(vine_v.min())
    global_v_max = float(vine_v.max())

    headings, valid_heading = estimate_headings(gps[["x", "y"]].to_numpy(dtype=float))
    gps["heading_x"] = headings[:, 0]
    gps["heading_y"] = headings[:, 1]
    gps["has_valid_heading"] = valid_heading

    x_offset_m = args.x_cm / 100.0
    y_offset_m = args.y_cm / 100.0
    left_vectors = np.column_stack((-gps["heading_y"], gps["heading_x"]))
    gps["camera_x"] = gps["x"] + x_offset_m * gps["heading_x"] + y_offset_m * left_vectors[:, 0]
    gps["camera_y"] = gps["y"] + x_offset_m * gps["heading_y"] + y_offset_m * left_vectors[:, 1]

    camera_lon, camera_lat = transformer.xy_to_lonlat(gps["camera_x"], gps["camera_y"])
    gps["Camera Longitude"] = camera_lon
    gps["Camera Latitude"] = camera_lat

    if y_offset_m >= 0:
        gps["camera_dir_x"] = left_vectors[:, 0]
        gps["camera_dir_y"] = left_vectors[:, 1]
    else:
        gps["camera_dir_x"] = -left_vectors[:, 0]
        gps["camera_dir_y"] = -left_vectors[:, 1]

    gps["camera_u"] = gps["camera_x"] * global_row_axis[0] + gps["camera_y"] * global_row_axis[1]
    gps["camera_v"] = gps["camera_x"] * global_cross_axis[0] + gps["camera_y"] * global_cross_axis[1]

    half_fov_rad = math.radians(args.fov_deg / 2.0)
    parallel_cos_threshold = math.cos(math.radians(args.parallel_angle_threshold_deg))
    matched_rows: list[str] = []
    matched_row_ids: list[float] = []
    visible_names: list[str] = []
    visible_coverage_percents: list[str] = []
    visible_counts: list[int] = []
    coverage_start_s: list[float] = []
    coverage_end_s: list[float] = []
    nearest_distances: list[float] = []
    status_notes: list[str] = []
    heading_degrees: list[float] = []
    outlier_flags: list[bool] = []
    motion_parallel_flags: list[bool] = []
    row_direction_angle_deg: list[float] = []
    within_row_end_margin_flags: list[bool] = []
    camera_within_row_band_flags: list[bool] = []
    camera_facing_labels: list[str] = []
    canopy_facing_labels: list[str] = []

    for row in progress_bar(
        gps.itertuples(index=False),
        total=len(gps),
        desc="Computing coverage",
        unit="frame",
    ):
        if row.has_valid_heading:
            heading_degrees.append(format_heading_deg(np.array([row.heading_x, row.heading_y])))
        else:
            heading_degrees.append(float("nan"))

        if not row.has_valid_heading:
            matched_rows.append("")
            matched_row_ids.append(float("nan"))
            visible_names.append("")
            visible_coverage_percents.append("")
            visible_counts.append(0)
            coverage_start_s.append(float("nan"))
            coverage_end_s.append(float("nan"))
            nearest_distances.append(float("nan"))
            outlier_flags.append(True)
            motion_parallel_flags.append(False)
            row_direction_angle_deg.append(float("nan"))
            within_row_end_margin_flags.append(False)
            camera_within_row_band_flags.append(False)
            camera_facing_labels.append("")
            canopy_facing_labels.append("")
            status_notes.append("invalid_heading")
            continue

        camera_point = np.array([row.camera_x, row.camera_y], dtype=float)
        camera_direction = np.array([row.camera_dir_x, row.camera_dir_y], dtype=float)

        candidates: list[tuple[float, RowModel, tuple[float, float]]] = []
        for row_model in row_models:
            interval = visible_interval_on_row(
                camera_point=camera_point,
                camera_direction=camera_direction,
                row=row_model,
                max_distance_m=args.max_distance_m,
                half_fov_rad=half_fov_rad,
            )
            if interval is None:
                continue
            distance = nearest_distance_to_interval(camera_point, row_model, interval)
            candidates.append((distance, row_model, interval))

        if not candidates:
            matched_rows.append("")
            matched_row_ids.append(float("nan"))
            visible_names.append("")
            visible_coverage_percents.append("")
            visible_counts.append(0)
            coverage_start_s.append(float("nan"))
            coverage_end_s.append(float("nan"))
            nearest_distances.append(float("nan"))
            matched_row = None
            interval = None
            visible_vines = pd.DataFrame(columns=["Name"])
            distance = float("nan")
        else:
            distance, matched_row, interval = min(candidates, key=lambda item: item[0])
            visible_vines = matched_row.vine_intervals.copy()
            visible_vines["coverage_length_m"] = (
                visible_vines["coverage_s_max"].astype(float)
                - visible_vines["coverage_s_min"].astype(float)
            )
            visible_vines["overlap_length_m"] = visible_vines.apply(
                lambda vine: overlap_length(
                    (float(vine["coverage_s_min"]), float(vine["coverage_s_max"])),
                    interval,
                ),
                axis=1,
            )
            visible_vines["coverage_percent"] = np.where(
                visible_vines["coverage_length_m"] > EPSILON,
                visible_vines["overlap_length_m"] / visible_vines["coverage_length_m"] * 100.0,
                0.0,
            )
            visible_vines = visible_vines[
                visible_vines["coverage_percent"] + EPSILON >= args.min_vine_coverage_percent
            ].copy()

            matched_rows.append(f"row_{matched_row.row_id}")
            matched_row_ids.append(float(matched_row.row_id))
            visible_names.append(";".join(visible_vines["Name"].tolist()))
            visible_coverage_percents.append(
                ";".join(
                    f"{vine.Name}:{float(vine.coverage_percent):.2f}"
                    for vine in visible_vines.itertuples(index=False)
                )
            )
            visible_counts.append(int(len(visible_vines)))
            coverage_start_s.append(float(interval[0]))
            coverage_end_s.append(float(interval[1]))
            nearest_distances.append(distance)

        reference_axis = matched_row.axis if matched_row is not None else global_row_axis
        alignment = abs(float(np.array([row.heading_x, row.heading_y]) @ reference_axis))
        alignment = max(-1.0, min(1.0, alignment))
        angle_to_row = math.degrees(math.acos(alignment))
        motion_parallel = alignment >= parallel_cos_threshold
        within_row_end_margin = (
            global_u_min - args.row_end_margin_m
            <= row.camera_u
            <= global_u_max + args.row_end_margin_m
        )
        camera_within_row_band = global_v_min <= row.camera_v <= global_v_max

        reasons: list[str] = []
        if matched_row is None:
            reasons.append("no_row_visible_in_fov")
        if not motion_parallel:
            reasons.append("motion_not_parallel_to_row")
        if not within_row_end_margin:
            reasons.append("outside_row_end_margin")

        outlier_flags.append(bool(reasons))
        motion_parallel_flags.append(motion_parallel)
        row_direction_angle_deg.append(angle_to_row)
        within_row_end_margin_flags.append(within_row_end_margin)
        camera_within_row_band_flags.append(camera_within_row_band)
        if matched_row is not None:
            camera_facing = nearest_cardinal_label(camera_direction, allowed_labels=row_side_labels)
            canopy_facing = OPPOSITE_CARDINAL[camera_facing] if camera_facing else ""
        else:
            camera_facing = ""
            canopy_facing = ""
        camera_facing_labels.append(camera_facing)
        canopy_facing_labels.append(canopy_facing)
        if not reasons and matched_row is not None and visible_vines.empty:
            status_notes.append("no_vine_meets_coverage_threshold")
        else:
            status_notes.append("ok" if not reasons else ";".join(reasons))

    result = gps[gps_original_columns].copy()
    result["Minimum Vine Coverage Percent Threshold"] = args.min_vine_coverage_percent
    result["Camera Latitude"] = gps["Camera Latitude"]
    result["Camera Longitude"] = gps["Camera Longitude"]
    result["Heading Degrees"] = heading_degrees
    result["Heading vs Row Angle (deg)"] = row_direction_angle_deg
    result["Motion Parallel To Row"] = motion_parallel_flags
    result["Within Row End Margin"] = within_row_end_margin_flags
    result["Camera Within Row Band"] = camera_within_row_band_flags
    result["Camera Row Coordinate U (m)"] = gps["camera_u"]
    result["Camera Row Coordinate V (m)"] = gps["camera_v"]
    result["Camera Facing To"] = camera_facing_labels
    result["Canopy Facing To"] = canopy_facing_labels
    result["Outlier"] = outlier_flags
    result["Matched Row"] = matched_rows
    result["Matched Row ID"] = matched_row_ids
    result["Row Coverage Start (m)"] = coverage_start_s
    result["Row Coverage End (m)"] = coverage_end_s
    result["Nearest Row Distance (m)"] = nearest_distances
    result["Visible Vine Count"] = visible_counts
    result["Visible Vines"] = visible_names
    result["Visible Vine Coverage Percents"] = visible_coverage_percents
    result["Status"] = status_notes

    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.min_vine_coverage_percent > 0:
        threshold_suffix = format_percent_suffix(args.min_vine_coverage_percent)
        output_name = (
            f"{args.gps_csv.stem}_camera_coverage_minvinecover_{threshold_suffix}pct.csv"
        )
    else:
        output_name = f"{args.gps_csv.stem}_camera_coverage.csv"
    output_path = args.output_dir / output_name
    result.to_csv(output_path, index=False)

    print(f"Camera coverage file written to: {output_path}")
    print(f"Frames processed: {len(result)}")
    print(f"Outliers: {int(result['Outlier'].sum())}")
    print(f"Frames matched to a row: {int((result['Matched Row'] != '').sum())}")

    if args.visualize:
        gps_for_plot = gps.copy()
        gps_for_plot["frame_id"] = gps["Frame ID"]
        gps_for_plot["matched_row_id"] = matched_row_ids
        gps_for_plot["coverage_start_s"] = coverage_start_s
        gps_for_plot["coverage_end_s"] = coverage_end_s
        gps_for_plot["visible_vines"] = visible_names
        gps_for_plot["camera_facing_to"] = camera_facing_labels
        gps_for_plot["canopy_facing_to"] = canopy_facing_labels
        gps_for_plot["status_note"] = status_notes
        visualization_dir = (
            args.visualization_dir
            if args.visualization_dir is not None
            else args.output_dir / f"{args.gps_csv.stem}_visualizations"
        )
        saved_count = visualize_frames(
            gps=gps_for_plot,
            row_models={row_model.row_id: row_model for row_model in row_models},
            max_distance_m=args.max_distance_m,
            fov_deg=args.fov_deg,
            output_dir=visualization_dir,
            visualize_every_n=args.visualize_every_n,
        )
        print(f"Visualization PNGs written to: {visualization_dir}")
        print(f"Visualization stride: every {args.visualize_every_n} frame(s)")
        print(f"Visualization PNG count: {saved_count}")


if __name__ == "__main__":
    main()
