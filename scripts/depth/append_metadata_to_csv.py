import csv
import argparse
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Append metadata columns from one CSV into another CSV by frame id."
    )
    parser.add_argument("--input_csv", type=str, required=True, help="existing csv to enrich")
    parser.add_argument("--metadata_csv", type=str, required=True, help="metadata csv source")
    parser.add_argument("--output_csv", type=str, required=True, help="output enriched csv")
    parser.add_argument(
        "--input_frame_id_column",
        type=str,
        default="frame_id",
        help="frame id column in input csv",
    )
    parser.add_argument(
        "--metadata_frame_id_column",
        type=str,
        default="Frame ID",
        help="frame id column in metadata csv",
    )
    parser.add_argument(
        "--metadata_columns",
        type=str,
        default="cameras facing to,Canopy facing to",
        help="comma-separated metadata columns to append",
    )
    parser.add_argument(
        "--output_columns",
        type=str,
        default="camera_facing_to,canopy_facing_to",
        help="comma-separated output column names",
    )
    parser.add_argument(
        "--input_frame_id_width",
        type=int,
        default=0,
        help="optional zero-pad width for input csv frame ids",
    )
    parser.add_argument(
        "--metadata_frame_id_width",
        type=int,
        default=0,
        help="optional zero-pad width for metadata csv frame ids",
    )
    return parser.parse_args()


def normalize_frame_id(frame_id: str, width: int):
    value = str(frame_id).strip()
    if value.endswith(".0"):
        value = value[:-2]
    if width > 0:
        value = value.zfill(width)
    return value


def load_metadata_map(metadata_csv: Path, frame_id_column: str, metadata_columns, output_columns, frame_id_width: int):
    metadata_map = {}
    with open(metadata_csv, newline="") as f:
        reader = csv.DictReader(f)
        if frame_id_column not in reader.fieldnames:
            raise ValueError(
                f"Column '{frame_id_column}' not found in metadata csv. Available: {reader.fieldnames}"
            )
        for col in metadata_columns:
            if col not in reader.fieldnames:
                raise ValueError(f"Column '{col}' not found in metadata csv. Available: {reader.fieldnames}")

        for row in reader:
            frame_id = normalize_frame_id(row[frame_id_column], frame_id_width)
            metadata_map[frame_id] = {
                out_col: row[src_col] for src_col, out_col in zip(metadata_columns, output_columns)
            }
    return metadata_map


def main():
    args = parse_args()
    input_csv = Path(args.input_csv)
    metadata_csv = Path(args.metadata_csv)
    output_csv = Path(args.output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    metadata_columns = [x.strip() for x in args.metadata_columns.split(",") if x.strip()]
    output_columns = [x.strip() for x in args.output_columns.split(",") if x.strip()]
    if len(metadata_columns) != len(output_columns):
        raise ValueError("--metadata_columns and --output_columns must have the same number of entries")

    metadata_map = load_metadata_map(
        metadata_csv,
        args.metadata_frame_id_column,
        metadata_columns,
        output_columns,
        args.metadata_frame_id_width,
    )

    with open(input_csv, newline="") as f_in, open(output_csv, "w", newline="") as f_out:
        reader = csv.DictReader(f_in)
        if args.input_frame_id_column not in reader.fieldnames:
            raise ValueError(
                f"Column '{args.input_frame_id_column}' not found in input csv. Available: {reader.fieldnames}"
            )

        fieldnames = list(reader.fieldnames)
        for col in output_columns:
            if col not in fieldnames:
                fieldnames.append(col)

        writer = csv.DictWriter(f_out, fieldnames=fieldnames)
        writer.writeheader()

        for row in reader:
            frame_id = normalize_frame_id(row[args.input_frame_id_column], args.input_frame_id_width)
            metadata = metadata_map.get(frame_id, {})
            for col in output_columns:
                row[col] = metadata.get(col, "")
            writer.writerow(row)

    print(f"Saved enriched csv to {output_csv}")


if __name__ == "__main__":
    main()
