#!/usr/bin/env python3
import argparse
import os
import sys

import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser(
        description="Select top-k motifs by absolute delta from motif CSV files.")
    parser.add_argument(
        "-i", "--input", required=True,
        help="Path to motif CSV (e.g., negative_motifs.csv).")
    parser.add_argument(
        "-k", "--topk", type=int, default=100,
        help="Number of motifs to keep (default: 100).")
    parser.add_argument(
        "-c", "--motif-col", choices=["motif_7nt", "motif_9nt"],
        default="motif_7nt",
        help="Which motif column to aggregate (default: motif_7nt).")
    parser.add_argument(
        "-o", "--output", default=None,
        help="Output CSV path; defaults to <input>.<motif_col>.top<k>.csv.")
    parser.add_argument(
        "--deduplicate", action="store_true",
        help="Return one row per unique motif (keeps the record with the largest abs_delta).")
    return parser.parse_args()


def main():
    args = parse_args()
    if not os.path.isfile(args.input):
        sys.exit(f"Input file not found: {args.input}")

    df = pd.read_csv(args.input)
    required_cols = {args.motif_col, "abs_delta"}
    missing = required_cols - set(df.columns)
    if missing:
        sys.exit(f"Missing required columns in CSV: {', '.join(missing)}")

    working_df = df.copy()
    if args.deduplicate:
        idx = working_df.groupby(args.motif_col)["abs_delta"].idxmax()
        working_df = working_df.loc[idx]

    top_df = (
        working_df.sort_values("abs_delta", ascending=False)
        .head(args.topk)
        .reset_index(drop=True)
    )

    output_path = args.output
    if output_path is None:
        base = os.path.basename(args.input)
        stem, ext = os.path.splitext(base)
        output_path = os.path.join(
            os.path.dirname(args.input),
            f"{stem}.{args.motif_col}.top{args.topk}{ext or '.csv'}",
        )

    top_df.to_csv(output_path, index=False)
    print(f"Saved top-{args.topk} motifs to {output_path}")


if __name__ == "__main__":
    main()
