#!/usr/bin/env python3
"""
Compute Spearman correlation between two prediction files.

expects:
- file A: columns id, prediction (id like row_0), separated by comma/whitespace
- file B: columns Unnamed0 (row index) and prediction (or another name)
"""
import argparse
import os
import re

import pandas as pd


def load_preds_a(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, sep=r"[,\s]+", engine="python")
    if "id" not in df.columns or "prediction" not in df.columns:
        raise ValueError(f"{path} missing required columns 'id' and 'prediction'")
    df["row_idx"] = df["id"].apply(lambda x: int(re.sub(r"^row_", "", str(x))))
    return df[["row_idx", "prediction"]].rename(columns={"prediction": "pred_a"})


def load_preds_b(path: str, pred_col: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    if pred_col not in df.columns:
        raise ValueError(f"{path} missing prediction column '{pred_col}'")
    row_col = "Unnamed0" if "Unnamed0" in df.columns else "id"
    df["row_idx"] = df[row_col].astype(int)
    return df[["row_idx", pred_col]].rename(columns={pred_col: "pred_b"})


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute Spearman correlation between two prediction files.")
    parser.add_argument("--file-a", required=True, help="First prediction file (id,prediction).")
    parser.add_argument("--file-b", required=True, help="Second prediction file (with prediction column).")
    parser.add_argument("--pred-col-b", default="prediction", help="Prediction column name in file B.")
    args = parser.parse_args()

    a = load_preds_a(args.file_a)
    b = load_preds_b(args.file_b, args.pred_col_b)

    merged = a.merge(b, on="row_idx", how="inner")
    if merged.empty:
        raise ValueError("No overlapping rows between the two files.")

    # sanity checks
    if len(merged) != len(a) or len(merged) != len(b):
        print(
            f"Warning: merged rows ({len(merged)}) differ from file A ({len(a)}) or file B ({len(b)}). "
            "IDs may not fully align."
        )
    dup_a = a["row_idx"].duplicated().sum()
    dup_b = b["row_idx"].duplicated().sum()
    if dup_a or dup_b:
        print(f"Warning: duplicates found — file A dup count {dup_a}, file B dup count {dup_b}.")

    spearman = merged["pred_a"].corr(merged["pred_b"], method="spearman")
    print(f"Rows compared: {len(merged)}")
    print(f"Spearman correlation: {spearman:.6f}")


if __name__ == "__main__":
    main()
