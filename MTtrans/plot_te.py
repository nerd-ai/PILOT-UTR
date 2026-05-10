#!/usr/bin/env python3
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def parse_args():
    p = argparse.ArgumentParser(description="Violin plot of TE distributions from two files.")
    p.add_argument("--pred1", required=True, help="prediction1.csv (with columns: id,prediction)")
    p.add_argument("--pred2", required=True, help="prediction2.csv (with columns: [index],seq,te)")
    p.add_argument("--label1", default="Natural", help="Label for pred1")
    p.add_argument("--label2", default="Generated", help="Label for pred2")
    p.add_argument("--title", default="Distribution of TE for Generated and Original Sequences")
    p.add_argument("--out", default="TE_Optimized_Natural.png", help="Output image path (.png/.pdf)")
    return p.parse_args()


def load_pred1(path: str) -> np.ndarray:
    """Read prediction1.csv -> numeric 1D array (column 'prediction' or second column)."""
    # Try flexible parsing with regex separator (comma OR whitespace)
    df = pd.read_csv(path, sep=r"[,\s]+", engine="python")

    # Accept common column layouts
    if "prediction" in df.columns:
        vals = pd.to_numeric(df["prediction"], errors="coerce")
    elif df.shape[1] >= 2:
        # If no header or only two columns, take the second one
        vals = pd.to_numeric(df.iloc[:, 1], errors="coerce")
    else:
        raise ValueError(f"{path} must contain at least two columns or a 'prediction' column")

    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        raise ValueError(f"{path} produced zero valid values after parsing (check delimiters).")
    return vals


def load_pred2(path: str) -> np.ndarray:
    """Read prediction1.csv -> numeric 1D array (column 'prediction' or second column)."""
    # Try flexible parsing with regex separator (comma OR whitespace)
    df = pd.read_csv(path, sep=r"[,\s]+", engine="python")

    # Accept common column layouts
    if "prediction" in df.columns:
        vals = pd.to_numeric(df["prediction"], errors="coerce")
    elif df.shape[1] >= 2:
        # If no header or only two columns, take the second one
        vals = pd.to_numeric(df.iloc[:, 1], errors="coerce")
    else:
        raise ValueError(f"{path} must contain at least two columns or a 'prediction' column")

    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        raise ValueError(f"{path} produced zero valid values after parsing (check delimiters).")
    return vals


# def load_pred2(path: str) -> np.ndarray:
#     """Read prediction2.csv -> numeric 1D array (column 'te')."""
#     # Your file has a leading unnamed index column. index_col=0 drops it.
#     df = pd.read_csv(path, index_col=0)
#     if "te" not in df.columns:
#         # Fallback: try using the last column if it's numeric
#         last = pd.to_numeric(df.iloc[:, -1], errors="coerce")
#         if last.notna().any():
#             vals = last.to_numpy()
#         else:
#             raise ValueError(f"{path} must contain a 'te' column (or a numeric last column).")
#     else:
#         vals = pd.to_numeric(df["te"], errors="coerce").to_numpy()
#     vals = vals[np.isfinite(vals)]
#     if vals.size == 0:
#         raise ValueError(f"{path} produced zero valid 'te' values after parsing.")
#     return vals


def main():
    args = parse_args()

    y1 = load_pred1(args.pred1)
    y2 = load_pred2(args.pred2)
    print(f"[info] Loaded {y1.size} values from {args.pred1}, {y2.size} from {args.pred2}")

    fig, ax = plt.subplots(figsize=(8, 4.2))

    parts = ax.violinplot([y1, y2],
                          positions=[1, 2],
                          showmeans=True,
                          showmedians=False,
                          showextrema=True)

    # Quartiles & median (dashed bars), like your reference figure
    for i, y in enumerate([y1, y2], start=1):
        q1, med, q3 = np.percentile(y, [25, 50, 75])
        ax.hlines([q1, med, q3], i - 0.25, i + 0.25, linestyles="--", linewidth=1)

    ax.set_xticks([1, 2])
    ax.set_xticklabels([args.label1, args.label2])
    ax.set_ylabel("Translation Efficiency (TE)")
    ax.set_title(args.title)
    ax.grid(True, axis="y", linestyle=":", linewidth=0.7, alpha=0.7)
    fig.tight_layout()
    fig.savefig(args.out, dpi=300)
    print(f"[done] Saved figure to {Path(args.out).resolve()}")


if __name__ == "__main__":
    main()
