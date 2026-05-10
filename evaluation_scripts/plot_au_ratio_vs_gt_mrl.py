#!/usr/bin/env python3
"""Plot mean A/U ratio across ground-truth MRL bins."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
DEFAULT_INPUT_CSV = REPO_ROOT / "data_and_model" / "val_dataset.csv"
DEFAULT_OUTPUT_DIR = SCRIPT_DIR


def configure_style() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 150,
            "savefig.dpi": 320,
            "font.family": "DejaVu Sans",
            "font.size": 12,
            "axes.labelsize": 13,
            "axes.labelweight": "bold",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 2.0,
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
        }
    )


def compute_au_ratio(seq: str) -> float:
    seq = str(seq).strip().upper().replace("U", "T")
    valid = [ch for ch in seq if ch in {"A", "C", "G", "T"}]
    if not valid:
        return float("nan")
    at_count = sum(ch in {"A", "T"} for ch in valid)
    return at_count / len(valid)


def build_rl_bins(frame: pd.DataFrame, gt_col: str) -> pd.DataFrame:
    filtered = frame[frame[gt_col] >= 0].copy()
    bin_edges = list(range(0, 11)) + [np.inf]
    bin_labels = [f"{i}-{i+1}" for i in range(0, 10)] + [">10"]
    filtered["mrl_bin"] = pd.cut(
        filtered[gt_col],
        bins=bin_edges,
        labels=bin_labels,
        right=False,
        include_lowest=True,
    )
    summary = (
        filtered.groupby("mrl_bin", observed=True)
        .agg(
            mean_au_ratio=("au_ratio", "mean"),
            std_au_ratio=("au_ratio", "std"),
            n=("au_ratio", "size"),
            gt_min=(gt_col, "min"),
            gt_max=(gt_col, "max"),
        )
        .reindex(bin_labels)
        .reset_index()
        .rename(columns={"mrl_bin": "gt_mrl_bin"})
    )
    summary["sem_au_ratio"] = summary["std_au_ratio"] / np.sqrt(summary["n"].clip(lower=1))
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Plot mean A/U ratio for fixed GT MRL bins."
    )
    parser.add_argument(
        "--input-csv",
        type=Path,
        default=DEFAULT_INPUT_CSV,
        help="CSV containing `utr` and `rl` columns.",
    )
    parser.add_argument(
        "--seq-col",
        type=str,
        default="utr",
        help="Sequence column.",
    )
    parser.add_argument(
        "--gt-col",
        type=str,
        default="rl",
        help="Ground-truth MRL column.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for saved outputs.",
    )
    parser.add_argument(
        "--stem",
        type=str,
        default="au_ratio_vs_gt_mrl",
        help="Output filename stem.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    configure_style()

    df = pd.read_csv(args.input_csv)
    required = [args.seq_col, args.gt_col]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in {args.input_csv}: {missing}")

    plot_df = df[[args.seq_col, args.gt_col]].copy()
    plot_df["au_ratio"] = plot_df[args.seq_col].map(compute_au_ratio)
    plot_df = plot_df.dropna(subset=[args.gt_col, "au_ratio"]).reset_index(drop=True)

    summary = build_rl_bins(plot_df, args.gt_col)
    plotted = summary.dropna(subset=["mean_au_ratio"]).copy()
    negative_rows = int((plot_df[args.gt_col] < 0).sum())

    fig, ax = plt.subplots(figsize=(8.2, 5.2))
    x = np.arange(len(plotted))
    ax.bar(
        x,
        plotted["mean_au_ratio"],
        width=0.62,
        color="#7DB7D5",
        edgecolor="#264653",
        linewidth=1.3,
        alpha=0.78,
        zorder=3,
    )
    ax.errorbar(
        x,
        plotted["mean_au_ratio"],
        yerr=plotted["sem_au_ratio"].fillna(0.0),
        fmt="none",
        ecolor="#264653",
        elinewidth=1.2,
        capsize=3,
        zorder=4,
    )

    ax.set_xlabel("GT MRL bin")
    ax.set_ylabel("Mean A/U ratio")
    ax.set_xticks(x)
    ax.set_xticklabels(plotted["gt_mrl_bin"], rotation=0)
    ax.set_ylim(0.0, max(1.0, float(plotted["mean_au_ratio"].max()) + 0.08))
    ax.grid(axis="y", linestyle=(0, (2, 3)), linewidth=0.85, alpha=0.22, zorder=0)
    ax.spines["left"].set_linewidth(2.3)
    ax.spines["bottom"].set_linewidth(2.3)
    if negative_rows:
        ax.text(
            0.02,
            0.98,
            f"Excluded rows with GT MRL < 0: {negative_rows}",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=10.5,
            color="#404040",
        )

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    out_png = output_dir / f"{args.stem}.png"
    out_pdf = output_dir / f"{args.stem}.pdf"
    out_csv = output_dir / f"{args.stem}_summary.csv"

    fig.subplots_adjust(left=0.12, right=0.92, bottom=0.14, top=0.98)
    fig.savefig(out_png, dpi=320, bbox_inches="tight", facecolor="white")
    fig.savefig(out_pdf, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    summary.to_csv(out_csv, index=False)

    print(f"Saved PNG figure to: {out_png}")
    print(f"Saved PDF figure to: {out_pdf}")
    print(f"Saved summary CSV to: {out_csv}")
    print(f"Rows used: {len(plot_df)}")
    print(f"Rows excluded because GT MRL < 0: {negative_rows}")


if __name__ == "__main__":
    main()
