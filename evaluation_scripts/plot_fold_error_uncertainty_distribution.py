#!/usr/bin/env python3
"""Single figure for RL vs 5-fold predictions + error/uncertainty distributions.

Figure layout (one PNG):
- Left panel: KDE curves of GT RL and each fold prediction distribution.
- Right panel: KDE curves of per-fold error (pred - GT) and uncertainty proxy
  (std across 5 fold predictions per sample).
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def infer_fold_columns(df: pd.DataFrame):
    cols = [c for c in df.columns if c.startswith("oracle_rl_fold")]
    if not cols:
        raise ValueError("No fold columns found. Expected columns like oracle_rl_fold1..N")
    return sorted(cols, key=lambda x: int("".join(ch for ch in x if ch.isdigit()) or 0))


def main():
    parser = argparse.ArgumentParser(description="Plot one figure for 5-fold prediction distributions, errors, and uncertainty.")
    parser.add_argument(
        "--csv_path",
        type=str,
        default="/home/xli263/xli/utr_design/DRAKES/drakes_dna/output_pretrained_rl/generated_optimized_4_base_50_utr_new_with_oracle_5_fold.csv",
        help="Path to CSV containing rl and oracle_rl_fold* columns.",
    )
    parser.add_argument("--target_col", type=str, default="rl", help="Ground-truth label column.")
    parser.add_argument(
        "--out_path",
        type=str,
        default="/home/xli263/xli/utr_design/DRAKES/drakes_rna/evaluation_scripts/fold_error_uncertainty_distribution_optimized_sequences.png",
        help="Output PNG path.",
    )
    parser.add_argument("--bw_adjust", type=float, default=1.0, help="KDE smoothing bandwidth adjustment.")
    args = parser.parse_args()

    df = pd.read_csv(args.csv_path)
    has_target = args.target_col in df.columns

    fold_cols = infer_fold_columns(df)
    keep_cols = fold_cols + ([args.target_col] if has_target else [])
    df = df[keep_cols].dropna().reset_index(drop=True)

    fold_preds = df[fold_cols].to_numpy(dtype=float)

    # Per-sample uncertainty proxy from fold disagreement.
    uncertainty = fold_preds.std(axis=1, ddof=0)
    uncertainty_mean = float(np.mean(uncertainty))

    # GT is only used for the left panel overlay when available.
    if has_target:
        y = df[args.target_col].to_numpy(dtype=float)

    sns.set_theme(style="whitegrid", context="notebook")
    fig, axes = plt.subplots(1, 2, figsize=(14.5, 5.4))

    # Panel 1: GT + fold prediction distributions (or folds only if GT is absent).
    ax = axes[0]
    if has_target:
        sns.kdeplot(y, ax=ax, label=args.target_col, linewidth=2.6, color="black", bw_adjust=args.bw_adjust)
    palette = sns.color_palette("tab10", n_colors=len(fold_cols))
    for c, color in zip(fold_cols, palette):
        sns.kdeplot(df[c].to_numpy(dtype=float), ax=ax, label=c, linewidth=1.8, bw_adjust=args.bw_adjust, color=color)
    ax.set_title("Label Distribution: GT vs 5-fold Predictions" if has_target else "Label Distribution: 5-fold Predictions")
    ax.set_xlabel("RL value")
    ax.set_ylabel("Density")
    ax.legend(fontsize=8)

    # Panel 2: uncertainty distribution only.
    ax = axes[1]
    sns.kdeplot(
        uncertainty,
        ax=ax,
        label="uncertainty (std across folds)",
        linewidth=2.6,
        linestyle="--",
        color="crimson",
        bw_adjust=args.bw_adjust,
    )
    ax.axvline(0.0, color="gray", linestyle=":", linewidth=1.0)
    ax.set_title("Uncertainty Distribution")
    ax.set_xlabel("Value")
    ax.set_ylabel("Density")
    ax.text(
        0.98,
        0.98,
        f"mean={uncertainty_mean:.4f}",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=10,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8, edgecolor="gray"),
    )
    ax.legend(fontsize=8)

    fig.tight_layout()

    out_path = Path(args.out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=260)
    plt.close(fig)

    print(f"Saved: {out_path}")
    print(f"Samples: {len(df)} | folds: {', '.join(fold_cols)}")
    print(f"Uncertainty mean (std across folds per sample): {uncertainty_mean:.6f}")
    if not has_target:
        print(f"Note: target column '{args.target_col}' not found; left panel uses fold predictions only.")


if __name__ == "__main__":
    main()
