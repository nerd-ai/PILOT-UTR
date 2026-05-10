#!/usr/bin/env python3
"""Visualize oracle fold prediction errors against ground-truth RL labels.

Generates:
- Parity scatter plots (GT vs pred) per fold + ensemble mean
- Residual distribution (violin + strip)
- Residual vs GT with linear trend
- ECDF of absolute error
- Bland-Altman plots
- Fold correlation heatmap
- MAE/RMSE/R2 bar chart

Also saves metrics tables as CSV.
"""

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

try:
    import seaborn as sns
except Exception as exc:
    raise ImportError("This script requires seaborn. Install with: pip install seaborn") from exc


def infer_fold_columns(df: pd.DataFrame):
    cols = [c for c in df.columns if c.startswith("oracle_rl_fold")]
    if not cols:
        raise ValueError("No fold columns found. Expected columns like oracle_rl_fold1..N")
    return sorted(cols, key=lambda x: int("".join(ch for ch in x if ch.isdigit()) or 0))


def safe_r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    denom = np.sum((y_true - y_true.mean()) ** 2)
    if denom <= 0:
        return float("nan")
    num = np.sum((y_true - y_pred) ** 2)
    return 1.0 - num / denom


def compute_metrics(df: pd.DataFrame, target_col: str, pred_cols):
    rows = []
    y = df[target_col].to_numpy(dtype=float)
    for col in pred_cols:
        p = df[col].to_numpy(dtype=float)
        e = p - y
        rows.append(
            {
                "model": col,
                "n": len(df),
                "bias_mean_error": float(np.mean(e)),
                "mae": float(np.mean(np.abs(e))),
                "rmse": float(np.sqrt(np.mean(e ** 2))),
                "r2": float(safe_r2(y, p)),
                "pearson": float(np.corrcoef(y, p)[0, 1]) if len(df) > 1 else float("nan"),
            }
        )
    return pd.DataFrame(rows)


def make_parity_plots(df: pd.DataFrame, target_col: str, pred_cols, out_path: Path):
    n = len(pred_cols)
    ncols = 3
    nrows = math.ceil(n / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.5 * ncols, 4.8 * nrows), squeeze=False)
    y = df[target_col].to_numpy(dtype=float)
    y_min = min(float(np.min(y)), *(float(df[c].min()) for c in pred_cols))
    y_max = max(float(np.max(y)), *(float(df[c].max()) for c in pred_cols))

    for i, col in enumerate(pred_cols):
        ax = axes[i // ncols][i % ncols]
        p = df[col].to_numpy(dtype=float)
        e = p - y
        mae = np.mean(np.abs(e))
        rmse = np.sqrt(np.mean(e ** 2))
        r2 = safe_r2(y, p)

        ax.scatter(y, p, s=14, alpha=0.55, edgecolors="none")
        ax.plot([y_min, y_max], [y_min, y_max], "r--", linewidth=1.2)
        ax.set_title(f"{col}\nMAE={mae:.4f}, RMSE={rmse:.4f}, R2={r2:.4f}")
        ax.set_xlabel("GT RL")
        ax.set_ylabel("Predicted RL")
        ax.grid(alpha=0.2)

    for j in range(n, nrows * ncols):
        axes[j // ncols][j % ncols].axis("off")

    fig.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def make_residual_violin(long_df: pd.DataFrame, out_path: Path):
    fig, ax = plt.subplots(figsize=(11, 5.2))
    sns.violinplot(data=long_df, x="model", y="residual", inner="quartile", cut=0, ax=ax)
    sns.stripplot(data=long_df, x="model", y="residual", size=2, alpha=0.2, color="black", ax=ax)
    ax.axhline(0.0, color="red", linestyle="--", linewidth=1)
    ax.set_title("Residual Distribution by Model (pred - gt)")
    ax.set_xlabel("")
    ax.set_ylabel("Residual")
    ax.tick_params(axis="x", rotation=25)
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def make_residual_vs_gt(long_df: pd.DataFrame, out_path: Path):
    g = sns.lmplot(
        data=long_df,
        x="gt",
        y="residual",
        col="model",
        col_wrap=3,
        height=3.4,
        scatter_kws={"s": 12, "alpha": 0.35},
        line_kws={"color": "red", "linewidth": 1.3},
        lowess=False,
        ci=None,
        facet_kws={"sharex": True, "sharey": True},
    )
    for ax in g.axes.flatten():
        ax.axhline(0.0, color="black", linestyle="--", linewidth=0.8)
        ax.grid(alpha=0.2)
    g.set_axis_labels("GT RL", "Residual")
    g.fig.suptitle("Residual vs GT by Model", y=1.02)
    g.fig.tight_layout()
    g.savefig(out_path, dpi=220)
    plt.close(g.fig)


def make_ecdf_abs_error(long_df: pd.DataFrame, out_path: Path):
    fig, ax = plt.subplots(figsize=(9, 5.2))
    for model_name, sub in long_df.groupby("model"):
        vals = np.sort(np.abs(sub["residual"].to_numpy(dtype=float)))
        ys = np.arange(1, len(vals) + 1) / len(vals)
        ax.step(vals, ys, where="post", label=model_name, linewidth=1.6)
    ax.set_xlabel("Absolute Error |pred - gt|")
    ax.set_ylabel("ECDF")
    ax.set_title("ECDF of Absolute Error")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8, ncols=2)
    fig.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def make_bland_altman(long_df: pd.DataFrame, out_path: Path):
    models = list(long_df["model"].unique())
    n = len(models)
    ncols = 3
    nrows = math.ceil(n / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.5 * ncols, 4.6 * nrows), squeeze=False)

    for i, model_name in enumerate(models):
        ax = axes[i // ncols][i % ncols]
        sub = long_df[long_df["model"] == model_name]
        mean_axis = (sub["pred"].to_numpy(dtype=float) + sub["gt"].to_numpy(dtype=float)) / 2.0
        diff = sub["residual"].to_numpy(dtype=float)
        mu = float(np.mean(diff))
        sd = float(np.std(diff, ddof=0))
        loa_lo = mu - 1.96 * sd
        loa_hi = mu + 1.96 * sd

        ax.scatter(mean_axis, diff, s=12, alpha=0.35)
        ax.axhline(mu, color="red", linewidth=1.2, label=f"bias={mu:.3f}")
        ax.axhline(loa_lo, color="gray", linestyle="--", linewidth=1.0)
        ax.axhline(loa_hi, color="gray", linestyle="--", linewidth=1.0)
        ax.set_title(f"{model_name}\nLoA: [{loa_lo:.3f}, {loa_hi:.3f}]")
        ax.set_xlabel("Mean of pred and gt")
        ax.set_ylabel("pred - gt")
        ax.grid(alpha=0.2)

    for j in range(n, nrows * ncols):
        axes[j // ncols][j % ncols].axis("off")

    fig.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def make_fold_corr_heatmap(df: pd.DataFrame, pred_cols, out_path: Path):
    corr = df[pred_cols].corr()
    fig, ax = plt.subplots(figsize=(7.2, 6.2))
    sns.heatmap(corr, annot=True, fmt=".3f", cmap="viridis", square=True, cbar=True, ax=ax)
    ax.set_title("Fold Prediction Correlation")
    fig.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def make_metrics_bar(metrics_df: pd.DataFrame, out_path: Path):
    plot_df = metrics_df.copy()
    melted = plot_df.melt(id_vars=["model"], value_vars=["mae", "rmse", "r2"], var_name="metric", value_name="value")
    fig, ax = plt.subplots(figsize=(10, 5.4))
    sns.barplot(data=melted, x="model", y="value", hue="metric", ax=ax)
    ax.set_title("Model Metrics")
    ax.set_xlabel("")
    ax.tick_params(axis="x", rotation=25)
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Visualize GT-vs-fold oracle prediction errors.")
    parser.add_argument(
        "--csv_path",
        type=str,
        default="/home/xli263/xli/utr_design/DRAKES/drakes_rna/data_and_model/mrl_8_to_8.6_dataset_with_oracle_5_fold.csv",
        help="Path to CSV containing columns: rl and oracle_rl_fold*",
    )
    parser.add_argument(
        "--target_col",
        type=str,
        default="rl",
        help="Ground-truth label column name.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/home/xli263/xli/utr_design/DRAKES/drakes_rna/evaluation_scripts/oracle_fold_error_plots",
        help="Directory to save figures and metrics.",
    )
    args = parser.parse_args()

    csv_path = Path(args.csv_path)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(csv_path)
    if args.target_col not in df.columns:
        raise ValueError(f"Target column '{args.target_col}' not found in CSV.")

    fold_cols = infer_fold_columns(df)
    keep_cols = [args.target_col] + fold_cols
    df = df[keep_cols].dropna().reset_index(drop=True)

    # Add ensemble mean prediction.
    df["oracle_rl_ensemble_mean"] = df[fold_cols].mean(axis=1)
    model_cols = fold_cols + ["oracle_rl_ensemble_mean"]

    # Long format for residual-based plots.
    long_parts = []
    for col in model_cols:
        tmp = pd.DataFrame(
            {
                "gt": df[args.target_col].to_numpy(dtype=float),
                "pred": df[col].to_numpy(dtype=float),
                "model": col,
            }
        )
        tmp["residual"] = tmp["pred"] - tmp["gt"]
        long_parts.append(tmp)
    long_df = pd.concat(long_parts, axis=0, ignore_index=True)

    metrics_df = compute_metrics(df, args.target_col, model_cols)
    metrics_df.to_csv(out_dir / "metrics_summary.csv", index=False)

    residual_summary = (
        long_df.groupby("model")["residual"]
        .agg(["mean", "std", "median", lambda x: np.mean(np.abs(x)), "min", "max"])
        .rename(columns={"<lambda_0>": "mae"})
        .reset_index()
    )
    residual_summary.to_csv(out_dir / "residual_summary.csv", index=False)

    sns.set_theme(style="whitegrid", context="notebook")

    make_parity_plots(df, args.target_col, model_cols, out_dir / "01_parity_scatter.png")
    make_residual_violin(long_df, out_dir / "02_residual_violin_strip.png")
    make_residual_vs_gt(long_df, out_dir / "03_residual_vs_gt.png")
    make_ecdf_abs_error(long_df, out_dir / "04_abs_error_ecdf.png")
    make_bland_altman(long_df, out_dir / "05_bland_altman.png")
    make_fold_corr_heatmap(df, fold_cols, out_dir / "06_fold_correlation_heatmap.png")
    make_metrics_bar(metrics_df, out_dir / "07_metrics_bar.png")

    print(f"Saved figures and tables to: {out_dir}")
    print(f"Models evaluated: {', '.join(model_cols)}")


if __name__ == "__main__":
    main()
