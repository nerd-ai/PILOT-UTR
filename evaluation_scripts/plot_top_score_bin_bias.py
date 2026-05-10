#!/usr/bin/env python3
"""Plot slim top-percent RL bars with hatched GT gap overlays.

Default workflow:
1. Use `data_and_model/val_dataset.csv` as the ground-truth table.
2. Generate UTR-LM predictions with `run_utrlm_hard_infer.py`.
   This intentionally reuses the default checkpoint defined in that script.
3. Generate FramePool predictions with the UTRGAN FramePool checkpoint.
   This runs under the `utrgan` Conda environment.
4. Rank sequences by ground-truth `rl`.
5. Plot Top 5% / Top 10% / Top 20% groups.

Each bar shows the model's average predicted score. The hatched overlay on the
same bar marks the gap to the average GT value for that RL-ranked subset.
"""

from __future__ import annotations

import argparse
import math
import subprocess
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
DEFAULT_SOURCE_CSV = REPO_ROOT / "data_and_model" / "val_dataset.csv"
DEFAULT_UTRLM_PRED_CSV = SCRIPT_DIR / "val_dataset_utrlm_predictions.csv"
DEFAULT_FRAMEPOOL_PRED_CSV = SCRIPT_DIR / "val_dataset_framepool_predictions.csv"
DEFAULT_ORACLE_PRED_CSV = SCRIPT_DIR / "val_dataset_oracle_predictions.csv"
DEFAULT_OUTPUT_DIR = SCRIPT_DIR
UTRLM_INFER_SCRIPT = REPO_ROOT / "run_utrlm_hard_infer.py"
FRAMEPOOL_PREDICT_SCRIPT = SCRIPT_DIR / "predict_framepool_scores.py"
FRAMEPOOL_ENV_NAME = "utrgan"
ORACLE_SCORE_SCRIPT = REPO_ROOT / "oracle_utr_new.py"
ORACLE_CHECKPOINT = (
    REPO_ROOT / "experiment" / "single_run_vanilla_mrl_50nt" / "vanilla_best.ckpt"
)

PAPER_COLORS = {
    "UTR-LM": "#4C78A8",
    "FramePool": "#D95F02",
    "Enformer": "#2A9D8F",
}


def configure_plot_style() -> None:
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
            "axes.linewidth": 2.2,
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
            "legend.fontsize": 11,
            "hatch.linewidth": 1.0,
        }
    )


def parse_model_columns(items: list[str]) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for item in items:
        if "=" not in item:
            raise ValueError(
                f"Invalid --model-col value '{item}'. Expected format: ModelName=column_name"
            )
        model_name, column_name = item.split("=", 1)
        model_name = model_name.strip()
        column_name = column_name.strip()
        if not model_name or not column_name:
            raise ValueError(
                f"Invalid --model-col value '{item}'. Model name and column must be non-empty."
            )
        mapping[model_name] = column_name
    if not mapping:
        raise ValueError("At least one --model-col entry is required.")
    return mapping


def maybe_generate_utrlm_predictions(
    source_csv: Path,
    pred_csv: Path,
    seq_col: str,
    force: bool,
) -> None:
    needs_generation = force or not pred_csv.exists()
    if not needs_generation:
        df = pd.read_csv(pred_csv, nrows=3)
        needs_generation = "utrlm_pred" not in df.columns

    if not needs_generation:
        return

    cmd = [
        sys.executable,
        str(UTRLM_INFER_SCRIPT),
        "--input-csv",
        str(source_csv),
        "--seq-column",
        seq_col,
        "--output-csv",
        str(pred_csv),
    ]
    subprocess.run(cmd, check=True)


def maybe_generate_framepool_predictions(
    source_csv: Path,
    pred_csv: Path,
    seq_col: str,
    force: bool,
) -> None:
    needs_generation = force or not pred_csv.exists()
    if not needs_generation:
        df = pd.read_csv(pred_csv, nrows=3)
        needs_generation = "framepool_pred" not in df.columns

    if not needs_generation:
        return

    cmd = [
        "conda",
        "run",
        "-n",
        FRAMEPOOL_ENV_NAME,
        "python",
        str(FRAMEPOOL_PREDICT_SCRIPT),
        "--input-csv",
        str(source_csv),
        "--output-csv",
        str(pred_csv),
        "--seq-col",
        seq_col,
    ]
    subprocess.run(cmd, check=True)


def maybe_generate_oracle_predictions(
    source_csv: Path,
    pred_csv: Path,
    seq_col: str,
    force: bool,
) -> None:
    needs_generation = force or not pred_csv.exists()
    if not needs_generation:
        df = pd.read_csv(pred_csv, nrows=3)
        needs_generation = "oracle_pred" not in df.columns

    if not needs_generation:
        return

    cmd = [
        sys.executable,
        str(ORACLE_SCORE_SCRIPT),
        "--checkpoint",
        str(ORACLE_CHECKPOINT),
        "--input-csv",
        str(source_csv),
        "--output-csv",
        str(pred_csv),
        "--seq-column",
        seq_col,
        "--score-column",
        "oracle_pred",
    ]
    subprocess.run(cmd, check=True)


def build_plot_frame(
    source_csv: Path,
    seq_col: str,
    gt_col: str,
    utrlm_pred_csv: Path | None,
    framepool_pred_csv: Path | None,
    oracle_pred_csv: Path | None,
    model_columns: dict[str, str],
) -> pd.DataFrame:
    df = pd.read_csv(source_csv)
    if seq_col not in df.columns:
        raise ValueError(f"Sequence column '{seq_col}' not found in {source_csv}")

    if utrlm_pred_csv is not None and "utrlm_pred" in model_columns.values():
        utr_df = pd.read_csv(utrlm_pred_csv)[[seq_col, "utrlm_pred"]].drop_duplicates(subset=[seq_col])
        df = df.merge(utr_df, on=seq_col, how="left")

    if framepool_pred_csv is not None and "framepool_pred" in model_columns.values():
        fp_df = pd.read_csv(framepool_pred_csv)[[seq_col, "framepool_pred"]].drop_duplicates(subset=[seq_col])
        df = df.merge(fp_df, on=seq_col, how="left")

    if oracle_pred_csv is not None and "oracle_pred" in model_columns.values():
        oracle_df = pd.read_csv(oracle_pred_csv)[[seq_col, "oracle_pred"]].drop_duplicates(subset=[seq_col])
        df = df.merge(oracle_df, on=seq_col, how="left")

    required_cols = [gt_col, *model_columns.values()]
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns after merge: {missing}")
    clean = df.dropna(subset=required_cols).copy()
    clean = clean.sort_values(gt_col, ascending=False).reset_index(drop=True)
    return clean


def summarize_top_percent(
    df: pd.DataFrame,
    gt_col: str,
    model_columns: dict[str, str],
    top_percents: list[float],
) -> pd.DataFrame:
    total_n = len(df)
    rows = []
    for pct in top_percents:
        k = max(1, math.ceil(total_n * pct / 100.0))
        subset = df.head(k)
        gt_mean = float(subset[gt_col].mean())
        label = f"Top {int(pct)}%"
        for model_name, pred_col in model_columns.items():
            pred_mean = float(subset[pred_col].mean())
            rows.append(
                {
                    "group": label,
                    "top_percent": pct,
                    "n": len(subset),
                    "series": model_name,
                    "pred_mean": pred_mean,
                    "gt_mean": gt_mean,
                    "gap_to_gt": gt_mean - pred_mean,
                    "gap_abs": abs(gt_mean - pred_mean),
                }
            )
    summary = pd.DataFrame(rows)
    summary["group"] = pd.Categorical(
        summary["group"],
        categories=[f"Top {int(p)}%" for p in top_percents],
        ordered=True,
    )
    return summary.sort_values(["top_percent", "series"]).reset_index(drop=True)


def compute_pearson_by_model(
    df: pd.DataFrame,
    gt_col: str,
    model_columns: dict[str, str],
) -> dict[str, float]:
    y_true = df[gt_col].to_numpy(dtype=float)
    out: dict[str, float] = {}
    for model_name, pred_col in model_columns.items():
        y_pred = df[pred_col].to_numpy(dtype=float)
        if len(y_true) < 2 or np.std(y_true) == 0.0 or np.std(y_pred) == 0.0:
            out[model_name] = float("nan")
        else:
            out[model_name] = float(np.corrcoef(y_true, y_pred)[0, 1])
    return out


def plot_top_percent_gap_bars(
    summary: pd.DataFrame,
    model_names: list[str],
    out_png: Path,
    out_pdf: Path,
) -> None:
    groups = list(summary["group"].cat.categories)
    x = np.arange(len(groups))
    width = 0.16 if len(model_names) == 1 else min(0.14, 0.42 / len(model_names))
    offsets = [
        (idx - (len(model_names) - 1) / 2.0) * (width * 1.9) for idx in range(len(model_names))
    ]

    fig, ax = plt.subplots(figsize=(6.8, 5.2))

    for offset, model_name in zip(offsets, model_names):
        sub = summary[summary["series"] == model_name].set_index("group").loc[groups].reset_index()
        xpos = x + offset
        pred_vals = sub["pred_mean"].to_numpy(dtype=float)
        gt_vals = sub["gt_mean"].to_numpy(dtype=float)
        gap_vals = gt_vals - pred_vals
        overlay_bottom = np.minimum(pred_vals, gt_vals)
        overlay_height = np.abs(gap_vals)

        bars = ax.bar(
            xpos,
            pred_vals,
            width=width,
            color=PAPER_COLORS.get(model_name, "#7F7F7F"),
            edgecolor="#222222",
            linewidth=1.0,
            alpha=0.7,
            zorder=3,
            label=model_name,
        )
        ax.bar(
            xpos,
            overlay_height,
            width=width,
            bottom=overlay_bottom,
            color="white",
            edgecolor=PAPER_COLORS.get(model_name, "#7F7F7F"),
            linewidth=1.0,
            hatch="////",
            zorder=4,
        )

    ax.set_ylabel("Average predicted score")
    ax.set_xlabel("Sequences ranked by GT MRL")
    ax.set_xticks(x)
    ax.set_xticklabels(groups)
    ax.grid(axis="y", linestyle=(0, (2, 3)), linewidth=0.9, alpha=0.35, zorder=0)
    ax.spines["left"].set_linewidth(2.4)
    ax.spines["bottom"].set_linewidth(2.4)

    legend_handles = [
        Patch(
            facecolor=PAPER_COLORS.get(model_name, "#7F7F7F"),
            edgecolor="#222222",
            label=model_name,
        )
        for model_name in model_names
    ]
    legend_handles.append(
        Patch(facecolor="white", edgecolor="#444444", hatch="////", label="Gap to GT")
    )
    ax.legend(
        handles=legend_handles,
        frameon=False,
        ncol=min(len(legend_handles), 4),
        loc="lower center",
        bbox_to_anchor=(0.5, 1.02),
        borderaxespad=0.0,
    )

    ax.margins(x=0.18)
    fig.subplots_adjust(left=0.13, right=0.98, bottom=0.14, top=0.90)

    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=320, bbox_inches="tight", facecolor="white")
    fig.savefig(out_pdf, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Plot slim bars of average predicted score with hatched GT gap overlays."
    )
    parser.add_argument(
        "--source-csv",
        type=Path,
        default=DEFAULT_SOURCE_CSV,
        help="Ground-truth validation CSV, ranked by RL.",
    )
    parser.add_argument(
        "--utrlm-pred-csv",
        type=Path,
        default=DEFAULT_UTRLM_PRED_CSV,
        help="CSV containing UTR-LM predictions. It will be generated if missing.",
    )
    parser.add_argument(
        "--framepool-pred-csv",
        type=Path,
        default=DEFAULT_FRAMEPOOL_PRED_CSV,
        help="CSV containing FramePool predictions. It will be generated if missing.",
    )
    parser.add_argument(
        "--oracle-pred-csv",
        type=Path,
        default=DEFAULT_ORACLE_PRED_CSV,
        help="CSV containing oracle_utr_new predictions. It will be generated if missing.",
    )
    parser.add_argument(
        "--seq-col",
        type=str,
        default="utr",
        help="Sequence column name passed into run_utrlm_hard_infer.py.",
    )
    parser.add_argument(
        "--gt-col",
        type=str,
        default="rl",
        help="Ground-truth RL column used for ranking.",
    )
    parser.add_argument(
        "--model-col",
        type=str,
        action="append",
        default=["UTR-LM=utrlm_pred", "FramePool=framepool_pred", "Enformer=oracle_pred"],
        help="Model definition in the form ModelName=column_name. Repeat for multiple models.",
    )
    parser.add_argument(
        "--top-percents",
        type=float,
        nargs="+",
        default=[5, 10, 20],
        help="Top percentages ranked by RL.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for saved figures and summary CSV.",
    )
    parser.add_argument(
        "--stem",
        type=str,
        default="top_rl_percent_gap_barplot",
        help="Output filename stem.",
    )
    parser.add_argument(
        "--force-utrlm-infer",
        action="store_true",
        help="Regenerate the UTR-LM prediction CSV even if it already exists.",
    )
    parser.add_argument(
        "--force-framepool-infer",
        action="store_true",
        help="Regenerate the FramePool prediction CSV even if it already exists.",
    )
    parser.add_argument(
        "--force-oracle-infer",
        action="store_true",
        help="Regenerate the oracle_utr_new prediction CSV even if it already exists.",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    configure_plot_style()
    model_columns = parse_model_columns(args.model_col)

    if "utrlm_pred" in model_columns.values():
        maybe_generate_utrlm_predictions(
            source_csv=args.source_csv,
            pred_csv=args.utrlm_pred_csv,
            seq_col=args.seq_col,
            force=args.force_utrlm_infer,
        )
    if "framepool_pred" in model_columns.values():
        maybe_generate_framepool_predictions(
            source_csv=args.source_csv,
            pred_csv=args.framepool_pred_csv,
            seq_col=args.seq_col,
            force=args.force_framepool_infer,
        )
    if "oracle_pred" in model_columns.values():
        maybe_generate_oracle_predictions(
            source_csv=args.source_csv,
            pred_csv=args.oracle_pred_csv,
            seq_col=args.seq_col,
            force=args.force_oracle_infer,
        )

    ranked_df = build_plot_frame(
        source_csv=args.source_csv,
        seq_col=args.seq_col,
        gt_col=args.gt_col,
        utrlm_pred_csv=args.utrlm_pred_csv if "utrlm_pred" in model_columns.values() else None,
        framepool_pred_csv=args.framepool_pred_csv if "framepool_pred" in model_columns.values() else None,
        oracle_pred_csv=args.oracle_pred_csv if "oracle_pred" in model_columns.values() else None,
        model_columns=model_columns,
    )
    summary_df = summarize_top_percent(
        ranked_df,
        gt_col=args.gt_col,
        model_columns=model_columns,
        top_percents=args.top_percents,
    )
    pearson_by_model = compute_pearson_by_model(
        ranked_df,
        gt_col=args.gt_col,
        model_columns=model_columns,
    )

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    out_png = output_dir / f"{args.stem}.png"
    out_pdf = output_dir / f"{args.stem}.pdf"
    out_csv = output_dir / f"{args.stem}_summary.csv"

    plot_top_percent_gap_bars(
        summary_df,
        list(model_columns.keys()),
        out_png,
        out_pdf,
    )
    summary_df.to_csv(out_csv, index=False)

    print(f"Saved PNG figure to: {out_png}")
    print(f"Saved PDF figure to: {out_pdf}")
    print(f"Saved summary table to: {out_csv}")
    if "utrlm_pred" in model_columns.values():
        print(f"UTR-LM prediction CSV: {args.utrlm_pred_csv}")
    if "framepool_pred" in model_columns.values():
        print(f"FramePool prediction CSV: {args.framepool_pred_csv}")
    if "oracle_pred" in model_columns.values():
        print(f"Oracle prediction CSV: {args.oracle_pred_csv}")
    print(f"Rows used after dropping missing values: {len(ranked_df)}")
    for model_name in model_columns:
        pearson = pearson_by_model.get(model_name, float("nan"))
        if np.isfinite(pearson):
            print(f"Pearson r for {model_name}: {pearson:.6f}")


if __name__ == "__main__":
    main()
