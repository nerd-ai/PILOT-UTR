#!/usr/bin/env python3
"""Plot optimization trajectories to illustrate reward hacking under different KL strengths."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
DEFAULT_OUTPUT_DIR = SCRIPT_DIR
DEFAULT_LOGS = {
    r"$\alpha = 0.001$": REPO_ROOT
    / "mdlm_50nt/seed9/alpha0.001_beta0.0_accum4_bsz32_truncate50_temp1.0_clip1.0_vanilla_grid_alpha0.001_beta0.0_20260425_003733/log_seed9_20260425_003733.txt",
    r"$\alpha = 0.005$": REPO_ROOT
    / "mdlm_50nt/seed9/alpha0.005_beta0.0_accum4_bsz32_truncate50_temp1.0_clip1.0_vanilla_grid_alpha0.005_beta0.0_20260425_004730/log_seed9_20260425_004730.txt",
    r"$\alpha = 0.01$": REPO_ROOT
    / "mdlm_50nt/seed9/alpha0.01_beta0.0_accum4_bsz32_truncate50_temp1.0_clip1.0_vanilla_grid_alpha0.01_beta0.0_20260425_004124/log_seed9_20260425_004124.txt",
}

COLORS = {
    r"$\alpha = 0.001$": "#1B9E77",
    r"$\alpha = 0.005$": "#D95F02",
    r"$\alpha = 0.01$": "#7570B3",
}

LINESTYLES = {
    r"$\alpha = 0.001$": "-",
    r"$\alpha = 0.005$": "--",
    r"$\alpha = 0.01$": "-.",
}

LINE_RE = re.compile(
    r"Epoch\s+(?P<epoch>\d+)\s+"
    r"Mean reward\s+(?P<reward>[-+eE0-9\.]+)\s+"
    r"Mean reward eval\s+(?P<reward_eval>[-+eE0-9\.]+).*?"
    r"Eval 3-mer corr\s+(?P<kmer_corr>[-+eE0-9\.]+)\s+"
    r"Gen A\+T fraction\s+(?P<at_frac>[-+eE0-9\.]+)"
)


def configure_style() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 150,
            "savefig.dpi": 600,
            "font.family": "DejaVu Sans",
            "font.size": 7.5,
            "axes.labelsize": 8,
            "axes.titlesize": 9,
            "axes.titleweight": "normal",
            "axes.spines.top": True,
            "axes.spines.right": True,
            "axes.linewidth": 0.8,
            "xtick.labelsize": 6.5,
            "ytick.labelsize": 7,
            "legend.fontsize": 6.5,
            "xtick.major.size": 3.0,
            "ytick.major.size": 3.0,
            "xtick.major.width": 0.8,
            "ytick.major.width": 0.8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def parse_log(path: Path, label: str) -> pd.DataFrame:
    rows: list[dict[str, float | int | str]] = []
    with path.open() as handle:
        for line in handle:
            match = LINE_RE.search(line)
            if not match:
                continue
            rows.append(
                {
                    "label": label,
                    "epoch": int(match.group("epoch")),
                    "reward": float(match.group("reward")),
                    "reward_eval": float(match.group("reward_eval")),
                    "kmer_corr": float(match.group("kmer_corr")),
                    "at_frac": float(match.group("at_frac")),
                }
            )
    if not rows:
        raise ValueError(f"No epoch metrics parsed from {path}")
    return pd.DataFrame(rows).sort_values("epoch").reset_index(drop=True)


def first_epoch_at_or_above(series: pd.Series, threshold: float) -> float:
    crossed = series >= threshold
    if not crossed.any():
        return float("nan")
    return float(series.index[crossed.argmax()])


def build_summary(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for label, sub in frame.groupby("label", sort=False):
        reward_cross = sub.loc[sub["reward_eval"] >= 8.0, "epoch"]
        rows.append(
            {
                "label": label,
                "max_reward_eval": float(sub["reward_eval"].max()),
                "min_kmer_corr": float(sub["kmer_corr"].min()),
                "final_reward_eval": float(sub["reward_eval"].iloc[-1]),
                "final_kmer_corr": float(sub["kmer_corr"].iloc[-1]),
                "epoch_first_reward_eval_ge_8": (
                    int(reward_cross.iloc[0]) if len(reward_cross) else np.nan
                ),
            }
        )
    return pd.DataFrame(rows)


def _add_shared_legend(ax: plt.Axes, loc: str, anchor: tuple[float, float]) -> None:
    handles, labels = ax.get_legend_handles_labels()
    legend = ax.legend(
        handles,
        labels,
        loc=loc,
        bbox_to_anchor=anchor,
        frameon=True,
        fancybox=False,
        framealpha=0.95,
        handlelength=1.8,
        borderaxespad=0.2,
        borderpad=0.35,
        labelspacing=0.2,
    )
    legend.get_frame().set_edgecolor("#BDBDBD")
    legend.get_frame().set_linewidth(0.8)


def _format_paper_axis(ax: plt.Axes) -> None:
    ax.grid(axis="both", linestyle="--", color="#D9D9D9", linewidth=0.55, alpha=0.9)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color("#555555")
        spine.set_linewidth(0.8)
    ax.tick_params(axis="both", colors="#222222", width=0.8, length=3.0, pad=2.0)
    ax.set_axisbelow(True)


def plot_reward_trajectory(
    frame: pd.DataFrame,
    out_png: Path,
    out_pdf: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(2.85, 2.1))
    for label in DEFAULT_LOGS:
        sub = frame[frame["label"] == label].sort_values("epoch")
        color = COLORS[label]
        ax.plot(
            sub["epoch"],
            sub["reward_eval"],
            color=color,
            linestyle=LINESTYLES[label],
            linewidth=1.65,
            label=label,
        )

    ax.set_xlabel("Number of Epochs")
    ax.set_ylabel("Estimated Reward")
    ax.set_xlim(0, 100)
    ax.set_ylim(6.0, 8.55)
    ax.set_xticks(np.arange(0, 101, 20))
    ax.set_yticks(np.arange(6.0, 8.51, 0.5))
    _format_paper_axis(ax)

    _add_shared_legend(ax, loc="lower right", anchor=(0.98, 0.04))

    fig.subplots_adjust(left=0.19, right=0.97, bottom=0.22, top=0.96)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=320, bbox_inches="tight", facecolor="white")
    fig.savefig(out_pdf, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_kmer_trajectory(
    frame: pd.DataFrame,
    out_png: Path,
    out_pdf: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(2.85, 2.1))
    for label in DEFAULT_LOGS:
        sub = frame[frame["label"] == label].sort_values("epoch")
        color = COLORS[label]
        ax.plot(
            sub["epoch"],
            sub["kmer_corr"],
            color=color,
            linestyle=LINESTYLES[label],
            linewidth=1.65,
            label=label,
        )

    ax.set_xlabel("Number of Epochs")
    ax.set_ylabel("3-mer correlation")
    ax.set_xlim(0, 100)
    ax.set_ylim(-0.45, 1.02)
    ax.set_xticks(np.arange(0, 101, 20))
    ax.set_yticks(np.arange(-0.4, 1.01, 0.2))
    _format_paper_axis(ax)

    _add_shared_legend(ax, loc="center right", anchor=(0.98, 0.6))

    fig.subplots_adjust(left=0.2, right=0.97, bottom=0.22, top=0.96)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=320, bbox_inches="tight", facecolor="white")
    fig.savefig(out_pdf, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Plot reward and 3-mer correlation trajectories for multiple KL strengths."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for figure and summary CSV.",
    )
    parser.add_argument(
        "--stem",
        type=str,
        default="rl_hacking_trajectories",
        help="Output filename stem.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    configure_style()

    frames = [parse_log(path, label) for label, path in DEFAULT_LOGS.items()]
    all_df = pd.concat(frames, ignore_index=True)
    summary = build_summary(all_df)

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    reward_png = output_dir / f"{args.stem}_reward.png"
    reward_pdf = output_dir / f"{args.stem}_reward.pdf"
    kmer_png = output_dir / f"{args.stem}_kmer_corr.png"
    kmer_pdf = output_dir / f"{args.stem}_kmer_corr.pdf"
    out_csv = output_dir / f"{args.stem}_summary.csv"

    plot_reward_trajectory(all_df, reward_png, reward_pdf)
    plot_kmer_trajectory(all_df, kmer_png, kmer_pdf)
    summary.to_csv(out_csv, index=False)

    print(f"Saved reward PNG figure to: {reward_png}")
    print(f"Saved reward PDF figure to: {reward_pdf}")
    print(f"Saved k-mer PNG figure to: {kmer_png}")
    print(f"Saved k-mer PDF figure to: {kmer_pdf}")
    print(f"Saved summary CSV to: {out_csv}")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
