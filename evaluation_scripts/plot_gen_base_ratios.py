#!/usr/bin/env python3
"""Plot generated base ratio trajectories from a finetuning log."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
DEFAULT_LOG = (
    REPO_ROOT
    / "mdlm_50nt/seed9/alpha0.005_beta0.0_accum4_bsz32_truncate50_temp1.0_clip1.0_vanilla_grid_alpha0.005_beta0.0_20260427_211555/log_seed9_20260427_211555.txt"
)
DEFAULT_OUTPUT_DIR = SCRIPT_DIR

LINE_RE = re.compile(
    r"Epoch\s+(?P<epoch>\d+).*?"
    r"Gen A ratio\s+(?P<A>[-+eE0-9\.]+)\s+"
    r"Gen C ratio\s+(?P<C>[-+eE0-9\.]+)\s+"
    r"Gen G ratio\s+(?P<G>[-+eE0-9\.]+)\s+"
    r"Gen T ratio\s+(?P<U>[-+eE0-9\.]+)"
)

COLORS = {
    "A": "#1B9E77",
    "C": "#D95F02",
    "G": "#7570B3",
    "U": "#E7298A",
}

LINESTYLES = {
    "A": "-",
    "C": "--",
    "G": "-.",
    "U": ":",
}


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


def parse_log(path: Path) -> dict[str, list[float]]:
    rows = {"epoch": [], "A": [], "C": [], "G": [], "U": []}
    with path.open() as handle:
        for line in handle:
            match = LINE_RE.search(line)
            if match is None:
                continue
            rows["epoch"].append(int(match.group("epoch")))
            for base in ("A", "C", "G", "U"):
                rows[base].append(float(match.group(base)))
    if not rows["epoch"]:
        raise ValueError(f"No generated base ratios parsed from {path}")
    return rows


def format_axis(ax: plt.Axes) -> None:
    ax.grid(axis="both", linestyle="--", color="#D9D9D9", linewidth=0.55, alpha=0.9)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color("#555555")
        spine.set_linewidth(0.8)
    ax.tick_params(axis="both", colors="#222222", width=0.8, length=3.0, pad=2.0)
    ax.set_axisbelow(True)


def plot_base_ratios(rows: dict[str, list[float]], out_png: Path, out_pdf: Path) -> None:
    fig, ax = plt.subplots(figsize=(2.85, 2.1))
    epochs = np.asarray(rows["epoch"])
    for base in ("A", "C", "G", "U"):
        ax.plot(
            epochs,
            rows[base],
            color=COLORS[base],
            linestyle=LINESTYLES[base],
            linewidth=1.65,
            label=base,
        )

    ax.set_xlabel("Number of Epochs")
    ax.set_ylabel("Generated Base Ratio")
    ax.set_xlim(0, 100)
    ax.set_ylim(0.0, 0.42)
    ax.set_xticks(np.arange(0, 101, 20))
    ax.set_yticks(np.arange(0.0, 0.41, 0.1))
    format_axis(ax)

    legend = ax.legend(
        loc="center right",
        bbox_to_anchor=(0.98, 0.42),
        ncol=2,
        frameon=True,
        fancybox=False,
        framealpha=0.95,
        handlelength=1.8,
        borderaxespad=0.2,
        borderpad=0.35,
        labelspacing=0.25,
        columnspacing=0.8,
    )
    legend.get_frame().set_edgecolor("#BDBDBD")
    legend.get_frame().set_linewidth(0.8)

    fig.subplots_adjust(left=0.19, right=0.97, bottom=0.22, top=0.96)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=320, bbox_inches="tight", facecolor="white")
    fig.savefig(out_pdf, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Plot generated A/C/G/U base ratio trajectories from a finetuning log."
    )
    parser.add_argument("--log", type=Path, default=DEFAULT_LOG, help="Input log file.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for saved figures.",
    )
    parser.add_argument(
        "--stem",
        type=str,
        default="gen_base_ratio_trajectory",
        help="Output filename stem.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    configure_style()
    rows = parse_log(args.log)
    output_dir = args.output_dir.resolve()
    out_png = output_dir / f"{args.stem}.png"
    out_pdf = output_dir / f"{args.stem}.pdf"
    plot_base_ratios(rows, out_png, out_pdf)
    print(f"Saved PNG figure to: {out_png}")
    print(f"Saved PDF figure to: {out_pdf}")


if __name__ == "__main__":
    main()
