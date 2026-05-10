import csv
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path("/home/xli263/xli/utr_design/DRAKES/drakes_rna/interpretable")
DEFAULT_VANILLA_CSV = ROOT / "vanilla" / "attention_visualizations" / "all_3mer_average_contribution.csv"
DEFAULT_AUGMENT_CSV = ROOT / "augment" / "attention_visualizations_augment" / "all_3mer_average_contribution.csv"
DEFAULT_OUTPUT_DIR = ROOT / "augment" / "attention_visualizations_augment"


def mean(vals):
    return sum(vals) / len(vals) if vals else 0.0


def load_table(path):
    with path.open(newline="") as handle:
        return {row["kmer"]: row for row in csv.DictReader(handle)}


def at_count(kmer):
    return sum(base in "AT" for base in kmer)


def g_count(kmer):
    return kmer.count("G")


def build_rows(vanilla_csv, augment_csv):
    vanilla = load_table(vanilla_csv)
    augment = load_table(augment_csv)
    rows = []
    for kmer in sorted(vanilla):
        if kmer not in augment:
            continue
        vanilla_mean = float(vanilla[kmer]["mean_score"])
        augment_mean = float(augment[kmer]["mean_score"])
        delta = augment_mean - vanilla_mean
        rows.append(
            {
                "kmer": kmer,
                "at_count": at_count(kmer),
                "g_count": g_count(kmer),
                "vanilla_mean_score": vanilla_mean,
                "augment_mean_score": augment_mean,
                "delta_mean_score": delta,
                "pct_delta_mean_score": 100.0 * delta / abs(vanilla_mean),
            }
        )
    return rows


def sem_95(vals):
    vals = np.asarray(vals, dtype=float)
    if len(vals) <= 1:
        return 0.0
    return 1.96 * float(vals.std(ddof=1)) / np.sqrt(len(vals))


def write_panel_data(group_defs, panel_data_csv):
    fieldnames = [
        "kmer",
        "group",
        "g_count",
        "vanilla_mean_score",
        "augment_mean_score",
        "delta_mean_score",
        "pct_delta_mean_score",
    ]
    panel_data_csv.parent.mkdir(parents=True, exist_ok=True)
    with panel_data_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for label, rows, _ in group_defs:
            for row in rows:
                writer.writerow(
                    {
                        "kmer": row["kmer"],
                        "group": label,
                        "g_count": row["g_count"],
                        "vanilla_mean_score": row["vanilla_mean_score"],
                        "augment_mean_score": row["augment_mean_score"],
                        "delta_mean_score": row["delta_mean_score"],
                        "pct_delta_mean_score": row["pct_delta_mean_score"],
                    }
                )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot standalone 3-mer contribution reweighting by G-content."
    )
    parser.add_argument("--vanilla-csv", type=Path, default=DEFAULT_VANILLA_CSV)
    parser.add_argument("--augment-csv", type=Path, default=DEFAULT_AUGMENT_CSV)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--output-prefix", type=str, default="paper_3mer_reweighting_by_g_content")
    return parser.parse_args()


def main():
    args = parse_args()
    out_prefix = args.output_dir / args.output_prefix
    panel_data_csv = args.output_dir / f"{args.output_prefix}_data.csv"

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 7,
            "axes.titlesize": 8,
            "axes.labelsize": 7.5,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 6.5,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "axes.linewidth": 0.7,
            "xtick.major.width": 0.7,
            "ytick.major.width": 0.7,
            "xtick.major.size": 2.6,
            "ytick.major.size": 2.6,
        }
    )

    rows = build_rows(args.vanilla_csv, args.augment_csv)
    no_g_color = "#c84b31"
    g_color = "#2a7f62"
    group_defs = [
        ("No G", [r for r in rows if r["g_count"] == 0], no_g_color),
        ("Contains G", [r for r in rows if r["g_count"] > 0], g_color),
    ]
    write_panel_data(group_defs, panel_data_csv)

    fig, ax = plt.subplots(figsize=(2.25, 2.05))
    ax.axhline(0, color="#4a4a4a", lw=0.7, zorder=1)

    for xi, (label, group, color) in enumerate(group_defs):
        vals = [r["pct_delta_mean_score"] for r in group]
        ordered = sorted(vals)
        offsets = np.linspace(-0.16, 0.16, len(ordered)) if len(ordered) > 1 else np.array([0.0])
        ax.scatter(
            xi + offsets,
            ordered,
            s=16,
            color=color,
            alpha=0.62,
            edgecolor="white",
            linewidth=0.25,
            zorder=2,
        )

        group_mean = mean(vals)
        ci = sem_95(vals)
        ax.errorbar(
            [xi],
            [group_mean],
            yerr=[[ci], [ci]],
            fmt="o",
            markersize=4.2,
            color="#111111",
            ecolor="#111111",
            elinewidth=1.05,
            capsize=3.0,
            capthick=1.05,
            zorder=4,
        )
        ax.text(
            xi,
            group_mean + (2.5 if group_mean >= 0 else -2.7),
            f"{group_mean:+.1f}%",
            ha="center",
            va="center",
            fontsize=7.5,
            fontweight="bold",
            color="#111111",
        )

    ax.set_xlim(-0.55, 1.55)
    ax.set_ylim(-18, 19)
    ax.set_xticks([0, 1])
    ax.set_xticklabels([f"{label}\n(n={len(group)})" for label, group, _ in group_defs])
    ax.set_ylabel("Contribution change (%)")
    ax.set_title("Reweighting by G content", loc="left", fontweight="bold", pad=4)
    ax.set_yticks([-15, -10, -5, 0, 5, 10, 15])
    ax.grid(axis="y", color="#d9d9d9", linewidth=0.45, alpha=0.65)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout(pad=0.5)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(f"{out_prefix}.pdf", bbox_inches="tight")
    fig.savefig(f"{out_prefix}.svg", bbox_inches="tight")
    fig.savefig(f"{out_prefix}.png", bbox_inches="tight", dpi=600)

    print(f"Vanilla CSV: {args.vanilla_csv}")
    print(f"Augment CSV: {args.augment_csv}")
    print(f"Wrote {out_prefix}.pdf")
    print(f"Wrote {out_prefix}.svg")
    print(f"Wrote {out_prefix}.png")
    print(f"Wrote {panel_data_csv}")


if __name__ == "__main__":
    main()
