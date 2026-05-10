import argparse
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path("/home/xli263/xli/utr_design/DRAKES/drakes_rna")
DEFAULT_NEGATIVE_CSV = ROOT / "generated_sequences_negative_samples" / "generated_negative_samples.csv"
DEFAULT_VANILLA_CSV = ROOT / "interpretable" / "vanilla" / "attention_visualizations" / "all_3mer_average_contribution.csv"
DEFAULT_AUGMENT_CSV = ROOT / "interpretable" / "augment" / "attention_visualizations_augment" / "all_3mer_average_contribution.csv"
DEFAULT_OUTPUT_DIR = ROOT / "interpretable" / "augment"


def top_kmers(sequence_csv: Path, seq_col: str, k: int, top_n: int):
    frame = pd.read_csv(sequence_csv)
    counts = Counter()
    for seq in frame[seq_col].astype(str).str.upper():
        for start in range(max(0, len(seq) - k + 1)):
            kmer = seq[start:start + k]
            if set(kmer) <= set("ACGT"):
                counts[kmer] += 1
    return counts.most_common(top_n), sum(counts.values())


def build_panel_data(args):
    top, total_kmers = top_kmers(args.sequence_csv, args.seq_col, args.k, args.top_n)
    vanilla = pd.read_csv(args.vanilla_csv).set_index("kmer")
    augment = pd.read_csv(args.augment_csv).set_index("kmer")

    rows = []
    for rank, (kmer, count) in enumerate(top, start=1):
        v = vanilla.loc[kmer]
        a = augment.loc[kmer]
        delta_mean = float(a["mean_score"] - v["mean_score"])
        delta_attn = float(a["mean_attention_score"] - v["mean_attention_score"])
        rows.append(
            {
                "rank": rank,
                "kmer": kmer,
                "count": count,
                "frequency": count / total_kmers,
                "vanilla_mean_score": float(v["mean_score"]),
                "augment_mean_score": float(a["mean_score"]),
                "delta_mean_score": delta_mean,
                "pct_delta_mean_score": 100.0 * delta_mean / abs(float(v["mean_score"])),
                "vanilla_mean_attention_score": float(v["mean_attention_score"]),
                "augment_mean_attention_score": float(a["mean_attention_score"]),
                "delta_mean_attention_score": delta_attn,
                "pct_delta_mean_attention_score": 100.0 * delta_attn / abs(float(v["mean_attention_score"])),
            }
        )
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(
        description="Plot contribution shifts for the most frequent k-mers in generated negative samples."
    )
    parser.add_argument("--sequence-csv", type=Path, default=DEFAULT_NEGATIVE_CSV)
    parser.add_argument("--seq-col", type=str, default="seq")
    parser.add_argument("--vanilla-csv", type=Path, default=DEFAULT_VANILLA_CSV)
    parser.add_argument("--augment-csv", type=Path, default=DEFAULT_AUGMENT_CSV)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--output-prefix", type=str, default="paper_top5_negative_3mer_contribution_change")
    parser.add_argument("--k", type=int, default=3)
    parser.add_argument("--top-n", type=int, default=5)
    args = parser.parse_args()

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

    data = build_panel_data(args)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    data_path = args.output_dir / f"{args.output_prefix}_data.csv"
    data.to_csv(data_path, index=False)

    x = np.arange(len(data))
    width = 0.34
    vanilla_color = "#6f7f91"
    augment_color = "#2a7f62"

    fig, ax = plt.subplots(figsize=(3.15, 2.15))
    ax.bar(
        x - width / 2,
        data["vanilla_mean_score"],
        width=width,
        color=vanilla_color,
        edgecolor="white",
        linewidth=0.35,
        label="Vanilla",
    )
    ax.bar(
        x + width / 2,
        data["augment_mean_score"],
        width=width,
        color=augment_color,
        edgecolor="white",
        linewidth=0.35,
        label="Augmented",
    )

    for idx, row in data.iterrows():
        y = max(row["vanilla_mean_score"], row["augment_mean_score"]) + 0.0022
        ax.text(
            idx,
            y,
            f"{row['pct_delta_mean_score']:+.1f}%",
            ha="center",
            va="bottom",
            fontsize=6.6,
            color="#111111",
        )
        ax.text(
            idx,
            -0.0125,
            f"{row['frequency'] * 100:.1f}%",
            ha="center",
            va="top",
            fontsize=6.2,
            color="#555555",
            clip_on=False,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(data["kmer"])
    ax.set_ylabel("Mean 3-mer contribution")
    ax.set_xlabel("Most frequent 3-mers in generated negatives")
    ax.set_title("Contribution shift of frequent negative-sample 3-mers", loc="left", fontweight="bold", pad=4)
    ax.set_ylim(0, max(data["vanilla_mean_score"].max(), data["augment_mean_score"].max()) + 0.013)
    ax.grid(axis="y", color="#d9d9d9", linewidth=0.45, alpha=0.65)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(frameon=False, loc="upper right", ncol=1, handlelength=1.0)
    ax.text(
        0.0,
        -0.34,
        "Small gray labels show frequency among all sliding 3-mers.",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=6.2,
        color="#555555",
    )

    fig.tight_layout(pad=0.55)
    out_prefix = args.output_dir / args.output_prefix
    fig.savefig(f"{out_prefix}.pdf", bbox_inches="tight")
    fig.savefig(f"{out_prefix}.svg", bbox_inches="tight")
    fig.savefig(f"{out_prefix}.png", bbox_inches="tight", dpi=600)

    print(f"Wrote {out_prefix}.pdf")
    print(f"Wrote {out_prefix}.svg")
    print(f"Wrote {out_prefix}.png")
    print(f"Wrote {data_path}")


if __name__ == "__main__":
    main()
