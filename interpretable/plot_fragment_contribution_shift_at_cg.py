from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path("/home/xli263/xli/utr_design/DRAKES/drakes_rna/interpretable/augment")
AT_CSV = ROOT / "top6_decreased_AT_only_3mers_after_augmentation.csv"
CG_CSV = ROOT / "top6_increased_CG_only_3mers_after_augmentation.csv"
OUT_PREFIX = ROOT / "paper_at_cg_3mer_contribution_shift"
OUT_DATA = ROOT / "paper_at_cg_3mer_contribution_shift_data.csv"


def rna_label(seq: str) -> str:
    return str(seq).replace("T", "U")


def main():
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 7,
            "axes.titlesize": 8,
            "axes.labelsize": 7.5,
            "xtick.labelsize": 6.8,
            "ytick.labelsize": 7.2,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "axes.linewidth": 0.7,
            "xtick.major.width": 0.7,
            "ytick.major.width": 0.7,
            "xtick.major.size": 2.6,
            "ytick.major.size": 0,
        }
    )

    at_df = pd.read_csv(AT_CSV).copy()
    cg_df = pd.read_csv(CG_CSV).copy()
    at_df["class"] = "A/T-only"
    cg_df["class"] = "C/G-only"
    at_df = at_df.sort_values("pct_delta_mean_score", ascending=True).head(5)
    cg_df = cg_df.sort_values("pct_delta_mean_score", ascending=False).head(5)
    combined = pd.concat([at_df, cg_df], ignore_index=True)
    combined.to_csv(OUT_DATA, index=False)

    plot_df = pd.concat([at_df, cg_df], ignore_index=True)
    colors = ["#c84b31" if cls == "A/T-only" else "#2a7f62" for cls in plot_df["class"]]

    fig, ax = plt.subplots(figsize=(3.35, 3.0), constrained_layout=True)
    y = list(range(len(plot_df)))
    ax.barh(
        y,
        plot_df["pct_delta_mean_score"],
        color=colors,
        alpha=0.9,
        height=0.62,
        edgecolor="white",
        linewidth=0.4,
        zorder=2,
    )

    ax.axvline(0, color="#333333", linewidth=0.8, zorder=1)
    ax.axhline(4.5, color="#bdbdbd", linewidth=0.55, zorder=1)
    ax.set_yticks(y)
    ax.set_yticklabels([rna_label(kmer) for kmer in plot_df["kmer"]])
    ax.invert_yaxis()
    ax.set_xlim(-12.6, 10.8)
    ax.set_xlabel("Contribution change after augmentation (%)")
    # Keep this panel title-free so it can be dropped into a multi-panel figure.
    ax.grid(axis="x", color="#d9d9d9", linewidth=0.45, alpha=0.65)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)

    for yi, (_, row) in enumerate(plot_df.iterrows()):
        val = row["pct_delta_mean_score"]
        if val < 0:
            x_text = 0.35
            ha = "left"
            text_color = "#111111"
        else:
            x_text = val + 0.35
            ha = "left"
            text_color = "#111111"
        bbox = (
            dict(boxstyle="round,pad=0.12", facecolor="white", edgecolor="none", alpha=0.78)
            if val < 0
            else None
        )
        ax.text(
            x_text,
            yi,
            f"{val:+.1f}%",
            va="center",
            ha=ha,
            fontsize=6.8,
            color=text_color,
            fontweight="bold" if abs(val) >= 5 else "normal",
            bbox=bbox,
        )

    ax.text(
        -12.45,
        -0.75,
        "A/U-only decreased",
        color="#9f3a25",
        fontsize=7,
        fontweight="bold",
        ha="left",
        va="center",
    )
    ax.text(
        -12.45,
        4.85,
        "C/G-only increased",
        color="#1f684f",
        fontsize=7,
        fontweight="bold",
        ha="left",
        va="center",
    )

    fig.savefig(f"{OUT_PREFIX}.pdf", bbox_inches="tight")
    fig.savefig(f"{OUT_PREFIX}.svg", bbox_inches="tight")
    fig.savefig(f"{OUT_PREFIX}.png", bbox_inches="tight", dpi=600)

    print(f"Wrote {OUT_PREFIX}.pdf")
    print(f"Wrote {OUT_PREFIX}.svg")
    print(f"Wrote {OUT_PREFIX}.png")
    print(f"Wrote {OUT_DATA}")


if __name__ == "__main__":
    main()
