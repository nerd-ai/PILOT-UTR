import csv
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from matplotlib.patches import Rectangle


ROOT = Path("/home/xli263/xli/utr_design/DRAKES/drakes_rna/interpretable")
VANILLA_CSV = ROOT / "attention_visualizations" / "all_3mer_average_contribution.csv"
AUGMENT_CSV = ROOT / "attention_visualizations_augment" / "all_3mer_average_contribution.csv"
OUT_DIR = ROOT / "attention_visualizations_augment"
OUT_PREFIX = OUT_DIR / "paper_3mer_at_contribution_comparison"
STATS_CSV = OUT_DIR / "paper_3mer_at_contribution_stats.csv"
HEATMAP_PREFIX = OUT_DIR / "paper_3mer_contribution_change_heatmap"

PARULA_COLORS = [
    "#352a87", "#0f5cdd", "#1485d4", "#06a4ca", "#20b7ad",
    "#49c16d", "#82c95a", "#b8cf4e", "#ebd84a", "#f9e95f",
]


def load_table(path):
    with path.open(newline="") as handle:
        return {row["kmer"]: row for row in csv.DictReader(handle)}


def as_float(row, key):
    return float(row[key])


def at_count(kmer):
    return sum(base in "AT" for base in kmer)


def g_count(kmer):
    return kmer.count("G")


def mean(vals):
    return sum(vals) / len(vals) if vals else 0.0


def build_rows():
    vanilla = load_table(VANILLA_CSV)
    augment = load_table(AUGMENT_CSV)
    rows = []
    for kmer in sorted(vanilla):
        if kmer not in augment:
            continue
        v = vanilla[kmer]
        a = augment[kmer]
        row = {
            "kmer": kmer,
            "at_count": at_count(kmer),
            "g_count": g_count(kmer),
            "vanilla_mean_score": as_float(v, "mean_score"),
            "augment_mean_score": as_float(a, "mean_score"),
            "vanilla_mean_attention_score": as_float(v, "mean_attention_score"),
            "augment_mean_attention_score": as_float(a, "mean_attention_score"),
        }
        row["delta_mean_score"] = row["augment_mean_score"] - row["vanilla_mean_score"]
        row["pct_delta_mean_score"] = 100.0 * row["delta_mean_score"] / abs(row["vanilla_mean_score"])
        row["delta_mean_attention_score"] = (
            row["augment_mean_attention_score"] - row["vanilla_mean_attention_score"]
        )
        row["pct_delta_mean_attention_score"] = (
            100.0 * row["delta_mean_attention_score"] / abs(row["vanilla_mean_attention_score"])
        )
        rows.append(row)
    return rows


def write_stats(rows, high_vanilla_at_rich):
    fieldnames = [
        "group",
        "n",
        "mean_vanilla_mean_score",
        "mean_augment_mean_score",
        "mean_delta_mean_score",
        "mean_pct_delta_mean_score",
        "weakened_mean_score",
        "mean_vanilla_mean_attention_score",
        "mean_augment_mean_attention_score",
        "mean_delta_mean_attention_score",
        "mean_pct_delta_mean_attention_score",
        "weakened_mean_attention_score",
    ]
    groups = [
        ("no_G", [r for r in rows if r["g_count"] == 0]),
        ("contains_G", [r for r in rows if r["g_count"] > 0]),
        ("AT_rich_no_G", [r for r in rows if r["at_count"] >= 2 and r["g_count"] == 0]),
        ("AT_rich_contains_G", [r for r in rows if r["at_count"] >= 2 and r["g_count"] > 0]),
        ("AT_rich_high_vanilla_top_quartile", high_vanilla_at_rich),
    ]
    with STATS_CSV.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for name, group in groups:
            writer.writerow(
                {
                    "group": name,
                    "n": len(group),
                    "mean_vanilla_mean_score": mean([r["vanilla_mean_score"] for r in group]),
                    "mean_augment_mean_score": mean([r["augment_mean_score"] for r in group]),
                    "mean_delta_mean_score": mean([r["delta_mean_score"] for r in group]),
                    "mean_pct_delta_mean_score": mean([r["pct_delta_mean_score"] for r in group]),
                    "weakened_mean_score": sum(r["delta_mean_score"] < 0 for r in group),
                    "mean_vanilla_mean_attention_score": mean(
                        [r["vanilla_mean_attention_score"] for r in group]
                    ),
                    "mean_augment_mean_attention_score": mean(
                        [r["augment_mean_attention_score"] for r in group]
                    ),
                    "mean_delta_mean_attention_score": mean(
                        [r["delta_mean_attention_score"] for r in group]
                    ),
                    "mean_pct_delta_mean_attention_score": mean(
                        [r["pct_delta_mean_attention_score"] for r in group]
                    ),
                    "weakened_mean_attention_score": sum(
                        r["delta_mean_attention_score"] < 0 for r in group
                    ),
                }
            )


def plot_heatmap(rows):
    row_order = [
        "AA",
        "AT",
        "TA",
        "TT",
        "AC",
        "CA",
        "CT",
        "TC",
        "CC",
        "AG",
        "GA",
        "GT",
        "TG",
        "CG",
        "GC",
        "GG",
    ]
    col_order = ["A", "T", "C", "G"]
    row_by_kmer = {row["kmer"]: row for row in rows}
    values = []
    labels = []
    for prefix in row_order:
        value_row = []
        label_row = []
        for suffix in col_order:
            kmer = prefix + suffix
            value_row.append(row_by_kmer[kmer]["pct_delta_mean_score"])
            label_row.append(kmer)
        values.append(value_row)
        labels.append(label_row)

    cmap = LinearSegmentedColormap.from_list("parula_like", PARULA_COLORS, N=256)
    norm = TwoSlopeNorm(vmin=-15.0, vcenter=0.0, vmax=15.0)

    fig, ax = plt.subplots(figsize=(4.2, 6.0))
    im = ax.imshow(values, cmap=cmap, norm=norm, aspect="auto")
    ax.set_xticks(range(len(col_order)))
    ax.set_xticklabels(col_order)
    ax.set_yticks(range(len(row_order)))
    ax.set_yticklabels(row_order)
    ax.set_xlabel("Third base")
    ax.set_ylabel("First two bases")
    ax.set_title("3-mer contribution change after augmentation", loc="left", fontweight="bold")

    for y, prefix in enumerate(row_order):
        for x, suffix in enumerate(col_order):
            kmer = labels[y][x]
            val = values[y][x]
            text_color = "white" if abs(val) >= 9.0 else "#111111"
            ax.text(
                x,
                y,
                f"{kmer}\n{val:+.1f}",
                ha="center",
                va="center",
                fontsize=6,
                color=text_color,
            )
            if "G" in kmer:
                ax.add_patch(
                    Rectangle(
                        (x - 0.48, y - 0.48),
                        0.96,
                        0.96,
                        fill=False,
                        edgecolor="#111111",
                        linewidth=0.45,
                    )
                )

    ax.set_xticks([i - 0.5 for i in range(1, len(col_order))], minor=True)
    ax.set_yticks([i - 0.5 for i in range(1, len(row_order))], minor=True)
    ax.grid(which="minor", color="white", linewidth=1.0)
    ax.tick_params(which="minor", bottom=False, left=False)
    for spine in ax.spines.values():
        spine.set_visible(False)

    cbar = fig.colorbar(im, ax=ax, fraction=0.045, pad=0.035)
    cbar.set_label("Augmented vs vanilla contribution change (%)")
    cbar.set_ticks([-15, -10, -5, 0, 5, 10, 15])
    cbar.ax.text(
        0.5,
        -0.07,
        "clipped at +/-15%",
        transform=cbar.ax.transAxes,
        ha="center",
        va="top",
        fontsize=6,
    )

    ax.text(
        0.0,
        -0.12,
        "Black outline marks 3-mers containing G.",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=7,
    )
    fig.savefig(f"{HEATMAP_PREFIX}.pdf", bbox_inches="tight")
    fig.savefig(f"{HEATMAP_PREFIX}.svg", bbox_inches="tight")
    fig.savefig(f"{HEATMAP_PREFIX}.png", bbox_inches="tight", dpi=300)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {
            "font.size": 7,
            "axes.titlesize": 8,
            "axes.labelsize": 7,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 7,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    rows = build_rows()
    at_rich = [r for r in rows if r["at_count"] >= 2]
    high_vanilla_at_rich = sorted(
        at_rich, key=lambda r: r["vanilla_mean_score"], reverse=True
    )[: max(1, len(at_rich) // 4)]
    high_vanilla_names = {r["kmer"] for r in high_vanilla_at_rich}
    write_stats(rows, high_vanilla_at_rich)

    g_color = "#2a7f62"
    no_g_color = "#c84b31"
    neutral_color = "#9a9a9a"

    fig = plt.figure(figsize=(7.2, 6.5))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 1.12], hspace=0.6, wspace=0.55)
    ax_scatter = fig.add_subplot(gs[0, 0])
    ax_group = fig.add_subplot(gs[0, 1])
    ax_at_rich = fig.add_subplot(gs[1, :])

    for contains_g, color, label in [
        (False, no_g_color, "No G"),
        (True, g_color, "Contains G"),
    ]:
        group = [r for r in rows if (r["g_count"] > 0) == contains_g]
        ax_scatter.scatter(
            [r["vanilla_mean_score"] for r in group],
            [r["augment_mean_score"] for r in group],
            s=34,
            color=color,
            edgecolor="white",
            linewidth=0.4,
            label=label,
            zorder=3,
        )
    lim_min = min(min(r["vanilla_mean_score"], r["augment_mean_score"]) for r in rows) - 0.005
    lim_max = max(max(r["vanilla_mean_score"], r["augment_mean_score"]) for r in rows) + 0.005
    ax_scatter.plot([lim_min, lim_max], [lim_min, lim_max], color="#333333", lw=1.0, ls="--")
    ax_scatter.set_xlim(lim_min, lim_max)
    ax_scatter.set_ylim(lim_min, lim_max)
    ax_scatter.set_xlabel("Vanilla 3-mer contribution")
    ax_scatter.set_ylabel("Augmented 3-mer contribution")
    ax_scatter.set_title("A. 3-mer contribution shift", loc="left", fontweight="bold")
    for r in sorted(rows, key=lambda x: x["delta_mean_score"], reverse=True)[:4]:
        ax_scatter.annotate(
            r["kmer"],
            (r["vanilla_mean_score"], r["augment_mean_score"]),
            textcoords="offset points",
            xytext=(3, 2),
            fontsize=6,
        )
    ax_scatter.legend(frameon=False, fontsize=8, loc="upper left")

    group_defs = [
        ("No G", [r for r in rows if r["g_count"] == 0], no_g_color),
        ("Contains G", [r for r in rows if r["g_count"] > 0], g_color),
    ]
    ax_group.axhline(0, color="#333333", lw=0.8)
    for xi, (label, group, color) in enumerate(group_defs):
        vals = [r["pct_delta_mean_score"] for r in group]
        offsets = [((i % 7) - 3) * 0.025 for i in range(len(vals))]
        ax_group.scatter(
            [xi + off for off in offsets],
            vals,
            color=color,
            alpha=0.55,
            s=18,
            edgecolor="white",
            linewidth=0.25,
            zorder=2,
        )
        ax_group.plot(
            [xi - 0.22, xi + 0.22],
            [mean(vals), mean(vals)],
            color="#111111",
            lw=2.4,
            zorder=4,
        )
        ax_group.text(
            xi,
            mean(vals) + (1.1 if mean(vals) >= 0 else -1.6),
            f"{mean(vals):+.1f}%",
            ha="center",
            va="center",
            fontsize=9,
            fontweight="bold",
        )
    ax_group.set_xticks([0, 1])
    ax_group.set_xticklabels([g[0] for g in group_defs])
    ax_group.set_ylabel("Contribution change (%)")
    ax_group.set_title("B. Reweighting by G content", loc="left", fontweight="bold")
    ax_group.text(
        0.03,
        0.05,
        "No G: 19/27 decreased\nContains G: 27/37 increased",
        transform=ax_group.transAxes,
        fontsize=7,
    )

    at_no_g = sorted(
        [r for r in rows if r["at_count"] >= 2 and r["g_count"] == 0],
        key=lambda r: r["pct_delta_mean_score"],
    )
    at_with_g = sorted(
        [r for r in rows if r["at_count"] >= 2 and r["g_count"] > 0],
        key=lambda r: r["pct_delta_mean_score"],
    )
    focus = at_no_g[:8] + at_with_g[-8:]
    focus = sorted(focus, key=lambda r: r["pct_delta_mean_score"])
    ax_at_rich.axvline(0, color="#333333", lw=0.8)
    ax_at_rich.barh(
        [r["kmer"] for r in focus],
        [r["pct_delta_mean_score"] for r in focus],
        color=[g_color if r["g_count"] > 0 else no_g_color for r in focus],
        height=0.68,
    )
    ax_at_rich.set_xlabel("Augmented vs vanilla contribution change (%)")
    ax_at_rich.set_title(
        "C. A/T-rich motifs split by G content",
        loc="left",
        fontweight="bold",
    )
    for ax in [ax_scatter, ax_group, ax_at_rich]:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    legend_handles = [
        Line2D([0], [0], color=g_color, lw=6, label="Contains G"),
        Line2D([0], [0], color=no_g_color, lw=6, label="No G"),
    ]
    ax_at_rich.legend(handles=legend_handles, frameon=False, fontsize=8, loc="lower right")

    fig.savefig(f"{OUT_PREFIX}.pdf", bbox_inches="tight")
    fig.savefig(f"{OUT_PREFIX}.svg", bbox_inches="tight")
    fig.savefig(f"{OUT_PREFIX}.png", bbox_inches="tight", dpi=300)
    plot_heatmap(rows)

    print(f"Wrote {OUT_PREFIX}.pdf")
    print(f"Wrote {OUT_PREFIX}.svg")
    print(f"Wrote {OUT_PREFIX}.png")
    print(f"Wrote {HEATMAP_PREFIX}.pdf")
    print(f"Wrote {HEATMAP_PREFIX}.svg")
    print(f"Wrote {HEATMAP_PREFIX}.png")
    print(f"Wrote {STATS_CSV}")


if __name__ == "__main__":
    main()
