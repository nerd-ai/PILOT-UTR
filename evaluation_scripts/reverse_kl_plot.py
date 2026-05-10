import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
OUT_PDF = SCRIPT_DIR / "reverse_kl_plot.pdf"
OUT_PNG = SCRIPT_DIR / "reverse_kl_plot.png"


def configure_style():
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


def format_axis(ax):
    ax.grid(axis="both", linestyle="--", color="#D9D9D9", linewidth=0.55, alpha=0.9)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color("#555555")
        spine.set_linewidth(0.8)
    ax.tick_params(axis="both", colors="#222222", width=0.8, length=3.0, pad=2.0)
    ax.set_axisbelow(True)


def gaussian_pdf(x, mu, sigma):
    return (1.0 / (np.sqrt(2 * np.pi) * sigma)) * np.exp(
        -0.5 * ((x - mu) / sigma) ** 2
    )


configure_style()

x = np.linspace(-5, 5, 1000)

# Bimodal target distribution
p_target = (
    0.5 * gaussian_pdf(x, -2.0, 0.55)
    + 0.5 * gaussian_pdf(x, 2.0, 0.55)
)

# Reverse-KL solution: mode-seeking approximation
q_reverse = gaussian_pdf(x, -2.0, 0.50)

# Normalize
p_target /= np.trapezoid(p_target, x)
q_reverse /= np.trapezoid(q_reverse, x)

fig, ax = plt.subplots(figsize=(2.85, 2.1))

ax.plot(
    x,
    p_target,
    linestyle="--",
    linewidth=1.65,
    color="black",
    label=r"Target",
)

ax.plot(
    x,
    q_reverse,
    linewidth=1.65,
    color="#D95F02",
    label=r"Reverse KL",
)

ax.fill_between(x, q_reverse, color="#D95F02", alpha=0.15)

ax.set_xlabel(r"$x$")
ax.set_ylabel("Density")

ax.set_xlim(-5, 5)
ax.set_ylim(0, 0.85)

format_axis(ax)
legend = ax.legend(
    loc="upper right",
    bbox_to_anchor=(0.98, 0.98),
    frameon=True,
    fancybox=False,
    framealpha=0.95,
    handlelength=1.8,
    borderaxespad=0.2,
    borderpad=0.35,
    labelspacing=0.25,
)
legend.get_frame().set_edgecolor("#BDBDBD")
legend.get_frame().set_linewidth(0.8)

fig.subplots_adjust(left=0.19, right=0.97, bottom=0.22, top=0.96)
fig.savefig(OUT_PNG, dpi=320, bbox_inches="tight", facecolor="white")
fig.savefig(OUT_PDF, bbox_inches="tight", facecolor="white")
plt.close(fig)

print(f"Saved PNG figure to: {OUT_PNG}")
print(f"Saved PDF figure to: {OUT_PDF}")
