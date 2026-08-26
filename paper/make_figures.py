"""Publication-quality figures for the paper PDF."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "results" / "results_paper.json"
OUT = ROOT / "paper" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

METHODS = ["random", "similarity", "complementary", "comp+rep", "claude"]
LABELS = {
    "random": "Random",
    "similarity": "Similarity",
    "complementary": "Complementary",
    "comp+rep": "Comp + probes",
    "claude": "Claude Sonnet 5",
}
COLORS = {
    "random": "#6b7280",
    "similarity": "#b91c1c",
    "complementary": "#1d4ed8",
    "comp+rep": "#047857",
    "claude": "#6d28d9",
}
MARKERS = {
    "random": "o",
    "similarity": "s",
    "complementary": "D",
    "comp+rep": "^",
    "claude": "v",
}


def load():
    return json.loads(DATA.read_text())


def style():
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 11,
            "axes.titlesize": 13,
            "axes.labelsize": 11,
            "legend.fontsize": 9,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.dpi": 200,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "axes.grid": True,
            "grid.alpha": 0.25,
            "grid.linestyle": "--",
        }
    )


def plot_success_vs_k(payload: dict) -> None:
    blocks = [payload["honest"], payload["spam"], payload["churn"]]
    titles = ["A. Honest ads", "B. Capability spam", "C. Churn + spam"]
    fig, axes = plt.subplots(1, 3, figsize=(11.2, 3.6), sharey=True)

    for ax, block, title in zip(axes, blocks, titles):
        ks = sorted(int(k) for k in block["by_k"])
        for m in METHODS:
            ys = [block["by_k"][str(k)][m]["success_rate"] for k in ks]
            ax.plot(
                ks,
                ys,
                marker=MARKERS[m],
                color=COLORS[m],
                label=LABELS[m],
                linewidth=2.2,
                markersize=7,
            )
        ax.set_title(title, fontweight="bold", pad=8)
        ax.set_xlabel("Team size k")
        ax.set_xticks(ks)
        ax.set_ylim(-0.03, 1.08)
        ax.set_xlim(min(ks) - 0.4, max(ks) + 0.4)

    axes[0].set_ylabel("Cover success rate")
    axes[2].legend(loc="lower right", frameon=True, fancybox=False, edgecolor="#ddd")
    fig.suptitle("Team cover success under open registry regimes", fontweight="bold", y=1.05)
    fig.tight_layout()
    fig.savefig(OUT / "success_vs_k.pdf")
    fig.savefig(OUT / "success_vs_k.png")
    plt.close(fig)


def plot_honest_vs_spam(payload: dict, k: int = 5) -> None:
    honest = payload["honest"]["by_k"][str(k)]
    spam = payload["spam"]["by_k"][str(k)]
    x = np.arange(len(METHODS))
    h = [honest[m]["success_rate"] for m in METHODS]
    s = [spam[m]["success_rate"] for m in METHODS]
    w = 0.36

    fig, ax = plt.subplots(figsize=(8.2, 4.0))
    ax.bar(x - w / 2, h, w, label="Honest ads", color="#1d4ed8", edgecolor="white")
    ax.bar(x + w / 2, s, w, label="Capability spam", color="#b91c1c", edgecolor="white")
    ax.set_xticks(x)
    ax.set_xticklabels([LABELS[m] for m in METHODS], rotation=12, ha="right")
    ax.set_ylabel("Cover success rate")
    ax.set_ylim(0, 1.12)
    ax.set_title(f"Honest ads vs capability spam (k = {k})", fontweight="bold", pad=10)
    ax.legend(frameon=True, fancybox=False, edgecolor="#ddd")
    for i, (hv, sv) in enumerate(zip(h, s)):
        ax.text(i - w / 2, hv + 0.02, f"{hv:.2f}", ha="center", va="bottom", fontsize=8, color="#1d4ed8")
        ax.text(i + w / 2, sv + 0.02, f"{sv:.2f}", ha="center", va="bottom", fontsize=8, color="#b91c1c")
    fig.tight_layout()
    fig.savefig(OUT / "honest_vs_spam.pdf")
    fig.savefig(OUT / "honest_vs_spam.png")
    plt.close(fig)


def plot_method_diagram() -> None:
    """Simple pipeline diagram for the paper."""
    fig, ax = plt.subplots(figsize=(10.5, 2.8))
    ax.set_xlim(0, 10.5)
    ax.set_ylim(0, 3)
    ax.axis("off")

    boxes = [
        (0.3, 1.0, 2.0, 1.2, "Open agent\nregistry", "#e0e7ff", "#3730a3"),
        (3.0, 1.0, 2.0, 1.2, "Task needs\nmulti skill cover", "#ffedd5", "#9a3412"),
        (5.7, 1.0, 2.0, 1.2, "Team selection\nmethod", "#dcfce7", "#166534"),
        (8.4, 1.0, 1.8, 1.2, "True skill\ncover check", "#fce7f3", "#9d174d"),
    ]
    for x, y, w, h, text, face, edge in boxes:
        rect = plt.Rectangle((x, y), w, h, facecolor=face, edgecolor=edge, linewidth=2, zorder=2)
        ax.add_patch(rect)
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=10, fontweight="bold", color=edge, zorder=3)

    for x0 in [2.3, 5.0, 7.7]:
        ax.annotate("", xy=(x0 + 0.65, 1.6), xytext=(x0, 1.6),
                    arrowprops=dict(arrowstyle="->", color="#374151", lw=1.8), zorder=1)

    ax.text(5.25, 2.55, "AgenticWeb Micro evaluation loop", ha="center", fontsize=13, fontweight="bold")
    ax.text(
        5.25,
        0.35,
        "Methods: Similarity  |  Complementary  |  Claude Sonnet 5  |  Comp + probes",
        ha="center",
        fontsize=9,
        color="#374151",
    )
    fig.tight_layout()
    fig.savefig(OUT / "pipeline.pdf")
    fig.savefig(OUT / "pipeline.png")
    plt.close(fig)


def plot_claim_diagram() -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 3.4))

    # Left: complementarity cartoon
    ax = axes[0]
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6)
    ax.axis("off")
    ax.set_title("Complementarity gap", fontweight="bold")
    ax.text(5, 5.4, "Task needs skills A + B + C", ha="center", fontsize=10)
    # specialists
    for i, (lab, col) in enumerate([("A only", "#1d4ed8"), ("B only", "#047857"), ("C only", "#6d28d9")]):
        ax.add_patch(plt.Circle((2.2, 3.6 - i * 1.1), 0.45, color=col, alpha=0.85))
        ax.text(3.0, 3.6 - i * 1.1, f"Specialist: {lab}", va="center", fontsize=9)
    ax.text(7.2, 4.2, "Near duplicate\nlooks like A+B+C\nbut misses one", ha="center", va="center",
            fontsize=9, bbox=dict(boxstyle="round,pad=0.4", facecolor="#fee2e2", edgecolor="#b91c1c"))
    ax.text(5, 0.45, "Similarity ranks lookalikes first.\nCover needs the specialists.",
            ha="center", fontsize=9, color="#111827")

    # Right: spam cartoon
    ax = axes[1]
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6)
    ax.axis("off")
    ax.set_title("Capability spam", fontweight="bold")
    ax.text(5, 5.4, "Ads say everything. Truth is narrow.", ha="center", fontsize=10)
    ax.add_patch(plt.Rectangle((1.2, 2.2), 3.2, 2.4, facecolor="#fee2e2", edgecolor="#b91c1c", lw=2))
    ax.text(2.8, 3.4, "Spam card\nclaims A to Z\ntrue: only A", ha="center", va="center", fontsize=9, color="#991b1b")
    ax.add_patch(plt.Rectangle((5.8, 2.2), 3.2, 2.4, facecolor="#dcfce7", edgecolor="#047857", lw=2))
    ax.text(7.4, 3.4, "Honest card\nclaims A\ntrue: A", ha="center", va="center", fontsize=9, color="#065f46")
    ax.text(5, 0.45, "Blind cover trusts spam.\nProbes + reputation recover.",
            ha="center", fontsize=9, color="#111827")

    fig.tight_layout()
    fig.savefig(OUT / "concepts.pdf")
    fig.savefig(OUT / "concepts.png")
    plt.close(fig)


if __name__ == "__main__":
    style()
    payload = load()
    plot_success_vs_k(payload)
    plot_honest_vs_spam(payload, k=5)
    plot_method_diagram()
    plot_claim_diagram()
    print(f"Wrote figures to {OUT}")
    for p in sorted(OUT.glob("*")):
        print(" ", p.name)
