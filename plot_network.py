"""
Draw the Bayesian Epistemic Routing network (v2) in the same style as the
original deck diagram. The layout is read from MODULES in
bayesian_epistemic_routing.py, so the picture stays in sync with the code.

Run:  python plot_network.py      -> writes bayesian_network_v2.png
"""

import warnings

warnings.filterwarnings("ignore")

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Circle, Polygon

from bayesian_epistemic_routing import MODULES, model

# Variables that were introduced in v2 (highlighted in the figure)
NEW_VARS = {"RerankerScore", "QueryReformulation", "UnresolvedReferences",
            "EntityPopularity", "TemporalSensitivity", "DomainShift"}

MODULE_TITLES = {
    "QueryAmbiguity": "Module 1 · Query Profiling",
    "ContextDeficit": "Module 2 · Retrieval Diagnostics",
    "IntentUncertainty": "Module 3 · User Context",
    "KnowledgeGap": "Module 4 · Knowledge Coverage",
}

DARK, BLUE, RED, GREEN, NEW = "#2b3340", "#2a6fbb", "#9b2a2a", "#2e6b45", "#e69b22"
LINE = "#1a1a1a"


def pretty(name):
    """'FAISSVectorDistance' -> 'FAISS Vector Distance'."""
    out = ""
    for i, ch in enumerate(name):
        if i and ch.isupper() and (not name[i - 1].isupper()
                                   or (i + 1 < len(name) and name[i + 1].islower())):
            out += " "
        out += ch
    return out


def box(ax, x, y, w, h, text, face, edge=None, lw=0, fs=10.5):
    ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h,
                                boxstyle="square,pad=0", fc=face,
                                ec=edge or face, lw=lw, zorder=3))
    ax.text(x, y, text, color="white", ha="center", va="center",
            fontsize=fs, zorder=4, family="DejaVu Sans")


def elbow(ax, x0, y0, x1, y1, xm):
    """Right-angle connector with an arrow head, like the original figure."""
    ax.plot([x0, xm, xm], [y0, y0, y1], color=LINE, lw=1.2, zorder=1)
    ax.annotate("", xy=(x1, y1), xytext=(xm, y1),
                arrowprops=dict(arrowstyle="-|>", color=LINE, lw=1.2,
                                mutation_scale=12), zorder=2)


def draw(path="bayesian_network_v2.png"):
    fig, ax = plt.subplots(figsize=(15, 16))
    ax.set_xlim(0, 15)
    ax.axis("off")

    in_x, in_w, h = 2.6, 3.4, 0.62
    bn_x, bn_w = 6.6, 2.7
    risk_x, risk_y, risk_r = 10.0, 0, 1.15
    route_x = 13.1

    y = 0.0
    bottleneck_pos = {}
    for bottleneck, inputs in MODULES.items():
        names = list(inputs)
        ys = [y - i * 1.0 for i in range(len(names))]
        ax.text(in_x - in_w / 2, ys[0] + 0.62, MODULE_TITLES[bottleneck],
                fontsize=11, color="#555", ha="left", va="bottom", style="italic")
        by = sum(ys) / len(ys)
        bottleneck_pos[bottleneck] = by
        for name, ny in zip(names, ys):
            new = name in NEW_VARS
            box(ax, in_x, ny, in_w, h, pretty(name), DARK,
                edge=NEW if new else None, lw=3 if new else 0)
            # spread arrow heads slightly so they don't overlap
            offset = (ny - by) * 0.06
            elbow(ax, in_x + in_w / 2, ny, bn_x - bn_w / 2, by + offset, bn_x - bn_w / 2 - 0.45)
        box(ax, bn_x, by, bn_w, 0.7, pretty(bottleneck), BLUE, fs=11.5)
        y = ys[-1] - 1.7

    # Hallucination Risk circle, vertically centred
    risk_y = (max(bottleneck_pos.values()) + min(bottleneck_pos.values())) / 2
    ax.add_patch(Circle((risk_x, risk_y), risk_r, fc=RED, ec="#e06060", lw=2, zorder=3))
    ax.text(risk_x, risk_y, "Hallucination\nRisk", color="white", ha="center",
            va="center", fontsize=13, zorder=4)

    for i, (b, by) in enumerate(bottleneck_pos.items()):
        target_y = risk_y + (1.5 - i) * 0.35
        dx = (risk_r ** 2 - (target_y - risk_y) ** 2) ** 0.5
        elbow(ax, bn_x + bn_w / 2, by, risk_x - dx, target_y, risk_x - risk_r - 0.6 - i * 0.12)

    # Optimal Route diamond
    d = 1.55
    ax.add_patch(Polygon([(route_x - d, risk_y), (route_x, risk_y + d),
                          (route_x + d, risk_y), (route_x, risk_y - d)],
                         closed=True, fc=GREEN, ec="#3fa066", lw=2, zorder=3))
    ax.text(route_x, risk_y, "Optimal Route\nServe / Search /\nClarify", color="white",
            ha="center", va="center", fontsize=11.5, zorder=4)
    ax.annotate("", xy=(route_x - d, risk_y), xytext=(risk_x + risk_r, risk_y),
                arrowprops=dict(arrowstyle="-|>", color=LINE, lw=1.4, mutation_scale=14))

    # Thresholds under the diamond
    ax.text(route_x, risk_y - d - 0.35,
            "P(H) < 0.2  →  Serve\n0.2 ≤ P(H) ≤ 0.6  →  Search\nP(H) > 0.6  →  Clarify",
            ha="center", va="top", fontsize=10.5, color="#333", linespacing=1.6)

    # Title + legend
    top = 1.9
    ax.text(0.9, top, "Bayesian Epistemic Routing — v2 network", fontsize=20,
            weight="bold", ha="left")
    ax.text(0.9, top - 0.6,
            f"{len(model.nodes())} nodes · {len(model.edges())} edges · "
            "all inputs observable before generation", fontsize=12, color="#555")
    ax.add_patch(FancyBboxPatch((9.6, top - 0.75), 0.45, 0.3, boxstyle="square,pad=0",
                                fc=DARK, ec=NEW, lw=3))
    ax.text(10.2, top - 0.6, "new in v2", fontsize=11, va="center")
    ax.text(11.6, top - 0.6, "“High” = trouble on every node", fontsize=11,
            va="center", color="#555")

    ax.set_ylim(y + 0.9, top + 0.8)
    fig.savefig(path, dpi=160, bbox_inches="tight", facecolor="white")
    print(f"Saved {path}")


if __name__ == "__main__":
    draw()
