"""Draws the v3 network: python plot_v3.py  ->  network_v3.png"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

import ber_v3 as B

POS = {
    # tier-0 symptoms
    "NameCollision": (0.2, 5.4), "Disambiguator": (1.9, 5.4), "TemporalAnchoring": (3.9, 5.4), "TimeWords": (5.6, 5.4),
    "ResidualReferences": (7.2, 5.4), "FollowUpForm": (8.9, 5.4), "MultiHopForm": (10.6, 5.4),
    # root causes
    "ReferentAmbiguity": (1.0, 4.2), "ConstraintAmbiguity": (3.4, 4.2), "TemporalMismatch": (5.6, 4.2),
    "ContextDependence": (7.8, 4.2), "ReasoningLoad": (10.2, 4.2), "ParametricGap": (12.4, 4.2),
    # evidence causes
    "ContextMismatch": (2.4, 3.0), "EvidenceConflict": (5.2, 3.0), "EvidenceGap": (8.4, 3.0),
    # tier-1 symptoms
    "RerankMargin": (0.0, 1.8), "TitleDiversity": (1.4, 1.8), "SubjectTypeMatch": (2.8, 1.8), "RerankerLogit": (4.2, 1.8),
    "CandidateAgreement": (5.6, 1.8), "EvidenceAge": (6.9, 1.8), "EntityCoverage": (8.1, 1.8),
    "AnswerTypePresence": (9.5, 1.8), "Answerability": (10.9, 1.8), "SupportSpread": (12.2, 1.8),
    # tier-2/3 symptoms
    "ParametricAgreement": (6.6, 0.6), "GroundedConfidence": (9.4, 0.6), "EntityFamiliarity": (11.6, 0.6),
    "ClosedBookConfidence": (13.2, 0.6),
    # decision layer
    "Outcome": (14.2, 3.0), "AnswerGrounding": (14.6, 1.8),
    "HistoryCandidates": (16.6, 5.5), "ClarifyRounds": (16.6, 4.75), "SearchRounds": (16.6, 4.0),
    "Decision": (16.6, 3.0), "Utility": (16.6, 1.8),
}
COLOR = {"cause": "#f4c7a1", "evidence_cause": "#f2a97e", "t0": "#cfe3f7", "t1": "#a9cdee", "t2": "#86b8e3",
         "t3": "#6aa3d6", "t4": "#c9b6e4", "outcome": "#f7a1a1", "context": "#d9d9d9", "decision": "#9ad19a",
         "utility": "#ffe08a"}


def kind(n):
    if n in B.CAUSES:
        return "evidence_cause" if "noisy_or" in B.CAUSES[n] else "cause"
    if n in B.SYMPTOMS:
        return f"t{B.SYMPTOMS[n]['tier']}"
    if n in B.CONTEXT:
        return "context"
    return {"Outcome": "outcome", "Decision": "decision", "Utility": "utility"}[n]


def label(n):
    import re
    return re.sub(r"(?<=[a-z])(?=[A-Z])", "\n", n) if len(n) > 12 else n


edges = []
for c in B.CAUSE_NAMES:
    edges += [(p, c, "solid") for p in B.cause_parents(c)]
for s, spec in B.SYMPTOMS.items():
    edges += [(p, s, "solid") for p in spec["parents"]]
edges += [(p, "Outcome", "solid") for p in B.OUTCOME["parents"]]
edges += [("Outcome", "Utility", "solid"), ("Decision", "Utility", "solid")]
edges += [(k, "Decision", "solid") for k in B.CONTEXT]
fixed = {k.split("@")[0] for a in ("Rewrite", "Search", "Clarify", "Decompose") for k in B.HANDSET_EFFECTS[a]} \
    | set(B.EVIDENCE_CAUSES)
edges += [("Decision", c, "dashed") for c in sorted(fixed)]

fig, ax = plt.subplots(figsize=(22, 10.5))
for a, b, style in edges:
    (x1, y1), (x2, y2) = POS[a], POS[b]
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="-|>", lw=0.9 if style == "solid" else 0.8,
                                color="#555" if style == "solid" else "#2e8b57", alpha=0.55,
                                linestyle="-" if style == "solid" else (0, (4, 3)),
                                shrinkA=22, shrinkB=22, connectionstyle="arc3,rad=0.04"))
for n, (x, y) in POS.items():
    k = kind(n)
    w, h = 1.12, 0.58
    ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h, boxstyle="round,pad=0.03,rounding_size=0.12",
                                fc=COLOR[k], ec="#333", lw=0.9, zorder=3))
    ax.text(x, y, label(n), ha="center", va="center", fontsize=7.6, zorder=4,
            fontweight="bold" if k in ("cause", "evidence_cause", "outcome", "decision", "utility") else "normal")

for y, txt in [(5.4, "TIER 0\nquestion"), (4.2, "CAUSES\n(query side)"), (3.0, "CAUSES\n(evidence side)"),
               (1.8, "TIER 1\nretrieval"), (0.6, "TIER 2-3\nLLM prefill /\nshort gen.")]:
    ax.text(-1.25, y, txt, ha="center", va="center", fontsize=9, color="#333", fontweight="bold")
ax.text(14.6, 1.25, "tier 4: after\nthe answer", ha="center", fontsize=7.5, color="#555")
ax.text(16.6, 6.1, "CONTEXT", ha="center", fontsize=9, fontweight="bold")
ax.text(15.2, 3.75, "dashed = the Decision\nintervenes on (fixes) a cause", fontsize=7.5, color="#2e8b57")
ax.set_xlim(-2.0, 17.6)
ax.set_ylim(0.05, 6.3)
ax.axis("off")
n_chance = len(B.CAUSES) + len(B.SYMPTOMS) + len(B.CONTEXT) + 1
ax.set_title(f"Bayesian Epistemic Routing v3 (online): {n_chance} chance nodes + Decision + Utility = {n_chance + 2} nodes",
             fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig("network_v3.png", dpi=150)
print("saved network_v3.png")
