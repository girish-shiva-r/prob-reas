"""Writes NODE_SPEC_v3.md from ber_v3.py (so the spec can never drift from the code)."""
import ber_v3 as B

L = ["# Bayesian Epistemic Routing v3: node specification", "",
     "Generated from `ber_v3.py` by `make_spec.py`. All numbers are expert priors, to be learned.", ""]
L += ["## Causes (9, hidden, Low/High)", "", "| Cause | Parents | Prior / noisy-OR | Fixed by | Training label |", "|---|---|---|---|---|"]
for c, s in B.CAUSES.items():
    pri = f"P(High) = {s['prior']}" if "prior" in s else "leak " + str(s["noisy_or"]["leak"]) + ", " + \
        ", ".join(f"{p} {v}" for p, v in s["noisy_or"].items() if p != "leak")
    L.append(f"| **{c}**: {s['desc']} | {', '.join(B.cause_parents(c)) or '-'} | {pri} | {s['fix']} | {s['weak_label']} |")
L += ["", "## Symptoms (22, observed)", "", "| Tier | Symptom | States | Parents | How it is measured |", "|---|---|---|---|---|"]
for n, s in B.SYMPTOMS.items():
    L.append(f"| {s['tier']} | **{n}** | {' / '.join(s['states'])} | {', '.join(s['parents'])} | {s['measure']} |")
L += ["", "Tier cost (utility units): " + ", ".join(f"tier {k}: {v}" for k, v in B.TIER_COST.items()), ""]
L += ["## Context (3, observed)", "", "| Node | States | Meaning |", "|---|---|---|"]
for n, s in B.CONTEXT.items():
    L.append(f"| **{n}** | {' / '.join(s['states'])} | {s['desc']} |")
L += ["", "## Outcome (answer model)", "", f"States {B.OUTCOME_STATES}; parents {', '.join(B.OUTCOME['parents'])}.",
      f"Multinomial logit, base (Abstain, Wrong) = {B.OUTCOME['base']}; effect of each parent being High (dAbstain, dWrong):", ""]
L += [f"- {p}: {v}" for p, v in B.OUTCOME["effects"].items()]
L += [f"- interaction {' & '.join(ps)}: {v}" for ps, v in B.OUTCOME["interactions"]]
E = B.HANDSET_EFFECTS
L += ["", "## Decision (6 actions) as interventions", "",
      "Every action except Serve/Abstain clears some question-side causes, then searches the web again, so the "
      "3 evidence-side causes are drawn anew from the web tables below. The same Outcome table scores the result.", "",
      "| Action | Cost | Clears (P success) |", "|---|---|---|"]
for a in B.ACTIONS:
    if a in ("Serve", "Abstain"):
        what = "nothing (answer now)" if a == "Serve" else "- (outcome = Abstain)"
    else:
        what = ", ".join(f"{k.replace('@typed', ' (only if the relation implies a type)')} {v}" for k, v in E[a].items())
    L.append(f"| **{a}** | {B.ACTION_COST[a]} | {what} |")
L += ["", f"Clarify success is scaled by user fatigue {E['clarify_fatigue']}; each earlier clarifying question costs "
      f"{B.CLARIFY_EXTRA_COST} more.", "", "Web retrieval (noisy-OR; leak by SearchRounds):", ""]
for c, spec in E["web"].items():
    L.append(f"- {c}: leak {spec['leak']}, " + ", ".join(f"{p} {v}" for p, v in spec.items() if p != "leak"))
L += ["", "## Utility (policy, not learned)", "",
      f"Correct +1, Abstain 0, Wrong -lambda; default lambda = {B.LAMBDA}. Serve beats Abstain when "
      "P(Wrong)/P(Correct) < 1/lambda, so lambda is a risk level: `learn_v3.choose_lambda` picks it from a target "
      f"share of wrong answers among served ones on validation data. Hedged answer after generation: {B.HEDGE_UTILITY}.",
      "", "## How the numbers are learned (learn_v3.py)", "",
      "1. Cut-offs of real-valued symptoms: information gain on the training split only.",
      "2. Weak cause labels make causes partly observed (each label wrong with probability "
      f"{__import__('learn_v3').LABEL_NOISE}).",
      "3. The tables above are only a Dirichlet prior; its strength alpha is chosen by grouped cross-validation "
      "(alpha = 0 ignores them).",
      "4. EM from the hand-set start and random starts; the spread between starts is reported.",
      "5. Action effects: MAP fit on questions where that action was actually run.",
      "6. recovery_test.py checks all of this on data from a known 'true' network.", ""]
open("NODE_SPEC_v3.md", "w").write("\n".join(L))
print("NODE_SPEC_v3.md written")
