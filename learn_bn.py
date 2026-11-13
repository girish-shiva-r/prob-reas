"""
Step 3 - Learn the Bayesian network's numbers from labelled data
-----------------------------------------------------------------
Input : labels/<llm>.jsonl  (one row per question: raw signals + graded LLM answer)
Output: profiles/<llm>.json (learned cut-offs + learned probability tables)

What is learned (the STRUCTURE stays the hand-designed one):
  1. Cut-offs. Each raw signal becomes a level, 0 = Low ... K-1 = High ("High" = trouble,
     same meaning as before). Most signals keep 2 levels (Low/High). The three strong
     retrieval signals get more levels (LEVELS below), because two levels throw away most
     of their information: a reranker score of 0.97 and one of 0.02 would both be "High".
     The direction is fixed by design (e.g. a LOW reranker score is trouble); only WHERE to
     cut is learned: the cuts that separate failures from successes best on the training
     data (largest information gain, added greedily one cut at a time). Signals that never
     vary keep the hand-set cut-off.
  2. Probability tables, by EM (expectation-maximisation):
       * the 14 inputs and the outcome are observed in every row,
       * the 4 bottlenecks (QueryAmbiguity, ContextDeficit, IntentUncertainty,
         KnowledgeGap) are never observed -> latent variables.
     E-step: for each question, the posterior over the 16 bottleneck combinations.
     M-step: re-estimate every table from those expected counts.
     The hand-set tables act as a PRIOR (Dirichlet pseudo-counts, `--prior-strength`
     questions' worth per table column): the data overrides them where it has evidence,
     and where it has none (e.g. PopQA has no follow-up questions) the expert values stay.
     This is MAP estimation; EM also starts from the hand-set values, so "High" keeps
     meaning "trouble" for every bottleneck.

Outcome node: HallucinationRisk is trained on "serving now would fail" = the answer was
wrong OR the model said "I don't know" (both mean the system could not answer from
what it had). Rows with empty replies are skipped.

    python learn_bn.py labels/Qwen_Qwen2.5-7B-Instruct-bfloat16.jsonl
"""

import argparse
import json
import math
import os
import re
from itertools import product

import numpy as np

import bayesian_epistemic_routing as ber

MODULE_NAMES = list(ber.MODULES)           # bottlenecks, in the network's order
INPUTS = [i for m in MODULE_NAMES for i in ber.MODULES[m]]
BOTTLENECKS = list(ber.CONVERGENCE_WEIGHTS)

# raw key, direction (+1: trouble when value is LARGE, -1: trouble when SMALL, None: yes/no rule)
SIGNALS = {
    "QueryLength":          ("word_count", -1),
    "QuestionType":         ("question_type", None),
    "EntitySpecificity":    ("named_entity_count", -1),
    "TopicComplexity":      ("sub_question_count", +1),
    "FAISSVectorDistance":  ("top1_faiss_distance", +1),
    "RerankerScore":        ("top1_rerank_score", -1),
    "ChunkDensity":         ("relevant_chunk_count", -1),
    "SourceAuthority":      ("mean_source_authority", -1),
    "FollowUpStatus":       ("is_ambiguous_followup", None),
    "QueryReformulation":   ("similarity_to_prev_query", +1),
    "UnresolvedReferences": ("unresolved_reference_count", +1),
    "EntityPopularity":     ("entity_monthly_views", -1),
    "TemporalSensitivity":  ("is_time_sensitive", None),
    "DomainShift":          ("kb_domain_similarity", -1),
}
# hand-set cut-offs, written in the same "value < cut / > cut" form
DEFAULT_CUTS = {"QueryLength": 4, "EntitySpecificity": 1, "TopicComplexity": 1.5, "FAISSVectorDistance": 0.8,
                "RerankerScore": 0.5, "ChunkDensity": 2, "SourceAuthority": 0.5, "QueryReformulation": 0.8,
                "UnresolvedReferences": 0.5, "EntityPopularity": 1000, "DomainShift": 0.3}
# number of levels per node in the LEARNED network (all others: 2 = Low/High)
# (chosen by 5-fold cross-validation among a few options; see compare_routers.py)
LEVELS = {"RerankerScore": 4, "FAISSVectorDistance": 3, "ChunkDensity": 3, "EntityPopularity": 3, "DomainShift": 3}
PRIOR_STRENGTH = 5.0       # the hand-set tables count as 5 questions' worth per table column


def state_names(k: int) -> list:
    return {2: ["Low", "High"], 3: ["Low", "Medium", "High"],
            4: ["Low", "Medium", "Elevated", "High"]}.get(k, [f"L{i}" for i in range(k)])


def cards(levels: dict) -> list:
    return [levels.get(n, 2) for n in INPUTS]


def failed(row) -> int:
    return int(not row["correct"])


# ----------------------------------------------------------------- cut-offs
def _entropy(p):
    return 0.0 if p <= 0 or p >= 1 else -(p * math.log2(p) + (1 - p) * math.log2(1 - p))


def info_gain(level, y):
    """Information gain of splitting y by the integer array `level`."""
    g = _entropy(y.mean())
    for l in np.unique(level):
        m = level == l
        g -= m.mean() * _entropy(y[m].mean())
    return g


def _level(vals, cuts, direction):
    return sum(((vals < c) if direction < 0 else (vals > c)).astype(int) for c in cuts) if cuts else \
        np.zeros(len(vals), dtype=int)


def learn_cutoffs(rows, levels=None, min_share=0.02):
    """Learned cut-offs per continuous signal (direction fixed by design), greedy one cut at a time."""
    levels = LEVELS if levels is None else levels
    y = np.array([failed(r) for r in rows])
    cuts = {}
    for node, (key, direction) in SIGNALS.items():
        if direction is None:
            continue
        k = levels.get(node, 2)
        vals = np.array([r["raw"].get(key) if isinstance(r["raw"].get(key), (int, float)) else np.nan
                         for r in rows], dtype=float).round(6)      # float noise (0.8000000000000002) is not a signal
        ok = ~np.isnan(vals)
        cuts[node] = {"cuts": [DEFAULT_CUTS[node]], "learned": False, "info_gain": 0.0}
        if ok.sum() < 50 or np.unique(vals[ok]).size < 2:
            continue
        v, yy = vals[ok], y[ok]
        cand = np.unique(np.quantile(v, np.linspace(0.02, 0.98, 49)))
        chosen, gain = [], 0.0
        for _ in range(k - 1):
            best = (gain, None)
            for q in cand:
                if q in chosen:
                    continue
                lv = _level(v, chosen + [q], direction)
                shares = np.bincount(lv, minlength=len(chosen) + 2) / len(lv)
                if shares.min() < min_share:
                    continue
                g = info_gain(lv, yy)
                if g > best[0] + 1e-3:
                    best = (g, float(q))
            if best[1] is None:
                break
            gain = best[0]
            chosen.append(best[1])
        if chosen:
            chosen = sorted(chosen, reverse=(direction < 0))        # cut 1 separates Low from the next level
            cuts[node] = {"cuts": chosen, "learned": True, "info_gain": round(gain, 4)}
    return cuts


def default_cuts():
    """The hand-set (binary) cut-offs, in the same format."""
    return {n: {"cuts": [c], "learned": False, "info_gain": 0.0} for n, c in DEFAULT_CUTS.items()}


def discretize(raw: dict, cuts: dict) -> dict:
    """raw measurements -> {node: level (0 = Low ... K-1 = High)}; unmeasurable signals are left out."""
    ev = {}
    for node, (key, direction) in SIGNALS.items():
        v = raw.get(key)
        if direction is None:
            if node == "QuestionType":
                if v is not None:
                    ev[node] = int(v in ber.OPEN_ENDED_TYPES)
            elif v is not None:
                ev[node] = int(bool(v))
            continue
        c = cuts.get(node, {}).get("cuts", [DEFAULT_CUTS[node]])
        if not isinstance(v, (int, float)):
            continue
        ev[node] = int(_level(np.array([float(v)]), c, direction)[0])
    return ev


def evidence_matrix(rows, cuts):
    """n x 14 matrix of levels; -1 = unknown."""
    X = np.full((len(rows), len(INPUTS)), -1, dtype=int)
    for r_i, r in enumerate(rows):
        ev = discretize(r["raw"], cuts)
        for c_i, node in enumerate(INPUTS):
            if node in ev:
                X[r_i, c_i] = ev[node]
    return X


# ----------------------------------------------------------------- the model as arrays
def configs(card_list):
    """All parent combinations in pgmpy column order (first parent varies slowest)."""
    return np.array(list(product(*[range(k) for k in card_list])), dtype=int)


def handset_params(levels=None):
    """The hand-designed network as arrays (prior + EM start). Extra levels share the 'High' weight evenly."""
    levels = {} if levels is None else levels
    clip = lambda p: min(max(p, 0.02), 0.98)
    pi = []
    for m in MODULE_NAMES:
        for n, (p_high, _, _) in ber.MODULES[m].items():
            k = levels.get(n, 2)
            pi.append(np.array([1 - p_high] + [p_high / (k - 1)] * (k - 1)))
    theta = []
    for m in MODULE_NAMES:
        names = list(ber.MODULES[m])
        w = [ber.MODULES[m][n][1] for n in names]
        ks = [levels.get(n, 2) for n in names]
        theta.append(np.array([clip(ber.BOTTLENECK_BASE + sum(wi * c / (k - 1) for c, wi, k in zip(cfg, w, ks)))
                               for cfg in configs(ks)]))
    w = list(ber.CONVERGENCE_WEIGHTS.values())
    phi = np.array([clip(ber.RISK_BASE + sum(wi for b, wi in zip(c, w) if b)) for c in configs([2] * 4)])
    return {"pi": pi, "theta": theta, "phi": phi}


class FastBN:
    """Exact inference for this two-layer network with numpy (all 16 bottleneck combinations)."""

    def __init__(self, params, levels=None):
        self.p = params
        levels = {} if levels is None else levels
        self.slices, start = [], 0
        for m in MODULE_NAMES:
            names = list(ber.MODULES[m])
            self.slices.append((start, start + len(names), configs([levels.get(n, 2) for n in names])))
            start += len(names)
        self.Z = configs([2] * 4)                                 # bottleneck combinations

    def _config_weights(self, X, j):
        """P(parent combination | evidence) for module j: indicator if observed, prior if not."""
        a, b, C = self.slices[j]
        w = np.ones((X.shape[0], C.shape[0]))
        for col, i in enumerate(range(a, b)):
            x = X[:, i][:, None]
            prior = self.p["pi"][i][C[:, col]][None, :]
            match = (C[None, :, col] == x).astype(float)
            w *= np.where(x < 0, prior, match)
        return w

    def bottleneck_marginals(self, X):
        out = []
        for j in range(len(MODULE_NAMES)):
            w = self._config_weights(X, j)
            hi = w @ self.p["theta"][j]
            out.append((np.stack([w.sum(1) - hi, hi], 1), w))
        return out

    def joint_z(self, X, y=None):
        """n x 16 posterior over bottleneck combinations (given the outcome if y is supplied)."""
        marg = self.bottleneck_marginals(X)
        logp = np.zeros((X.shape[0], 16))
        for j, (m, _) in enumerate(marg):
            logp += np.log(np.clip(m[:, self.Z[:, j]], 1e-300, None))
        if y is not None:
            phi = self.p["phi"][None, :]
            logp += np.log(np.where(y[:, None] == 1, phi, 1 - phi))
        ll = np.logaddexp.reduce(logp, axis=1)
        return np.exp(logp - ll[:, None]), ll, marg

    def p_fail(self, X):
        q, _, _ = self.joint_z(X)
        return q @ self.p["phi"]

    def diagnose(self, X):
        """P(each bottleneck High | evidence, serving FAILS): which problem would be to blame."""
        q, _, _ = self.joint_z(X, y=np.ones(X.shape[0], dtype=int))
        return {b: q[:, self.Z[:, j] == 1].sum(1) for j, b in enumerate(BOTTLENECKS)}


def em(X, y, levels=None, prior_strength=PRIOR_STRENGTH, max_iter=300, tol=1e-6, verbose=True):
    """MAP-EM with the hand-set network as Dirichlet prior and starting point."""
    levels = LEVELS if levels is None else levels
    prior = handset_params(levels)
    params = {"pi": [p.copy() for p in prior["pi"]], "theta": [t.copy() for t in prior["theta"]],
              "phi": prior["phi"].copy()}
    a = prior_strength
    last = -np.inf
    for it in range(max_iter):
        bn = FastBN(params, levels)
        q, ll, marg = bn.joint_z(X, y)
        loglik = ll.sum()
        n1, n = (q * y[:, None]).sum(0), q.sum(0)                                   # outcome table
        params["phi"] = (n1 + a * prior["phi"]) / (n + a)
        new_pi = [p.copy() for p in params["pi"]]
        for j, (start, end, C) in enumerate(bn.slices):
            m, w = marg[j]
            qj = np.stack([q[:, bn.Z[:, j] == b].sum(1) for b in (0, 1)], 1)       # P(B_j | e, y)
            th = params["theta"][j]
            r1 = w * th[None, :] / np.clip(m[:, 1:2], 1e-300, None) * qj[:, 1:2]     # P(config, B=High | e, y)
            r0 = w * (1 - th)[None, :] / np.clip(m[:, 0:1], 1e-300, None) * qj[:, 0:1]
            N1, N = r1.sum(0), (r1 + r0).sum(0)
            params["theta"][j] = (N1 + a * prior["theta"][j]) / (N + a)
            rc = r1 + r0                                                               # P(config | e, y)
            for col, i in enumerate(range(start, end)):
                k = len(prior["pi"][i])
                counts = np.array([(rc * (C[None, :, col] == s)).sum() for s in range(k)])
                new_pi[i] = (counts + a * prior["pi"][i]) / (len(X) + a)
        params["pi"] = new_pi
        if verbose and it % 25 == 0:
            print(f"   EM iteration {it:3d}  log-likelihood {loglik:.2f}")
        if abs(loglik - last) < tol * max(1, abs(loglik)):
            if verbose:
                print(f"   converged after {it + 1} iterations (log-likelihood {loglik:.2f})")
            break
        last = loglik
    return params


def choose_threshold(p, y, target_fail):
    """Largest t such that questions with P(fail) < t fail at most target_fail of the time."""
    best = 0.0
    for t in np.unique(p):
        served = p < t
        if served.any() and y[served].mean() <= target_fail:
            best = float(t)
    return best


# ----------------------------------------------------------------- save / load / pgmpy export
def to_profile(params, cuts, levels, meta):
    return {"meta": meta, "levels": {n: levels.get(n, 2) for n in INPUTS}, "cutoffs": cuts,
            "inputs": {n: params["pi"][i].round(5).tolist() for i, n in enumerate(INPUTS)},
            "bottlenecks": {m: {"parents": list(ber.MODULES[m]), "p_high": params["theta"][j].round(5).tolist()}
                            for j, m in enumerate(MODULE_NAMES)},
            "risk": {"parents": BOTTLENECKS, "p_high": params["phi"].round(5).tolist()}}


def params_from_profile(profile):
    return {"pi": [np.array(profile["inputs"][n]) for n in INPUTS],
            "theta": [np.array(profile["bottlenecks"][m]["p_high"]) for m in MODULE_NAMES],
            "phi": np.array(profile["risk"]["p_high"])}


def load_profile(path):
    profile = json.load(open(path))
    return profile, FastBN(params_from_profile(profile), profile["levels"])


def to_pgmpy(profile):
    """The learned network as a pgmpy model (same structure as bayesian_epistemic_routing.model)."""
    from pgmpy.factors.discrete import TabularCPD
    from pgmpy.models import DiscreteBayesianNetwork
    lv = profile["levels"]
    S = lambda n: state_names(lv.get(n, 2))
    model, cpds = DiscreteBayesianNetwork(), []
    for m in MODULE_NAMES:
        parents = profile["bottlenecks"][m]["parents"]
        model.add_edges_from((p, m) for p in parents)
        for p in parents:
            probs = profile["inputs"][p]
            cpds.append(TabularCPD(p, len(probs), [[v] for v in probs], state_names={p: S(p)}))
        hi = profile["bottlenecks"][m]["p_high"]
        cpds.append(TabularCPD(m, 2, [[1 - v for v in hi], hi], evidence=parents,
                               evidence_card=[lv.get(p, 2) for p in parents],
                               state_names={m: ber.STATES, **{p: S(p) for p in parents}}))
    model.add_edges_from((b, "HallucinationRisk") for b in BOTTLENECKS)
    hi = profile["risk"]["p_high"]
    cpds.append(TabularCPD("HallucinationRisk", 2, [[1 - v for v in hi], hi], evidence=BOTTLENECKS,
                           evidence_card=[2] * 4, state_names={"HallucinationRisk": ber.STATES,
                                                                **{b: ber.STATES for b in BOTTLENECKS}}))
    model.add_edge("HallucinationRisk", "OptimalRoute")
    cpds.append(ber.route_cpd())
    model.add_cpds(*cpds)
    assert model.check_model()
    return model


# ----------------------------------------------------------------- routing with the learned network
def decide(p_fail, why, tau):
    """
    Vectorised routing rule:
      Serve   if P(fail) < tau
      Clarify if, given that serving would fail, the QUESTION itself is probably to blame
              (P(QueryAmbiguity or IntentUncertainty High | evidence, fail) >= 0.5):
              searching cannot fix a question the system does not understand
      Search  otherwise (the context / knowledge is to blame)
    """
    q_side = np.maximum(why["QueryAmbiguity"], why["IntentUncertainty"])
    return np.where(p_fail < tau, "Serve", np.where(q_side >= 0.5, "Clarify", "Search"))


def route(profile, raw: dict, bn: "FastBN" = None, serve_below: float = None) -> dict:
    """
    Serve if P(fail) is below the profile's serve threshold; otherwise ask the network WHY
    (diagnosis = posterior of each bottleneck given that serving would fail) - see decide().
    """
    bn = bn or FastBN(params_from_profile(profile), profile["levels"])
    ev = discretize(raw, profile["cutoffs"])
    X = np.array([[ev.get(n, -1) for n in INPUTS]])
    p = float(bn.p_fail(X)[0])
    why = {b: float(v[0]) for b, v in bn.diagnose(X).items()}
    tau = serve_below if serve_below is not None else profile["meta"].get("serve_below", 0.2)
    decision = str(decide(np.array([p]), {k: np.array([v]) for k, v in why.items()}, tau)[0])
    named = {n: state_names(profile["levels"].get(n, 2))[l] for n, l in ev.items()}
    return {"route": decision, "p_fail": round(p, 4), "serve_below": tau,
            "blame": {k: round(v, 3) for k, v in why.items()}, "evidence": named}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("labels", nargs="+", help="one or more labels files (e.g. PopQA + hybrid)")
    ap.add_argument("--prior-strength", type=float, default=PRIOR_STRENGTH,
                    help="how many questions' worth the hand-set tables count for (per table column)")
    ap.add_argument("--target-fail", type=float, default=0.2,
                    help="serve threshold: served questions may fail at most this often (training data)")
    ap.add_argument("--binary", action="store_true", help="all signals Low/High (the original 2-level network)")
    ap.add_argument("--out", default="profiles")
    args = ap.parse_args()
    levels = {} if args.binary else LEVELS

    rows = [json.loads(l) for path in args.labels for l in open(path, encoding="utf-8")]
    rows = [r for r in rows if not r.get("empty")]
    from collections import Counter
    print("datasets:", dict(Counter(r.get("dataset") for r in rows)))
    y = np.array([failed(r) for r in rows])
    print(f"{len(rows)} labelled questions, serving fails on {y.mean():.1%}")

    cuts = learn_cutoffs(rows, levels)
    print("\nCut-offs (level 0 = Low ... last = High = trouble):")
    for node, c in cuts.items():
        d = SIGNALS[node][1]
        sign = "<" if d < 0 else ">"
        txt = ", ".join(f"{sign} {x:.4g}" for x in c["cuts"])
        print(f"   {node:<22} {SIGNALS[node][0]:<26} {len(c['cuts']) + 1} levels, cuts {txt:<32}"
              f"{'learned (info gain ' + str(c['info_gain']) + ')' if c['learned'] else 'hand-set (no variation / no gain)'}")

    X = evidence_matrix(rows, cuts)
    print("\nEM (bottlenecks are latent):")
    params = em(X, y, levels, prior_strength=args.prior_strength)

    p = FastBN(params, levels).p_fail(X)
    tau = choose_threshold(p, y, args.target_fail)
    served = p < tau
    print(f"\nServe threshold P(fail) < {tau:.3f}: serves {served.mean():.1%} of questions, "
          f"of which {y[served].mean() if served.any() else float('nan'):.1%} fail (training data)")

    name = os.path.basename(args.labels[0]).replace(".jsonl", "").split("__")[0]
    if len(args.labels) > 1:
        name += "__combined"
    llm_prof = args.labels[0].replace(".jsonl", ".profile.json")
    meta = {"labels": [os.path.abspath(p) for p in args.labels], "n": len(rows), "fail_rate": round(float(y.mean()), 4),
            "target": "serving fails = wrong answer OR 'I don't know'", "prior_strength": args.prior_strength,
            "serve_below": round(float(tau), 4), "target_fail": args.target_fail,
            "llm_profile": json.load(open(llm_prof)) if os.path.exists(llm_prof) else None}
    profile = to_profile(params, cuts, levels, meta)
    os.makedirs(args.out, exist_ok=True)
    out = os.path.join(args.out, re.sub(r"[^\w.-]+", "_", name) + ".json")
    json.dump(profile, open(out, "w"), indent=1)
    print(f"\nLearned network saved -> {out}")

    print("\nWhat the network learned - P(serving fails | only this bottleneck High):")
    for j, b in enumerate(BOTTLENECKS):
        idx = 1 << (3 - j)
        print(f"   {b:<18} {params['phi'][idx]:.2f}")
    print(f"   {'(none High)':<18} {params['phi'][0]:.2f}")


if __name__ == "__main__":
    main()
