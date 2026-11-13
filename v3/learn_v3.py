"""
Learning the v3 network from labelled data - so that the hand-set numbers do NOT decide the result
====================================================================================================
Why the old (v2) numbers mattered so much
  * the 4 bottlenecks were fully hidden: EM can only find A local optimum, and WHICH one depends on
    the starting values -> the hand-set tables chose the answer, and the meaning of the hidden
    nodes could drift (QueryAmbiguity became a generic "will fail" node)
  * the prior strength (alpha = 5) and the cut-offs were fixed by hand, never tested

What this file does instead
  1. CUT-OFFS for real-valued signals are learned on the training split only (information gain
     against the symptom's own parent cause label where available, else against the outcome).
  2. WEAK CAUSE LABELS make the causes partly OBSERVED. Each label is a noisy child of its cause
     (P(label wrong) = LABEL_NOISE), so a wrong label cannot force a wrong answer, but the causes
     keep their meaning and EM no longer depends on where it starts.
  3. The hand-set tables are only a Dirichlet prior. Its STRENGTH alpha (pseudo-questions per table
     column) is chosen by GROUPED cross-validation on held-out outcome log-loss. alpha = 0 means the
     hand-set numbers are ignored entirely.
  4. RESTARTS: EM is run from the hand-set values AND from random starting points; the spread of the
     results is reported. A large spread = that part of the network is not identified by the data.
  5. ACTION EFFECTS are fitted (maximum likelihood with a weak Beta prior) on the questions where
     that action was actually run.
  6. LAMBDA is not learned (it is a policy). choose_lambda() picks it from a target risk:
     "at most X% of served answers may be wrong", using validation data.

Input: one JSON row per question (see `ROW FORMAT` below).
    python learn_v3.py labels_v3.jsonl --target-risk 0.05 --out profiles/v3_profile.json
"""

from __future__ import annotations

import argparse
import json
import math

import numpy as np

import ber_v3 as B

LABEL_NOISE = 0.05            # assumed error rate of a weak cause label (it is a noisy child of the cause)
ALPHAS = [0.0, 1.0, 3.0, 10.0, 30.0, 100.0]

# ----------------------------------------------------------------------------- ROW FORMAT
# {"id": ..., "group": "<entity / conversation / story id: rows of one group never split>",
#  "dataset": "musique",
#  "raw": {"rerank_logit": 4.2, ...},            # real-valued signals (discretised with learned cut-offs)
#  "symptoms": {"MultiHopForm": "Bridge", ...},  # categorical symptoms, already states
#  "labels": {"ReasoningLoad": 1, ...},          # weak cause labels (any subset, 0/1)
#  "outcome": "Correct" | "Abstain" | "Wrong",   # what Serve produced (graded)
#  "context": {"HistoryCandidates": "One", ...},
#  "action": {"name": "Search", "outcome": "Correct"}   # optional: another action actually run
# }

# ----------------------------------------------------------------------------- 1. cut-offs
# symptom -> (raw key, direction, default cuts). direction +1: state index rises with the value,
# -1: state index rises as the value falls. States are in the order listed in ber_v3.SYMPTOMS.
RAW_SPEC = {
    "NameCollision":        ("name_collision_count", +1, [1.5, 3.5]),
    "ResidualReferences":   ("residual_reference_count", +1, [0.5, 1.5]),
    "RerankerLogit":        ("rerank_logit", -1, [6.0, 2.0, -2.0]),
    "RerankMargin":         ("rerank_margin", -1, [3.0, 1.0]),
    "TitleDiversity":       ("title_diversity", +1, [1.5, 2.5]),
    "EntityCoverage":       ("entity_coverage", -1, [0.999, 0.001]),
    "Answerability":        ("answerability", -1, [0.7, 0.3]),
    "EntityFamiliarity":    ("entity_logprob", -1, [-2.0, -4.0]),
    "ClosedBookConfidence": ("closed_book_p_idk", +1, [0.1, 0.5]),
    "GroundedConfidence":   ("grounded_p_idk", +1, [0.1, 0.5]),
}


def to_state(v, cuts, direction):
    return int(sum((v < c) if direction < 0 else (v > c) for c in cuts))


def _entropy_bits(y):
    p = np.bincount(y, minlength=2) / max(len(y), 1)
    p = p[p > 0]
    return float(-(p * np.log2(p)).sum())


def info_gain(levels, y):
    g = _entropy_bits(y)
    for l in np.unique(levels):
        m = levels == l
        g -= m.mean() * _entropy_bits(y[m])
    return g


def learn_cuts(values, target, n_states, direction, min_share=0.03):
    """Greedy: add the cut that raises information gain most, until n_states - 1 cuts."""
    v, y = np.asarray(values, float), np.asarray(target, int)
    cand = np.unique(np.quantile(v, np.linspace(0.03, 0.97, 60)))
    chosen, gain = [], 0.0
    for _ in range(n_states - 1):
        best = (gain + 1e-4, None)
        for c in cand:
            if c in chosen:
                continue
            cuts = sorted(chosen + [c], reverse=direction < 0)
            lv = np.array([to_state(x, cuts, direction) for x in v])
            if (np.bincount(lv, minlength=len(cuts) + 1) / len(lv)).min() < min_share:
                continue
            g = info_gain(lv, y)
            if g > best[0]:
                best = (g, float(c))
        if best[1] is None:
            break
        gain = best[0]
        chosen.append(best[1])
    return sorted(chosen, reverse=direction < 0), gain


def fit_cutoffs(rows):
    """Cut-offs per real-valued symptom, learned on THESE rows only (call on the training split)."""
    out = {}
    for s, (key, direction, default) in RAW_SPEC.items():
        n_states = len(B.SYMPTOMS[s]["states"])
        vals, tgt = [], []
        parents = B.SYMPTOMS[s]["parents"]
        for r in rows:
            v = r.get("raw", {}).get(key)
            if not isinstance(v, (int, float)):
                continue
            lab = [r.get("labels", {}).get(p) for p in parents]
            if any(l is not None for l in lab):          # the symptom's own cause(s): the right target
                t = int(any(l == 1 for l in lab if l is not None))
            elif r.get("outcome") is not None:
                t = int(r["outcome"] != "Correct")
            else:
                continue
            vals.append(v)
            tgt.append(t)
        if len(vals) >= 200 and len(set(tgt)) == 2:
            cuts, gain = learn_cuts(vals, tgt, n_states, direction)
            if len(cuts) == n_states - 1:
                out[s] = {"cuts": cuts, "learned": True, "info_gain": round(gain, 4)}
                continue
        out[s] = {"cuts": default, "learned": False, "info_gain": 0.0}
    return out


def discretize(row, cuts):
    """raw values + categorical symptoms -> {symptom: state name}."""
    ev = dict(row.get("symptoms", {}))
    for s, (key, direction, _) in RAW_SPEC.items():
        v = row.get("raw", {}).get(key)
        if isinstance(v, (int, float)) and s not in ev:
            ev[s] = B.SYMPTOMS[s]["states"][to_state(v, cuts[s]["cuts"], direction)]
    return ev


# ----------------------------------------------------------------------------- 2. data as arrays
OUT_IDX = {s: i for i, s in enumerate(B.OUTCOME_STATES)}


class Data:
    """Rows -> integer arrays. Accepts rows from ber_v3.sample (already discrete) or real rows + cuts."""

    def __init__(self, rows, cuts=None):
        n = len(rows)
        self.n = n
        self.X = {s: np.full(n, -1, dtype=int) for s in B.SYMPTOM_NAMES}
        self.L = np.full((n, len(B.CAUSE_NAMES)), -1, dtype=int)
        self.Y = np.full(n, -1, dtype=int)
        self.ctx = {k: np.zeros(n, dtype=int) for k in B.CONTEXT}
        self.typed = np.zeros(n, dtype=bool)
        self.act = np.full(n, -1, dtype=int)
        self.act_y = np.full(n, -1, dtype=int)
        self.groups = []
        for i, r in enumerate(rows):
            ev = discretize(r, cuts) if cuts is not None else r["symptoms"]
            for s, v in ev.items():
                if s in self.X:
                    self.X[s][i] = B.SYMPTOMS[s]["states"].index(v)
            for c, v in r.get("labels", {}).items():
                self.L[i, B.CI[c]] = int(v)
            o = r.get("outcome")
            if o is not None:
                self.Y[i] = o if isinstance(o, int) else OUT_IDX[o]
            ctx = {**B.DEFAULT_CONTEXT, **r.get("context", {})}
            for k in B.CONTEXT:
                self.ctx[k][i] = B.CONTEXT[k]["states"].index(ctx[k])
            self.typed[i] = ev.get("SubjectTypeMatch") in ("Match", "Mismatch")
            a = r.get("action")
            if a:
                name, out = (a if isinstance(a, (list, tuple)) else (a["name"], a["outcome"]))
                self.act[i] = B.ACTIONS.index(name)
                self.act_y[i] = out if isinstance(out, int) else OUT_IDX[out]
            self.groups.append(r.get("group", r.get("id", i)))
        self.groups = np.array(self.groups, dtype=object)

    def subset(self, idx):
        d = Data.__new__(Data)
        d.n = len(idx)
        d.X = {s: v[idx] for s, v in self.X.items()}
        d.L, d.Y, d.typed = self.L[idx], self.Y[idx], self.typed[idx]
        d.ctx = {k: v[idx] for k, v in self.ctx.items()}
        d.act, d.act_y, d.groups = self.act[idx], self.act_y[idx], self.groups[idx]
        return d


# ----------------------------------------------------------------------------- 3. EM with weak labels
def log_joint(router, d: Data, use_labels=True, use_outcome=True):
    """(n, 512): log P(cause combination, everything observed for the row)."""
    lp = np.tile(router.log_prior, (d.n, 1))
    for s in B.SYMPTOM_NAMES:
        x = d.X[s]
        m = x >= 0
        if m.any():
            lp[m] += router.log_lik[s].T[x[m]]
    if use_labels:
        for c in B.CAUSE_NAMES:
            l = d.L[:, B.CI[c]]
            m = l >= 0
            if m.any():
                agree = B.Z[None, :, B.CI[c]] == l[m][:, None]
                lp[m] += np.log(np.where(agree, 1 - LABEL_NOISE, LABEL_NOISE))
    if use_outcome:
        m = d.Y >= 0
        if m.any():
            lp[m] += np.log(router.O.T[d.Y[m]])
    return lp


def _normalise(lp):
    ll = np.logaddexp.reduce(lp, axis=1)
    return np.exp(lp - ll[:, None]), ll


def _map(counts, prior_rows, alpha):
    """Dirichlet MAP per column: (counts + alpha * prior) / (total + alpha); empty columns keep the prior."""
    tot = counts.sum(1, keepdims=True)
    out = (counts + alpha * prior_rows) / np.clip(tot + alpha, 1e-12, None)
    return np.where(tot + alpha > 1e-9, out, prior_rows)


def m_step(q, d: Data, prior: B.Params, alpha, effects):
    w = q.sum(0)
    cause = {}
    for c in B.CAUSE_NAMES:
        k = len(prior.cause[c])
        hi = np.bincount(B.PIDX[c], weights=w * B.Z[:, B.CI[c]], minlength=k)
        tot = np.bincount(B.PIDX[c], weights=w, minlength=k)
        counts = np.stack([tot - hi, hi], 1)
        pr = np.stack([1 - prior.cause[c], prior.cause[c]], 1)
        cause[c] = np.clip(_map(counts, pr, alpha)[:, 1], 1e-4, 1 - 1e-4)
    symptom = {}
    for s in B.SYMPTOM_NAMES:
        k_states = len(B.SYMPTOMS[s]["states"])
        n_cfg = prior.symptom[s].shape[0]
        x = d.X[s]
        m = x >= 0
        counts = np.zeros((n_cfg, k_states))
        if m.any():
            onehot = np.eye(k_states)[x[m]]                       # (n_obs, k)
            per_z = onehot.T @ q[m]                               # (k, 512)
            for st in range(k_states):
                counts[:, st] = np.bincount(B.PIDX[s], weights=per_z[st], minlength=n_cfg)
        symptom[s] = np.clip(_map(counts, prior.symptom[s], alpha), 1e-4, None)
        symptom[s] /= symptom[s].sum(1, keepdims=True)
    symptom["AnswerGrounding"] = prior.symptom["AnswerGrounding"].copy()
    m = d.Y >= 0
    counts = np.zeros_like(prior.outcome)
    if m.any():
        per_z = np.eye(3)[d.Y[m]].T @ q[m]
        for st in range(3):
            counts[:, st] = np.bincount(B.PIDX["Outcome"], weights=per_z[st], minlength=len(prior.outcome))
    outcome = np.clip(_map(counts, prior.outcome, alpha), 1e-4, None)
    outcome /= outcome.sum(1, keepdims=True)
    return B.Params(cause, symptom, outcome, effects)


def log_prior_density(P: B.Params, prior: B.Params, alpha):
    if alpha <= 0:
        return 0.0
    s = sum(alpha * (prior.cause[c] * np.log(P.cause[c]) + (1 - prior.cause[c]) * np.log(1 - P.cause[c])).sum()
            for c in B.CAUSE_NAMES)
    s += sum((alpha * prior.symptom[k] * np.log(P.symptom[k])).sum() for k in B.SYMPTOM_NAMES)
    s += (alpha * prior.outcome * np.log(P.outcome)).sum()
    return float(s)


def em(d: Data, prior: B.Params, alpha, init: B.Params = None, max_iter=150, tol=1e-5, verbose=False):
    """MAP-EM. Stops when the objective changes by less than tol per question."""
    P = (init or prior).copy()
    last = -np.inf
    for it in range(max_iter):
        r = B.Router(P)
        q, ll = _normalise(log_joint(r, d))
        obj = ll.sum() + log_prior_density(P, prior, alpha)
        P = m_step(q, d, prior, alpha, P.effects)
        if verbose and it % 20 == 0:
            print(f"      EM {it:3d}  objective {obj:.1f}")
        if abs(obj - last) < tol * d.n:
            break
        last = obj
    return P, float(obj), it + 1


def random_init(prior: B.Params, rng, concentration=None):
    """Random starting tables: Dirichlet around the prior (concentration) or uniform-random (None)."""
    def draw(row):
        a = np.full(len(row), 1.0) if concentration is None else np.clip(row * concentration, 0.05, None)
        return rng.dirichlet(a)
    cause = {c: np.clip(np.array([draw(np.array([1 - p, p]))[1] for p in v]), 0.02, 0.98)
             for c, v in prior.cause.items()}
    symptom = {s: np.array([draw(row) for row in v]) for s, v in prior.symptom.items()}
    outcome = np.array([draw(row) for row in prior.outcome])
    return B.Params(cause, symptom, outcome, prior.effects)


def fit_with_restarts(d, prior, alpha, n_random=4, seed=0, verbose=False):
    """EM from the hand-set start + random starts. Returns the best fit and the spread across starts."""
    rng = np.random.default_rng(seed)
    starts = [("hand-set", prior)] + [(f"random {i + 1}", random_init(prior, rng)) for i in range(n_random)]
    fits = []
    for name, init in starts:
        P, obj, iters = em(d, prior, alpha, init)
        fits.append((obj, name, P, iters))
        if verbose:
            print(f"      start {name:<10} objective {obj:12.1f}  ({iters} iterations)")
    fits.sort(key=lambda f: -f[0])
    best = fits[0][2]
    spread = max(table_distance(best, f[2]) for f in fits)
    return best, {"objective": [round(f[0], 1) for f in fits], "starts": [f[1] for f in fits],
                  "max_distance_between_starts": round(spread, 4)}


def table_distance(P1: B.Params, P2: B.Params):
    """Mean absolute difference over all symptom, cause and outcome table entries."""
    diffs = [np.abs(P1.cause[c] - P2.cause[c]).ravel() for c in B.CAUSE_NAMES]
    diffs += [np.abs(P1.symptom[s] - P2.symptom[s]).ravel() for s in B.SYMPTOM_NAMES]
    diffs.append(np.abs(P1.outcome - P2.outcome).ravel())
    return float(np.concatenate(diffs).mean())


# ----------------------------------------------------------------------------- 4. alpha by grouped CV
def predict_outcome(P: B.Params, d: Data):
    """P(Serve outcome | symptoms only) - what the router knows at decision time."""
    r = B.Router(P)
    q, _ = _normalise(log_joint(r, d, use_labels=False, use_outcome=False))
    return q @ r.O


def outcome_logloss(P, d):
    m = d.Y >= 0
    p = predict_outcome(P, d.subset(np.where(m)[0]))
    return float(-np.log(np.clip(p[np.arange(m.sum()), d.Y[m]], 1e-12, None)).mean())


def grouped_folds(groups, k, seed=0):
    uniq = np.array(sorted(set(groups), key=str), dtype=object)
    rng = np.random.default_rng(seed)
    rng.shuffle(uniq)
    fold_of = {g: i % k for i, g in enumerate(uniq)}
    f = np.array([fold_of[g] for g in groups])
    return [(np.where(f != i)[0], np.where(f == i)[0]) for i in range(k)]


def select_alpha(d: Data, prior, alphas=ALPHAS, k=5, seed=0, verbose=True):
    res = {}
    for a in alphas:
        losses = []
        for tr, te in grouped_folds(d.groups, k, seed):
            P, _, _ = em(d.subset(tr), prior, a)
            losses.append(outcome_logloss(P, d.subset(te)))
        res[a] = float(np.mean(losses))
        if verbose:
            print(f"   alpha {a:>6}: held-out outcome log-loss {res[a]:.4f}")
    best = min(res, key=res.get)
    return best, res


# ----------------------------------------------------------------------------- 5. action effects
def _flatten(eff):
    """Learnable effect probabilities as (path, value) pairs (clarify_fatigue['0'] stays 1)."""
    out = []
    for a in ("Rewrite", "Search", "Clarify", "Decompose"):
        for k, v in eff[a].items():
            if isinstance(v, dict):
                out += [((a, k, kk), vv) for kk, vv in v.items()]
            else:
                out.append(((a, k), v))
    out += [(("clarify_fatigue", k), v) for k, v in eff["clarify_fatigue"].items() if k != "0"]
    for c, spec in eff["web"].items():
        for k, v in spec.items():
            if isinstance(v, dict):
                out += [(("web", c, k, kk), vv) for kk, vv in v.items()]
            else:
                out.append((("web", c, k), v))
    return out


def _set(eff, path, value):
    import copy
    e = eff
    for p in path[:-1]:
        e = e[p]
    e[path[-1]] = float(value)


def post_action_batch(router, Q, action, d: Data):
    """Vectorised over rows: (n, 512) cause distribution after the action (matches Router.transition)."""
    n = Q.shape[0]
    e = router.p.effects
    hc = np.array(B.CONTEXT["HistoryCandidates"]["states"])[d.ctx["HistoryCandidates"]]
    cr = np.array(B.CONTEXT["ClarifyRounds"]["states"])[d.ctx["ClarifyRounds"]]
    sr = np.array(B.CONTEXT["SearchRounds"]["states"])[d.ctx["SearchRounds"]]
    Qt = Q.reshape((n,) + (2,) * len(B.CAUSE_NAMES))
    for k, v in e.get(action, {}).items():
        c, _, cond = k.partition("@")
        if isinstance(v, dict):
            p = np.array([v[h] for h in hc])
        else:
            p = np.full(n, float(v))
        if cond == "typed":
            p = np.where(d.typed, p, 0.0)
        if action == "Clarify":
            p = p * np.array([e["clarify_fatigue"][x] for x in cr])
        ax = 1 + B.CI[c]
        low, high = np.take(Qt, 0, axis=ax), np.take(Qt, 1, axis=ax)
        pb = p.reshape((n,) + (1,) * (len(B.CAUSE_NAMES) - 1))
        Qt = np.stack([low + pb * high, (1 - pb) * high], axis=ax)
    Q2 = Qt.reshape(n, -1)
    # evidence causes: marginalise the old ones, draw new ones from the web tables
    qbits = B.Z[:, [B.CI[c] for c in B.QUERY_CAUSES]]
    qkey = (qbits * (2 ** np.arange(qbits.shape[1])[::-1])).sum(1)          # 0..63 per combination
    Eagg = np.eye(64)[qkey]                                                  # (512, 64)
    F = {}
    for s in ("0", "1", "2+"):
        f = np.ones(len(B.Z))
        for c, spec in e["web"].items():
            p_low = np.full(len(B.Z), 1 - spec["leak"][s])
            for par, st in spec.items():
                if par != "leak":
                    p_low = p_low * np.where(B.Z[:, B.CI[par]] == 1, 1 - st, 1.0)
            f *= np.where(B.Z[:, B.CI[c]] == 1, 1 - p_low, p_low)
        F[s] = f
    Fr = np.stack([F[s] for s in sr])
    return (Q2 @ Eagg @ Eagg.T) * Fr


def effects_loglik(P: B.Params, d: Data, Q):
    r = B.Router(P)
    total = 0.0
    for ai, a in enumerate(B.ACTIONS):
        m = d.act == ai
        if not m.any():
            continue
        Qa = post_action_batch(r, Q[m], a, d.subset(np.where(m)[0]))
        p = (Qa @ r.O)[np.arange(m.sum()), d.act_y[m]]
        total += np.log(np.clip(p, 1e-12, None)).sum()
    return float(total)


def fit_effects(P: B.Params, d: Data, prior_effects, strength=10.0, verbose=True):
    """MAP fit of the action-effect probabilities on rows where an action was actually run."""
    from scipy.optimize import minimize
    m = d.act >= 0
    if m.sum() < 50:
        if verbose:
            print("   fewer than 50 action rows: keeping the hand-set action effects")
        return P
    sub = d.subset(np.where(m)[0])
    Q, _ = _normalise(log_joint(B.Router(P), sub))         # everything known about these questions
    flat = _flatten(prior_effects)
    paths = [f[0] for f in flat]
    x0 = np.array([f[1] for f in flat])

    def unpack(x):
        import copy
        eff = copy.deepcopy(prior_effects)
        for p, v in zip(paths, x):
            _set(eff, p, v)
        return eff

    def nll(x):
        Pt = B.Params(P.cause, P.symptom, P.outcome, unpack(x))
        lp = strength * (x0 * np.log(x) + (1 - x0) * np.log(1 - x)).sum()     # Beta prior
        return -(effects_loglik(Pt, sub, Q) + lp)

    res = minimize(nll, x0, method="L-BFGS-B", bounds=[(0.01, 0.99)] * len(x0), options={"maxiter": 200})
    if verbose:
        print(f"   action effects: {len(x0)} parameters from {m.sum()} action rows, "
              f"NLL {nll(x0):.1f} -> {res.fun:.1f}")
    return B.Params(P.cause, P.symptom, P.outcome, unpack(res.x))


# ----------------------------------------------------------------------------- 6. lambda from a target risk
LAMBDA_GRID = tuple(np.round(np.geomspace(0.5, 20, 25), 2))


def risk_coverage(P: B.Params, d: Data, lambdas=LAMBDA_GRID):
    """Serve-vs-abstain on validation rows: share served and share of served answers that are wrong."""
    m = d.Y >= 0
    sub = d.subset(np.where(m)[0])
    p = predict_outcome(P, sub)
    rows = []
    for lam in lambdas:
        serve = p[:, 0] - lam * p[:, 2] > 0
        cov = serve.mean()
        risk = (sub.Y[serve] == 2).mean() if serve.any() else 0.0
        acc = (sub.Y[serve] == 0).mean() if serve.any() else float("nan")
        rows.append({"lambda": float(lam), "coverage": round(float(cov), 4), "risk": round(float(risk), 4),
                     "accuracy_served": round(float(acc), 4)})
    return rows


def choose_lambda(P, d, target_risk, min_coverage=0.3):
    """
    LAMBDA is a policy, not a fact: it is the exchange rate between "a wrong answer" and "no answer".
    Serve beats Abstain when P(Wrong)/P(Correct) < 1/lambda, so choosing lambda = choosing a risk level.
    Rule: the smallest lambda (most questions served) whose served answers are wrong at most target_risk
    on validation data. If that needs serving fewer than min_coverage of the questions, the target is not
    realistic for this LLM + retriever: we say so and use the lowest-risk lambda that still serves
    min_coverage. Always report the whole table (risk-coverage curve) in the paper.
    """
    table = risk_coverage(P, d)
    ok = [r for r in table if r["risk"] <= target_risk and r["coverage"] >= min_coverage]
    if ok:
        return ok[0]["lambda"], table, None
    feasible = [r for r in table if r["coverage"] >= min_coverage] or table
    best = min(feasible, key=lambda r: (r["risk"], -r["coverage"]))
    note = (f"target risk {target_risk:.0%} is not reachable while serving >= {min_coverage:.0%} of questions; "
            f"using lambda {best['lambda']} (serves {best['coverage']:.0%}, wrong among served {best['risk']:.1%})")
    return best["lambda"], table, note


# ----------------------------------------------------------------------------- 7. the whole pipeline
def learn(rows, target_risk=0.05, alphas=ALPHAS, restarts=4, val_share=0.2, seed=0, verbose=True):
    groups = np.array([r.get("group", r.get("id", i)) for i, r in enumerate(rows)], dtype=object)
    tr_idx, val_idx = grouped_folds(groups, int(round(1 / val_share)), seed)[0]
    train_rows = [rows[i] for i in tr_idx]
    cuts = fit_cutoffs(train_rows)                         # training split only
    d = Data(rows, cuts)
    dtr, dval = d.subset(tr_idx), d.subset(val_idx)
    prior = B.handset_params()
    if verbose:
        print(f"{len(rows)} rows: {len(tr_idx)} train / {len(val_idx)} validation (grouped)")
        print("1. prior strength alpha by grouped 5-fold CV on the training split:")
    alpha, alpha_table = select_alpha(dtr, prior, alphas, verbose=verbose)
    if verbose:
        print(f"   -> alpha = {alpha}\n2. EM from the hand-set start + {restarts} random starts:")
    P, restart_info = fit_with_restarts(dtr, prior, alpha, restarts, seed, verbose)
    if verbose:
        print(f"   max table distance between starts: {restart_info['max_distance_between_starts']} "
              f"(near 0 = the data, not the start, decides)\n3. action effects:")
    P = fit_effects(P, dtr, prior.effects, verbose=verbose)
    lam, rc, note = choose_lambda(P, dval, target_risk)
    if verbose:
        print(f"4. lambda for at most {target_risk:.0%} wrong among served answers (validation): {lam}")
        if note:
            print(f"   NOTE: {note}")
        for r in rc[::3]:
            print(f"      lambda {r['lambda']:>6}: serves {r['coverage']:.1%}, wrong among served {r['risk']:.1%}")
        print(f"5. held-out outcome log-loss: learned {outcome_logloss(P, dval):.4f}, "
              f"hand-set {outcome_logloss(prior, dval):.4f}")
    return {"params": P.to_json(), "cutoffs": cuts, "alpha": alpha, "alpha_cv": alpha_table,
            "restarts": restart_info, "lambda": lam, "lambda_note": note, "risk_coverage": rc,
            "target_risk": target_risk}


def load_profile(path):
    prof = json.load(open(path))
    return prof, B.Router(B.Params.from_json(prof["params"]), lam=prof["lambda"])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("labels", nargs="+")
    ap.add_argument("--target-risk", type=float, default=0.05, help="max share of served answers that are wrong")
    ap.add_argument("--restarts", type=int, default=4)
    ap.add_argument("--out", default="profiles/v3_profile.json")
    args = ap.parse_args()
    rows = [json.loads(l) for p in args.labels for l in open(p, encoding="utf-8")]
    prof = learn(rows, args.target_risk, restarts=args.restarts)
    import os
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    json.dump(prof, open(args.out, "w"), indent=1)
    print(f"saved -> {args.out}")


if __name__ == "__main__":
    main()
