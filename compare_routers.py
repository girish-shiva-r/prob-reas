"""
Step 4 - Is the Bayesian network any good? Compare it with simpler routers.
---------------------------------------------------------------------------
Every router outputs a RISK score per question ("how likely is serving now to fail").
5-fold cross-validation: everything learnable (cut-offs, tables, thresholds, regression
weights) is fitted on 4/5 of the questions and scored on the held-out 1/5.

Routers
  hand-set BN        the original network: expert tables, expert cut-offs, no data
  reranker only      one signal: 1 - top reranker score (a common adaptive-RAG baseline)
  logistic reg.      all raw signals, continuous (strong, simple ML baseline)
  gradient boosting  all raw signals (strong non-linear ML baseline)
  learned BN, 2 lv.  same structure, cut-offs + tables learned, every signal Low/High
  learned BN         same, with 3-4 levels for the informative signals (learn_bn.LEVELS)

Metrics (held-out questions)
  AUROC              how well risk separates failures from successes (0.5 = coin flip)
  Brier              accuracy of the probabilities themselves (lower = better; probabilistic routers only)
  serve@20%          share of questions that can be served directly while keeping their failure
                     rate <= 20% (threshold chosen on the training folds)
  fail@50%           failure rate among the 50% of questions judged safest
Robustness
  "retrieval signals missing": the same models when FAISS / reranker / chunk signals are
  unavailable at prediction time (e.g. deciding before retrieval runs). The BN simply
  marginalises them out; the regression has to fall back to training averages.

    python compare_routers.py labels/Qwen_Qwen2.5-7B-Instruct-bfloat16.jsonl
"""

import argparse
import json
import math
import warnings

import numpy as np

warnings.filterwarnings("ignore")

from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import learn_bn as L

RAW_COLS = ["word_count", "named_entity_count", "sub_question_count", "top1_faiss_distance",
            "top1_rerank_score", "relevant_chunk_count", "kb_domain_similarity", "log_views",
            "views_missing", "is_time_sensitive", "unresolved_reference_count", "open_ended",
            "is_ambiguous_followup", "similarity_to_prev_query"]
RETRIEVAL_RAW = ["top1_faiss_distance", "top1_rerank_score", "relevant_chunk_count"]
RETRIEVAL_NODES = ["FAISSVectorDistance", "RerankerScore", "ChunkDensity", "SourceAuthority"]


def raw_matrix(rows):
    X = []
    for r in rows:
        raw = r["raw"]
        v = raw.get("entity_monthly_views")
        row = {**{k: raw.get(k) for k in RAW_COLS},
               "log_views": math.log10(v + 1) if isinstance(v, (int, float)) else np.nan,
               "views_missing": float(not isinstance(v, (int, float))),
               "open_ended": float(raw.get("question_type") in L.ber.OPEN_ENDED_TYPES)}
        X.append([float(row[k]) if isinstance(row[k], (int, float, bool)) else np.nan for k in RAW_COLS])
    return np.array(X)


def impute(train, test):
    mean = np.nanmean(train, axis=0)
    mean = np.where(np.isnan(mean), 0, mean)
    fill = lambda A: np.where(np.isnan(A), mean[None, :], A)
    return fill(train), fill(test), mean


def metrics(y, risk, train_y=None, train_risk=None, prob=True):
    out = {"AUROC": roc_auc_score(y, risk)}
    out["Brier"] = brier_score_loss(y, np.clip(risk, 0, 1)) if prob else float("nan")
    if train_risk is not None:                       # threshold from TRAINING folds only
        tau = L.choose_threshold(train_risk, train_y, 0.20)
        served = risk < tau
        out["serve@20%"] = served.mean()
        out["fail_served"] = y[served].mean() if served.any() else float("nan")
    order = np.argsort(risk, kind="stable")
    out["fail@50%"] = y[order[:len(y) // 2]].mean()
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("labels", nargs="+", help="one or more labels files (e.g. PopQA + hybrid)")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rows = [json.loads(l) for path in args.labels for l in open(path, encoding="utf-8")]
    rows = [r for r in rows if not r.get("empty")]
    y = np.array([L.failed(r) for r in rows])
    ds = np.array([r.get("dataset") or "?" for r in rows])
    oof = {n: np.zeros(len(rows)) for n in ["reranker only", "logistic reg.", "gradient boosting", "learned BN"]}
    oof_route = np.empty(len(rows), dtype=object)
    R = raw_matrix(rows)
    print(f"{len(rows)} questions, serving fails on {y.mean():.1%}  ({args.folds}-fold cross-validation)\n")

    names = ["hand-set BN", "reranker only", "logistic reg.", "gradient boosting", "learned BN, 2 lv.", "learned BN"]
    res = {n: [] for n in names}
    res_missing = {n: [] for n in ["hand-set BN", "logistic reg.", "logistic reg. (retrained w/o)", "learned BN"]}
    routes = {"Serve": [], "Search": [], "Clarify": []}
    handset = L.FastBN(L.handset_params({}), {})
    X_hand = L.evidence_matrix(rows, L.default_cuts())            # expert cut-offs, 2 levels
    hide = [L.INPUTS.index(n) for n in RETRIEVAL_NODES]
    keep_raw = [i for i, c in enumerate(RAW_COLS) if c not in RETRIEVAL_RAW]

    for k, (tr, te) in enumerate(StratifiedKFold(args.folds, shuffle=True, random_state=args.seed).split(R, y)):
        rtr = [rows[i] for i in tr]
        # hand-set BN
        res["hand-set BN"].append(metrics(y[te], handset.p_fail(X_hand[te]), y[tr], handset.p_fail(X_hand[tr])))
        # reranker only
        s = 1 - R[:, RAW_COLS.index("top1_rerank_score")]
        res["reranker only"].append(metrics(y[te], s[te], y[tr], s[tr], prob=False))
        oof["reranker only"][te] = s[te]
        # logistic regression + gradient boosting on raw signals
        A_tr, A_te, mean = impute(R[tr], R[te])
        lr = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)).fit(A_tr, y[tr])
        res["logistic reg."].append(metrics(y[te], lr.predict_proba(A_te)[:, 1], y[tr], lr.predict_proba(A_tr)[:, 1]))
        oof["logistic reg."][te] = lr.predict_proba(A_te)[:, 1]
        gb = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.05, random_state=0).fit(R[tr], y[tr])
        res["gradient boosting"].append(metrics(y[te], gb.predict_proba(R[te])[:, 1], y[tr], gb.predict_proba(R[tr])[:, 1]))
        oof["gradient boosting"][te] = gb.predict_proba(R[te])[:, 1]
        # learned BN, all signals 2 levels
        c2 = L.learn_cutoffs(rtr, {})
        X2tr, X2te = L.evidence_matrix(rtr, c2), L.evidence_matrix([rows[i] for i in te], c2)
        bn2 = L.FastBN(L.em(X2tr, y[tr], {}, verbose=False), {})
        res["learned BN, 2 lv."].append(metrics(y[te], bn2.p_fail(X2te), y[tr], bn2.p_fail(X2tr)))
        # learned BN, multi-level retrieval signals
        cuts = L.learn_cutoffs(rtr, L.LEVELS)
        Xtr, Xte = L.evidence_matrix(rtr, cuts), L.evidence_matrix([rows[i] for i in te], cuts)
        params = L.em(Xtr, y[tr], L.LEVELS, verbose=False)
        bn = L.FastBN(params, L.LEVELS)
        p_tr, p_te = bn.p_fail(Xtr), bn.p_fail(Xte)
        res["learned BN"].append(metrics(y[te], p_te, y[tr], p_tr))
        oof["learned BN"][te] = p_te
        # 3-way routing with the learned BN (diagnosis decides Search vs Clarify)
        tau = L.choose_threshold(p_tr, y[tr], 0.20)
        dec = L.decide(p_te, bn.diagnose(Xte), tau)
        oof_route[te] = dec
        for d in routes:
            routes[d] += list(y[te][dec == d])

        # robustness: retrieval signals unavailable at prediction time
        Xte_m, Xh_m = Xte.copy(), X_hand[te].copy()
        Xte_m[:, hide] = -1
        Xh_m[:, hide] = -1
        res_missing["learned BN"].append(metrics(y[te], bn.p_fail(Xte_m)))
        res_missing["hand-set BN"].append(metrics(y[te], handset.p_fail(Xh_m)))
        A_te_m = A_te.copy()
        for c in RETRIEVAL_RAW:
            A_te_m[:, RAW_COLS.index(c)] = mean[RAW_COLS.index(c)]
        res_missing["logistic reg."].append(metrics(y[te], lr.predict_proba(A_te_m)[:, 1]))
        lr2 = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)).fit(A_tr[:, keep_raw], y[tr])
        res_missing["logistic reg. (retrained w/o)"].append(metrics(y[te], lr2.predict_proba(A_te[:, keep_raw])[:, 1]))

    avg = lambda lst, key: np.nanmean([d[key] for d in lst])
    print(f"{'router':<20}{'AUROC':>8}{'Brier':>8}{'serve@20%':>11}{'(fail)':>8}{'fail@50%':>10}")
    for n in names:
        b = avg(res[n], "Brier")
        print(f"{n:<20}{avg(res[n], 'AUROC'):>8.3f}{'   -' if np.isnan(b) else f'{b:8.3f}':>8}"
              f"{avg(res[n], 'serve@20%'):>10.1%}{avg(res[n], 'fail_served'):>8.1%}{avg(res[n], 'fail@50%'):>10.1%}")
    print(f"\n(base rates: serving everything fails {y.mean():.1%})")

    print("\nRetrieval signals MISSING at prediction time (AUROC):")
    for n, lst in res_missing.items():
        print(f"   {n:<32}{avg(lst, 'AUROC'):.3f}")

    print("\nLearned BN, three-way routing on held-out questions:")
    tot = sum(len(v) for v in routes.values())
    for d, v in routes.items():
        if v:
            print(f"   {d:<8} {len(v):>5} questions ({len(v) / tot:.0%})   serving them now would fail {np.mean(v):.0%}")

    if len(set(ds)) > 1:
        print("\nPer dataset (held-out predictions, AUROC):")
        print(f"   {'dataset':<10}{'n':>6}{'fail':>7}" + "".join(f"{n:>19}" for n in oof))
        for d in sorted(set(ds)):
            m = ds == d
            if len(set(y[m])) < 2:
                continue
            print(f"   {d:<10}{m.sum():>6}{y[m].mean():>7.0%}" +
                  "".join(f"{roc_auc_score(y[m], oof[n][m]):>19.3f}" for n in oof))
        print("\nRouting per dataset (learned BN):")
        for d in sorted(set(ds)):
            m = ds == d
            parts = []
            for r in ("Serve", "Search", "Clarify"):
                k = m & (oof_route == r)
                parts.append(f"{r} {k.sum() / m.sum():4.0%} (fail {y[k].mean() if k.any() else float('nan'):.0%})")
            print(f"   {d:<10} " + "   ".join(parts))


if __name__ == "__main__":
    main()
