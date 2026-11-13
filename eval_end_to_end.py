"""
End-to-end: plain RAG vs RAG with the Bayesian router
------------------------------------------------------
Same questions, same LLM, same knowledge base, three systems:

  A  Naive RAG      retrieve from the KB, answer. Always.
  B  Reactive RAG   A, and if the model says "I don't know", search Wikipedia and answer again
                    (repair AFTER a wasted generation)
  C  Routed RAG     ours: the Bayesian network decides BEFORE generating:
                      Serve   -> answer from the KB
                      Search  -> search Wikipedia first, then answer once
                      Clarify -> (no user in a benchmark) treated like Search

Routing uses 5-fold cross-validation (each question is routed by a network learned WITHOUT
it), so nothing is evaluated on its own training data.
Reported: correct / wrong (hallucinated) / "I don't know", LLM calls and time per question.

    python eval_end_to_end.py labels/Qwen_Qwen2.5-7B-Instruct-bfloat16.jsonl --n 200
    (needs the Ollama app running with qwen2.5:7b, and internet for Wikipedia)
"""

import argparse
import os

# macOS: torch, faiss and scikit-learn each bring an OpenMP runtime; several at once can segfault
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import json
import random
import time
import warnings

warnings.filterwarnings("ignore")

import numpy as np
from sklearn.model_selection import StratifiedKFold

import learn_bn as L
from search_action import search_pool
from feature_extractors import FeatureExtractor, KnowledgeBase
from label_dataset import grade
from llm_backends import build_prompt, make_backend


def oof_routes(rows):
    y = np.array([L.failed(r) for r in rows])
    route = np.empty(len(rows), dtype=object)
    p_out = np.zeros(len(rows))
    for tr, te in StratifiedKFold(5, shuffle=True, random_state=0).split(y, y):
        rtr = [rows[i] for i in tr]
        cuts = L.learn_cutoffs(rtr)
        Xtr, Xte = L.evidence_matrix(rtr, cuts), L.evidence_matrix([rows[i] for i in te], cuts)
        bn = L.FastBN(L.em(Xtr, y[tr], verbose=False), L.LEVELS)
        tau = L.choose_threshold(bn.p_fail(Xtr), y[tr], 0.20)
        p = bn.p_fail(Xte)
        route[te] = L.decide(p, bn.diagnose(Xte), tau)
        p_out[te] = p
    return route, p_out


def outcome(g):
    return "correct" if g["correct"] else "idk" if (g["abstained"] or g["empty"]) else "wrong"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("labels")
    ap.add_argument("--n", type=int, default=200, help="questions to evaluate (random sample)")
    ap.add_argument("--kb", default="kb")
    ap.add_argument("--llm", default="openai:qwen2.5:7b@http://localhost:11434/v1")
    ap.add_argument("--out", default=None, help="default: labels/end_to_end_<search>.jsonl")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--search", choices=["better", "simple"], default="better",
                    help="the Search action: better = entity search + full pages (v2); simple = v1")
    ap.add_argument("--fake-models", action="store_true", help=argparse.SUPPRESS)   # offline test
    args = ap.parse_args()

    args.out = args.out or f"labels/end_to_end_{args.search}.jsonl"
    rows = [json.loads(l) for l in open(args.labels, encoding="utf-8")]
    rows = [r for r in rows if not r.get("empty")]
    print("Routing every question with a network learned without it (5-fold) ...")
    route, p_fail = oof_routes(rows)
    idx = sorted(random.Random(args.seed).sample(range(len(rows)), args.n))
    print("routes in the sample:", {r: int(sum(route[i] == r for i in idx)) for r in ("Serve", "Search", "Clarify")})

    kb = KnowledgeBase.load(args.kb)
    popularity = json.load(open(f"{args.kb}/popularity.json"))
    if args.fake_models:
        import test_extractors as t
        fx = FeatureExtractor(kb, embed=t.fake_embed, rerank=t.fake_rerank, nlp=t.nlp, popularity=popularity)
    else:
        fx = FeatureExtractor(kb, popularity=popularity)
    fx.extract("warm-up question?")
    llm = make_backend(args.llm)

    def ask(question, chunks, scores):
        top = [c["text"] for _, c in sorted(zip(scores, chunks), key=lambda x: -x[0])[:3]]
        t0 = time.time()
        reply = llm.generate(build_prompt(question, top, 3), max_new_tokens=48)
        return reply, time.time() - t0

    def searched(question):
        t0 = time.time()
        pool, _ = search_pool(question, fx, kb, args.search)
        pool = pool[:15]
        scores = list(fx.rerank(question, [c["text"] for c in pool]))
        return pool, scores, time.time() - t0

    stats = {s: {"correct": 0, "wrong": 0, "idk": 0, "calls": 0, "secs": 0.0} for s in "ABC"}
    with open(args.out, "w", encoding="utf-8") as out:
        for k, i in enumerate(idx, 1):
            r = rows[i]
            q = r["question"]
            t0 = time.time()
            res = fx.extract(q)
            t_route = time.time() - t0
            # A: naive
            a_reply, a_t = ask(q, res.details["chunks"], res.details["rerank_scores"])
            a = outcome(grade(a_reply, r["answers"]))
            # search answer (needed by B when A says "I don't know", and by C when not Serve)
            s_reply = s_t = None
            if a == "idk" or route[i] != "Serve":
                pool, scores, t_search = searched(q)
                s_reply, s_llm = ask(q, pool, scores)
                s_t = t_search + s_llm
                s_out = outcome(grade(s_reply, r["answers"]))
            # B: reactive
            b = s_out if a == "idk" else a
            stats["B"]["calls"] += 1 + (a == "idk")
            stats["B"]["secs"] += t_route + a_t + (s_t if a == "idk" else 0)
            # C: routed (routing cost included)
            if route[i] == "Serve":
                c, c_t, calls = a, t_route + a_t, 1
            else:
                c, c_t, calls = s_out, t_route + s_t, 1
            stats["C"]["calls"] += calls
            stats["C"]["secs"] += c_t
            stats["A"]["calls"] += 1
            stats["A"]["secs"] += t_route + a_t
            for s, o in (("A", a), ("B", b), ("C", c)):
                stats[s][o] += 1
            out.write(json.dumps({"id": r["id"], "question": q, "answers": r["answers"], "route": route[i],
                                  "p_fail": float(p_fail[i]), "naive": [a_reply, a], "searched": [s_reply,
                                  s_out if s_reply is not None else None], "reactive": b, "routed": c},
                                 ensure_ascii=False) + "\n")
            if k % 20 == 0 or k == len(idx):
                print(f"  {k}/{len(idx)}  " + "  ".join(f"{s}: {st['correct'] / k:.0%} correct" for s, st in stats.items()))

    n = len(idx)
    names = {"A": "A  naive RAG", "B": "B  reactive (search after 'I don't know')", "C": "C  routed (ours)"}
    print(f"\n{n} questions, same LLM ({args.llm.split('@')[0]}), same KB, search action: {args.search}")
    print(f"{'system':<44}{'correct':>9}{'wrong':>8}{'IDK':>7}{'LLM calls/q':>13}{'sec/q':>8}")
    for s, st in stats.items():
        print(f"{names[s]:<44}{st['correct'] / n:>9.1%}{st['wrong'] / n:>8.1%}{st['idk'] / n:>7.1%}"
              f"{st['calls'] / n:>13.2f}{st['secs'] / n:>8.1f}")
    print(f"\nper-question results -> {args.out}")


if __name__ == "__main__":
    main()
