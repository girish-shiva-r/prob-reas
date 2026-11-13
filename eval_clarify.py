"""
Does asking the user help?  Clarify, end to end, on CoQA follow-up questions
---------------------------------------------------------------------------
CoQA follow-ups ("What did she buy?") are where a raw question retrieves badly: the
pronoun hides the subject. Same LLM, same knowledge base, same conversation shown to the LLM:

  A  Naive conversational RAG   retrieve with the raw follow-up, answer
  B  Always rewrite             retrieve with "previous question + previous answer + follow-up"
                                for EVERY follow-up (a common fixed rule, no router)
  C  Ours (full conversational pipeline)
                                  1. a pronoun whose target the previous turn names unambiguously is
                                     filled in automatically (clarify_loop.resolve_from_history), no question
                                  2. otherwise the Bayesian network routes:
                                     Serve / Search -> as A
                                     Clarify -> ask the user; the SIMULATED user replies with what the
                                                conversation was about (previous question and answer - which
                                                the real user knows); the reply is merged into the query
                                                (clarify_loop.merge_clarification) and retrieval is redone
NOTE: in CoQA the system already has the history, so B can rewrite without asking anyone. B is
therefore a strong baseline here; Clarify earns its keep when the history does NOT resolve the
question (no history, or several candidates) - which this benchmark contains little of.

Routing uses 5-fold cross-validation over ALL labelled questions (PopQA + hybrid).
Reported: correct / wrong / "I don't know", and how often the user had to answer a question.

    python eval_clarify.py labels/<popqa>.jsonl labels/<hybrid merged>.jsonl --kb kb_hybrid --n 200
    (kb_hybrid: run `python build_hybrid.py --hotpot 1500 --coqa-stories 250` once on the Mac -
     same seed, so it is identical to the KB the labels were made with)
"""

import argparse
import os

# macOS: torch, faiss and scikit-learn each bring an OpenMP runtime; several at once can segfault
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import json
import random
import warnings

warnings.filterwarnings("ignore")

import numpy as np
from sklearn.model_selection import StratifiedKFold

import learn_bn as L
from clarify_loop import merge_clarification, resolve_from_history
from feature_extractors import FeatureExtractor, KnowledgeBase
from label_dataset import grade
from llm_backends import build_prompt, make_backend


def outcome(g):
    return "correct" if g["correct"] else "idk" if (g["abstained"] or g["empty"]) else "wrong"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("labels", nargs="+")
    ap.add_argument("--kb", default="kb_hybrid")
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--llm", default="openai:qwen2.5:7b@http://localhost:11434/v1")
    ap.add_argument("--out", default=None, help="default: labels/clarify_eval[_convsearch].jsonl")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--conv-search", action="store_true",
                    help="(added after the first run) the Search action uses the conversation: for a follow-up it "
                         "re-retrieves with 'previous turn + question' instead of the raw question")
    ap.add_argument("--fake-models", action="store_true", help=argparse.SUPPRESS)
    args = ap.parse_args()

    args.out = args.out or ("labels/clarify_eval_convsearch.jsonl" if args.conv_search else "labels/clarify_eval.jsonl")
    rows = [json.loads(l) for p in args.labels for l in open(p, encoding="utf-8")]
    rows = [r for r in rows if not r.get("empty")]
    y = np.array([L.failed(r) for r in rows])
    route = np.empty(len(rows), dtype=object)
    print(f"Routing {len(rows)} questions with networks learned without them (5-fold) ...")
    for tr, te in StratifiedKFold(5, shuffle=True, random_state=0).split(y, y):
        rtr = [rows[i] for i in tr]
        cuts = L.learn_cutoffs(rtr)
        Xtr, Xte = L.evidence_matrix(rtr, cuts), L.evidence_matrix([rows[i] for i in te], cuts)
        bn = L.FastBN(L.em(Xtr, y[tr], verbose=False), L.LEVELS)
        tau = L.choose_threshold(bn.p_fail(Xtr), y[tr], 0.20)
        route[te] = L.decide(bn.p_fail(Xte), bn.diagnose(Xte), tau)

    followups = [i for i, r in enumerate(rows) if r.get("dataset") == "coqa" and (r.get("turn") or 0) >= 1]
    for d in ("popqa", "hotpotqa", "coqa"):
        m = [i for i, r in enumerate(rows) if r.get("dataset") == d]
        if m:
            print(f"   {d:<9} routed Clarify: {np.mean([route[i] == 'Clarify' for i in m]):.1%}")
    print(f"   CoQA follow-ups routed Clarify: {np.mean([route[i] == 'Clarify' for i in followups]):.1%}")
    idx = sorted(random.Random(args.seed).sample(followups, min(args.n, len(followups))))

    convo = {}
    for line in open(f"{args.kb}/questions.jsonl", encoding="utf-8"):
        q = json.loads(line)
        convo[q["id"]] = q.get("conversation", [])
    kb = KnowledgeBase.load(args.kb)
    pop = json.load(open(f"{args.kb}/popularity.json"))
    if args.fake_models:
        import test_extractors as t
        fx = FeatureExtractor(kb, embed=t.fake_embed, rerank=t.fake_rerank, nlp=t.nlp, popularity=pop)
    else:
        fx = FeatureExtractor(kb, popularity=pop)
    fx.extract("warm-up question about Paris?")          # load spaCy, embedder and reranker up front
    llm = make_backend(args.llm)

    def answer(query, question, history, conversation):
        res = fx.extract(query, history=history)
        top = [c["text"] for _, c in sorted(zip(res.details["rerank_scores"], res.details["chunks"]),
                                             key=lambda x: -x[0])[:3]]
        return llm.generate(build_prompt(question, top, 3, conversation), max_new_tokens=48), res

    stats = {s: {"correct": 0, "wrong": 0, "idk": 0} for s in "ABC"}
    asked = n_auto = 0
    with open(args.out, "w", encoding="utf-8") as out:
        for k, i in enumerate(idx, 1):
            r = rows[i]
            q, hist, conv = r["question"], r.get("history", []), convo.get(r["id"], [])
            a_reply, res = answer(q, q, hist, conv)
            a = outcome(grade(a_reply, r["answers"]))
            prev = f"{conv[-1][0]} {conv[-1][1]}" if conv else ""
            b_reply, _ = answer(f"{prev} {q}".strip(), q, hist, conv)
            b = outcome(grade(b_reply, r["answers"]))
            resolved, auto = resolve_from_history(q, hist, fx.nlp, fx.popularity)
            if auto:                                               # filled in from history, nobody asked
                merged = resolved
                c_reply, _ = answer(merged, q, hist, conv)
                c = outcome(grade(c_reply, r["answers"]))
                n_auto += 1
            elif route[i] == "Clarify" and conv:
                asked += 1
                merged = merge_clarification(q, prev, "FollowUpStatus", res.details, nlp=fx.nlp)   # keep all reply words
                c_reply, _ = answer(merged, q, hist, conv)
                c = outcome(grade(c_reply, r["answers"]))
            elif args.conv_search and route[i] == "Search" and conv:
                merged, c_reply, c = f"{prev} {q}", b_reply, b          # same retrieval as B: reuse its answer
            else:
                merged, c_reply, c = None, a_reply, a
            for s, o in (("A", a), ("B", b), ("C", c)):
                stats[s][o] += 1
            out.write(json.dumps({"id": r["id"], "question": q, "answers": r["answers"], "route": route[i],
                                  "A": [a_reply, a], "B": [b_reply, b], "C": [c_reply, c], "merged": merged},
                                 ensure_ascii=False) + "\n")
            if k % 20 == 0 or k == len(idx):
                print(f"  {k}/{len(idx)}  " + "  ".join(f"{s}: {v['correct'] / k:.0%}" for s, v in stats.items()))

    n = len(idx)
    names = {"A": "A  naive (raw follow-up)", "B": "B  always rewrite with previous turn", "C": "C  ours (clarify when routed)"}
    print(f"\n{n} CoQA follow-up questions")
    print(f"{'system':<40}{'correct':>9}{'wrong':>8}{'IDK':>7}{'user asked':>12}")
    for s, v in stats.items():
        ask = f"{asked / n:.0%}" if s == "C" else ("0%" if s == "A" else "0%")
        print(f"{names[s]:<40}{v['correct'] / n:>9.1%}{v['wrong'] / n:>8.1%}{v['idk'] / n:>7.1%}{ask:>12}")
    print(f"(ours: {n_auto / n:.0%} resolved automatically from the history, {asked / n:.0%} asked the user)")
    print(f"\nper-question results -> {args.out}")


if __name__ == "__main__":
    main()
