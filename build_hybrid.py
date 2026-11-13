"""
Hybrid dataset: questions PopQA cannot provide
----------------------------------------------
PopQA only has one kind of question (simple, one fact, one entity), so half of the
network never sees anything happen. This adds two datasets that exercise the rest:

  HotpotQA (distractor, validation)  multi-hop questions ("Which magazine was started first,
                                     X or Y?") -> TopicComplexity, QuestionType, ContextDeficit.
                                     KB = each question's 10 Wikipedia paragraphs (2 gold +
                                     8 distractors). 25% of questions have their GOLD paragraphs
                                     left out of the KB (knowledge gaps, as for PopQA).
  CoQA (validation)                  CONVERSATIONS about a passage ("Where did she go?" ...
                                     "Did he come back?") -> follow-ups, unresolved pronouns,
                                     FollowUpStatus / UnresolvedReferences / QueryReformulation.
                                     KB = the stories; 20% of stories left out of the KB.
                                     Each question carries its conversation so far: the router
                                     sees the previous questions, the LLM sees the previous Q&A.

Output folder (default kb_hybrid/): docs.jsonl, embeddings.npy, centroids.npy, questions.jsonl,
popularity.json (null: page views unknown for these datasets), profile.json.
Deterministic (fixed seed): every Kaggle account that runs this builds the identical KB.

    python build_hybrid.py                     # on Kaggle (needs internet for the datasets)
"""

import argparse
import json
import os
import random
import re
import time


def chunk_text(title, text, max_words=100):
    sents = re.split(r"(?<=[.!?])\s+", re.sub(r"\s+", " ", text).strip())
    chunks, cur = [], []
    for s in sents:
        if cur and len(" ".join(cur + [s]).split()) > max_words:
            chunks.append(" ".join(cur))
            cur = []
        cur.append(s)
    if cur:
        chunks.append(" ".join(cur))
    return [f"{title}: {c}" for c in chunks if len(c.split()) >= 5]


def load_hotpot():
    from datasets import load_dataset
    for name in ("hotpotqa/hotpot_qa", "hotpot_qa"):
        try:
            return load_dataset(name, "distractor", split="validation")
        except Exception as e:
            print(f"  could not load {name}: {type(e).__name__}: {str(e)[:120]}")
    raise SystemExit("HotpotQA could not be loaded (internet on?)")


def load_coqa():
    from datasets import load_dataset
    return load_dataset("stanfordnlp/coqa", split="validation")


def build_hotpot(n, holdout, rng):
    ds = load_hotpot()
    idx = sorted(rng.sample(range(len(ds)), n))
    held = set(rng.sample(idx, int(n * holdout)))
    paragraphs, questions = {}, []
    for i in idx:
        ex = ds[i]
        gold = list(dict.fromkeys(ex["supporting_facts"]["title"]))
        for title, sents in zip(ex["context"]["title"], ex["context"]["sentences"]):
            if i in held and title in gold:
                continue                                   # knowledge gap: the gold paragraphs are missing
            paragraphs.setdefault(title, " ".join(sents))
        questions.append({"id": f"hotpot-{ex['id']}", "dataset": "hotpotqa", "question": ex["question"],
                          "answers": [ex["answer"]], "subject_title": gold[0] if gold else None,
                          "gold_titles": gold, "type": ex["type"], "level": ex["level"],
                          "in_kb": i not in held, "history": [], "s_pop": None})
    # a gold paragraph of a held-out question may still be in the KB via another question
    for q in questions:
        if not q["in_kb"] and all(t in paragraphs for t in q["gold_titles"]):
            q["in_kb"] = True
    docs = []
    for title, text in paragraphs.items():
        for j, c in enumerate(chunk_text(title, text)):
            docs.append({"id": f"hotpot:{title}#{j}", "title": title, "text": c, "source_type": "wikipedia"})
    return docs, questions


def build_coqa(n_stories, max_turns, holdout, rng):
    ds = load_coqa()
    idx = sorted(rng.sample(range(len(ds)), n_stories))
    held = set(rng.sample(idx, int(n_stories * holdout)))
    docs, questions = [], []
    for i in idx:
        ex = ds[i]
        title = f"Story {i} ({ex['source']})"
        if i not in held:
            for j, c in enumerate(chunk_text(title, ex["story"])):
                docs.append({"id": f"coqa:{i}#{j}", "title": title, "text": c,
                             "source_type": "wikipedia" if ex["source"] == "wikipedia" else "unknown"})
        qs, ans = ex["questions"], ex["answers"]["input_text"]
        for t in range(min(max_turns, len(qs))):
            questions.append({"id": f"coqa-{i}-{t}", "dataset": "coqa", "question": qs[t], "answers": [ans[t]],
                              "subject_title": title, "in_kb": i not in held, "turn": t,
                              "history": qs[:t],                                  # what the router sees
                              "conversation": [[qs[k], ans[k]] for k in range(t)],  # what the LLM sees
                              "s_pop": None})
    return docs, questions


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--hotpot", type=int, default=600, help="HotpotQA questions")
    ap.add_argument("--coqa-stories", type=int, default=100, help="CoQA conversations")
    ap.add_argument("--coqa-turns", type=int, default=6, help="questions used per conversation")
    ap.add_argument("--holdout", type=float, default=0.2)
    ap.add_argument("--topics", type=int, default=16)
    ap.add_argument("--out", default="kb_hybrid")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    print("Loading HotpotQA ...")
    hd, hq = build_hotpot(args.hotpot, 0.25, random.Random(args.seed))
    print(f"  {len(hq)} questions, {len(hd)} passages, {sum(not q['in_kb'] for q in hq)} with gold paragraphs held out")
    print("Loading CoQA ...")
    cd, cq = build_coqa(args.coqa_stories, args.coqa_turns, args.holdout, random.Random(args.seed + 1))
    print(f"  {len(cq)} questions from {args.coqa_stories} conversations, {len(cd)} passages, "
          f"{sum(not q['in_kb'] for q in cq)} questions whose story is held out")

    docs, questions = hd + cd, hq + cq
    with open(os.path.join(args.out, "questions.jsonl"), "w", encoding="utf-8") as f:
        for q in questions:
            f.write(json.dumps(q, ensure_ascii=False) + "\n")
    with open(os.path.join(args.out, "popularity.json"), "w") as f:
        json.dump(None, f)                                   # no page-view data for these datasets

    from feature_extractors import ExtractorConfig, KnowledgeBase, load_default_embedder
    t0 = time.time()
    kb = KnowledgeBase(docs, load_default_embedder(), n_topics=args.topics, seed=args.seed)
    kb.save(args.out)
    with open(os.path.join(args.out, "profile.json"), "w") as f:
        json.dump({"embedding_model": ExtractorConfig().embedding_model, "passages": len(docs),
                   "questions": len(questions), "datasets": {"hotpotqa": len(hq), "coqa": len(cq)},
                   "seed": args.seed}, f, indent=2)
    print(f"Saved {args.out}/: {len(questions)} questions, {len(docs)} passages ({time.time() - t0:.0f}s to embed)")


if __name__ == "__main__":
    main()
