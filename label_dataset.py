"""
Step 2 - Label questions: features + LLM answer + hallucinated yes/no
---------------------------------------------------------------------
For every question in kb/questions.jsonl:
  1. measure the 14 inputs (feature_extractors.py) and retrieve passages
  2. ask the LLM with the top passages (llm_backends.py)
  3. grade the answer against the dataset's accepted answers:
        correct      - an accepted answer appears in the reply
        abstained    - the model said "I don't know"
        hallucinated - neither: it answered, and the answer is wrong
  4. append one JSON row to the output file (safe to stop and resume)

Pilot first (measures speed and answer quality before committing to 5,000):
  python label_dataset.py --llm qwen2.5-7b --limit 200

Full run on the same file later just continues where the pilot stopped:
  python label_dataset.py --llm qwen2.5-7b

Outputs: labels/<llm name>.jsonl (rows) and labels/<llm name>.profile.json
(which models produced them; learned parameters are only valid for this profile).
"""

import argparse
import json
import math
import os
import re
import statistics
import string
import time
import warnings

warnings.filterwarnings("ignore")

from bayesian_epistemic_routing import extract_evidence, route_query
from feature_extractors import ExtractorConfig, FeatureExtractor, KnowledgeBase
from llm_backends import build_prompt, make_backend


def normalize(s: str) -> str:
    """SQuAD-style: lower-case, drop punctuation and articles, squeeze spaces."""
    s = s.lower()
    s = "".join(ch for ch in s if ch not in set(string.punctuation))
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())


STOP = {"and", "of", "in", "on", "for", "to", "by", "with", "or", "jr", "de", "genre"}


def _close_match(r: str, a: str) -> bool:
    """
    Lenient but conservative matches that exact matching misses:
      * the reply is a shortened form of the accepted answer and covers at least half of its
        words: "Cal Sr." ~ "Cal Ripken Sr.", "Ptolemy XII" ~ "Ptolemy XII Auletes"
      * near-identical spelling (>= 90% similar, both 6+ letters):
        "Nagathihalli Chandrashekar" ~ "Nagathihalli Chandrashekhar"
    """
    from difflib import SequenceMatcher
    rt = [t for t in r.split() if t not in STOP]
    at = [t for t in a.split() if t not in STOP]
    if rt and at and set(rt) <= set(at) and len(set(rt)) / len(set(at)) >= 0.5 and max(map(len, rt)) >= 3:
        return True
    return len(r) >= 6 and len(a) >= 6 and SequenceMatcher(None, r, a).ratio() >= 0.9


def grade(reply: str, answers: list[str]) -> dict:
    """
    correct      - an accepted answer appears in the reply (whole words), or the reply is a
                   close form of an accepted answer (see _close_match)
    abstained    - the model explicitly said it doesn't know
    hallucinated - it answered, wrongly
    empty        - no answer text at all: a generation problem, NOT a model judgement;
                   such rows are excluded from learning and flagged in the summary
    """
    r = normalize(reply)
    empty = r == ""
    # an explicit "I don't know" is never counted as correct (a short alias could appear inside it)
    abstained = (not empty) and ("i dont know" in r or "i do not know" in r or r == "unknown")
    padded, squeezed = f" {r} ", r.replace(" ", "")   # whole-word match; squeezed catches "EmperorTenmu"
    golds = [normalize(a) for a in answers if normalize(a)]
    correct = (not empty) and (not abstained) and (
        any(f" {a} " in padded or (len(a) >= 6 and a.replace(" ", "") in squeezed) for a in golds)
        or any(_close_match(r, a) for a in golds))
    return {"correct": correct, "abstained": abstained, "empty": empty,
            "hallucinated": not (correct or abstained or empty)}


def summarize(rows: list[dict]) -> None:
    n = len(rows)
    if not n:
        return
    pct = lambda k, rs: 100 * sum(r[k] for r in rs) / max(len(rs), 1)
    print(f"\n{n} questions labelled")
    empty = sum(r.get("empty", normalize(r["reply"]) == "") for r in rows)
    if empty:
        print(f"  WARNING: {empty} empty replies ({100 * empty / n:.0f}%) - a generation problem, not model "
              "abstentions; they are excluded from learning")
    print(f"  correct {pct('correct', rows):.1f}%   abstained {pct('abstained', rows):.1f}%   "
          f"hallucinated {pct('hallucinated', rows):.1f}%")
    for flag, label in ((True, "page IN knowledge base"), (False, "page NOT in knowledge base")):
        sub = [r for r in rows if r.get("in_kb") == flag]
        if sub:
            print(f"  {label:<28} n={len(sub):<5} correct {pct('correct', sub):5.1f}%  "
                  f"hallucinated {pct('hallucinated', sub):5.1f}%")
    bands = {}
    for r in rows:
        if r.get("s_pop") is not None:
            bands.setdefault(int(math.log10(max(1, r["s_pop"]))), []).append(r)
    if bands:
        print("  by subject popularity (monthly page views):")
        for b in sorted(bands):
            sub = bands[b]
            print(f"    10^{b}-10^{b + 1}  n={len(sub):<5} correct {pct('correct', sub):5.1f}%  "
                  f"hallucinated {pct('hallucinated', sub):5.1f}%")
    routes = {}
    for r in rows:
        routes.setdefault(r["route_v2_placeholder"], []).append(r)
    print("  current (hand-set) router vs reality:")
    for route in ("Serve", "Search", "Clarify"):
        sub = routes.get(route, [])
        if sub:
            print(f"    routed {route:<8} n={len(sub):<5} hallucinated {pct('hallucinated', sub):5.1f}%")
    gen = [r["timing_ms"]["generate"] for r in rows]
    feat = [r["timing_ms"]["features"] for r in rows]
    per_q = statistics.median(gen) + statistics.median(feat)
    print(f"  time per question (median): features {statistics.median(feat):.0f} ms + LLM {statistics.median(gen):.0f} ms")
    print(f"  -> 5,000 questions would take about {5000 * per_q / 3.6e6:.1f} hours on this machine")
    ink = [r for r in rows if r.get("in_kb")]
    if ink:
        print(f"  retrieval: gold page in top passages for {pct('gold_page_retrieved', ink):.0f}% of in-KB questions")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--llm", default="qwen2.5-7b",
                    help="qwen2.5-7b[-nf4] | llama3.1-8b | other names in llm_backends.MODEL_ALIASES | hf:<id> | "
                         "stripedhyena[-nf4][-noflash] | openai:<model>@<url> | echo")
    ap.add_argument("--compute-dtype", default=None, help="4-bit compute dtype: float16 | bfloat16 (match the DL project)")
    ap.add_argument("--param-dtype", default=None,
                    help="precision of the non-4-bit parts (default = compute dtype); float32 = safest on T4")
    ap.add_argument("--double-quant", action="store_true", help="bitsandbytes double quantisation (match the DL project)")
    ap.add_argument("--adapter", default=None, help="LoRA/PEFT adapter the DL project fine-tuned, if any")
    ap.add_argument("--kb", default="kb")
    ap.add_argument("--questions", default=None, help="default: <kb>/questions.jsonl")
    ap.add_argument("--out", default="labels")
    ap.add_argument("--limit", type=int, default=0, help="label at most this many (0 = all)")
    ap.add_argument("--passages", type=int, default=3, help="passages given to the LLM")
    ap.add_argument("--fake-models", action="store_true", help="offline test: stand-in embedder/reranker")
    ap.add_argument("--checkpoint", default=None, metavar="REPO",
                    help="back up labels to a private HF dataset ('auto' = <user>/prob-reas-labels); needs HF_TOKEN")
    ap.add_argument("--checkpoint-every", type=int, default=50, help="questions between backups")
    ap.add_argument("--shard", type=int, default=0, help="this machine's share (0-based) when splitting over accounts")
    ap.add_argument("--num-shards", type=int, default=1, help="number of machines/accounts sharing the work")
    args = ap.parse_args()
    ckpt = None
    if args.checkpoint:
        from checkpoint import HFCheckpoint
        ckpt = HFCheckpoint(args.checkpoint, args.checkpoint_every)

    kb = KnowledgeBase.load(args.kb)
    with open(os.path.join(args.kb, "popularity.json"), encoding="utf-8") as f:
        popularity = json.load(f)
    cfg = ExtractorConfig()
    kb_profile_path = os.path.join(args.kb, "profile.json")
    if os.path.exists(kb_profile_path):
        kb_profile = json.load(open(kb_profile_path))
        if kb_profile.get("embedding_model") != cfg.embedding_model and not args.fake_models:
            raise SystemExit(f"KB was embedded with {kb_profile.get('embedding_model')} but the extractor "
                             f"uses {cfg.embedding_model}. Rebuild the KB or change ExtractorConfig.")

    if args.fake_models:
        import test_extractors as t
        fx = FeatureExtractor(kb, embed=t.fake_embed, rerank=t.fake_rerank, nlp=t.nlp, popularity=popularity)
    else:
        fx = FeatureExtractor(kb, popularity=popularity, config=cfg)

    # Load (and on first run download) the extractor models BEFORE timing anything,
    # so question 1 isn't charged with a 50-second download.
    fx.extract("warm-up question about Python?")
    print(f"Loading LLM backend {args.llm!r} ...")
    llm_kwargs = {}
    if not args.llm.startswith(("openai:", "echo")):
        llm_kwargs = {k: v for k, v in {"compute_dtype": args.compute_dtype, "dtype": args.param_dtype,
                                        "adapter": args.adapter}.items() if v}
        if args.double_quant:
            llm_kwargs["double_quant"] = True
    llm = make_backend(args.llm, **llm_kwargs)
    tag = re.sub(r"[^\w.-]+", "_", llm.name)
    os.makedirs(args.out, exist_ok=True)
    kb_name = os.path.basename(os.path.normpath(args.kb))
    if kb_name != "kb":                      # another question set / KB -> its own file name
        tag += "__" + kb_name
    if args.num_shards > 1:                  # each account writes (and backs up) its own part
        tag += f".part{args.shard}of{args.num_shards}"
    out_path = os.path.join(args.out, f"{tag}.jsonl")
    profile = {"llm": llm.name, "embedding_model": "fake" if args.fake_models else cfg.embedding_model,
               "reranker_model": "fake" if args.fake_models else cfg.reranker_model,
               "spacy_model": cfg.spacy_model, "kb": os.path.abspath(args.kb),
               "kb_passages": len(kb.docs), "passages_in_prompt": args.passages}
    with open(os.path.join(args.out, f"{tag}.profile.json"), "w") as f:
        json.dump(profile, f, indent=2)

    profile_path = os.path.join(args.out, f"{tag}.profile.json")
    if ckpt:
        ckpt.pull(out_path)                     # continue from the backup if this machine is a fresh one
        ckpt.watch(out_path, profile_path)

    q_path = args.questions or os.path.join(args.kb, "questions.jsonl")
    questions = [json.loads(line) for line in open(q_path, encoding="utf-8")]
    done = {}
    if os.path.exists(out_path):
        for line in open(out_path, encoding="utf-8"):
            row = json.loads(line)
            done[row["id"]] = row
    questions = [q for k, q in enumerate(questions) if k % args.num_shards == args.shard]
    todo = [q for q in questions if q["id"] not in done]
    if args.limit:
        todo = todo[:max(0, args.limit - len(done))]
    print(f"{len(done)} already labelled, labelling {len(todo)} now -> {out_path}")

    with open(out_path, "a", encoding="utf-8") as out:
        for i, q in enumerate(todo, 1):
            t0 = time.perf_counter()
            res = fx.extract(q["question"], history=q.get("history", []))
            ranked = sorted(zip(res.details["rerank_scores"], res.details["chunks"]),
                            key=lambda x: -x[0])
            passages = [c["text"] for _, c in ranked]
            t1 = time.perf_counter()
            reply = llm.generate(build_prompt(q["question"], passages, args.passages, q.get("conversation")))
            t2 = time.perf_counter()

            evidence = extract_evidence(res.raw)
            row = {
                "id": q["id"], "dataset": q.get("dataset"), "question": q["question"],
                "answers": q["answers"], "in_kb": q.get("in_kb"), "s_pop": q.get("s_pop"),
                "history": q.get("history", []), "turn": q.get("turn"), "qtype": q.get("type"),
                "raw": res.raw, "evidence": evidence,
                "retrieved_titles": [c.get("title") for _, c in ranked[:args.passages]],
                "gold_page_retrieved": q.get("subject_title") in {c.get("title") for _, c in ranked[:args.passages]},
                "reply": reply, "raw_reply": getattr(llm, "last_raw", None), **grade(reply, q["answers"]),
                "route_v2_placeholder": route_query(evidence)["route"],
                "timing_ms": {"features": round(1000 * (t1 - t0)), "generate": round(1000 * (t2 - t1))},
            }
            out.write(json.dumps(row, ensure_ascii=False) + "\n")
            out.flush()
            done[row["id"]] = row
            if i % 25 == 0 or i == len(todo):
                print(f"  {i}/{len(todo)}  last: {q['question']!r} -> {reply!r} "
                      f"({'correct' if row['correct'] else 'abstained' if row['abstained'] else 'WRONG'})")
            if ckpt:
                ckpt.maybe_push(i)

    if ckpt:
        ckpt.push()                             # final backup

    summarize(list(done.values()))


if __name__ == "__main__":
    main()
