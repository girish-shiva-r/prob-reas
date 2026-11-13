"""
Live demo - Bayesian Epistemic Routing
--------------------------------------
Type a question. Before any answer is generated, the router:
  1. measures the 14 signals (query, retrieval, conversation, knowledge coverage)
  2. turns them into levels with the LEARNED cut-offs
  3. computes P(serving now fails) with the LEARNED Bayesian network, and - if risky -
     which bottleneck is to blame (diagnosis: posterior given that serving would fail)
  4. routes:
       Serve   -> the LLM answers from the retrieved passages
       Search  -> live Wikipedia search, re-measure, re-route, then answer
       Clarify -> asks you a targeted question; your reply is merged into the query

    ollama pull qwen2.5:7b          # once (about 4.7 GB); keep the Ollama app running
    python demo.py                            # interactive
    python demo.py --script                   # runs a prepared sequence of examples
    python demo.py --llm none                 # routing only, no answers

Commands while running:  /new  (new conversation)   /quit
"""

import argparse
import os

# macOS: torch, faiss and scikit-learn each bring an OpenMP runtime; several at once can segfault
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import json
import os
import re
import sys
import time
import warnings

warnings.filterwarnings("ignore")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np

import learn_bn as L
from clarify_loop import Conversation
from feature_extractors import FeatureExtractor, KnowledgeBase
from llm_backends import build_prompt, make_backend

WIKI_API = "https://en.wikipedia.org/w/api.php"
UA = "BayesianEpistemicRouting/0.1 (university course project demo)"

SCRIPT = [
    "Who was the director of The Inspector General?",
    "Who was the screenwriter for Rent?",
    "/new",
    "Who is the author of Babette's Feast?",
    "/new",
    "When was it released?",
    "the film Chinese Coffee",
    "/new",
    "Tell me about Larry Kramer.",
    "Where was he born?",
]

# ---------------------------------------------------------------- pretty printing
B, DIM, RST = "\033[1m", "\033[2m", "\033[0m"
COL = {"Serve": "\033[92m", "Search": "\033[93m", "Clarify": "\033[96m"}


def bar(p, width=24):
    n = int(round(p * width))
    return "█" * n + "░" * (width - n)


SHOW = [("RerankerScore", "top1_rerank_score", "best passage relevance (cross-encoder)", "{:.3f}"),
        ("FAISSVectorDistance", "top1_faiss_distance", "distance to nearest passage", "{:.3f}"),
        ("ChunkDensity", "relevant_chunk_count", "relevant passages found", "{}"),
        ("EntityPopularity", "entity_monthly_views", "monthly page views of the entity", "{:,.0f}"),
        ("DomainShift", "kb_domain_similarity", "similarity to the knowledge base's topics", "{:.2f}"),
        ("UnresolvedReferences", "unresolved_reference_count", "pronouns with no clear target", "{}"),
        ("FollowUpStatus", "is_ambiguous_followup", "ambiguous follow-up", "{}"),
        ("QueryLength", "word_count", "words in the question", "{}"),
        ("TopicComplexity", "sub_question_count", "sub-questions detected", "{}"),
        ("EntitySpecificity", "named_entity_count", "specific entities named", "{}"),
        ("TemporalSensitivity", "is_time_sensitive", "asks about recent events", "{}")]


def show_routing(res, tau, title="Routing"):
    print(f"\n{B}── {title} ──{RST}  {DIM}(query used for retrieval: {res['query']!r}){RST}")
    raw, ev = res["raw"], res["evidence"]
    for node, key, label, fmt in SHOW:
        v = raw.get(key)
        if v is None:
            val = "unknown"
        else:
            try:
                val = fmt.format(v)
            except (ValueError, TypeError):
                val = str(v)
        state = ev.get(node, "unknown")
        mark = "" if state in ("Low", "unknown") else f"  {B}← {state}{RST}"
        print(f"   {label:<44} {val:>10}   {node} = {state}{mark}")
    p = res["p_fail"]
    print(f"\n   P(serving now fails) = {B}{p:.2f}{RST}  {bar(p)}   (serve below {tau:.2f})")
    if res["route"] != "Serve":
        print(f"   {DIM}if it failed, the cause would be:{RST}")
        for b, v in sorted(res["blame"].items(), key=lambda x: -x[1]):
            print(f"      {b:<18} {v:.2f} {bar(v, 16)}")
    print(f"\n   Decision: {COL[res['route']]}{B}{res['route'].upper()}{RST}")
    if title.startswith("After search") and res["route"] != "Serve":
        print(f"   {DIM}(already searched / asked once - answering now, flagged as low confidence){RST}")


# ---------------------------------------------------------------- actions
def ranked_passages(details, n=3):
    pairs = sorted(zip(details["rerank_scores"], details["chunks"]), key=lambda x: -x[0])
    return [c for _, c in pairs[:n]]


def answer(llm, question, passages):
    if llm is None:
        return None
    t0 = time.time()
    reply = llm.generate(build_prompt(question, [c["text"] for c in passages], 3), max_new_tokens=48)
    return reply, time.time() - t0


def wiki_search(query, k=3):
    """Live Wikipedia search -> intro passages (the 'Search' action)."""
    import requests
    s = requests.Session()
    s.headers["User-Agent"] = UA
    r = s.get(WIKI_API, params={"action": "query", "list": "search", "srsearch": query, "srlimit": k,
                                "format": "json"}, timeout=10)
    titles = [h["title"] for h in r.json()["query"]["search"]]
    if not titles:
        return []
    r = s.get(WIKI_API, params={"action": "query", "prop": "extracts", "exintro": 1, "explaintext": 1,
                                "redirects": 1, "titles": "|".join(titles), "format": "json",
                                "formatversion": 2}, timeout=10)
    out = []
    for p in r.json()["query"]["pages"]:
        text = re.sub(r"\s+", " ", p.get("extract", "") or "").strip()
        sents = re.split(r"(?<=[.!?])\s+", text)
        chunk, chunks = [], []
        for sent in sents:
            if chunk and len(" ".join(chunk + [sent]).split()) > 100:
                chunks.append(" ".join(chunk))
                chunk = []
            chunk.append(sent)
        if chunk:
            chunks.append(" ".join(chunk))
        out += [{"title": p["title"], "text": f"{p['title']}: {c}", "source_type": "wikipedia", "web": True}
                for c in chunks if len(c.split()) >= 8]
    return out


def search_and_reroute(fx, kb, conv, res, router, tau, use_web):
    if not use_web:
        print(f"   {DIM}(live search disabled){RST}")
        return res
    q = res["query"]
    from search_action import search_pool
    pool, web = search_pool(q, fx, kb, "better")
    print(f"   searched Wikipedia: {', '.join(sorted({w['title'] for w in web})) or 'nothing found'}")
    if not web:
        return res
    pool = pool[:15]
    scores = fx.rerank(q, [c["text"] for c in pool])
    pool = [c for _, c in sorted(zip(scores, pool), key=lambda x: -x[0])]   # best passages first
    result = fx.extract(q, history=conv.history[:-1], retrieved=pool[:fx.cfg.retrieve_k])
    new = {**router(result.raw), "raw": result.raw, "details": result.details, "query": q}
    show_routing(new, tau, "After search: re-measured")
    return new


# ---------------------------------------------------------------- main loop
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--profile", default="profiles/Qwen_Qwen2.5-7B-Instruct-bfloat16.json")
    ap.add_argument("--llm", default="openai:qwen2.5:7b@http://localhost:11434/v1",
                    help="answering model (default: Qwen2.5-7B via Ollama); 'none' = routing only")
    ap.add_argument("--kb", default="kb")
    ap.add_argument("--no-web", action="store_true", help="Search action without live Wikipedia")
    ap.add_argument("--script", action="store_true", help="run the prepared examples")
    ap.add_argument("--max-clarify", type=int, default=1,
                    help="clarifying questions per user question before searching anyway (default 1)")
    ap.add_argument("--fake-models", action="store_true", help=argparse.SUPPRESS)   # offline test
    args = ap.parse_args()

    import clarify_loop
    clarify_loop.MAX_CLARIFY_ROUNDS = args.max_clarify
    print("Loading knowledge base, signal extractors and the learned network ...")
    kb = KnowledgeBase.load(args.kb)
    popularity = json.load(open(os.path.join(args.kb, "popularity.json")))
    if args.fake_models:
        import test_extractors as t
        fx = FeatureExtractor(kb, embed=t.fake_embed, rerank=t.fake_rerank, nlp=t.nlp, popularity=popularity)
    else:
        fx = FeatureExtractor(kb, popularity=popularity)
    fx.extract("warm-up question about Paris?")
    profile, bn = L.load_profile(args.profile)
    tau = profile["meta"]["serve_below"]
    router = lambda raw: L.route(profile, raw, bn)
    llm = None
    if args.llm != "none":
        llm = make_backend(args.llm) if not args.llm == "echo" else make_backend("echo")
        try:
            llm.generate("Context:\n[1] Paris is the capital of France.\n\nQuestion: What is the capital of France?")
        except Exception as e:
            print(f"Answering model not reachable ({type(e).__name__}): is the Ollama app running and "
                  f"'ollama pull qwen2.5:7b' done? Continuing with routing only.")
            llm = None
    print(f"Ready. Network learned from {profile['meta']['n']} labelled questions; "
          f"serve when P(fail) < {tau:.2f}. Answers: {args.llm if llm else 'off'}\n")

    conv = Conversation(fx, router=router)
    inputs = iter(SCRIPT) if args.script else None
    while True:
        try:
            if inputs is not None:
                msg = next(inputs)
                print(f"\n{B}You:{RST} {msg}")
            else:
                msg = input(f"\n{B}You:{RST} ").strip()
        except (StopIteration, EOFError, KeyboardInterrupt):
            print()
            break
        if not msg:
            continue
        if msg == "/quit":
            break
        if msg == "/new":
            conv = Conversation(fx, router=router)
            print(f"{DIM}(new conversation){RST}")
            continue

        t0 = time.time()
        res = conv.handle(msg)
        route_ms = (time.time() - t0) * 1000
        show_routing(res, tau)
        print(f"   {DIM}routing took {route_ms:.0f} ms{RST}")

        if res["route"] == "Clarify":
            print(f"\n{B}Assistant:{RST} {res['ask_user']}")
            continue
        if res["route"] == "Search":
            res = search_and_reroute(fx, kb, conv, res, router, tau, not args.no_web)
        passages = ranked_passages(res["details"])
        print(f"\n   passages used: " + "; ".join(f"{c.get('title', '?')}{' (web)' if c.get('web') else ''}"
                                                for c in passages))
        out = answer(llm, res["query"], passages)
        if out:
            reply, secs = out
            caveat = "" if res["route"] == "Serve" else f"  {DIM}(low confidence: risk still {res['p_fail']:.2f}){RST}"
            print(f"\n{B}Assistant:{RST} {reply}{caveat}   {DIM}[{secs:.1f}s]{RST}")
            conv.transcript.append({"role": "assistant", "text": reply})


if __name__ == "__main__":
    main()
