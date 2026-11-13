"""
Step 1 - Build the knowledge base (KB) and question set from PopQA
------------------------------------------------------------------
Run ONCE on a machine with internet; the output folder then works offline.

What it does
  1. Samples questions from PopQA, spread evenly across entity popularity
     (rare -> famous), so EntityPopularity has real variation.
  2. Fetches the Wikipedia intro of each question's subject page.
  3. HOLDS OUT ~25% of subjects: their pages are NOT put in the KB. Those
     questions have no supporting document, so the network can learn what
     missing context looks like (ContextDeficit / DomainShift).
  4. Splits pages into ~100-word passages, embeds them, builds the FAISS
     index and DomainShift topic centroids, and saves everything.
  5. Writes the offline EntityPopularity table (page views per entity).

Outputs (in --out, default kb/):
  docs.jsonl, embeddings.npy, centroids.npy   - the KB (load with KnowledgeBase.load)
  questions.jsonl                             - sampled questions + answers + in_kb flag
  popularity.json                             - entity -> monthly page views
  wiki_cache.jsonl                            - raw fetched pages (re-runs skip fetching)

Usage
  python build_kb.py --popqa data/popQA.tsv --sample 3000
  (PopQA: https://github.com/AlexTMallen/adaptive-retrieval , file data/popQA.tsv)
"""

import argparse
import csv
import json
import math
import os
import random
import re
import sys
import time
from collections import defaultdict

WIKI_API = "https://en.wikipedia.org/w/api.php"
# Wikimedia asks every client to identify itself.
USER_AGENT = "BayesianEpistemicRouting/0.1 (university course project; contact via project repo)"


# --------------------------------------------------------------------------
# PopQA loading and sampling
# --------------------------------------------------------------------------

def clean_title(title: str) -> str:
    """'John Mayne (journalist)' -> 'John Mayne' (how people actually write the name)."""
    return re.sub(r"\s*\([^)]*\)\s*$", "", title).strip()


def load_popqa(path: str) -> list[dict]:
    csv.field_size_limit(sys.maxsize)
    with open(path, encoding="utf-8") as f:
        rows = list(csv.DictReader(f, delimiter="\t"))
    out = []
    for r in rows:
        out.append({
            "id": f"popqa-{r['id']}",
            "dataset": "popqa",
            "question": r["question"],
            "answers": json.loads(r["possible_answers"]),
            "subject": r["subj"],
            "subject_title": r["s_wiki_title"],
            "relation": r["prop"],
            "s_pop": int(r["s_pop"]),
            "o_title": r["o_wiki_title"],
            "o_pop": int(r["o_pop"]),
        })
    return out


def stratified_sample(rows: list[dict], n: int, bins: int, seed: int) -> list[dict]:
    """Equal numbers of questions from each log-popularity band."""
    rng = random.Random(seed)
    by_bin = defaultdict(list)
    lo = math.log10(max(1, min(r["s_pop"] for r in rows)))
    hi = math.log10(max(r["s_pop"] for r in rows))
    for r in rows:
        b = min(bins - 1, int((math.log10(max(1, r["s_pop"])) - lo) / (hi - lo + 1e-9) * bins))
        by_bin[b].append(r)
    per_bin = n // bins
    sample = []
    for b in range(bins):
        group = by_bin[b]
        rng.shuffle(group)
        sample += group[:per_bin]
    if len(sample) < n:                       # top up from the leftovers (sparse bins)
        chosen = {r["id"] for r in sample}
        rest = [r for r in rows if r["id"] not in chosen]
        rng.shuffle(rest)
        sample += rest[:n - len(sample)]
    rng.shuffle(sample)
    return sample


def build_popularity(rows: list[dict]) -> dict:
    """Entity name -> monthly page views, from PopQA's own popularity columns."""
    pop = {}
    for r in rows:
        for title, views in ((r["subject_title"], r["s_pop"]), (r["o_title"], r["o_pop"])):
            for name in {clean_title(title).lower(), r["subject"].lower() if title == r["subject_title"] else ""}:
                if name:
                    pop[name] = max(pop.get(name, 0), views)
    return pop


# --------------------------------------------------------------------------
# Wikipedia fetching (cached, polite, batched)
# --------------------------------------------------------------------------

def load_cache(path: str) -> dict:
    cache = {}
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            for line in f:
                rec = json.loads(line)
                cache[rec["requested"]] = rec
    return cache


def fetch_intros(titles: list[str], cache_path: str, batch: int = 20, pause: float = 1.0) -> dict:
    """Plain-text intro section for each title. Results are appended to a cache file."""
    import requests
    cache = load_cache(cache_path)
    todo = [t for t in dict.fromkeys(titles) if t not in cache]
    session = requests.Session()
    session.headers["User-Agent"] = USER_AGENT
    with open(cache_path, "a", encoding="utf-8") as out:
        for i in range(0, len(todo), batch):
            chunk = todo[i:i + batch]
            params = {"action": "query", "format": "json", "formatversion": 2, "prop": "extracts",
                      "exintro": 1, "explaintext": 1, "redirects": 1, "titles": "|".join(chunk)}
            for attempt in range(8):
                try:
                    resp = session.get(WIKI_API, params=params, timeout=30)
                    if resp.status_code == 429:             # rate limited: wait as long as asked
                        wait = int(resp.headers.get("Retry-After", 0) or 0) or min(60, 5 * 2 ** attempt)
                        print(f"  rate limited, waiting {wait}s")
                        time.sleep(wait)
                        pause = min(pause * 1.5, 5.0)       # and slow down for the rest of the run
                        continue
                    resp.raise_for_status()
                    data = resp.json()["query"]
                    break
                except Exception as e:                      # network hiccup
                    wait = min(60, 2 ** attempt)
                    print(f"  retry in {wait}s ({type(e).__name__})")
                    time.sleep(wait)
            else:
                print(f"  skipped batch starting {chunk[0]!r} (will be retried on the next run)")
                continue
            # map requested title -> final title through normalisation + redirects
            alias = {t: t for t in chunk}
            for step in ("normalized", "redirects"):
                for m in data.get(step, []):
                    for k, v in alias.items():
                        if v == m["from"]:
                            alias[k] = m["to"]
            pages = {p["title"]: p for p in data.get("pages", [])}
            for req in chunk:
                page = pages.get(alias[req], {})
                rec = {"requested": req, "title": page.get("title", alias[req]),
                       "text": page.get("extract", "") or "", "missing": "missing" in page}
                cache[req] = rec
                out.write(json.dumps(rec, ensure_ascii=False) + "\n")
            done = min(i + batch, len(todo))
            if done % 500 < batch or done == len(todo):
                print(f"  fetched {done}/{len(todo)} pages")
            time.sleep(pause)
    return cache


# --------------------------------------------------------------------------
# Chunking
# --------------------------------------------------------------------------

def chunk_page(title: str, text: str, max_words: int = 100) -> list[str]:
    """Split on sentence ends into ~max_words passages; prefix the title (helps retrieval)."""
    sentences = re.split(r"(?<=[.!?])\s+", re.sub(r"\s+", " ", text).strip())
    chunks, cur = [], []
    for s in sentences:
        if cur and len(" ".join(cur + [s]).split()) > max_words:
            chunks.append(" ".join(cur))
            cur = []
        cur.append(s)
    if cur:
        chunks.append(" ".join(cur))
    return [f"{title}: {c}" for c in chunks if len(c.split()) >= 8]


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--popqa", default="data/popQA.tsv")
    ap.add_argument("--sample", type=int, default=3000, help="number of PopQA questions")
    ap.add_argument("--holdout", type=float, default=0.25, help="share of subjects left OUT of the KB")
    ap.add_argument("--bins", type=int, default=6, help="popularity bands for stratified sampling")
    ap.add_argument("--topics", type=int, default=16, help="topic centroids for DomainShift")
    ap.add_argument("--out", default="kb")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-embed", action="store_true", help="stop after fetching + chunking")
    ap.add_argument("--allow-missing", action="store_true", help="build even if some pages failed to fetch")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    rng = random.Random(args.seed)

    rows = load_popqa(args.popqa)
    print(f"PopQA: {len(rows)} questions")
    sample = stratified_sample(rows, args.sample, args.bins, args.seed)

    # Hold out whole SUBJECTS (not single questions), so a held-out entity has no page at all.
    subjects = sorted({r["subject_title"] for r in sample})
    rng.shuffle(subjects)
    held_out = set(subjects[:int(len(subjects) * args.holdout)])
    for r in sample:
        r["in_kb"] = r["subject_title"] not in held_out
    print(f"Sampled {len(sample)} questions on {len(subjects)} subjects; "
          f"{len(held_out)} subjects held out of the KB")

    with open(os.path.join(args.out, "popularity.json"), "w", encoding="utf-8") as f:
        json.dump(build_popularity(rows), f)

    kb_titles = [t for t in subjects if t not in held_out]
    print(f"Fetching {len(kb_titles)} Wikipedia intros (cached, ~0.5 s per 20 pages)...")
    pages = fetch_intros(kb_titles, os.path.join(args.out, "wiki_cache.jsonl"))
    not_fetched = [t for t in kb_titles if t not in pages]
    if not_fetched and not args.allow_missing:
        print(f"\n{len(not_fetched)} pages could not be fetched (rate limits). They are NOT held out on")
        print("purpose, so stopping here to keep the 25% hold-out honest. Just run the same command")
        print("again: fetched pages are cached and only the missing ones are requested.")
        return

    docs = []
    for t in kb_titles:
        page = pages.get(t)
        if not page or page["missing"] or not page["text"]:
            continue
        for j, text in enumerate(chunk_page(page["title"], page["text"])):
            docs.append({"id": f"{page['title']}#{j}", "title": page["title"], "text": text,
                         "source_type": "wikipedia"})
    missing = sum(1 for t in kb_titles if not pages.get(t, {}).get("text"))
    print(f"KB: {len(docs)} passages from {len(kb_titles) - missing} pages ({missing} pages empty/missing)")

    # Questions whose page turned out empty are effectively held out too.
    have_text = {t for t in kb_titles if pages.get(t, {}).get("text")}
    for r in sample:
        r["in_kb"] = r["subject_title"] in have_text
    with open(os.path.join(args.out, "questions.jsonl"), "w", encoding="utf-8") as f:
        for r in sample:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"Questions: {len(sample)} ({sum(r['in_kb'] for r in sample)} with their page in the KB)")

    if args.no_embed:
        with open(os.path.join(args.out, "docs.jsonl"), "w", encoding="utf-8") as f:
            for d in docs:
                f.write(json.dumps(d, ensure_ascii=False) + "\n")
        print("Stopped before embedding (--no-embed).")
        return

    from feature_extractors import KnowledgeBase, load_default_embedder
    print("Embedding passages with all-MiniLM-L6-v2 (first run downloads ~90 MB)...")
    t0 = time.time()
    kb = KnowledgeBase(docs, load_default_embedder(), n_topics=args.topics, seed=args.seed)
    kb.save(args.out)
    from feature_extractors import ExtractorConfig
    with open(os.path.join(args.out, "profile.json"), "w") as f:     # labels must use the same embedder
        json.dump({"embedding_model": ExtractorConfig().embedding_model, "passages": len(docs),
                   "questions": len(sample), "holdout": args.holdout, "seed": args.seed}, f, indent=2)
    print(f"Saved KB to {args.out}/ in {time.time() - t0:.0f}s "
          f"({kb.embeddings.shape[0]} vectors x {kb.embeddings.shape[1]} dims)")


if __name__ == "__main__":
    main()
