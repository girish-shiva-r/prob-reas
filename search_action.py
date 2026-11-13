"""
The "Search" action: what the system does when the router says the context is not good enough.

simple_search  (v1)  the raw question typed into Wikipedia search; top 3 pages; INTRO only
better_search  (v2)  1. search by the question's ENTITY ("Ship Breaker"), plus the question itself
                     2. FULL page text (answers like a film's composer are often below the intro)
                     3. keep only the passages closest to the question (embeddings), then let the
                        cross-encoder pick the best (same reranker the router uses)

Both return passages shaped like knowledge-base chunks: {"title", "text", "source_type", "web": True}.
"""

import re

import numpy as np

WIKI_API = "https://en.wikipedia.org/w/api.php"
import os
# Wikimedia asks API clients to identify themselves with a contact; set WIKI_CONTACT to an email/URL of yours
UA = f"BayesianEpistemicRouting/0.3 (university course project; contact: {os.environ.get('WIKI_CONTACT', 'not set')})"


def _session():
    import requests
    s = requests.Session()
    s.headers["User-Agent"] = UA
    return s


def _get(s, params, tries=4):
    """GET the Wikipedia API politely: wait and retry when rate-limited (429/5xx or a non-JSON error page)."""
    import time
    last = ""
    for attempt in range(tries):
        try:
            r = s.get(WIKI_API, params=params, timeout=10)
            if r.status_code == 200:
                data = r.json()
                time.sleep(0.3)                   # stay well under Wikipedia's rate limits
                return data
            last = f"HTTP {r.status_code}: {r.text[:80]!r}"
            wait = float(r.headers.get("Retry-After") or 0) or 1.5 * (attempt + 1)
        except ValueError as e:                   # 200 but not JSON (an HTML error page)
            last, wait = f"not JSON ({e})", 1.5 * (attempt + 1)
        except Exception as e:
            last, wait = f"{type(e).__name__}: {e}", 1.5 * (attempt + 1)
        time.sleep(min(wait, 6))
    raise RuntimeError(f"Wikipedia API: {last}")


def _chunks(title, text, max_words=100, max_total_words=4000):
    text = re.sub(r"\n=+[^=\n]+=+\n", "\n", text or "")            # drop "== Section ==" headings
    words = text.split()
    text = " ".join(words[:max_total_words])
    sents = re.split(r"(?<=[.!?])\s+", re.sub(r"\s+", " ", text).strip())
    out, cur = [], []
    for s in sents:
        if cur and len(" ".join(cur + [s]).split()) > max_words:
            out.append(" ".join(cur))
            cur = []
        cur.append(s)
    if cur:
        out.append(" ".join(cur))
    return [{"title": title, "text": f"{title}: {c}", "source_type": "wikipedia", "web": True}
            for c in out if len(c.split()) >= 8]


def _search_titles(s, query, k):
    data = _get(s, {"action": "query", "list": "search", "srsearch": query, "srlimit": k, "format": "json"})
    return [h["title"] for h in data["query"]["search"] if "(disambiguation)" not in h["title"]]


def simple_search(query, k=3):
    """v1: raw question -> top k pages -> intros."""
    s = _session()
    titles = _search_titles(s, query, k)
    if not titles:
        return []
    data = _get(s, {"action": "query", "prop": "extracts", "exintro": 1, "explaintext": 1, "redirects": 1,
                    "titles": "|".join(titles), "format": "json", "formatversion": 2})
    out = []
    for p in data["query"]["pages"]:
        out += _chunks(p["title"], p.get("extract", ""))
    return out


def better_search(question, fx, max_pages=2, keep=10):
    """v2: entity + question search -> full pages -> passages closest to the question."""
    s = _session()
    doc = fx.nlp(question)
    from feature_extractors import specific_entities
    entities = sorted(specific_entities(doc, fx.popularity), key=len, reverse=True)   # "Babette's Feast" before "Feast"
    titles = []
    for q in entities[:1] + [question]:
        for t in _search_titles(s, q, 2):
            if t not in titles:
                titles.append(t)
    passages = []
    for t in titles[:max_pages]:
        data = _get(s, {"action": "query", "prop": "extracts", "explaintext": 1, "redirects": 1,
                        "titles": t, "format": "json", "formatversion": 2})
        for p in data["query"]["pages"]:
            passages += _chunks(p["title"], p.get("extract", ""))
    if not passages:
        return []
    q_emb = fx.embed([question])[0]
    embs = fx.embed([p["text"] for p in passages])
    sims = embs @ q_emb
    for p, sim in zip(passages, sims):
        p["faiss_distance"] = float(1 - sim)
    return [passages[i] for i in np.argsort(-sims)[:keep]]


def search_pool(question, fx, kb, mode="better"):
    """KB passages + web passages, each with a faiss_distance, closest first."""
    q_emb = fx.embed([question])[0]
    try:
        web = better_search(question, fx) if mode == "better" else simple_search(question)
    except Exception as e:
        print(f"   (search failed: {str(e)[:110]}; trying the simple search)")
        try:
            web = simple_search(question)
        except Exception as e2:
            print(f"   (simple search failed too: {str(e2)[:110]})")
            web = []
    for w in web:
        if "faiss_distance" not in w:
            w["faiss_distance"] = float(1 - np.dot(q_emb, fx.embed([w["text"]])[0]))
    return sorted(kb.search(q_emb, fx.cfg.retrieve_k) + web, key=lambda c: c["faiss_distance"]), web
