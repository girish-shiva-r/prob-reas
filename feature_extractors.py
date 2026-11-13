"""
Feature extractors for Bayesian Epistemic Routing (v2)
------------------------------------------------------
Computes the 14 raw measurements that
bayesian_epistemic_routing.extract_evidence() turns into Low/High states.

Everything here runs OFFLINE on a CPU once the models are on disk:
  * spaCy  en_core_web_sm                         (~12 MB)  NER, POS, parsing
  * sentence-transformers all-MiniLM-L6-v2        (~90 MB)  query/chunk embeddings
  * cross-encoder ms-marco-MiniLM-L-6-v2          (~90 MB)  reranker
  * FAISS (optional; falls back to NumPy search)
No API calls, no closed-source models, no internet at runtime.

To ship inside the offline OS: download the two Hugging Face models once,
copy their folders onto the device, point ExtractorConfig at the local
paths, and set HF_HUB_OFFLINE=1.

Usage
    from feature_extractors import FeatureExtractor, KnowledgeBase, load_default_embedder
    embed = load_default_embedder()
    kb = KnowledgeBase(docs, embed)              # docs: [{"text":..., "source_type":...}, ...]
    fx = FeatureExtractor(kb, embed=embed)       # reranker + spaCy load lazily
    result = fx.extract("How does Tokio handle async in Rust?", history=[])
    evidence = extract_evidence(result.raw)      # -> route_query(evidence)

Every model is injectable (embed=, rerank=, nlp=) so the team can swap in
whatever local models the OS project already uses, or test with stand-ins.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Callable, Optional, Sequence

import numpy as np

Embedder = Callable[[Sequence[str]], np.ndarray]                 # texts -> (n, d), L2-normalised
Reranker = Callable[[str, Sequence[str]], np.ndarray]            # (query, passages) -> scores in [0, 1]


# --------------------------------------------------------------------------
# 0. CONFIG
# --------------------------------------------------------------------------

@dataclass
class ExtractorConfig:
    spacy_model: str = "en_core_web_sm"
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"   # or a local folder path
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"      # or a local folder path
    retrieve_k: int = 5              # chunks retrieved and reranked
    relevance_cutoff: float = 0.5    # reranker score that counts a chunk as relevant
    model_knowledge_cutoff_year: int = 2024   # training cut-off of the generator LLM
    # Authority per source type, stored as metadata on each KB chunk.
    source_authority: dict = field(default_factory=lambda: {
        "official_docs": 0.95, "textbook": 0.9, "paper": 0.85, "wikipedia": 0.8,
        "internal_wiki": 0.7, "blog": 0.5, "forum": 0.3, "unknown": 0.5,
    })


# --------------------------------------------------------------------------
# 1. MODEL LOADERS (lazy, offline-capable)
# --------------------------------------------------------------------------

def load_default_embedder(name_or_path: str = ExtractorConfig.embedding_model) -> Embedder:
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(name_or_path, device="cpu")

    def embed(texts):
        return np.asarray(model.encode(list(texts), normalize_embeddings=True,
                                       convert_to_numpy=True), dtype="float32")
    return embed


def load_default_reranker(name_or_path: str = ExtractorConfig.reranker_model) -> Reranker:
    from sentence_transformers import CrossEncoder
    model = CrossEncoder(name_or_path, device="cpu")

    def rerank(query, passages):
        if not passages:
            return np.zeros(0)
        scores = np.asarray(model.predict([(query, p) for p in passages]), dtype="float64")
        # Some versions return raw logits, others probabilities: normalise to [0, 1].
        if scores.min() < 0 or scores.max() > 1:
            scores = 1 / (1 + np.exp(-scores))
        return scores
    return rerank


def load_default_nlp(name: str = ExtractorConfig.spacy_model):
    import spacy
    return spacy.load(name)


def load_popularity_table(path: str) -> dict:
    """
    Offline replacement for the Wikipedia page-view API.
    JSON file: {"python": 250000, "tokio": 900, ...} (monthly views, lower-case keys).
    Build it once while online (e.g. from PopQA's popularity column or a
    Wikipedia page-view dump) and ship the file with the OS.
    """
    with open(path, encoding="utf-8") as f:
        return {k.lower(): int(v) for k, v in json.load(f).items()}


# --------------------------------------------------------------------------
# 2. KNOWLEDGE BASE (FAISS + topic centroids for DomainShift)
# --------------------------------------------------------------------------

class KnowledgeBase:
    """
    docs: list of dicts with at least "text"; optional "source_type", "id".
    Embeddings are L2-normalised, so inner product = cosine similarity and
    faiss_distance = 1 - cosine (0 = identical, 1 = unrelated).
    """

    def __init__(self, docs: list[dict], embed: Embedder, n_topics: int = 8, seed: int = 0):
        self.docs = docs
        self.embeddings = embed([d["text"] for d in docs]).astype("float32")
        self.index = None
        try:
            import faiss
            self.index = faiss.IndexFlatIP(self.embeddings.shape[1])
            self.index.add(self.embeddings)
        except ImportError:
            pass   # NumPy fallback below
        self.centroids = self._topic_centroids(n_topics, seed)

    def search(self, query_emb: np.ndarray, k: int) -> list[dict]:
        q = query_emb.reshape(1, -1).astype("float32")
        k = min(k, len(self.docs))
        if self.index is not None:
            sims, ids = self.index.search(q, k)
            sims, ids = sims[0], ids[0]
        else:
            all_sims = self.embeddings @ q[0]
            ids = np.argsort(-all_sims)[:k]
            sims = all_sims[ids]
        return [{**self.docs[i], "faiss_distance": float(1 - s)} for i, s in zip(ids, sims)]

    # ---- persistence: build once (slow), load instantly on the device -------
    def save(self, folder: str) -> None:
        import os
        os.makedirs(folder, exist_ok=True)
        with open(os.path.join(folder, "docs.jsonl"), "w", encoding="utf-8") as f:
            for d in self.docs:
                f.write(json.dumps(d, ensure_ascii=False) + "\n")
        np.save(os.path.join(folder, "embeddings.npy"), self.embeddings)
        np.save(os.path.join(folder, "centroids.npy"), self.centroids)

    @classmethod
    def load(cls, folder: str) -> "KnowledgeBase":
        """Load a saved KB without re-embedding anything (the embedder is only needed for queries)."""
        import os
        kb = cls.__new__(cls)
        with open(os.path.join(folder, "docs.jsonl"), encoding="utf-8") as f:
            kb.docs = [json.loads(line) for line in f]
        kb.embeddings = np.load(os.path.join(folder, "embeddings.npy")).astype("float32")
        kb.centroids = np.load(os.path.join(folder, "centroids.npy"))
        kb.index = None
        try:
            import faiss
            kb.index = faiss.IndexFlatIP(kb.embeddings.shape[1])
            kb.index.add(kb.embeddings)
        except ImportError:
            pass
        return kb

    def domain_similarity(self, query_emb: np.ndarray) -> float:
        """Highest cosine similarity between the query and any KB topic centroid."""
        return float(np.max(self.centroids @ query_emb))

    def _topic_centroids(self, n_topics: int, seed: int) -> np.ndarray:
        X = self.embeddings
        if len(X) <= n_topics:
            C = X.copy()
        else:
            try:
                from sklearn.cluster import KMeans
                C = KMeans(n_clusters=n_topics, n_init=10, random_state=seed).fit(X).cluster_centers_
            except ImportError:
                rng = np.random.default_rng(seed)
                C = X[rng.choice(len(X), n_topics, replace=False)]
                for _ in range(20):
                    labels = np.argmax(X @ C.T, axis=1)
                    C = np.stack([X[labels == j].mean(0) if np.any(labels == j) else C[j]
                                  for j in range(n_topics)])
        return C / np.linalg.norm(C, axis=1, keepdims=True)


# --------------------------------------------------------------------------
# 3. TEXT HEURISTICS (pure functions, easy to unit-test and tune)
# --------------------------------------------------------------------------

NON_SPECIFIC_ENT_LABELS = {"CARDINAL", "ORDINAL", "QUANTITY", "PERCENT", "MONEY", "TIME", "DATE"}
REFERRING_PRONOUNS = {"it", "its", "they", "them", "their", "this", "that", "these", "those",
                      "he", "she", "him", "her", "his"}
FOLLOWUP_START = re.compile(r"^(and|also|what about|how about|but|so|then|ok|okay|same for)\b", re.I)
OPINION = re.compile(r"\b(best|better|worse|worst|should (i|we)|recommend\w*|prefer\w*|opinion|"
                     r"worth it|pros and cons|good idea)\b", re.I)
OPEN_START = re.compile(r"^(explain|describe|discuss|tell me about|compare|elaborate|overview|"
                        r"what do you think|summari[sz]e)\b", re.I)
HOW_FACTOID = re.compile(r"^how (many|much|old|long|far|often|big|tall|large)\b", re.I)
TIME_WORDS = re.compile(r"\b(latest|current(ly)?|now|today|tonight|recent(ly)?|this (year|month|week)|"
                        r"newest|upcoming|right now|as of|so far|nowadays|breaking|new release|"
                        r"still)\b", re.I)
YEAR = re.compile(r"\b(19\d{2}|20\d{2})\b")


def count_words(text: str) -> int:
    return len(re.findall(r"[\w'-]+", text))


def specific_entities(doc, known_terms: Optional[dict] = None) -> list[str]:
    """
    Named entities that make a query specific: spaCy entities (minus numbers/
    dates), proper nouns spaCy missed, and any 1-3 word phrase found in the
    popularity table (catches lower-case tech names like 'tokio').
    """
    found, covered = [], set()
    # 1. Names from the popularity table first, LONGEST match wins:
    #    "Princess Tōchi" beats "Tōchi"; "The Island" beats "island".
    #    A one-word match only counts if it is capitalised in the query,
    #    so common words ("help", "island") are not taken as entities.
    if known_terms:
        words = [t.text.lower() for t in doc]
        for n in (6, 5, 4, 3, 2, 1):
            for i in range(len(words) - n + 1):
                span = range(i, i + n)
                if covered.intersection(span):
                    continue
                phrase = doc[i:i + n].text.lower()
                if phrase not in known_terms:
                    continue
                if n == 1 and (not doc[i].text[:1].isupper() or doc[i].is_stop):
                    continue                    # "Who", "What" at the start of a question
                found.append(doc[i:i + n].text)
                covered.update(span)
    # 2. spaCy entities and leftover proper nouns that don't overlap a table name
    for e in doc.ents:
        if e.label_ not in NON_SPECIFIC_ENT_LABELS and not covered.intersection(range(e.start, e.end)):
            found.append(e.text)
            covered.update(range(e.start, e.end))
    found += [t.text for t in doc if t.pos_ == "PROPN" and t.i not in covered]
    seen, out = set(), []
    for f in found:
        if f.lower() not in seen:
            seen.add(f.lower())
            out.append(f)
    return out


def clause_counts(doc, text: str) -> tuple[int, bool]:
    """
    Returns (sub_question_count, has_coordination).
    A 'sub-question' = a main verb, a verb joined by and/or, or a relative
    clause ('the company THAT MAKES X' -> a second hop), or an extra '?'.
    """
    verbs = [t for t in doc if t.pos_ in ("VERB", "AUX")
             and (t.dep_ == "ROOT" or t.dep_ in ("conj", "relcl"))]
    coordinated = any(t.dep_ == "conj" for t in verbs) or text.count("?") >= 2
    return max(1, len(verbs), text.count("?")), coordinated


def classify_question(text: str, coordinated: bool) -> str:
    t = text.strip().lower()
    if coordinated:
        return "compound"
    if OPINION.search(t):
        return "opinion"
    if OPEN_START.search(t):
        return "open_ended"
    if t.startswith("why"):
        return "why"
    if t.startswith("how") and not HOW_FACTOID.search(t):
        return "how"
    return "factoid"


def is_time_sensitive(text: str, cutoff_year: int) -> bool:
    if TIME_WORDS.search(text):
        return True
    return any(int(y) >= cutoff_year for y in YEAR.findall(text))


def referring_pronouns(doc) -> list:
    """Pronouns that point at something (excludes 'it' in 'is it raining')."""
    return [t for t in doc if t.text.lower() in REFERRING_PRONOUNS
            and t.pos_ == "PRON" and t.dep_ != "expl"]


PERSONAL_PRONOUNS = {"he", "him", "his", "she", "her"}


def pronouns_needing_resolution(doc) -> list[str]:
    """
    Pronouns with no antecedent inside the query itself.
    he/she/his/her need an earlier PERSON-like proper noun ("Was this the end
    of his career?" - "the end" is not who "his" is); it/they/this/that can
    point back at any earlier noun phrase ("Python is great, why is it slow?").
    """
    needs = []
    for p in referring_pronouns(doc):
        if p.text.lower() in PERSONAL_PRONOUNS:
            earlier = [t for t in doc[:p.i] if t.pos_ == "PROPN"]
        else:
            earlier = [nc for nc in doc.noun_chunks if nc.end <= p.i and nc.root.pos_ != "PRON"]
        if not earlier:
            needs.append(p.text)
    return needs


def unresolved_references(doc, prev_entities: list[str]) -> list[str]:
    """
    Pronouns the system cannot resolve on its own: no antecedent in the
    query, and the previous turn does not name exactly ONE entity.
    (With exactly one candidate, clarify_loop substitutes it automatically.)
    """
    if len(prev_entities) == 1:
        return []
    return pronouns_needing_resolution(doc)


# --------------------------------------------------------------------------
# 4. THE EXTRACTOR
# --------------------------------------------------------------------------

@dataclass
class ExtractionResult:
    raw: dict        # the 14 measurements -> extract_evidence(raw)
    details: dict    # what was found (entities, pronouns, scores) -> used by the clarify step


class FeatureExtractor:
    def __init__(self, kb: KnowledgeBase, embed: Optional[Embedder] = None,
                 rerank: Optional[Reranker] = None, nlp=None,
                 popularity: Optional[dict] = None, config: Optional[ExtractorConfig] = None):
        self.cfg = config or ExtractorConfig()
        self.kb = kb
        self._embed, self._rerank, self._nlp = embed, rerank, nlp
        self.popularity = popularity

    # lazy loading keeps start-up fast and lets tests inject stand-ins
    @property
    def embed(self):
        if self._embed is None:
            self._embed = load_default_embedder(self.cfg.embedding_model)
        return self._embed

    @property
    def rerank(self):
        if self._rerank is None:
            self._rerank = load_default_reranker(self.cfg.reranker_model)
        return self._rerank

    @property
    def nlp(self):
        if self._nlp is None:
            self._nlp = load_default_nlp(self.cfg.spacy_model)
        return self._nlp

    def extract(self, query: str, history: Sequence[str] = (),
                retrieved: Optional[list[dict]] = None,
                is_clarification_reply: bool = False) -> ExtractionResult:
        """
        query:    the user's current message (after any clarification merge)
        history:  previous USER messages in this conversation, oldest first
        retrieved: chunks already retrieved elsewhere (each needs "text",
                   "faiss_distance", optional "source_type"); if None, the KB
                   is searched here
        is_clarification_reply: True when this turn answers OUR clarifying
                   question - then it must not count as a rephrase.
        """
        cfg = self.cfg
        doc = self.nlp(query)
        prev_doc = self.nlp(history[-1]) if history else None

        # ---- Module 1: Query Profiling --------------------------------------
        entities = specific_entities(doc, self.popularity)
        sub_q, coordinated = clause_counts(doc, query)
        qtype = classify_question(query, coordinated)

        # ---- Module 2: Retrieval Diagnostics --------------------------------
        q_emb = self.embed([query])[0]
        chunks = retrieved if retrieved is not None else self.kb.search(q_emb, cfg.retrieve_k)
        chunks = chunks[:cfg.retrieve_k]
        scores = np.asarray(self.rerank(query, [c["text"] for c in chunks]), dtype=float)
        relevant = [c for c, s in zip(chunks, scores) if s >= cfg.relevance_cutoff]
        auth_pool = relevant or chunks[:3]
        authority = (float(np.mean([cfg.source_authority.get(c.get("source_type", "unknown"),
                                                             cfg.source_authority["unknown"])
                                    for c in auth_pool])) if auth_pool else None)

        # ---- Module 3: User Context -----------------------------------------
        prev_entities = specific_entities(prev_doc, self.popularity) if prev_doc is not None else []
        unresolved = unresolved_references(doc, prev_entities)
        if history:
            is_followup = bool(FOLLOWUP_START.search(query.strip())
                               or referring_pronouns(doc) or count_words(query) <= 4)
            # A follow-up is ambiguous when it doesn't name its own subject and
            # the previous turn offers zero or several candidates.
            ambiguous_followup = is_followup and not entities and len(prev_entities) != 1
            if is_clarification_reply:
                sim_prev = 0.0         # answering our question is not a rephrase
            else:
                prev_emb = self.embed([history[-1]])[0]
                sim_prev = float(np.dot(q_emb, prev_emb))
        else:
            ambiguous_followup, sim_prev = False, None   # None -> marginalised

        # ---- Module 4: Knowledge Coverage -----------------------------------
        if self.popularity is not None and entities:
            # the rarest KNOWN entity drives the risk; if none is in the table
            # at all, treat the query's subject as rare (0 views)
            known = [self.popularity[e.lower()] for e in entities if e.lower() in self.popularity]
            views = min(known) if known else 0
        else:
            views = None
        time_sensitive = is_time_sensitive(query, cfg.model_knowledge_cutoff_year)
        domain_sim = self.kb.domain_similarity(q_emb)

        raw = {
            # Module 1
            "word_count": count_words(query),
            "question_type": qtype,
            "named_entity_count": len(entities),
            "sub_question_count": sub_q,
            # Module 2
            "top1_faiss_distance": min(c["faiss_distance"] for c in chunks) if chunks else None,
            "top1_rerank_score": float(scores.max()) if len(scores) else None,
            "relevant_chunk_count": len(relevant),
            "mean_source_authority": authority,
            # Module 3
            "is_ambiguous_followup": ambiguous_followup,
            "similarity_to_prev_query": sim_prev,
            "unresolved_reference_count": len(unresolved),
            # Module 4
            "entity_monthly_views": views,
            "is_time_sensitive": time_sensitive,
            "kb_domain_similarity": domain_sim,
        }
        details = {
            "entities": entities,
            "previous_turn_entities": prev_entities,
            "unresolved_pronouns": unresolved,
            "rerank_scores": [round(float(s), 3) for s in scores],
            "chunks": chunks,
        }
        return ExtractionResult(raw=raw, details=details)
