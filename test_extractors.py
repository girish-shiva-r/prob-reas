"""
Offline tests for feature_extractors.py and clarify_loop.py.

The two neural models are replaced by tiny stand-ins (bag-of-words hashing
embedder, word-overlap reranker) so the tests run anywhere in seconds with
no downloads. Only spaCy's small English model is needed:
    python -m spacy download en_core_web_sm
Run:  python test_extractors.py      (or: pytest test_extractors.py)
"""

import re
import warnings
import zlib

warnings.filterwarnings("ignore")

import numpy as np

from bayesian_epistemic_routing import extract_evidence, route_query
from clarify_loop import Conversation, merge_clarification
from feature_extractors import FeatureExtractor, KnowledgeBase, load_default_nlp

STOP = {"the", "a", "an", "is", "are", "how", "what", "why", "does", "do", "in", "of", "to",
        "my", "i", "it", "can", "you", "and", "for", "on", "with", "this", "that", "me"}


def tokens(text):
    return [w for w in re.findall(r"[a-z0-9]+", text.lower()) if w not in STOP]


def fake_embed(texts, dim=512):
    out = np.zeros((len(texts), dim), dtype="float32")
    for i, t in enumerate(texts):
        for w in tokens(t):
            out[i, zlib.crc32(w.encode()) % dim] += 1   # stable across runs/machines
        n = np.linalg.norm(out[i])
        out[i] = out[i] / n if n else out[i]
    return out


def fake_rerank(query, passages):
    q = set(tokens(query))
    return np.array([len(q & set(tokens(p))) / max(len(q), 1) for p in passages])


DOCS = [
    {"text": "Tokio is an asynchronous runtime for Rust that schedules async tasks.", "source_type": "official_docs"},
    {"text": "Rust async functions return futures that the Tokio runtime polls.", "source_type": "official_docs"},
    {"text": "Python memory is managed by reference counting and a garbage collector.", "source_type": "wikipedia"},
    {"text": "Slow PostgreSQL queries can often be fixed by adding an index.", "source_type": "blog"},
    {"text": "PostgreSQL EXPLAIN ANALYZE shows the query plan and timing.", "source_type": "official_docs"},
    {"text": "Django and Flask are Python web frameworks.", "source_type": "wikipedia"},
    {"text": "Flask is a lightweight microframework for Python.", "source_type": "forum"},
]
POPULARITY = {"rust": 400000, "tokio": 900, "python": 2000000, "postgresql": 150000,
              "django": 300000, "flask": 120000, "pytorch": 350000}

nlp = load_default_nlp()
kb = KnowledgeBase(DOCS, fake_embed, n_topics=3)
fx = FeatureExtractor(kb, embed=fake_embed, rerank=fake_rerank, nlp=nlp, popularity=POPULARITY)


def raw(q, history=(), **kw):
    return fx.extract(q, history=list(history), **kw)


def test_query_profiling():
    r = raw("How does Tokio handle async in Rust?").raw
    assert r["named_entity_count"] >= 2 and r["question_type"] == "how"
    assert raw("python memory").raw["word_count"] == 2
    assert raw("Who founded the company that makes PyTorch?").raw["sub_question_count"] >= 2
    assert raw("What is Django and how do I install Flask?").raw["question_type"] == "compound"
    assert raw("What is the best Python web framework?").raw["question_type"] == "opinion"
    assert raw("How many threads does Tokio use?").raw["question_type"] == "factoid"
    assert raw("Explain garbage collection").raw["question_type"] == "open_ended"


def test_retrieval_diagnostics():
    good = raw("How does Tokio run async tasks in Rust?").raw
    bad = raw("What is the tax rate on capital gains?").raw
    assert good["top1_rerank_score"] > bad["top1_rerank_score"]
    assert good["relevant_chunk_count"] >= 1 and bad["relevant_chunk_count"] == 0
    assert good["top1_faiss_distance"] < bad["top1_faiss_distance"]
    assert good["mean_source_authority"] >= 0.9   # official docs


def test_user_context():
    # no history: 'it' cannot be resolved
    assert raw("Can you make it faster?").raw["unresolved_reference_count"] == 1
    # one candidate in the previous turn: resolved
    r = raw("Can you make it faster?", ["Why is my PostgreSQL query slow?"]).raw
    assert r["unresolved_reference_count"] == 0 and not r["is_ambiguous_followup"]
    # two candidates in the previous turn: ambiguous
    r = raw("What about its performance?", ["Compare Django and Flask"]).raw
    assert r["is_ambiguous_followup"] and r["unresolved_reference_count"] == 1
    # rephrase vs answering our clarifying question
    prev = ["How do I fix slow PostgreSQL queries?"]
    assert raw("How can I fix slow PostgreSQL queries?", prev).raw["similarity_to_prev_query"] > 0.8
    assert raw("PostgreSQL slow queries fix", prev, is_clarification_reply=True).raw["similarity_to_prev_query"] == 0.0


def test_knowledge_coverage():
    assert raw("How does Tokio handle async?").raw["entity_monthly_views"] == 900
    assert raw("What is the latest Rust release?").raw["is_time_sensitive"]
    assert raw("What changed in Python in 2025?").raw["is_time_sensitive"]
    assert not raw("What is reference counting in Python?").raw["is_time_sensitive"]
    assert raw("How does Tokio schedule Rust tasks?").raw["kb_domain_similarity"] > \
           raw("What is the tax rate on capital gains?").raw["kb_domain_similarity"]


def test_merge_without_model():
    d = {"unresolved_pronouns": ["it"]}
    assert merge_clarification("Can you make it faster?", "the PostgreSQL query",
                               "UnresolvedReferences", d) == "Can you make the PostgreSQL query faster?"
    assert merge_clarification("python memory", "memory leaks", "QueryLength", {}) == \
           "python memory \u2014 memory leaks"


def test_merge_conversational_replies():
    """People answer in sentences, not clean noun phrases (the critique's cases)."""
    def m(orig, reply, pron="it", reason="UnresolvedReferences"):
        d = {"unresolved_pronouns": [pron]} if pron else {}
        return merge_clarification(orig, reply, reason, d, nlp=nlp)
    assert m("Can you make it faster?", "I meant the database") == "Can you make the database faster?"
    assert m("Can you make it faster?", "oh sorry, I was talking about the slow PostgreSQL queries") \
           == "Can you make the slow PostgreSQL queries faster?"
    assert m("Can you make it faster?", "it is the query on the orders table") \
           == "Can you make the query on the orders table faster?"
    assert m("Can you make it faster?", "my script keeps timing out when it reads the CSV") \
           == "Can you make my script faster?"
    # agreement and possessives
    assert m("Why is it slow?", "the queries") == "Why are the queries slow?"
    assert m("Does it support async?", "the Django and Flask frameworks") \
           == "Do the Django and Flask frameworks support async?"
    assert m("What about its performance?", "I was asking about Flask, sorry", "its") \
           == "What about Flask's performance?"
    # non-pronoun reasons read naturally; empty answers change nothing
    assert m("What is the best web framework?", "mostly speed I guess", None, "QuestionType") \
           == "What is the best web framework \u2014 speed?"
    assert m("Can you make it faster?", "yes") == "Can you make it faster?"


def test_merge_fixes_found_by_qrecc():
    def m(orig, reply, pron):
        return merge_clarification(orig, reply, "UnresolvedReferences", {"unresolved_pronouns": [pron]}, nlp=nlp)
    assert m("What was his biggest hit?", "Robin Thicke", "his") == "What was Robin Thicke's biggest hit?"
    assert m("What about her sexual life caused controversy?", "Simone de Beauvoir", "her") \
           == "What about Simone de Beauvoir's sexual life caused controversy?"
    assert m("Did they ever meet her?", "Marie Curie", "her") == "Did they ever meet Marie Curie?"
    assert m("Was it a success?", "No Favors by Big Sean", "it") == "Was No Favors by Big Sean a success?"
    # other gaps keep every word of the reply, not just one noun phrase
    assert merge_clarification("who was president", "when the United States entered World War 1",
                               "FollowUpStatus", {}, nlp=nlp) == \
           "who was president \u2014 when the United States entered World War 1"


def test_history_auto_resolution():
    from clarify_loop import resolve_from_history
    q, changed = resolve_from_history("When was he born?", ["Who composed the Brandenburg Concertos, Bach?"], nlp)
    assert not changed          # two candidates (Brandenburg Concertos, Bach): leave it to Clarify
    q, changed = resolve_from_history("When was he born?", ["What is Johann Sebastian Bach known for?"], nlp)
    assert changed and q == "When was Johann Sebastian Bach born?"
    q, changed = resolve_from_history("Was this the end of his career?", ["Who is Pierre Monteux?"], nlp)
    assert changed and q == "Was this the end of Pierre Monteux's career?"


def test_transcript_kept_for_generation():
    conv = Conversation(fx)
    conv.handle("Can you make it faster?")
    out = conv.handle("I meant the PostgreSQL query")
    assert out["retrieval_query"] == "Can you make the PostgreSQL query faster?"
    assert [t["role"] for t in out["transcript"]] == ["user", "assistant", "user"]
    assert out["transcript"][-1]["text"] == "I meant the PostgreSQL query"


def test_conversation_loop():
    conv = Conversation(fx)
    first = conv.handle("Can you make it faster?")
    assert first["route"] == "Clarify", first
    second = conv.handle("the PostgreSQL query")
    assert second["route"] != "Clarify", second
    assert second["query"] == "Can you make the PostgreSQL query faster?"


if __name__ == "__main__":
    tests = [v for k, v in dict(globals()).items() if k.startswith("test_")]
    for t in tests:
        t()
        print(f"PASS {t.__name__}")

    print("\nEnd-to-end examples:")
    for q, h in [("How does Tokio handle async in Rust?", []),
                 ("python memory", []),
                 ("What is the latest PostgreSQL release?", []),
                 ("Can you make it faster?", [])]:
        res = fx.extract(q, history=h)
        dec = route_query(extract_evidence(res.raw))
        print(f"  {q:<42} P(H)={dec['p_hallucination_high']:.2f} -> {dec['route']}")

    print("\nClarify loop:")
    conv = Conversation(fx)
    for msg in ["Can you make it faster?", "the PostgreSQL query"]:
        out = conv.handle(msg)
        print(f"  user: {msg!r:<28} -> {out['route']:<8}"
              + (f" ask: {out['ask_user']}" if "ask_user" in out else f" query: {out['query']}"))
