"""
Hand-picked checks for the tier-0 / tier-1 signal extractors.
    python test_signals_v3.py              # logic checks (spaCy only; reranker and QA are stand-ins)
    python test_signals_v3.py --models     # also sanity-checks the REAL reranker and QA models
Each case states what a careful human would say the signal should be.
"""

import argparse
import datetime as dt
import sys

import numpy as np
import spacy

import signals_v3 as S

TITLES = S.TitleIndex({"pulse": 9, "belly": 5, "escape": 14, "inception": 2}, {"pulse", "escape"})


# ----------------------------------------------------------------------------- tier 0 cases
# (question, history, {path: expected}) ; path = "symptoms.X" | "raw.X" | "context.X"
T0 = [
    ("What genre is Pulse?", [], {"raw.name_collision_count": 9, "symptoms.Disambiguator": "Absent",
                                  "symptoms.TemporalAnchoring": "Stable", "symptoms.MultiHopForm": "Single",
                                  "raw.residual_reference_count": 0}),
    ("Who directed Pulse, the 2001 film?", [], {"symptoms.Disambiguator": "Present"}),
    ("Who wrote the novel Babette's Feast?", [], {"symptoms.Disambiguator": "Present",
                                                  "symptoms.MultiHopForm": "Single"}),
    ("Who is the current CEO of Twitter?", [], {"symptoms.TemporalAnchoring": "VolatileAnchored",
                                                "symptoms.TimeWords": "Present"}),
    ("Who won the World Cup?", [], {"symptoms.TemporalAnchoring": "VolatileUnanchored",
                                    "symptoms.TimeWords": "Absent"}),
    ("Who won the 2010 World Cup?", [], {"symptoms.TemporalAnchoring": "VolatileAnchored",
                                         "symptoms.TimeWords": "Absent"}),
    ("What is the population of Chennai?", [], {"symptoms.TemporalAnchoring": "VolatileUnanchored"}),
    ("When was the Eiffel Tower built?", [], {"symptoms.TemporalAnchoring": "Stable",
                                              "symptoms.TimeWords": "Absent"}),
    ("Tell me the latest news about SpaceX", [], {"symptoms.TimeWords": "Present"}),
    ("What is the capital of France?", [], {"symptoms.MultiHopForm": "Single",
                                            "symptoms.TemporalAnchoring": "Stable"}),
    ("When was he born?", [], {"raw.residual_reference_count": 1, "symptoms.FollowUpForm": "No",
                               "context.HistoryCandidates": "None"}),
    ("When was he born?", [("Who directed Inception?", "Christopher Nolan")],
     {"raw.residual_reference_count": 1, "symptoms.FollowUpForm": "Yes", "context.HistoryCandidates": "One"}),
    ("Where was she born?", [("Who starred in Titanic?", "Leonardo DiCaprio and Kate Winslet")],
     {"symptoms.FollowUpForm": "Yes", "context.HistoryCandidates": "Many"}),
    ("What about Paris?", [("What is the capital of France?", "Paris")],
     {"symptoms.FollowUpForm": "Yes", "raw.residual_reference_count": 0}),
    ("And the second one?", [("Name two rivers in India.", "The Ganges and the Yamuna")],
     {"symptoms.FollowUpForm": "Yes", "raw.residual_reference_count": 1}),
    ("Is it raining in London?", [], {"raw.residual_reference_count": 0}),
    ("Python is popular, so why is it slow?", [], {"raw.residual_reference_count": 0}),
    ("What did they do after the war?", [], {"raw.residual_reference_count": 1}),
    ("Which magazine was started first, Arthur's Magazine or First for Women?", [],
     {"symptoms.MultiHopForm": "Comparison"}),
    ("Are Christopher Nolan and Steven Spielberg both directors?", [], {"symptoms.MultiHopForm": "Comparison"}),
    ("Who is the father of the director of the film Inception?", [], {"symptoms.MultiHopForm": "Bridge"}),
    ("In which county is the town where Jane Austen was born?", [], {"symptoms.MultiHopForm": "Bridge"}),
    ("Who was born first, Albert Einstein or Niels Bohr?", [], {"symptoms.MultiHopForm": "Comparison"}),
    ("What sport does Turkish Seniors Open play?", [], {"symptoms.MultiHopForm": "Single"}),
]


def get(out, path):
    sec, key = path.split(".", 1)
    return out.get(sec, {}).get(key, "<missing>")


# ----------------------------------------------------------------------------- tier 1 cases
def fake_rerank(table):
    return lambda q, texts: np.array([table.get(t[:30], -5.0) for t in texts])


def fake_qa(table):
    return lambda q, texts: [table.get(t[:30], (0.05, "")) for t in texts]


P = {
    "inc": {"title": "Inception", "text": "Inception is a 2010 science fiction film written and directed by Christopher Nolan."},
    "int": {"title": "Interstellar", "text": "Interstellar is a 2014 epic science fiction film directed by Christopher Nolan."},
    "dun": {"title": "Dunkirk (2017 film)", "text": "Dunkirk is a 2017 war film written, directed and produced by Christopher Nolan."},
    "belly_band": {"title": "Belly (band)", "text": "Belly is an American alternative rock band formed in 1991 by Tanya Donelly."},
    "belly_film": {"title": "Belly (1998 film)", "text": "Belly is a 1998 American crime film written and directed by Hype Williams."},
    "pulse_film": {"title": "Pulse (2001 film)", "text": "Pulse is a 2001 Japanese techno-horror film directed by Kiyoshi Kurosawa."},
    "pulse_band": {"title": "Pulse (band)", "text": "Pulse is a British progressive rock band formed in London."},
    "pulse_album": {"title": "Pulse (Pink Floyd album)", "text": "Pulse is a live album by the English rock band Pink Floyd, released in 1995."},
    "arthur": {"title": "Arthur's Magazine", "text": "Arthur's Magazine was an American literary periodical first published in 1844."},
    "first": {"title": "First for Women", "text": "First for Women is a woman's magazine published by Bauer Media Group, started in 1989."},
    "ceo_old": {"title": "Twitter", "text": "Jack Dorsey is the chief executive officer of Twitter.", "date": "2019-05-01"},
    "ceo_new": {"title": "X Corp", "text": "Linda Yaccarino became the chief executive officer of X Corp.", "date": "2026-09-20"},
    "rnd": {"title": "Banana", "text": "A banana is an elongated, edible fruit produced by several kinds of plants."},
}
K = lambda k: P[k]["text"][:30]
T1 = [
    ("Who directed Inception?", ["inc", "int", "dun"], {K("inc"): 8.5, K("int"): 3.0, K("dun"): 2.0},
     {K("inc"): (0.95, "Christopher Nolan")},
     {"raw.rerank_logit": 8.5, "raw.title_diversity": 1, "raw.entity_coverage": 1.0,
      "symptoms.SubjectTypeMatch": "Match", "symptoms.AnswerTypePresence": "Present",
      "symptoms.CandidateAgreement": "Single", "symptoms.SupportSpread": "Concentrated"}),
    ("Who was the screenwriter for Belly?", ["belly_band", "belly_film", "rnd"],
     {K("belly_band"): 6.0, K("belly_film"): 5.5, K("rnd"): -8.0}, {K("belly_film"): (0.9, "Hype Williams")},
     {"symptoms.SubjectTypeMatch": "Mismatch", "raw.title_diversity": 2}),
    ("What genre is Pulse?", ["pulse_film", "pulse_band", "pulse_album"],
     {K("pulse_film"): 4.0, K("pulse_band"): 3.6, K("pulse_album"): 3.1},
     {K("pulse_film"): (0.8, "techno-horror"), K("pulse_band"): (0.8, "progressive rock")},
     {"raw.title_diversity": 3, "symptoms.SubjectTypeMatch": "Unknown", "symptoms.AnswerTypePresence": "<missing>",
      "symptoms.CandidateAgreement": "Disagree"}),
    ("Who directed Interstellar and Dunkirk?", ["int", "dun", "inc"], {K("int"): 7.0, K("dun"): 6.5, K("inc"): 1.0},
     {K("int"): (0.9, "Christopher Nolan"), K("dun"): (0.9, "Christopher Nolan")},
     {"symptoms.CandidateAgreement": "Agree"}),
    ("Which magazine was started first, Arthur's Magazine or First for Women?", ["arthur", "first", "rnd"],
     {K("arthur"): 5.0, K("first"): 4.8, K("rnd"): -9.0}, {},
     {"symptoms.SupportSpread": "Spread", "raw.entity_coverage": 1.0}),
    ("Who is the current CEO of Twitter?", ["ceo_old", "rnd"], {K("ceo_old"): 6.0, K("rnd"): -9.0}, {},
     {"symptoms.EvidenceAge": "Stale"}),
    ("Who is the current CEO of X Corp?", ["ceo_new", "rnd"], {K("ceo_new"): 6.0, K("rnd"): -9.0}, {},
     {"symptoms.EvidenceAge": "Fresh"}),
    ("Who directed Inception?", ["rnd"], {K("rnd"): -9.0}, {},
     {"raw.entity_coverage": 0.0, "raw.rerank_logit": -9.0, "symptoms.AnswerTypePresence": "Absent"}),
]


def run(models=False):
    nlp = spacy.load("en_core_web_sm")
    fails, total = [], 0
    print("TIER 0")
    for q, hist, exp in T0:
        out = S.tier0(nlp, q, hist, TITLES, S.Tier0Config(model_cutoff_year=2025))
        for path, want in exp.items():
            got = get(out, path)
            total += 1
            ok = got == want or (isinstance(want, float) and isinstance(got, float) and abs(got - want) < 1e-6)
            if not ok:
                fails.append((q, path, want, got, out["details"]))
            print(f"  {'ok ' if ok else 'FAIL'} {q[:58]:<58} {path:<34} want {want!s:<20} got {got}")
    print("\nTIER 1 (stand-in reranker / QA)")
    for q, keys, logits, qa, exp in T1:
        passages = [P[k] for k in keys]
        ents = S.question_entities(nlp(q))
        out = S.tier1(nlp, q, passages, fake_rerank(logits), fake_qa(qa), ents, today=dt.date(2026, 11, 2))
        for path, want in exp.items():
            got = get(out, path)
            total += 1
            ok = got == want or (isinstance(want, float) and isinstance(got, (int, float)) and abs(got - want) < 1e-6)
            if not ok:
                fails.append((q, path, want, got, {k: v for k, v in out["details"].items() if k != "ranked"}))
            print(f"  {'ok ' if ok else 'FAIL'} {q[:58]:<58} {path:<34} want {want!s:<20} got {got}")
    if models:
        print("\nREAL MODELS")
        rr, qa = S.load_reranker(), S.load_qa()
        l = rr("Who directed Inception?", [P["inc"]["text"], P["rnd"]["text"]])
        total += 1
        ok = l[0] > 3 and l[1] < 0
        print(f"  {'ok ' if ok else 'FAIL'} reranker raw logits relevant {l[0]:.2f} > 3, irrelevant {l[1]:.2f} < 0")
        fails += [] if ok else [("reranker", "", "", l, {})]
        a = qa("Who directed Inception?", [P["inc"]["text"], P["rnd"]["text"]])
        total += 1
        ok = a[0][0] > 0.5 and "Nolan" in a[0][1] and a[1][0] < 0.5
        print(f"  {'ok ' if ok else 'FAIL'} QA has-answer {a[0][0]:.2f} ('{a[0][1]}'), unanswerable {a[1][0]:.2f}")
        fails += [] if ok else [("qa", "", "", a, {})]
    print(f"\n{total - len(fails)}/{total} checks pass")
    for q, path, want, got, det in fails:
        print(f"  FAIL {q} | {path}: want {want}, got {got}\n       details {det}")
    return not fails


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", action="store_true")
    sys.exit(0 if run(ap.parse_args().models) else 1)
