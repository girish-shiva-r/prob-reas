"""
Signal extractors for Bayesian Epistemic Routing v3 - tier 0 (question) and tier 1 (retrieval)
==============================================================================================
Produces the symptoms of ber_v3.SYMPTOMS in the row format learn_v3.py reads:
    {"symptoms": {categorical symptom: state}, "raw": {real-valued signal: number},
     "context": {"HistoryCandidates": ...}, "details": {...what was found, for debugging}}
Real-valued signals go to "raw" and are discretised later with LEARNED cut-offs (learn_v3.RAW_SPEC);
categorical ones are decided here. A signal that does not apply is LEFT OUT (the network
marginalises it), never guessed.

Tier 0 (question text + history, ~ms):   NameCollision, Disambiguator, TemporalAnchoring, TimeWords,
                                         ResidualReferences, FollowUpForm, MultiHopForm, HistoryCandidates
Tier 1 (retrieved passages, CPU/GPU):    RerankerLogit, RerankMargin, TitleDiversity, EntityCoverage,
                                         SubjectTypeMatch, AnswerTypePresence, Answerability,
                                         CandidateAgreement, SupportSpread, EvidenceAge

Models (all injectable, so tests can use stand-ins):
    spaCy en_core_web_sm                       NER / parsing
    cross-encoder/ms-marco-MiniLM-L-6-v2       reranker, RAW logits (no sigmoid)
    deepset/roberta-base-squad2                extractive QA with a no-answer head (answerability)
    TitleIndex (build_title_index.py)          how many Wikipedia pages share a name
"""

from __future__ import annotations

import datetime as dt
import gzip
import math
import re
from dataclasses import dataclass, field
from typing import Callable, Optional, Sequence

import numpy as np

# ============================================================================ shared text helpers
NON_SPECIFIC = {"CARDINAL", "ORDINAL", "QUANTITY", "PERCENT", "MONEY", "TIME", "DATE"}
WH = {"who", "whom", "whose", "what", "which", "when", "where", "why", "how"}
PAREN = re.compile(r"\s*\(([^)]*)\)\s*$")


def base_name(title: str) -> str:
    """'Pulse (2001 film)' -> 'pulse'."""
    return PAREN.sub("", title).strip().lower()


def norm(s: str) -> str:
    s = s.lower()
    s = re.sub(r"[^\w\s]", " ", s)
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())


def question_entities(doc) -> list[str]:
    """
    Named things the question is about: spaCy entities (minus numbers/dates), runs of proper nouns,
    and Title-Case runs that spaCy missed ('What genre is Pulse?', 'The Inspector General').
    Longest spans first, no overlaps.
    """
    spans, covered = [], set()

    def add(start, end):
        if end <= start or covered.intersection(range(start, end)):
            return
        text = doc[start:end].text.strip(" '\"?.,")
        if text and text.lower() not in WH:
            spans.append(text)
            covered.update(range(start, end))

    for e in sorted(doc.ents, key=lambda e: -(e.end - e.start)):
        if e.label_ not in NON_SPECIFIC:
            add(e.start, e.end)
    i = 0
    while i < len(doc):                                    # Title-Case / PROPN runs
        t = doc[i]
        cap = t.text[:1].isupper() and t.text.lower() not in WH and not (i == 0 and t.is_stop)
        if (t.pos_ == "PROPN" or (cap and i > 0)) and not t.is_punct:
            j = i
            while j + 1 < len(doc) and (doc[j + 1].pos_ == "PROPN" or doc[j + 1].text[:1].isupper()
                                        or (doc[j + 1].lower_ in {"of", "the", "and", "de", "'s"}
                                            and j + 2 < len(doc) and doc[j + 2].text[:1].isupper())):
                j += 1
            start = i - 1 if i > 0 and doc[i - 1].lower_ == "the" and doc[i - 1].text[0].isupper() else i
            add(start, j + 1)
            i = j + 1
        else:
            i += 1
    for m in re.finditer(r"[\"“']([^\"”']{2,60})[\"”']", doc.text):   # quoted titles
        s = doc.char_span(m.start(1), m.end(1), alignment_mode="expand")
        if s is not None:
            add(s.start, s.end)
    return spans


# ============================================================================ title index
class TitleIndex:
    """
    base name -> number of Wikipedia pages sharing it, and whether a '(disambiguation)' page exists.
    Only names with 2+ pages are stored; anything else counts as 1 (unique or unknown).
    Build once with build_title_index.py from enwiki-latest-all-titles-in-ns0.gz.
    """

    def __init__(self, counts: Optional[dict] = None, disambig: Optional[set] = None):
        self.counts = counts or {}
        self.disambig = disambig or set()

    @classmethod
    def from_titles(cls, titles):
        from collections import Counter
        counts, dis = Counter(), set()
        for t in titles:
            t = t.replace("_", " ")
            m = PAREN.search(t)
            if m and m.group(1).lower() == "disambiguation":
                dis.add(base_name(t))
                continue
            counts[base_name(t)] += 1
        return cls({k: v for k, v in counts.items() if v >= 2}, dis)

    def save(self, path):
        with gzip.open(path, "wt", encoding="utf-8") as f:
            for k, v in self.counts.items():
                f.write(f"{k}\t{v}\t{int(k in self.disambig)}\n")
            for k in self.disambig - set(self.counts):
                f.write(f"{k}\t1\t1\n")

    @classmethod
    def load(cls, path):
        counts, dis = {}, set()
        with gzip.open(path, "rt", encoding="utf-8") as f:
            for line in f:
                k, v, d = line.rstrip("\n").split("\t")
                counts[k] = int(v)
                if d == "1":
                    dis.add(k)
        return cls(counts, dis)

    def collisions(self, name: str) -> int:
        k = base_name(name)
        n = self.counts.get(k, 1)
        return max(n, 4) if k in self.disambig else n        # a disambiguation page = many meanings


# ============================================================================ tier 0
TYPE_WORDS = (r"film|movie|novel|book|album|song|single|band|group|tv series|series|show|play|opera|musical|"
              r"video game|game|company|city|town|village|river|mountain|county|state|country|province|"
              r"magazine|newspaper|ship|footballer|politician|actor|actress|singer|painting|poem|church|"
              r"school|university|team|club|station|airport|album")
DISAMBIGUATOR = re.compile(rf"\b(the|a)?\s*(\d{{4}}\s+)?({TYPE_WORDS})\b|\(\s*[^)]+\)|\b(19|20)\d{{2}}\b", re.I)
VOLATILE = re.compile(
    r"\b(ceo|chief executive|president|prime minister|chancellor|governor|mayor|chairman|chair|leader|head coach|"
    r"coach|manager|captain|owner|population|record|champion|champions|winner|won the (last|latest|most recent)|"
    r"current|currently|latest|newest|most recent|recent|price|stock|ranking|ranked|tallest|largest|richest|"
    r"won|wins|winner|winners|title holder|holds the title|"
    r"oldest living|plays for|signed (with|for)|married to|spouse|wife|husband|members?|squad|roster|"
    r"how many (people|users|employees|subscribers|members)|net worth|salary|age of)\b", re.I)
TIME_ANCHOR = re.compile(r"\b(in|by|during|as of|before|after|since|until|of)\s+(the\s+)?(\d{4}|\d{4}s|"
                         r"(january|february|march|april|may|june|july|august|september|october|november|december)"
                         r"( \d{1,2})?,? ?\d{0,4})\b|\b(19|20)\d{2}\b|\b(now|today|currently|current|right now|"
                         r"this year|at present|presently)\b", re.I)
TIME_WORDS = re.compile(r"\b(latest|current(ly)?|now|today|tonight|recent(ly)?|this (year|month|week)|"
                        r"newest|upcoming|right now|as of|so far|nowadays|breaking|new release|still|"
                        r"at present|presently|these days)\b", re.I)
YEAR = re.compile(r"\b(19\d{2}|20\d{2})\b")
FOLLOWUP_START = re.compile(r"^(and|also|what about|how about|but|so|then|ok|okay|same for|what else|"
                            r"anything else|why|how come|really)\b", re.I)
ELLIPSIS = re.compile(r"^(and|what about|how about|same for|what else)\b", re.I)
REFERRING = {"it", "its", "they", "them", "their", "this", "that", "these", "those", "he", "she", "him", "her",
             "his", "hers", "theirs"}
PERSONAL = {"he", "him", "his", "she", "her", "hers"}
COMPARISON = re.compile(
    r"\b(which|who|what)\b[^?]*\b(first|earlier|earliest|older|oldest|younger|youngest|later|latest|longer|"
    r"shorter|larger|bigger|smaller|more|less|fewer|higher|lower|taller|closer|further|farther|faster|"
    r"slower|heavier|lighter)\b[^?]*\bor\b"
    r"|\b(are|were|is|was|do|did|does)\b[^?]+\band\b[^?]+\b(both|same|different|either)\b"
    r"|\bboth\b[^?]*\band\b|\b(same|different) (nationality|country|genre|year|city|state|profession|occupation)\b"
    r"|\bin common\b", re.I)


WEATHER_VERBS = {"rain", "snow", "hail", "drizzle", "pour", "thunder", "sleet", "storm", "freeze"}
DUMMY_IT_ADJ = {"sunny", "cold", "hot", "warm", "windy", "cloudy", "late", "early", "dark", "light", "time",
                "possible", "true", "likely", "necessary", "important", "hard", "easy", "safe"}


def _dummy_it(p) -> bool:
    """'Is it raining?', 'Is it cold in Oslo?', 'Is it true that ...?' - 'it' points at nothing."""
    if p.lower_ != "it":
        return False
    h = p.head
    if h.lemma_ in WEATHER_VERBS:
        return True
    return any(c.lower_ in DUMMY_IT_ADJ and c.dep_ in ("acomp", "attr") for c in h.children)


def pronouns_needing_resolution(doc) -> list:
    """Referring pronouns with no antecedent inside the question itself."""
    out = []
    for p in doc:
        if p.lower_ not in REFERRING or p.dep_ == "expl" or _dummy_it(p):
            continue
        if p.lower_ in {"that", "this", "these", "those"} and p.pos_ not in ("PRON",):
            continue                                    # determiner: "that film" names its own noun
        if p.pos_ not in ("PRON", "DET") and p.tag_ not in ("PRP", "PRP$"):
            continue
        if p.lower_ in PERSONAL:
            earlier = [t for t in doc[:p.i] if t.ent_type_ == "PERSON" or t.pos_ == "PROPN"]
        else:
            earlier = [nc for nc in doc.noun_chunks if nc.end <= p.i and nc.root.pos_ not in ("PRON",)
                       and nc.root.lower_ not in WH]
        if not earlier:
            out.append(p.text)
    return out


LIGHT_NOUNS = {"name", "title", "number", "type", "kind", "part", "one", "year", "date", "member", "sort"}


def _in_name(tok) -> bool:
    """Part of a capitalised name ('Special Region of Yogyakarta', 'William III of England')."""
    return tok.text[:1].isupper() and tok.i > 0


def _of_depth(tok) -> int:
    """'father of the director of the film Inception' -> 2 (light nouns like 'name of' do not count,
    nor 'of' inside a name such as 'William III of England')."""
    best = 0
    for prep in tok.children:
        if prep.dep_ == "prep" and prep.lower_ == "of":
            for obj in prep.children:
                if obj.dep_ == "pobj":
                    inside = _in_name(tok) and _in_name(obj)
                    step = 0 if (tok.lemma_.lower() in LIGHT_NOUNS or inside) else 1
                    best = max(best, _of_depth(obj) + step)
    return best


def _asked_noun(doc):
    """The noun the question asks ABOUT: 'Who is THE MAN that founded Apple' -> man (skip light nouns)."""
    root = next((t for t in doc if t.dep_ == "ROOT"), None)
    if root is None or root.lemma_ != "be":
        return None
    if not any(c.lower_ in WH and c.dep_ in ("attr", "nsubj") for c in root.children):
        return None
    n = next((c for c in root.children if c.dep_ in ("nsubj", "attr") and c.pos_ in ("NOUN", "PROPN")), None)
    while n is not None and n.lemma_.lower() in LIGHT_NOUNS:
        n = next((g for prep in n.children if prep.dep_ == "prep" and prep.lower_ == "of"
                  for g in prep.children if g.dep_ == "pobj"), None)
    return n


def multi_hop_form(doc, text) -> tuple[str, dict]:
    """
    Comparison: 'which ... first, X or Y', 'are X and Y both ...'
    Bridge:     a chain of 2+ 'of' links ('father of the director of Inception'), a possessive chain
                ('X's wife's father'), or a relative clause hanging on a noun OTHER than the one the
                question asks about ('the county of THE TOWN where Austen was born').
    """
    if COMPARISON.search(text):
        return "Comparison", {"pattern": "comparison"}
    depth = max((_of_depth(t) for t in doc if t.pos_ in ("NOUN", "PROPN", "PRON")), default=0)
    poss = len(re.findall(r"\w's\b", text))
    asked = _asked_noun(doc)
    relcl = [t for t in doc if t.dep_ in ("relcl", "acl") and t.head.pos_ in ("NOUN", "PROPN")
             and t.head != asked and t.head.lemma_.lower() not in LIGHT_NOUNS and not _in_name(t.head)]
    if depth >= 2 or poss >= 2 or relcl:
        return "Bridge", {"of_depth": depth, "possessives": poss, "relcl_on": [t.head.text for t in relcl]}
    return "Single", {}


def history_candidates(nlp, history_turns: Sequence[tuple[str, str]], pronoun: Optional[str]) -> tuple[str, list]:
    """
    Antecedents the previous turn offers for the unresolved pronoun: entities in the previous
    question AND answer, filtered by type (he/she -> people; it -> not people). -> None / One / Many
    """
    if not history_turns:
        return "None", []
    q, a = history_turns[-1]
    cands = []
    for part in (q, a or ""):
        d = nlp(part)
        for e in d.ents:
            if e.label_ in NON_SPECIFIC:
                continue
            is_person = e.label_ == "PERSON"
            if pronoun and pronoun.lower() in PERSONAL and not is_person:
                continue
            if pronoun and pronoun.lower() in {"it", "its"} and is_person:
                continue
            cands.append(e.text)
        if not d.ents and not (pronoun and pronoun.lower() in PERSONAL):
            cands += [s for s in question_entities(d)]     # no NER hit: Title-Case runs (not for he/she)
    seen, uniq = set(), []
    for c in cands:
        k = norm(c)
        if k and not any(k in s or s in k for s in seen):
            seen.add(k)
            uniq.append(c)
    return ("None" if not uniq else "One" if len(uniq) == 1 else "Many"), uniq


@dataclass
class Tier0Config:
    model_cutoff_year: int = 2025           # knowledge cut-off of the answering LLM


def tier0(nlp, question: str, history: Sequence[tuple[str, str]] = (), titles: Optional[TitleIndex] = None,
          cfg: Tier0Config = Tier0Config()) -> dict:
    """history: earlier turns as (question, answer) pairs, oldest first."""
    doc = nlp(question)
    text = question.strip()
    ents = question_entities(doc)
    sym, raw, det = {}, {}, {"entities": ents}

    # C1 ReferentAmbiguity
    if titles is not None and ents:
        counts = {e: titles.collisions(e) for e in ents}
        raw["name_collision_count"] = max(counts.values())
        det["collisions"] = counts
    if ents:
        sym["Disambiguator"] = "Present" if DISAMBIGUATOR.search(text) else "Absent"

    # C2 / C9 time
    volatile = bool(VOLATILE.search(text))
    anchored = bool(TIME_ANCHOR.search(text))
    sym["TemporalAnchoring"] = ("Stable" if not volatile else "VolatileAnchored" if anchored
                                else "VolatileUnanchored")
    sym["TimeWords"] = "Present" if (TIME_WORDS.search(text) or any(int(y) >= cfg.model_cutoff_year
                                                                    for y in YEAR.findall(text))) else "Absent"

    # C3 ContextDependence
    unresolved = pronouns_needing_resolution(doc)
    ellipsis = bool(ELLIPSIS.search(text)) and len(ents) == 0
    raw["residual_reference_count"] = len(unresolved) + int(ellipsis)
    det["unresolved"] = unresolved + (["<ellipsis>"] if ellipsis else [])
    if history:
        short = len(re.findall(r"[\w'-]+", text)) <= 4
        sym["FollowUpForm"] = "Yes" if (FOLLOWUP_START.search(text) or short or unresolved) else "No"
    else:
        sym["FollowUpForm"] = "No"
    hc, cands = history_candidates(nlp, history, unresolved[0] if unresolved else None)
    det["history_candidates"] = cands

    # C8 ReasoningLoad
    form, why = multi_hop_form(doc, text)
    sym["MultiHopForm"] = form
    det["multi_hop"] = why
    return {"symptoms": sym, "raw": raw, "context": {"HistoryCandidates": hc}, "details": det}


# ============================================================================ tier 1
# relation phrase in the question -> the TYPE its subject must have
IMPLIED_TYPE = [
    (r"\b(screenwriter|screenplay|directed|director|cinematograph|starred|starring|cast of|box office)\b", "film"),
    (r"\b(producer of|produced)\b", "film"),
    (r"\b(author of|written by|who wrote|publisher|published|novel|illustrat)\b", "book"),
    (r"\b(record label|album|lead singer|band member|guitarist|drummer|performed by|recorded)\b", "music"),
    (r"\b(capital of|population of|located in|mayor of|what country is|in what country|which country is|"
     r"what state is|in which state)\b", "place"),
    (r"\b(born|died|spouse|wife|husband|occupation|father of|mother of|educated at|alma mater|plays for|"
     r"position does|what sport does .+ play|nationality)\b", "person"),
    (r"\b(founded|headquarter|ceo of|subsidiary|parent company)\b", "org"),
]
PAGE_TYPE = [
    (r"\b(\d{4} )?([a-z-]+ )?(film|movie|documentary|tv series|television series|sitcom|miniseries|episode)\b", "film"),
    (r"\b(novel|book|short story|novella|poem|play|memoir|comic|manga)\b", "book"),
    (r"\b(album|song|single|band|ep|rapper|singer|musician|composer|group|opera|symphony)\b", "music"),
    (r"\b(city|town|village|municipality|commune|county|province|state|country|district|river|mountain|"
     r"island|capital|region|neighbourhood|neighborhood)\b", "place"),
    (r"\b(footballer|politician|actor|actress|writer|painter|player|athlete|cricketer|scientist|director|"
     r"businessman|businesswoman|journalist|poet|bishop|general|officer|lawyer|judge|engineer)\b", "person"),
    (r"\b(company|corporation|organization|organisation|university|school|club|team|airline|party|"
     r"agency|brand|manufacturer|publisher)\b", "org"),
]
EXPECTED_ANSWER = [
    (r"^(who|whom)\b|\bwhose\b|\bwhich (person|actor|actress|author|player|director|writer|singer)\b",
     {"PERSON", "ORG", "NORP"}),
    (r"^when\b|\bwhat (year|date|day|month|decade|century)\b|\bin which year\b|\bwhen did\b", {"DATE", "TIME"}),
    (r"^where\b|\b(in )?(what|which) (city|country|state|town|place|region|continent|county|province)\b",
     {"GPE", "LOC", "FAC"}),
    (r"^how (many|much)\b|\bhow (old|long|tall|far|big|large)\b|\bwhat (number|percentage|amount)\b",
     {"CARDINAL", "QUANTITY", "MONEY", "PERCENT", "DATE", "TIME"}),
]


def implied_subject_type(question: str) -> Optional[str]:
    for pat, t in IMPLIED_TYPE:
        if re.search(pat, question, re.I):
            return t
    return None


def page_type(title: str, text: str) -> Optional[str]:
    """Coarse type of a page from its title qualifier, else from its first sentence."""
    m = PAREN.search(title or "")
    probes = [m.group(1)] if m else []
    first = re.split(r"(?<=[.!?])\s", re.sub(r"^[^:]{0,120}:\s*", "", text or "").strip(), maxsplit=1)[0]
    m2 = re.search(r"\b(is|was|are|were)\s+(a|an|the)\s+([^,.;]{0,80})", first, re.I)
    if m2:
        probes.append(m2.group(3))
    for p in probes:
        for pat, t in PAGE_TYPE:
            if re.search(pat, p, re.I):
                return t
    return None


def expected_answer_labels(question: str) -> Optional[set]:
    q = question.strip().lower()
    for pat, labels in EXPECTED_ANSWER:
        if re.search(pat, q):
            return labels
    return None


def _token_f1(a, b):
    a, b = norm(a).split(), norm(b).split()
    if not a or not b:
        return 0.0
    common = sum(min(a.count(w), b.count(w)) for w in set(a))
    if common == 0:
        return 0.0
    p, r = common / len(a), common / len(b)
    return 2 * p * r / (p + r)


def _covered(entity, text):
    try:
        from rapidfuzz import fuzz
        return fuzz.partial_ratio(norm(entity), norm(text)) >= 90
    except ImportError:
        return norm(entity) in norm(text)


Reranker = Callable[[str, Sequence[str]], np.ndarray]          # raw logits
QA = Callable[[str, Sequence[str]], list]                       # [(has_answer_prob, span), ...]


def load_reranker(name="cross-encoder/ms-marco-MiniLM-L-6-v2", device=None) -> Reranker:
    """RAW logits (no sigmoid): the sigmoid squeezed most scores into 0.99-1.0 in v2."""
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    tok = AutoTokenizer.from_pretrained(name)
    model = AutoModelForSequenceClassification.from_pretrained(name).to(device).eval()

    def rerank(query, passages, batch=32):
        out = []
        for i in range(0, len(passages), batch):
            enc = tok([query] * len(passages[i:i + batch]), list(passages[i:i + batch]), truncation=True,
                      max_length=512, padding=True, return_tensors="pt").to(device)
            with torch.no_grad():
                out.append(model(**enc).logits[:, 0].float().cpu().numpy())
        return np.concatenate(out) if out else np.zeros(0)
    return rerank


def load_qa(name="deepset/roberta-base-squad2", device=None, max_answer_tokens=30) -> QA:
    """Extractive QA with a no-answer option: P(has answer) = sigmoid(best span score - null score)."""
    import torch
    from transformers import AutoModelForQuestionAnswering, AutoTokenizer
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    tok = AutoTokenizer.from_pretrained(name)
    model = AutoModelForQuestionAnswering.from_pretrained(name).to(device).eval()

    def qa(question, passages):
        if not passages:
            return []
        enc = tok([question] * len(passages), list(passages), truncation="only_second", max_length=384,
                  padding=True, return_offsets_mapping=True, return_tensors="pt")
        offsets = enc.pop("offset_mapping")
        seq_ids = [enc.sequence_ids(i) for i in range(len(passages))]
        with torch.no_grad():
            o = model(**{k: v.to(device) for k, v in enc.items()})
        S, E = o.start_logits.float().cpu().numpy(), o.end_logits.float().cpu().numpy()
        res = []
        for i, p in enumerate(passages):
            ctx = [j for j, s in enumerate(seq_ids[i]) if s == 1]
            null = S[i, 0] + E[i, 0]
            best, span = -1e9, ""
            top_s = sorted(ctx, key=lambda j: -S[i, j])[:20]
            top_e = sorted(ctx, key=lambda j: -E[i, j])[:20]
            for a in top_s:
                for b in top_e:
                    if a <= b <= a + max_answer_tokens and S[i, a] + E[i, b] > best:
                        best = S[i, a] + E[i, b]
                        span = p[int(offsets[i, a, 0]):int(offsets[i, b, 1])]
            res.append((float(1 / (1 + math.exp(-(best - null)))), span))
        return res
    return qa


@dataclass
class Tier1Config:
    top_passages: int = 3            # what the LLM will be shown
    diversity_window: float = 2.0    # logits below the top passage that still count as "tied"
    fresh_days: int = 90
    answer_threshold: float = 0.5    # has-answer probability for a passage to count as answering


def tier1(nlp, question: str, passages: list[dict], rerank: Reranker, qa: QA, entities: Optional[list] = None,
          cfg: Tier1Config = Tier1Config(), today: Optional[dt.date] = None) -> dict:
    """
    passages: [{"title", "text", optional "date" (YYYY-MM-DD)}], any order (re-ranked here).
    Returns the ranked passages in details["ranked"] (the LLM gets the top cfg.top_passages).
    """
    sym, raw, det = {}, {}, {}
    if not passages:
        return {"symptoms": sym, "raw": raw, "details": {"ranked": []}}
    entities = entities if entities is not None else question_entities(nlp(question))
    logits = np.asarray(rerank(question, [p["text"] for p in passages]), dtype=float)
    order = np.argsort(-logits)
    ranked = [{**passages[i], "logit": float(logits[i])} for i in order]
    top = ranked[0]
    det["ranked"] = ranked

    # RerankerLogit / RerankMargin / TitleDiversity
    raw["rerank_logit"] = top["logit"]
    other = [p["logit"] for p in ranked[1:] if p.get("title") != top.get("title")]
    raw["rerank_margin"] = top["logit"] - other[0] if other else 10.0
    near = {p.get("title") for p in ranked if p["logit"] >= top["logit"] - cfg.diversity_window}
    raw["title_diversity"] = len(near)

    shown = ranked[:cfg.top_passages]
    # EntityCoverage
    if entities:
        hit = [e for e in entities if any(_covered(e, p.get("title", "") + " " + p["text"]) for p in shown)]
        raw["entity_coverage"] = len(hit) / len(entities)
        det["entities_found"] = hit
    # SubjectTypeMatch
    want = implied_subject_type(question)
    have = page_type(top.get("title", ""), top["text"])
    det["implied_type"], det["top_page_type"] = want, have
    sym["SubjectTypeMatch"] = ("Unknown" if want is None or have is None
                               else "Match" if want == have else "Mismatch")
    # AnswerTypePresence
    labels = expected_answer_labels(question)
    if labels is not None:
        found = {e.label_ for p in shown for e in nlp(p["text"]).ents}
        sym["AnswerTypePresence"] = "Present" if labels & found else "Absent"
        det["expected_answer_labels"] = sorted(labels)
    # Answerability / CandidateAgreement
    qa_out = qa(question, [p["text"] for p in shown])
    if qa_out:
        raw["answerability"] = max(p for p, _ in qa_out)
        spans = [s for p, s in qa_out if p >= cfg.answer_threshold and s.strip()]
        det["candidate_answers"] = spans
        if len(spans) == 1:
            sym["CandidateAgreement"] = "Single"
        elif len(spans) >= 2:
            agree = all(_token_f1(spans[0], s) >= 0.5 for s in spans[1:])
            sym["CandidateAgreement"] = "Agree" if agree else "Disagree"
    # SupportSpread
    if len(entities) >= 2:
        per = [{e for e in entities if _covered(e, p["text"])} for p in shown]
        union = set().union(*per)
        sym["SupportSpread"] = ("Spread" if len(union) >= 2 and not any(s >= union for s in per)
                                else "Concentrated")
    else:
        sym["SupportSpread"] = "Concentrated"
    # EvidenceAge
    dates = []
    for p in shown:
        try:
            dates.append(dt.date.fromisoformat(str(p.get("date"))[:10]))
        except (TypeError, ValueError):
            pass
    if dates:
        today = today or dt.date.today()
        sym["EvidenceAge"] = "Fresh" if (today - max(dates)).days <= cfg.fresh_days else "Stale"
    return {"symptoms": sym, "raw": raw, "details": det}


def merge(*parts) -> dict:
    """Combine tier outputs into one learn_v3 row (symptoms / raw / context / details)."""
    row = {"symptoms": {}, "raw": {}, "context": {}, "details": {}}
    for p in parts:
        for k in row:
            row[k].update(p.get(k, {}))
    return row
