"""
Clarify loop for Bayesian Epistemic Routing - model-free by default
-------------------------------------------------------------------
When the router says "Clarify", we need (1) a clarifying question and
(2) a way to fold the user's answer back into the query. Both are done
here with templates and rules, so no extra model runs on the device.

Why rules work here: the network already tells us WHY it wants to
clarify (which inputs came out High). A question aimed at that one gap
gets a short, targeted answer ("the database"), and a targeted answer
can be merged mechanically:
  * a pronoun was unclear  -> replace the pronoun with the answer
  * anything else          -> append the answer to the original query
Dense retrievers handle "original query + extra detail" well, so the
merged text does not need to be fluent English to retrieve the right
chunks; the generator LLM later sees the full conversation anyway.

Optional: pass rewrite_fn= to reuse the SAME local LLM the OS already
runs for answering (no new model) when the rule-based merge isn't enough.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Callable, Optional

from bayesian_epistemic_routing import extract_evidence, route_query
from feature_extractors import FeatureExtractor, pronouns_needing_resolution, specific_entities

MAX_CLARIFY_ROUNDS = 2

# Which High input to ask about first: the most specific, easiest to answer.
CLARIFY_PRIORITY = [
    "UnresolvedReferences", "FollowUpStatus", "QueryReformulation",
    "EntitySpecificity", "QueryLength", "TopicComplexity", "QuestionType",
]


def high_inputs(evidence: dict) -> list[str]:
    return [k for k, v in evidence.items() if v == "High"]


def make_clarifying_question(query: str, evidence: dict, details: dict) -> tuple[str, str]:
    """Returns (question_text, reason) where reason is the input we asked about."""
    high = set(high_inputs(evidence))
    reason = next((r for r in CLARIFY_PRIORITY if r in high), "QueryLength")
    options = details.get("previous_turn_entities", [])
    pick = f": {' or '.join(options[:3])}?" if len(options) >= 2 else "?"

    if reason == "UnresolvedReferences":
        pron = (details.get("unresolved_pronouns") or ["it"])[0]
        q = f'When you say "{pron}", what are you referring to{pick}'
    elif reason == "FollowUpStatus":
        q = f"Which earlier topic is this about{pick}"
    elif reason == "QueryReformulation":
        q = "It looks like my last answer missed. What was wrong or missing?"
    elif reason in ("EntitySpecificity", "QueryLength"):
        q = "Could you be more specific - which person, place, work or thing do you mean?"
    elif reason == "TopicComplexity":
        q = "Your question has several parts. Which part should I answer first?"
    else:  # QuestionType: opinion / open-ended
        q = "What matters most for you here (for example speed, cost, ease of use)?"
    return q, reason


# Conversational wrappers people put around the actual answer:
# "I meant the database", "oh sorry, I was talking about X", "it's the one about X".
LEAD_FILLER = re.compile(
    r"^(?:(?:(?:yes|yeah|no|nope|oh|ah|sorry|well|ok|okay|um+|uh+|so|actually|basically)\s*)+[,!.]\s*"
    r"|(?:oh|ah|um+|uh+)\s+"
    r"|(?:mostly|mainly|probably|definitely|really|i think)\s+(?=\S)"
    r"|i\s*(?:'m|am|was)?\s*(?:meant|mean|meaning|talking about|referring to|asking about|thinking of)\s+"
    r"|(?:it|that|this)\s*(?:'s|is|was)\s+(?:about\s+)?"
    r"|(?:the one|the thing)\s+(?:about|with|for|on)\s+"
    r"|(?:talking|referring|asking)\s+about\s+|about\s+|regarding\s+)",
    re.I)
ONLY_FILLER = re.compile(r"^(?:(?:yes|yeah|yep|no|nope|oh|ah|sorry|well|ok|okay|sure|um+|uh+|hmm+)[\s,.!?]*)*$", re.I)
TRAIL_FILLER = re.compile(r"[\s,]*(?:sorry|please|thanks|thank you|i guess|i think)?[\s.!?]*$", re.I)

PLURAL_FIX = {"is": "are", "was": "were", "has": "have", "does": "do",
              "isn't": "aren't", "wasn't": "weren't", "doesn't": "don't", "hasn't": "haven't"}
SINGULAR_PRONOUNS = {"it", "this", "that", "its"}


def clean_reply(reply: str) -> str:
    """Strip conversational wrappers: 'oh sorry, I was talking about X, sorry' -> 'X'."""
    text = reply.strip()
    for _ in range(4):                          # peel nested fillers: "oh sorry, I meant ..."
        stripped = LEAD_FILLER.sub("", text, count=1).strip()
        if stripped == text:
            break
        text = stripped
    text = TRAIL_FILLER.sub("", text).strip()
    return "" if ONLY_FILLER.match(text) else text


def answer_phrase(reply: str, nlp=None) -> tuple[Optional[str], bool]:
    """
    Pull the clean noun phrase out of a conversational reply.
      "I meant the database"                       -> ("the database", singular)
      "oh sorry, I was talking about the slow queries" -> ("the slow queries", plural)
    Returns (phrase or None, is_plural). None = no clean phrase found.
    """
    text = clean_reply(reply)
    if not text:
        return None, False                       # reply was only filler ("yes", "sorry")
    if nlp is None:                              # no parser: trust the cleaned text if it is short
        return (text, text.lower().endswith("s")) if len(text.split()) <= 6 else (None, False)

    doc = nlp(text)
    root = doc[:].root
    has_verb = any(t.pos_ in ("VERB", "AUX") for t in doc)
    has_subject = any(t.dep_ in ("nsubj", "nsubjpass") and t.head == root for t in doc)
    # The small parser sometimes reads a plural noun at the end as a verb
    # ("the Django and Flask frameworks" -> frameworks/VBZ). A verb that ends
    # the reply with nothing after it is treated as that noun.
    plural_noun_misread = root.i == len(doc) - 1 and root.tag_ == "VBZ"
    if (len(doc) <= 3 or root.pos_ in ("NOUN", "PROPN") or plural_noun_misread
            or not (has_verb and has_subject)):
        span = doc[:]                               # the whole remainder is the answer phrase
        if plural_noun_misread:
            root = None
    else:
        # a full sentence is left: take the biggest noun phrase with its modifiers
        chunks = [nc for nc in doc.noun_chunks if nc.root.pos_ != "PRON"]
        if not chunks:
            return None, False
        best = max(chunks, key=lambda nc: len(list(nc.root.subtree)))
        sub = list(best.root.subtree)
        span, root = doc[sub[0].i: sub[-1].i + 1], best.root
    # Capitalised plural-looking words are usually titles/names ("No Favors"): singular.
    plural = (root is None or (root.tag_ == "NNS" and not root.text[0].isupper())
              or any(t.lower_ == "and" for t in span))
    return span.text, plural


POSSESSIVE = {"its", "their", "his"}


def _possessive(phrase: str) -> str:
    if phrase.endswith("'s") or phrase.endswith("'"):
        return phrase
    return phrase + ("'" if phrase.endswith("s") and not phrase.endswith("ss") else "'s")


def _is_possessive(original: str, pron: str, nlp) -> bool:
    if pron.lower() in POSSESSIVE:
        return True
    if pron.lower() != "her":
        return False
    if nlp is not None:                         # "her career" (PRP$) vs "about her" (PRP)
        return any(t.lower_ == "her" and t.tag_ == "PRP$" for t in nlp(original))
    m = re.search(r"\bher\s+(\w+)", original, re.I)
    return bool(m) and m.group(1).lower() not in {"to", "in", "on", "at", "for", "with", "about", "from", "and", "or"}


def _replace_pronoun(original: str, pron: str, phrase: str, plural: bool, nlp=None) -> Optional[str]:
    """Swap the pronoun for the phrase, fixing possessives and is/are agreement."""
    possessive = _is_possessive(original, pron, nlp)
    words = original.split()
    for i, w in enumerate(words):
        core = re.sub(r"^\W+|\W+$", "", w)
        if core.lower() != pron.lower():
            continue
        lead = w[:w.lower().find(core.lower())]
        trail = w[len(lead) + len(core):]
        new = _possessive(phrase) if possessive else phrase
        if i == 0:
            new = new[0].upper() + new[1:]
        words[i] = lead + new + trail
        if plural and core.lower() in SINGULAR_PRONOUNS:
            for j in (i - 1, i + 1):             # "Why IS it slow" / "it IS slow"
                if 0 <= j < len(words):
                    bare = re.sub(r"\W+$", "", words[j])
                    fix = PLURAL_FIX.get(bare.lower())
                    if fix:
                        fix = fix.capitalize() if bare[0].isupper() else fix
                        words[j] = fix + words[j][len(bare):]
        return " ".join(words)
    return None


def merge_clarification(original: str, reply: str, reason: str, details: dict,
                        rewrite_fn: Optional[Callable[[str, str], str]] = None,
                        nlp=None) -> str:
    """
    Fold the user's reply into the original query. Order of attempts:
      1. pull a clean noun phrase out of the reply (spaCy, already loaded)
         - unclear pronoun -> substitute it (with possessive + is/are fixes)
         - other reasons   -> "original — <cleaned reply>" (all of its words)
      2. nothing usable and a local LLM is available -> rewrite_fn
      3. otherwise keep the original (the next round asks again)
    The merged query is used for RETRIEVAL and ROUTING. The answering LLM
    gets the real conversation (see Conversation.transcript), never this string.
    """
    if reason == "UnresolvedReferences" and details.get("unresolved_pronouns"):
        phrase, plural = answer_phrase(reply, nlp)
        if phrase:
            merged = _replace_pronoun(original, details["unresolved_pronouns"][0], phrase, plural, nlp)
            if merged:
                return merged
    # Any other gap: keep ALL the words the user gave (minus filler) - a
    # retriever needs "when the United States entered World War 1", not
    # just its biggest noun phrase.
    extra = clean_reply(reply)
    if extra:
        core = original.rstrip(" ?.!")
        end = "?" if original.rstrip().endswith("?") else ""
        return f"{core} — {extra}{end}"
    if rewrite_fn is not None:
        return rewrite_fn(original, reply)
    return original                              # "yes"/"sorry" adds nothing: ask again next round


def resolve_from_history(query: str, history: list[str], nlp, known_terms=None) -> tuple[str, bool]:
    """
    'When was he born?' after 'Who composed the Brandenburg Concertos? -> Bach'
    style turns: if the query has a pronoun with no antecedent and the
    previous (already resolved) question names exactly ONE entity, substitute
    it directly - no clarifying question needed. Returns (query, changed).
    """
    if not history:
        return query, False
    # Demonstratives (this/that/these/those) usually point at an event or the
    # previous ANSWER ("Was this the end of his career?"), not at an entity,
    # so only personal pronouns and it/its/they/them/their are auto-filled.
    needs = [p for p in pronouns_needing_resolution(nlp(query))
             if p.lower() not in {"this", "that", "these", "those"}]
    if not needs:
        return query, False
    prev = specific_entities(nlp(history[-1]), known_terms)
    if len(prev) != 1:
        return query, False
    merged = _replace_pronoun(query, needs[0], prev[0], False, nlp)
    return (merged, True) if merged else (query, False)


@dataclass
class Conversation:
    """Keeps the per-conversation state the router needs."""
    extractor: FeatureExtractor
    history: list[str] = field(default_factory=list)       # resolved USER queries (for routing)
    transcript: list[dict] = field(default_factory=list)   # what was actually said (for the answering LLM)
    pending: Optional[dict] = None                          # an open clarification
    rewrite_fn: Optional[Callable[[str, str], str]] = None
    # router(raw) -> {"route", "evidence": {node: state name}, ...}; default = hand-set network.
    # demo.py passes the LEARNED network here (learn_bn.route).
    router: Optional[Callable[[dict], dict]] = None

    def handle(self, message: str) -> dict:
        self.transcript.append({"role": "user", "text": message})
        is_reply = self.pending is not None
        if is_reply:
            query = merge_clarification(self.pending["query"], message,
                                        self.pending["reason"], self.pending["details"],
                                        self.rewrite_fn, nlp=self.extractor.nlp)
            rounds = self.pending["rounds"]
        else:
            query, rounds = message, 0
            query, _ = resolve_from_history(query, self.history, self.extractor.nlp,
                                            self.extractor.popularity)

        result = self.extractor.extract(query, history=self.history,
                                        is_clarification_reply=is_reply)
        if self.router is not None:
            decision = self.router(result.raw)
            evidence = decision["evidence"]
        else:
            evidence = extract_evidence(result.raw)
            decision = route_query(evidence)
        decision = {**decision, "raw": result.raw, "details": result.details}

        # Cap the loop: after MAX rounds, search and answer with a caveat.
        if decision["route"] == "Clarify" and rounds >= MAX_CLARIFY_ROUNDS:
            decision = {**decision, "route": "Search", "caveat": True}

        if decision["route"] == "Clarify":
            question, reason = make_clarifying_question(query, evidence, result.details)
            self.pending = {"query": query, "reason": reason,
                            "details": result.details, "rounds": rounds + 1}
            self.transcript.append({"role": "assistant", "text": question})
            return {**decision, "query": query, "ask_user": question, "reason": reason}

        self.pending = None
        self.history.append(query)
        # retrieval_query: the merged text, used only for search/routing.
        # transcript: the real words, which the answering LLM receives.
        return {**decision, "query": query, "retrieval_query": query,
                "transcript": list(self.transcript), "chunks": result.details["chunks"]}
