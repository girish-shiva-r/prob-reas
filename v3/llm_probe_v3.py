"""
Tier-2/3 signals from the answering LLM, through a llama.cpp server (llama-server)
==================================================================================
The SAME .gguf file is used for labelling (Kaggle) and for the demo (local RTX 3050), so the learned
network describes exactly the deployed model.

    EntityFamiliarity     mean log-prob per token of the entity name after "Here is a fact about"
    ClosedBookConfidence  P("I don't know") as the answer to the bare question
    GroundedConfidence    P("I don't know") as the answer to the RAG prompt (question + passages)
    ParametricAgreement   short closed-book answer vs the extractive span from the passages (tier 3)

P("I don't know") is computed by TEACHER FORCING: the probability of the exact token sequence
"I don't know", one token at a time (the server returns the top-N next-token probabilities).
That is exact when each forced token is in the top N; otherwise the token gets the floor value
(smallest listed probability) and the result is an upper bound - flagged in `floored`.

Thinking is OFF (Qwen3.5 thinks by default): prompts are rendered with an empty think block,
exactly what the chat template produces with enable_thinking=False. The smoke test checks this.
"""

from __future__ import annotations

import json
import math
import re
import urllib.request
from dataclasses import dataclass
from typing import Optional

ANSWER_INSTRUCTION = (
    "Answer the question using the context passages if they help. "
    "Reply with the answer only, in a few words. "
    "If you do not know the answer, reply exactly: I don't know.\n\n"
    "Context:\n{context}\n\nQuestion: {question}"
)
CLOSED_BOOK_INSTRUCTION = (
    "Answer the question from your own knowledge. Reply with the answer only, in a few words. "
    "If you do not know the answer, reply exactly: I don't know.\n\nQuestion: {question}"
)
IDK = "I don't know"


def render_qwen(user: str, system: Optional[str] = None) -> str:
    """Qwen chat format with thinking disabled (empty think block), ready for the raw /completion API."""
    s = f"<|im_start|>system\n{system}<|im_end|>\n" if system else ""
    return s + f"<|im_start|>user\n{user}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"


def rag_prompt(question, passages, conversation=None, k=3):
    ctx = "\n".join(f"[{i + 1}] {p}" for i, p in enumerate(passages[:k])) or "(none)"
    user = ANSWER_INSTRUCTION.format(context=ctx, question=question)
    if conversation:
        turns = "\n".join(f"Q: {q}\nA: {a}" for q, a in conversation)
        user = user.replace("\n\nQuestion: ", f"\n\nConversation so far:\n{turns}\n\nQuestion: ", 1)
    return render_qwen(user)


def clean(text: str) -> str:
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S).strip()
    return text.split("\n")[0].strip()


@dataclass
class Score:
    logprob: float          # sum over the continuation tokens
    tokens: int
    floored: int            # tokens not in the top-N list (logprob is then an upper bound)

    @property
    def prob(self):
        return math.exp(self.logprob)

    @property
    def mean(self):
        return self.logprob / max(self.tokens, 1)


class LlamaServer:
    def __init__(self, url="http://127.0.0.1:8080", n_probs=100, timeout=600):
        self.url, self.n_probs, self.timeout = url.rstrip("/"), n_probs, timeout

    # ---- transport ----------------------------------------------------------------------
    def _post(self, path, payload):
        req = urllib.request.Request(self.url + path, data=json.dumps(payload).encode(),
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=self.timeout) as r:
            return json.loads(r.read())

    def tokenize(self, text):
        return self._post("/tokenize", {"content": text, "add_special": False})["tokens"]

    def complete(self, prompt, n_predict=24, slot=None, n_probs=0, stop=("<|im_end|>", "\n")):
        payload = {"prompt": prompt, "n_predict": n_predict, "temperature": 0.0, "top_k": 1,
                   "cache_prompt": True, "n_probs": n_probs, "stop": list(stop)}
        if slot is not None:
            payload["id_slot"] = slot
        return self._post("/completion", payload)

    @staticmethod
    def _top(resp):
        """Top next-token log-probs {token_id: logprob} from either llama-server response format."""
        cp = resp.get("completion_probabilities") or []
        if not cp:
            return {}
        first = cp[0]
        if "top_logprobs" in first:                                   # newer servers
            return {t["id"]: t["logprob"] for t in first["top_logprobs"]}
        if "probs" in first:                                          # older servers
            return {t.get("id", t.get("tok_str")): math.log(max(t["prob"], 1e-12)) for t in first["probs"]}
        return {}

    # ---- teacher forcing ----------------------------------------------------------------
    def score(self, prompt: str, continuation: str, slot=None) -> Score:
        """log P(continuation | prompt), token by token (each step reuses the cached prefix)."""
        p_ids = self.tokenize(prompt)
        full = self.tokenize(prompt + continuation)
        if full[:len(p_ids)] != p_ids:                     # tokens merged across the boundary
            c_ids = self.tokenize(continuation)
        else:
            c_ids = full[len(p_ids):]
        total, floored, ids = 0.0, 0, list(p_ids)
        for t in c_ids:
            top = self._top(self.complete(ids, n_predict=1, slot=slot, n_probs=self.n_probs, stop=()))
            if t in top:
                total += top[t]
            else:
                total += min(top.values()) if top else math.log(1e-6)
                floored += 1
            ids.append(t)
        return Score(total, len(c_ids), floored)

    # ---- the signals --------------------------------------------------------------------
    def answer(self, question, passages, conversation=None, slot=None, n_predict=24):
        resp = self.complete(rag_prompt(question, passages, conversation), n_predict=n_predict, slot=slot)
        return clean(resp.get("content", "")), resp.get("timings", {})

    def grounded_p_idk(self, question, passages, conversation=None, slot=None):
        return self.score(rag_prompt(question, passages, conversation), IDK, slot)

    def closed_book(self, question, slot=None, n_predict=12):
        prompt = render_qwen(CLOSED_BOOK_INSTRUCTION.format(question=question))
        resp = self.complete(prompt, n_predict=n_predict, slot=slot)
        return clean(resp.get("content", "")), self.score(prompt, IDK, slot)

    def entity_logprob(self, entity, slot=None):
        return self.score("Here is a fact about", " " + entity, slot)

    def tier2(self, question, passages, entities, conversation=None, slot=None, with_closed_book_answer=True):
        """Raw tier-2 (+ the closed-book answer for tier 3) in learn_v3 row format."""
        raw, det = {}, {}
        if entities:
            sc = [self.entity_logprob(e, slot) for e in entities]
            raw["entity_logprob"] = min(s.mean for s in sc)          # the least familiar entity
            det["entity_logprob"] = {e: round(s.mean, 3) for e, s in zip(entities, sc)}
        cb_answer, cb = self.closed_book(question, slot)
        raw["closed_book_p_idk"] = cb.prob
        g = self.grounded_p_idk(question, passages, conversation, slot)
        raw["grounded_p_idk"] = g.prob
        det.update({"closed_book_answer": cb_answer, "floored_tokens": cb.floored + g.floored})
        return {"raw": raw, "details": det}


def parametric_agreement(closed_book_answer: str, extractive_span: Optional[str]) -> Optional[str]:
    """Tier 3: Agree / NoParametric / Disagree (None if there is no extractive span to compare)."""
    if not closed_book_answer or IDK.lower() in closed_book_answer.lower():
        return "NoParametric"
    if not extractive_span:
        return None
    from signals_v3 import _token_f1, norm
    a, b = norm(closed_book_answer), norm(extractive_span)
    return "Agree" if (a in b or b in a or _token_f1(a, b) >= 0.5) else "Disagree"
