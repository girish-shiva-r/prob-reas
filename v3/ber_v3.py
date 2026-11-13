"""
Bayesian Epistemic Routing - v3 (online)
========================================
A causal, decision-theoretic Bayesian network for routing questions in a RAG system.

    9 CAUSES (hidden)  ->  22 SYMPTOMS (measured)          "what is wrong, and how do we see it"
    7 of the CAUSES      ->  OUTCOME {Correct, Abstain, Wrong}   "what happens if we answer now"
    DECISION (6 actions) intervenes on the causes, then OUTCOME -> UTILITY   "what should we do"
    3 CONTEXT nodes (history, clarify rounds, search rounds) shape what each action can fix

    35 chance nodes (built in pgmpy) + Decision + Utility = 37 nodes.

Direction of the arrows: CAUSE -> SYMPTOM. Ambiguity *causes* tied reranker scores; a short
query does not cause ambiguity. Consequences:
  * diagnosis = posterior over the 9 causes, each of which maps to an action that fixes it
  * every symptom table has <= 2 parents (small, learnable)
  * a symptom that was not measured is simply left out (anytime routing, see value_of_tier)

ALL NUMBERS IN SECTIONS 1-4b ARE EXPERT GUESSES. They are used for exactly two things:
  1. the starting point of EM, and
  2. a Dirichlet prior whose STRENGTH is chosen by cross-validation (learn_v3.py), so the data
     decides how much they count - possibly not at all.
recovery_test.py checks that the learned network does NOT depend on them when weak cause labels
are available. LAMBDA (section 5) is not learned: it is a policy choice, picked from a target risk.

    python ber_v3.py            # check the model, run the sanity scenarios

"""

from __future__ import annotations

from itertools import product

import numpy as np

# ============================================================================
# 1. CAUSES (hidden, binary Low/High; "High" = the problem is present)
# ============================================================================
LH = ["Low", "High"]

CAUSES = {
    # roots: prior P(High)
    "ReferentAmbiguity":   {"prior": 0.12, "fix": "Clarify (which one?)",
                            "desc": "the name refers to several things ('What genre is Pulse?')",
                            "weak_label": "the question's name is shared by 2+ KB/Wikipedia pages (AmbigDocs, AmbigQA)"},
    "ConstraintAmbiguity": {"prior": 0.06, "fix": "Clarify (which time / version / place?)",
                            "desc": "the entity is clear but the answer depends on an unstated time/place/scope",
                            "weak_label": "AmbigQA multi-answer questions, SituatedQA"},
    "ContextDependence":   {"prior": 0.12, "fix": "Rewrite from history, else Clarify",
                            "desc": "the question cannot stand alone ('Where did she go?')",
                            "weak_label": "QReCC / TopiOCQA turn whose gold rewrite differs from the raw question"},
    "ParametricGap":       {"prior": 0.40, "fix": "(decides whether bad evidence matters)",
                            "desc": "the LLM does not know the answer without context",
                            "weak_label": "closed-book run of the LLM is wrong"},
    "ReasoningLoad":       {"prior": 0.15, "fix": "Decompose",
                            "desc": "needs 2+ facts: bridge / comparison / composition",
                            "weak_label": "MuSiQue hops >= 2, 2WikiMultiHopQA type"},
    "TemporalMismatch":    {"prior": 0.06, "fix": "Search (fresh web results)",
                            "desc": "the true answer changed; KB or model is out of date",
                            "weak_label": "FreshQA fast-changing questions"},
    # caused by other causes: noisy-OR (leak + one strength per parent)
    "EvidenceGap":         {"noisy_or": {"leak": 0.20, "ReferentAmbiguity": 0.30, "ConstraintAmbiguity": 0.25,
                                         "ContextDependence": 0.45, "ReasoningLoad": 0.35},
                            "fix": "Search (deeper / full pages)",
                            "desc": "the retrieved passages do not contain the answer",
                            "weak_label": "no gold answer string / alias in the top-k passages"},
    "ContextMismatch":     {"noisy_or": {"leak": 0.05, "ReferentAmbiguity": 0.40, "ContextDependence": 0.20},
                            "fix": "Search with entity + type ('Belly film')",
                            "desc": "passages look relevant but are about a same-name, different entity",
                            "weak_label": "top passage's page shares the name but is not the gold page"},
    "EvidenceConflict":    {"noisy_or": {"leak": 0.05, "ContextMismatch": 0.45, "TemporalMismatch": 0.45},
                            "fix": "Search for more support, or hedge",
                            "desc": "passages disagree with each other or with the model's memory",
                            "weak_label": "ClashEval perturbed passages; gold answer in context but model still wrong"},
}
CAUSE_NAMES = list(CAUSES)          # topological order (roots first)
ROOTS = [c for c in CAUSE_NAMES if "prior" in CAUSES[c]]

# ============================================================================
# 2. CONTEXT (always observed, set by the pipeline)
# ============================================================================
CONTEXT = {
    "HistoryCandidates": {"states": ["None", "One", "Many"], "prior": [0.6, 0.25, 0.15],
                          "desc": "antecedents for a pronoun in the previous question AND answer"},
    "ClarifyRounds":     {"states": ["0", "1", "2+"], "prior": [0.85, 0.12, 0.03],
                          "desc": "clarifying questions already asked for this question"},
    "SearchRounds":      {"states": ["0", "1", "2+"], "prior": [0.8, 0.15, 0.05],
                          "desc": "searches already done for this question"},
}

# ============================================================================
# 3. SYMPTOMS (observed). cpt: parent states (in `parents` order) -> P(states)
#    tier = cost of measuring: 0 query only, 1 after retrieval (CPU), 2 one LLM prefill,
#           3 short LLM generation, 4 after the answer is generated
# ============================================================================
SYMPTOMS = {
    # ---------------- tier 0: the question text (+ history) ----------------
    "NameCollision": {
        "tier": 0, "parents": ["ReferentAmbiguity"], "states": ["Unique", "Few", "Many"],
        "measure": "Wikipedia/KB pages sharing the question entity's base name (strip '(film)' etc.): 1 / 2-3 / 4+",
        "cpt": {("Low",): [.80, .15, .05], ("High",): [.10, .45, .45]}},
    "Disambiguator": {
        "tier": 0, "parents": ["ReferentAmbiguity"], "states": ["Present", "Absent"],
        "measure": "query carries a type word or year next to the entity ('the film', '(1999)', 'the band')",
        "cpt": {("Low",): [.35, .65], ("High",): [.05, .95]}},
    "TemporalAnchoring": {
        "tier": 0, "parents": ["ConstraintAmbiguity", "TemporalMismatch"],
        "states": ["Stable", "VolatileAnchored", "VolatileUnanchored"],
        "measure": "relation changes over time (CEO, population, champion, record)? if so, is a time given?",
        "cpt": {("Low", "Low"): [.85, .10, .05], ("Low", "High"): [.30, .45, .25],
                ("High", "Low"): [.40, .05, .55], ("High", "High"): [.10, .10, .80]}},
    "TimeWords": {
        "tier": 0, "parents": ["TemporalMismatch"], "states": ["Absent", "Present"],
        "measure": "'current', 'latest', 'now', 'this year', or a year >= the model's knowledge cut-off",
        "cpt": {("Low",): [.97, .03], ("High",): [.40, .60]}},
    "ResidualReferences": {
        "tier": 0, "parents": ["ContextDependence"], "states": ["None", "One", "Several"],
        "measure": "pronouns / ellipsis still unresolved inside the question itself",
        "cpt": {("Low",): [.95, .04, .01], ("High",): [.25, .55, .20]}},
    "FollowUpForm": {
        "tier": 0, "parents": ["ContextDependence"], "states": ["No", "Yes"],
        "measure": "elliptical / 'and what about...' / <= 4 words while a conversation is running",
        "cpt": {("Low",): [.85, .15], ("High",): [.25, .75]}},
    "MultiHopForm": {
        "tier": 0, "parents": ["ReasoningLoad"], "states": ["Single", "Bridge", "Comparison"],
        "measure": "comparison pattern ('which ... first', 'both'), bridge relative clause, possessive chain",
        "cpt": {("Low",): [.92, .06, .02], ("High",): [.15, .50, .35]}},
    # ---------------- tier 1: retrieval (hybrid BM25 + dense, top 50 reranked) ----------------
    "RerankerLogit": {
        "tier": 1, "parents": ["EvidenceGap", "ContextMismatch"],
        "states": ["VeryHigh", "High", "Mid", "Low"],
        "measure": "top-1 RAW cross-encoder logit: > 6 / 2..6 / -2..2 / < -2 (ms-marco-MiniLM scale; re-learn)",
        "cpt": {("Low", "Low"): [.55, .30, .10, .05], ("Low", "High"): [.40, .35, .17, .08],
                ("High", "Low"): [.05, .15, .30, .50], ("High", "High"): [.20, .35, .27, .18]}},
    "RerankMargin": {
        "tier": 1, "parents": ["ReferentAmbiguity", "ContextMismatch"], "states": ["Clear", "Narrow", "Tied"],
        "measure": "top-1 logit minus best logit from a DIFFERENT page: > 3 / 1..3 / < 1",
        "cpt": {("Low", "Low"): [.60, .28, .12], ("Low", "High"): [.25, .35, .40],
                ("High", "Low"): [.20, .35, .45], ("High", "High"): [.10, .30, .60]}},
    "TitleDiversity": {
        "tier": 1, "parents": ["ReferentAmbiguity", "ContextMismatch"], "states": ["One", "Two", "ThreePlus"],
        "measure": "distinct pages among passages within 2 logits of the top passage",
        "cpt": {("Low", "Low"): [.65, .25, .10], ("Low", "High"): [.30, .40, .30],
                ("High", "Low"): [.20, .35, .45], ("High", "High"): [.10, .30, .60]}},
    "EntityCoverage": {
        "tier": 1, "parents": ["EvidenceGap"], "states": ["Full", "Partial", "None"],
        "measure": "share of question entities found (fuzzy) in the top-3 passages: all / some / none",
        "cpt": {("Low",): [.80, .15, .05], ("High",): [.30, .35, .35]}},
    "SubjectTypeMatch": {
        "tier": 1, "parents": ["ContextMismatch"], "states": ["Match", "Unknown", "Mismatch"],
        "measure": "type implied by the relation (screenwriter -> film) vs the top page's type (title/first sentence/Wikidata)",
        "cpt": {("Low",): [.70, .25, .05], ("High",): [.12, .28, .60]}},
    "AnswerTypePresence": {
        "tier": 1, "parents": ["EvidenceGap"], "states": ["Present", "Absent"],
        "measure": "expected answer type (who->PERSON, when->DATE, where->GPE/LOC, how many->CARDINAL) in top-3 NER",
        "cpt": {("Low",): [.85, .15], ("High",): [.35, .65]}},
    "Answerability": {
        "tier": 1, "parents": ["EvidenceGap", "ReasoningLoad"], "states": ["High", "Mid", "Low"],
        "measure": "max has-answer probability of an extractive SQuAD2 model over the top-3: > .7 / .3-.7 / < .3",
        "cpt": {("Low", "Low"): [.70, .20, .10], ("Low", "High"): [.35, .35, .30],
                ("High", "Low"): [.08, .22, .70], ("High", "High"): [.05, .20, .75]}},
    "CandidateAgreement": {
        "tier": 1, "parents": ["EvidenceConflict"], "states": ["Agree", "Single", "Disagree"],
        "measure": "extractive answers from the top-3 passages: same / only one passage answers / different",
        "cpt": {("Low",): [.55, .40, .05], ("High",): [.10, .35, .55]}},
    "SupportSpread": {
        "tier": 1, "parents": ["ReasoningLoad"], "states": ["Concentrated", "Spread"],
        "measure": "question entities split across different pages and no single passage covers them all",
        "cpt": {("Low",): [.85, .15], ("High",): [.30, .70]}},
    "EvidenceAge": {
        "tier": 1, "parents": ["TemporalMismatch"], "states": ["Fresh", "Stale"],
        "measure": "newest supporting passage's date (page revision / article date) within 90 days?",
        "cpt": {("Low",): [.70, .30], ("High",): [.25, .75]}},
    # ---------------- tier 2: one LLM prefill, no generation ----------------
    "EntityFamiliarity": {
        "tier": 2, "parents": ["ParametricGap"], "states": ["Familiar", "Vague", "Unfamiliar"],
        "measure": "mean token log-prob of the entity name in 'Here is a fact about <entity>:' (terciles, learned)",
        "cpt": {("Low",): [.65, .28, .07], ("High",): [.10, .35, .55]}},
    "ClosedBookConfidence": {
        "tier": 2, "parents": ["ParametricGap"], "states": ["Confident", "Unsure", "IDK"],
        "measure": "P(first token starts 'I don't know') prefilling the bare question: < .1 / .1-.5 / > .5",
        "cpt": {("Low",): [.70, .22, .08], ("High",): [.10, .30, .60]}},
    "GroundedConfidence": {
        "tier": 2, "parents": ["EvidenceGap", "ParametricGap"], "states": ["Confident", "Unsure", "IDK"],
        "measure": "same, prefilling the full RAG prompt (its KV cache is reused when we Serve)",
        "cpt": {("Low", "Low"): [.85, .12, .03], ("Low", "High"): [.70, .20, .10],
                ("High", "Low"): [.45, .30, .25], ("High", "High"): [.08, .22, .70]}},
    # ---------------- tier 3: short generation ----------------
    "ParametricAgreement": {
        "tier": 3, "parents": ["EvidenceConflict", "ParametricGap"], "states": ["Agree", "NoParametric", "Disagree"],
        "measure": "closed-book answer (<= 8 tokens) vs the extractive span from the passages",
        "cpt": {("Low", "Low"): [.75, .15, .10], ("Low", "High"): [.15, .75, .10],
                ("High", "Low"): [.12, .13, .75], ("High", "High"): [.05, .60, .35]}},
    # ---------------- tier 4: after the answer is generated ----------------
    "AnswerGrounding": {
        "tier": 4, "parents": ["Outcome"], "states": ["Supported", "Partial", "Unsupported"],
        "measure": "generated answer found in / entailed by the passages (alias match + small NLI model)",
        "cpt": {("Correct",): [.75, .18, .07], ("Abstain",): [.34, .33, .33], ("Wrong",): [.10, .25, .65]}},
}
SYMPTOM_NAMES = [s for s in SYMPTOMS if SYMPTOMS[s]["tier"] < 4]       # pre-generation symptoms
TIERS = {t: [s for s in SYMPTOMS if SYMPTOMS[s]["tier"] == t] for t in range(5)}
TIER_COST = {1: 0.005, 2: 0.015, 3: 0.04}      # utility units (1 = one correct answer)

# ============================================================================
# 4. OUTCOME = the answer model: what happens if the LLM answers NOW from the current passages
#    {Correct, Abstain, Wrong}, multinomial logit:
#    logit(Correct) = 0, logit(Abstain) = base_A + sum dA, logit(Wrong) = base_W + sum dW
# ============================================================================
OUTCOME_STATES = ["Correct", "Abstain", "Wrong"]
OUTCOME = {
    "parents": ["ConstraintAmbiguity", "EvidenceGap", "ContextMismatch", "EvidenceConflict",
                "ParametricGap", "ReasoningLoad", "TemporalMismatch"],
    "base": (-2.6, -2.8),
    "effects": {"ConstraintAmbiguity": (0.2, 1.4), "EvidenceGap": (1.5, 0.5), "ContextMismatch": (0.3, 2.4),
                "EvidenceConflict": (0.3, 1.8), "ParametricGap": (0.9, 0.5), "ReasoningLoad": (0.9, 1.2),
                "TemporalMismatch": (0.4, 2.0)},
    # no evidence AND no memory -> "I don't know" (the model knows it does not know)
    "interactions": [(("EvidenceGap", "ParametricGap"), (1.6, 0.6))],
}

# ============================================================================
# 4b. ACTIONS as INTERVENTIONS on the causes (a two-slice decision network)
#     Every action CLEARS some question-side causes (with a success probability) and then
#     RE-RETRIEVES from the web, so the three evidence-side causes are drawn anew from
#     web-retrieval tables. The SAME answer model (OUTCOME) then scores the result.
#     -> one answer model + a handful of effect probabilities, instead of 5 outcome tables that
#        could contradict each other; every effect is learnable by running that action on a subset.
# ============================================================================
EVIDENCE_CAUSES = ["EvidenceGap", "ContextMismatch", "EvidenceConflict"]
QUERY_CAUSES = [c for c in CAUSES if c not in EVIDENCE_CAUSES]
ACTIONS = ["Serve", "Rewrite", "Search", "Decompose", "Clarify", "Abstain"]

HANDSET_EFFECTS = {
    "Rewrite":   {"ContextDependence": {"None": 0.10, "One": 0.85, "Many": 0.30}},   # by HistoryCandidates
    "Search":    {"TemporalMismatch": 0.85, "ReferentAmbiguity@typed": 0.80},       # typed query 'Belly film'
    "Clarify":   {"ReferentAmbiguity": 0.90, "ConstraintAmbiguity": 0.90, "ContextDependence": 0.90},
    "Decompose": {"ReasoningLoad": 0.80},
    "clarify_fatigue": {"0": 1.0, "1": 0.7, "2+": 0.4},     # the user's reply helps less each round
    # evidence-side causes after a web search: noisy-OR, leak rises after earlier failed searches
    "web": {
        "EvidenceGap":      {"leak": {"0": 0.10, "1": 0.30, "2+": 0.55}, "ReferentAmbiguity": 0.30,
                             "ConstraintAmbiguity": 0.25, "ContextDependence": 0.50, "ReasoningLoad": 0.45},
        "ContextMismatch":  {"leak": {"0": 0.03, "1": 0.05, "2+": 0.08}, "ReferentAmbiguity": 0.45,
                             "ContextDependence": 0.25},
        "EvidenceConflict": {"leak": {"0": 0.04, "1": 0.05, "2+": 0.06}, "ContextMismatch": 0.45,
                             "TemporalMismatch": 0.45},
    },
}

# ============================================================================
# 5. UTILITY  (NOT learned - a policy choice; see choose_lambda in learn_v3.py)
#    Serve beats Abstain  <=>  P(Correct) - LAMBDA * P(Wrong) > 0  <=>  P(Wrong)/P(Correct) < 1/LAMBDA
#    so LAMBDA is just another way of writing the risk you accept. Pick it from a target risk on
#    validation data (learn_v3.choose_lambda) and report the whole risk-coverage curve.
# ============================================================================
LAMBDA = 2.5
ACTION_COST = {"Serve": 0.0, "Rewrite": 0.10, "Search": 0.08, "Decompose": 0.15, "Clarify": 0.20, "Abstain": 0.0}
CLARIFY_EXTRA_COST = 0.10                 # per clarifying question already asked (user patience)
HEDGE_UTILITY = {"Correct": 0.6, "Abstain": 0.0, "Wrong": -0.5}   # "I think it's X, but I'm not sure"
DEFAULT_CONTEXT = {"HistoryCandidates": "None", "ClarifyRounds": "0", "SearchRounds": "0"}


def utility(lam=LAMBDA):
    return np.array([1.0, 0.0, -lam])


def states_of(node):
    if node in CAUSES:
        return LH
    if node in CONTEXT:
        return CONTEXT[node]["states"]
    if node in SYMPTOMS:
        return SYMPTOMS[node]["states"]
    if node == "Outcome":
        return OUTCOME_STATES
    raise KeyError(node)


def cause_parents(c):
    return [p for p in CAUSES[c].get("noisy_or", {}) if p != "leak"]


def parents_of(node):
    if node in CAUSES:
        return cause_parents(node)
    if node in SYMPTOMS:
        return SYMPTOMS[node]["parents"]
    if node == "Outcome":
        return OUTCOME["parents"]
    return []


# ============================================================================
# 6. PARAMETERS: every number the network uses, in one learnable object
#    cause[c]   : (n_parent_configs,)           P(c = High | parents)
#    symptom[s] : (n_parent_configs, n_states)  P(s | parents)
#    outcome    : (2^7, 3)                      P(Outcome | its 7 cause parents)
#    effects    : action effect probabilities (HANDSET_EFFECTS layout)
#    Parent configurations are in pgmpy order (first parent varies slowest).
# ============================================================================
def parent_configs(node):
    return list(product(*[states_of(p) for p in parents_of(node)]))


class Params:
    def __init__(self, cause, symptom, outcome, effects):
        self.cause, self.symptom, self.outcome, self.effects = cause, symptom, outcome, effects

    def copy(self):
        import copy
        return Params({k: v.copy() for k, v in self.cause.items()}, {k: v.copy() for k, v in self.symptom.items()},
                      self.outcome.copy(), copy.deepcopy(self.effects))

    def to_json(self):
        return {"cause": {k: v.round(6).tolist() for k, v in self.cause.items()},
                "symptom": {k: v.round(6).tolist() for k, v in self.symptom.items()},
                "outcome": self.outcome.round(6).tolist(), "effects": self.effects}

    @classmethod
    def from_json(cls, d):
        return cls({k: np.array(v) for k, v in d["cause"].items()}, {k: np.array(v) for k, v in d["symptom"].items()},
                   np.array(d["outcome"]), d["effects"])


def _noisy_or(spec, a, rounds=None):
    leak = spec["leak"][rounds] if isinstance(spec["leak"], dict) else spec["leak"]
    p_low = 1 - leak
    for p, s in spec.items():
        if p != "leak" and a[p] == "High":
            p_low *= 1 - s
    return 1 - p_low


def _outcome_logit(a):
    al, w = OUTCOME["base"]
    for p, (da, dw) in OUTCOME["effects"].items():
        if a[p] == "High":
            al, w = al + da, w + dw
    for ps, (da, dw) in OUTCOME["interactions"]:
        if all(a[p] == "High" for p in ps):
            al, w = al + da, w + dw
    z = np.exp([0.0, al, w])
    return z / z.sum()


def handset_params():
    """The expert guesses (sections 1-4b) as tables: the PRIOR and EM starting point, nothing more."""
    cause = {}
    for c, spec in CAUSES.items():
        if "prior" in spec:
            cause[c] = np.array([spec["prior"]])
        else:
            cause[c] = np.array([_noisy_or(spec["noisy_or"], dict(zip(cause_parents(c), cfg)))
                                 for cfg in parent_configs(c)])
    symptom = {s: np.array([spec["cpt"][cfg] for cfg in parent_configs(s)], dtype=float)
               for s, spec in SYMPTOMS.items()}
    outcome = np.array([_outcome_logit(dict(zip(OUTCOME["parents"], cfg))) for cfg in parent_configs("Outcome")])
    import copy
    return Params(cause, symptom, outcome, copy.deepcopy(HANDSET_EFFECTS))


# ============================================================================
# 7. THE NETWORK IN PGMPY (structure check, export, visualisation)
# ============================================================================
def build_pgmpy(params=None):
    from pgmpy.factors.discrete import TabularCPD
    from pgmpy.models import DiscreteBayesianNetwork

    P = params or handset_params()
    model, cpds = DiscreteBayesianNetwork(), []
    model.add_nodes_from(CAUSE_NAMES + list(CONTEXT) + list(SYMPTOMS) + ["Outcome"])

    def cpd(node, table):
        ps = parents_of(node)
        model.add_edges_from((p, node) for p in ps)
        return TabularCPD(node, len(states_of(node)), np.asarray(table).T.tolist(), evidence=ps or None,
                          evidence_card=[len(states_of(p)) for p in ps] or None,
                          state_names={node: states_of(node), **{p: states_of(p) for p in ps}})

    for c in CAUSE_NAMES:
        cpds.append(cpd(c, np.stack([1 - P.cause[c], P.cause[c]], 1)))
    for k, spec in CONTEXT.items():
        cpds.append(cpd(k, [spec["prior"]]))
    cpds.append(cpd("Outcome", P.outcome))
    for s in SYMPTOMS:
        cpds.append(cpd(s, P.symptom[s]))
    model.add_cpds(*cpds)
    assert model.check_model(), "model invalid"
    return model


# ============================================================================
# 8. FAST EXACT INFERENCE over the 2^9 = 512 cause combinations (vectorised)
# ============================================================================
Z = np.array(list(product([0, 1], repeat=len(CAUSE_NAMES))), dtype=np.int8)      # (512, 9), 1 = High
CI = {c: i for i, c in enumerate(CAUSE_NAMES)}


def _pidx(node):
    """For each cause combination: index of the node's parent configuration (pgmpy order)."""
    ps = parents_of(node)
    idx = np.zeros(len(Z), dtype=int)
    for p in ps:
        idx = idx * 2 + Z[:, CI[p]]
    return idx


PIDX = {n: _pidx(n) for n in CAUSE_NAMES + SYMPTOM_NAMES + ["Outcome"]}


class Router:
    def __init__(self, params: Params = None, lam: float = LAMBDA):
        self.p = params or handset_params()
        self.lam = lam
        self.u = utility(lam)
        lp = np.zeros(len(Z))
        for c in CAUSE_NAMES:
            ph = self.p.cause[c][PIDX[c]]
            lp += np.log(np.where(Z[:, CI[c]] == 1, ph, 1 - ph))
        self.log_prior = lp
        self.log_lik = {s: np.log(self.p.symptom[s][PIDX[s]]) for s in SYMPTOM_NAMES}   # (512, k)
        self.O = self.p.outcome[PIDX["Outcome"]]                                         # (512, 3)
        self._TO = {}

    # ---- posterior over causes -----------------------------------------------------------
    def log_post(self, evidence: dict):
        lp = self.log_prior.copy()
        for s, v in evidence.items():
            if s in self.log_lik:
                lp += self.log_lik[s][:, SYMPTOMS[s]["states"].index(v)]
        return lp

    def posterior(self, evidence: dict) -> np.ndarray:
        lp = self.log_post(evidence)
        return np.exp(lp - np.logaddexp.reduce(lp))

    def diagnose(self, evidence):
        q = self.posterior(evidence)
        return {c: float(q @ Z[:, CI[c]]) for c in CAUSE_NAMES}

    # ---- actions as interventions --------------------------------------------------------
    def clear_probs(self, action, context, typed):
        e = self.p.effects
        out = {}
        for k, v in e.get(action, {}).items():
            c, _, cond = k.partition("@")
            if isinstance(v, dict):
                v = v[context["HistoryCandidates"]]
            if cond == "typed" and not typed:
                v = 0.0
            if action == "Clarify":
                v *= e["clarify_fatigue"][context["ClarifyRounds"]]
            out[c] = v
        return out

    def transition(self, action, context, typed=False):
        """(512 x 512) P(cause combination after the action | before)."""
        if action == "Serve":
            return np.eye(len(Z))
        clear = self.clear_probs(action, context, typed)
        T = np.ones((len(Z), len(Z)))
        for c in QUERY_CAUSES:
            hi, hi2 = Z[:, CI[c]][:, None] == 1, Z[:, CI[c]][None, :] == 1
            p = clear.get(c, 0.0)
            T *= np.where(hi, np.where(hi2, 1 - p, p), np.where(hi2, 0.0, 1.0))
        f = np.ones(len(Z))                      # evidence causes: fresh web retrieval, depend on z' only
        for c, spec in self.p.effects["web"].items():
            leak = spec["leak"][context["SearchRounds"]]
            p_low = np.full(len(Z), 1 - leak)
            for par, s in spec.items():
                if par != "leak":
                    p_low = p_low * np.where(Z[:, CI[par]] == 1, 1 - s, 1.0)
            f *= np.where(Z[:, CI[c]] == 1, 1 - p_low, p_low)
        return T * f[None, :]

    def TO(self, action, context, typed=False):
        """(512 x 3) P(outcome after the action | cause combination now). Cached."""
        key = (action, tuple(sorted(context.items())), typed)
        if key not in self._TO:
            self._TO[key] = self.transition(action, context, typed) @ self.O
        return self._TO[key]

    def outcome(self, action, q, context, typed=False):
        if action == "Abstain":
            return np.array([0.0, 1.0, 0.0])
        return q @ self.TO(action, context, typed)

    def expected_utility(self, action, q, context, typed=False):
        cost = ACTION_COST[action]
        if action == "Clarify":
            cost += CLARIFY_EXTRA_COST * CONTEXT["ClarifyRounds"]["states"].index(context["ClarifyRounds"])
        return float(self.outcome(action, q, context, typed) @ self.u) - cost

    def decide(self, evidence, context=None):
        context = {**DEFAULT_CONTEXT, **(context or {})}
        q = self.posterior(evidence)
        typed = evidence.get("SubjectTypeMatch") in ("Match", "Mismatch")     # the relation implies a type
        eu = {a: self.expected_utility(a, q, context, typed) for a in ACTIONS}
        best = max(eu, key=eu.get)
        return {"action": best, "eu": {k: round(v, 3) for k, v in eu.items()},
                "p_serve": dict(zip(OUTCOME_STATES, self.outcome("Serve", q, context))),
                "causes": {c: float(q @ Z[:, CI[c]]) for c in CAUSE_NAMES}}

    # ---- anytime routing: is the next tier of signals worth measuring? --------------------
    def value_of_tier(self, evidence, tier, context=None, n=300, seed=0):
        """Expected gain in utility from measuring all symptoms of `tier` (Monte Carlo)."""
        context = {**DEFAULT_CONTEXT, **(context or {})}
        rng = np.random.default_rng(seed)
        q = self.posterior(evidence)
        best = lambda qq, ev: max(self.expected_utility(a, qq, context,
                                                        ev.get("SubjectTypeMatch") in ("Match", "Mismatch"))
                                  for a in ACTIONS)
        now = best(q, evidence)
        new = [s for s in TIERS[tier] if s not in evidence and s in self.log_lik]
        after = []
        for z in rng.choice(len(q), size=n, p=q):
            ev = dict(evidence)
            for s in new:
                p = np.exp(self.log_lik[s][z])
                ev[s] = SYMPTOMS[s]["states"][rng.choice(len(p), p=p / p.sum())]
            after.append(best(self.posterior(ev), ev))
        return float(np.mean(after) - now)

    # ---- after generation: accept, hedge, or search again ---------------------------------
    def after_answer(self, evidence, grounding, context=None):
        context = {**DEFAULT_CONTEXT, **(context or {})}
        q = self.posterior(evidence)
        prior_o = self.outcome("Serve", q, context)
        g = self.p.symptom["AnswerGrounding"]
        lik = g[:, SYMPTOMS["AnswerGrounding"]["states"].index(grounding)]
        post_o = prior_o * lik / (prior_o * lik).sum()
        nxt = {**context, "SearchRounds": "1" if context["SearchRounds"] == "0" else "2+"}
        opts = {"Accept": float(post_o @ self.u),
                "Hedge": float(post_o @ [HEDGE_UTILITY[s] for s in OUTCOME_STATES]),
                "SearchAgain": self.expected_utility("Search", q, nxt)}
        return {"decision": max(opts, key=opts.get), "eu": opts,
                "p_outcome": {k: round(float(v), 3) for k, v in zip(OUTCOME_STATES, post_o)}}


# ============================================================================
# 9. SAMPLING (for the synthetic parameter-recovery test, and for sanity checks)
# ============================================================================
def sample(params: Params, n: int, seed=0, missing=None, label_rate=0.0, action_rate=0.0, label_noise=0.05):
    """
    Draw n questions from the network. Returns a list of rows:
      symptoms  {name: state}          (some missing, by tier: missing={tier: rate})
      labels    {cause: 0/1}           weak cause labels, each present with prob label_rate, flipped w.p. label_noise
      outcome   0/1/2                  Serve outcome
      context   {...}
      action    (name, outcome)        for a share action_rate of rows: one other action actually run
      z         true cause combination (for evaluation only)
    """
    rng = np.random.default_rng(seed)
    missing = missing or {}
    zz = np.zeros((n, len(CAUSE_NAMES)), dtype=np.int8)
    for c in CAUSE_NAMES:                                  # topological order
        idx = np.zeros(n, dtype=int)
        for p in parents_of(c):
            idx = idx * 2 + zz[:, CI[p]]
        zz[:, CI[c]] = rng.random(n) < params.cause[c][idx]
    zi = (zz * (2 ** np.arange(len(CAUSE_NAMES))[::-1])).sum(1)
    r = Router(params)
    rows = []
    for k in range(n):
        z = zi[k]
        sym = {}
        for s in SYMPTOM_NAMES:
            if rng.random() < missing.get(SYMPTOMS[s]["tier"], 0.0):
                continue
            p = params.symptom[s][PIDX[s][z]]
            sym[s] = SYMPTOMS[s]["states"][rng.choice(len(p), p=p)]
        labels = {}
        for c in CAUSE_NAMES:
            if rng.random() < label_rate:
                v = int(zz[k, CI[c]])
                labels[c] = 1 - v if rng.random() < label_noise else v
        ctx = {kk: CONTEXT[kk]["states"][rng.choice(3, p=CONTEXT[kk]["prior"])] for kk in CONTEXT}
        out = int(rng.choice(3, p=r.O[z]))
        row = {"symptoms": sym, "labels": labels, "outcome": out, "context": ctx, "z": int(z)}
        if rng.random() < action_rate:
            a = ACTIONS[1 + rng.integers(4)]                # Rewrite / Search / Decompose / Clarify
            typed = sym.get("SubjectTypeMatch") in ("Match", "Mismatch")
            T = r.transition(a, ctx, typed)[z]
            z2 = rng.choice(len(Z), p=T / T.sum())
            row["action"] = (a, int(rng.choice(3, p=r.O[z2])))
        rows.append(row)
    return rows


# ============================================================================
# 8. SANITY SCENARIOS
# ============================================================================
CLEAN_RETRIEVAL = {"RerankerLogit": "VeryHigh", "RerankMargin": "Clear", "TitleDiversity": "One",
                   "EntityCoverage": "Full", "SubjectTypeMatch": "Match", "AnswerTypePresence": "Present",
                   "Answerability": "High", "CandidateAgreement": "Agree", "SupportSpread": "Concentrated"}
PLAIN_QUERY = {"NameCollision": "Unique", "Disambiguator": "Absent", "TemporalAnchoring": "Stable",
               "TimeWords": "Absent", "ResidualReferences": "None", "FollowUpForm": "No", "MultiHopForm": "Single"}

SCENARIOS = [
    ("Clear factoid, good passages", {**PLAIN_QUERY, **CLEAN_RETRIEVAL}, {}, "Serve"),
    ("'What genre is Pulse?' (ambiguous name)",
     {**PLAIN_QUERY, "NameCollision": "Many", "RerankerLogit": "High", "RerankMargin": "Tied",
      "TitleDiversity": "ThreePlus", "SubjectTypeMatch": "Unknown", "CandidateAgreement": "Disagree"}, {}, "Clarify"),
    ("Rare entity, nothing useful retrieved",
     {**PLAIN_QUERY, "RerankerLogit": "Low", "EntityCoverage": "None", "AnswerTypePresence": "Absent",
      "Answerability": "Low", "EntityFamiliarity": "Unfamiliar", "ClosedBookConfidence": "IDK"}, {}, "Search"),
    ("'When was he born?', history names one person",
     {**PLAIN_QUERY, "ResidualReferences": "One", "FollowUpForm": "Yes", "RerankerLogit": "Mid",
      "EntityCoverage": "None"}, {"HistoryCandidates": "One"}, "Rewrite"),
    ("'When was he born?', history names several people",
     {**PLAIN_QUERY, "ResidualReferences": "One", "FollowUpForm": "Yes", "RerankerLogit": "Mid",
      "EntityCoverage": "None"}, {"HistoryCandidates": "Many"}, "Clarify"),
    ("'Which magazine started first, X or Y?'",
     {**PLAIN_QUERY, "MultiHopForm": "Comparison", "SupportSpread": "Spread", "Answerability": "Mid",
      "RerankerLogit": "High", "TitleDiversity": "Two"}, {}, "Decompose"),
    ("'Who is the current CEO of X?' with old passages",
     {**PLAIN_QUERY, **CLEAN_RETRIEVAL, "TemporalAnchoring": "VolatileAnchored", "TimeWords": "Present",
      "EvidenceAge": "Stale"}, {}, "Search"),
    ("Same-name trap: relevant-looking passage, wrong entity",
     {**PLAIN_QUERY, **CLEAN_RETRIEVAL, "NameCollision": "Few", "RerankMargin": "Narrow", "TitleDiversity": "Two",
      "SubjectTypeMatch": "Mismatch", "ParametricAgreement": "Disagree"}, {}, "Search"),
    ("Still nothing after two searches",
     {**PLAIN_QUERY, "RerankerLogit": "Low", "EntityCoverage": "None", "Answerability": "Low",
      "AnswerTypePresence": "Absent", "ClosedBookConfidence": "IDK", "EntityFamiliarity": "Unfamiliar"},
     {"SearchRounds": "2+"}, "Abstain"),
]



def main():
    model = build_pgmpy()
    print(f"pgmpy model OK: {len(model.nodes())} chance nodes, {len(model.edges())} edges "
          f"(+ Decision + Utility = {len(model.nodes()) + 2} nodes)")
    n_par = sum((len(states_of(n)) - 1) * int(np.prod([len(states_of(p)) for p in model.get_parents(n)] or [1]))
                for n in model.nodes())
    print(f"free parameters as full tables: {n_par}")

    r = Router()
    print(f"\n{'scenario':<56}{'expected':>10}{'got':>11}   P(serve: correct/IDK/wrong)   top causes")
    ok = 0
    for title, ev, ctx, want in SCENARIOS:
        d = r.decide(ev, ctx)
        top = sorted(d["causes"].items(), key=lambda x: -x[1])[:2]
        ps = d["p_serve"]
        ok += d["action"] == want
        print(f"{title:<56}{want:>10}{d['action']:>11}   {ps['Correct']:.2f}/{ps['Abstain']:.2f}/{ps['Wrong']:.2f}"
              f"            " + ", ".join(f"{c} {p:.2f}" for c, p in top))
    print(f"\n{ok}/{len(SCENARIOS)} scenarios route as designed")

    print("\nAnytime routing: value of measuring the next tier (Monte Carlo)")
    for title, ev in [("clear factoid, query only", PLAIN_QUERY),
                      ("follow-up, query only", {**PLAIN_QUERY, "ResidualReferences": "One", "FollowUpForm": "Yes"}),
                      ("clear factoid + retrieval", {**PLAIN_QUERY, **CLEAN_RETRIEVAL})]:
        vals = {t: r.value_of_tier(ev, t) for t in (1, 2, 3)}
        print(f"   {title:<30} " + "   ".join(f"tier {t}: {v:+.3f} (cost {TIER_COST[t]})" for t, v in vals.items()))

    print("\nAfter the answer (clear factoid served):")
    for g in ("Supported", "Partial", "Unsupported"):
        a = r.after_answer({**PLAIN_QUERY, **CLEAN_RETRIEVAL}, g)
        print(f"   grounding {g:<12} -> {a['decision']:<12} P(outcome) {a['p_outcome']}")

    print("\nSensitivity of the routing to LAMBDA (the only number that is a choice, not a fact):")
    for lam in (1.0, 2.5, 5.0, 10.0):
        rl = Router(lam=lam)
        acts = [rl.decide(ev, ctx)["action"] for _, ev, ctx, _ in SCENARIOS]
        print(f"   lambda {lam:>4}: " + " ".join(f"{a[:4]:<5}" for a in acts))


if __name__ == "__main__":
    main()
