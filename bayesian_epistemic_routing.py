"""
Bayesian Epistemic Routing for RAG Pipelines
---------------------------------------------
Phase 1 scaffold (v2): revised variable set, 20-node network.

ALL probability values below are PLACEHOLDERS (hand-tuned guesses, not
learned from data). They make the network complete, keep check_model()
happy, and make the sanity scenarios at the bottom route sensibly.
Replace them with CPDs learned from labelled data in Phase 2.

Changes from v1
  * Every input is observable BEFORE generation starts, so routing can
    happen before the LLM is called.
      - LogitEntropy            -> DomainShift          (needed generation)
      - ModelTemperature        -> EntityPopularity     (constant setting)
      - SystemPromptStrictness  -> TemporalSensitivity  (constant setting)
      - AccountTier             -> UnresolvedReferences (not an epistemic signal)
      - SessionLength           -> QueryReformulation   (weak signal)
      - DocumentRecency         -> RerankerScore        (similarity != relevance)
  * Bottlenecks renamed so that "High" reads as "trouble" everywhere:
      ContextQuality    -> ContextDeficit
      IntentClarity     -> IntentUncertainty
      ModelPreparedness -> KnowledgeGap
  * OptimalRoute is now a real node (the 20th). Its CPD is built so that
    argmax P(OptimalRoute) reproduces the deck's thresholds exactly:
        P(H) < 0.2         -> Serve
        0.2 <= P(H) <= 0.6 -> Search
        P(H) > 0.6         -> Clarify
  * extract_evidence() turns raw pipeline measurements into Low/High states.
  * The Variable Elimination engine is built once, not on every call.

Structure (14 inputs + 4 bottlenecks + HallucinationRisk + OptimalRoute = 20)
  Module 1 Query Profiling:      QueryLength, QuestionType, EntitySpecificity,
                                 TopicComplexity              -> QueryAmbiguity
  Module 2 Retrieval Diagnostics: FAISSVectorDistance, RerankerScore,
                                 ChunkDensity, SourceAuthority -> ContextDeficit
  Module 3 User Context:         FollowUpStatus, QueryReformulation,
                                 UnresolvedReferences          -> IntentUncertainty
  Module 4 Knowledge Coverage:   EntityPopularity, TemporalSensitivity,
                                 DomainShift                   -> KnowledgeGap
  Convergence: 4 bottlenecks -> HallucinationRisk -> OptimalRoute

Convention: every binary node has states ("Low", "High") and "High" ALWAYS
means "more evidence of trouble".
"""

from itertools import product

from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination

STATES = ["Low", "High"]
ROUTES = ["Serve", "Search", "Clarify"]

# --------------------------------------------------------------------------
# 1. MODULE DEFINITIONS
#    One place that lists every input, what "High" means, and how to
#    measure it. The DAG, priors and evidence extraction are all built
#    from this table.
#
#    Each input: name -> (prior P(High), weight into its bottleneck,
#                          meaning of "High")
# --------------------------------------------------------------------------

MODULES = {
    "QueryAmbiguity": {
        "QueryLength":       (0.30, 0.30, "very short query (< 4 words)"),
        "QuestionType":      (0.35, 0.35, "open-ended / opinion / compound question"),
        "EntitySpecificity": (0.40, 0.30, "no named entities in the query"),
        "TopicComplexity":   (0.25, 0.30, "multi-hop: needs 2+ separate facts"),
    },
    "ContextDeficit": {
        "FAISSVectorDistance": (0.35, 0.20, "top-1 chunk is far from the query"),
        "RerankerScore":       (0.35, 0.35, "cross-encoder says chunks are not relevant"),
        "ChunkDensity":        (0.40, 0.20, "fewer than 2 of top-k chunks pass relevance cutoff"),
        "SourceAuthority":     (0.30, 0.15, "chunks come from low-authority sources"),
    },
    "IntentUncertainty": {
        "FollowUpStatus":       (0.30, 0.40, "follow-up that depends on unclear earlier context"),
        "QueryReformulation":   (0.20, 0.35, "user is rephrasing -> last answer missed"),
        "UnresolvedReferences": (0.25, 0.45, "pronouns like 'it' / 'that one' with no clear target"),
    },
    "KnowledgeGap": {
        "EntityPopularity":    (0.30, 0.30, "rare / long-tail entity"),
        "TemporalSensitivity": (0.20, 0.30, "asks about recent events / after model cutoff"),
        "DomainShift":         (0.25, 0.30, "query is outside the knowledge base's domain"),
    },
}

# How strongly each bottleneck pushes HallucinationRisk toward "High".
# QueryAmbiguity and IntentUncertainty carry the most weight because no
# amount of extra context fixes a question the system doesn't understand;
# those cases should end in "Clarify". Context / knowledge problems alone
# should land in the "Search" band.
CONVERGENCE_WEIGHTS = {
    "QueryAmbiguity":    0.70,
    "ContextDeficit":    0.22,
    "IntentUncertainty": 0.65,
    "KnowledgeGap":      0.22,
}

BOTTLENECK_BASE = 0.03   # P(bottleneck=High) when all its inputs are Low
RISK_BASE = 0.03         # P(HallucinationRisk=High) when all bottlenecks are Low


# --------------------------------------------------------------------------
# 2. CPD HELPERS
# --------------------------------------------------------------------------

def root_cpd(name, p_high):
    """Binary input node with prior P(High) = p_high."""
    return TabularCPD(
        variable=name,
        variable_card=2,
        values=[[1 - p_high], [p_high]],
        state_names={name: STATES},
    )


def weighted_cpd(name, parents, weights, base):
    """
    Binary node whose P(High | parents) = base + sum of weights of parents
    that are "High", clipped to [0.02, 0.98].

    Placeholder scoring rule: coherent and monotonic (more trouble in ->
    more trouble out) without hand-writing 8- or 16-row tables.
    """
    high_col = []
    for combo in product([0, 1], repeat=len(parents)):  # 0 = Low, 1 = High
        score = base + sum(w for bit, w in zip(combo, weights) if bit)
        high_col.append(min(max(score, 0.02), 0.98))
    return TabularCPD(
        variable=name,
        variable_card=2,
        values=[[1 - h for h in high_col], high_col],
        evidence=list(parents),
        evidence_card=[2] * len(parents),
        state_names={name: STATES, **{p: STATES for p in parents}},
    )


def route_cpd():
    """
    P(OptimalRoute | HallucinationRisk), built so that argmax over routes
    matches the deck's thresholds. With p = P(HallucinationRisk=High):

        P(Serve)   = 0.5 - 0.5p
        P(Search)  = 0.4
        P(Clarify) = 0.1 + 0.5p

    Serve == Search at p = 0.2 and Search == Clarify at p = 0.6, so the
    most likely route is Serve below 0.2, Search from 0.2 to 0.6, and
    Clarify above 0.6.
    """
    return TabularCPD(
        variable="OptimalRoute",
        variable_card=3,
        #        H=Low  H=High
        values=[[0.5,   0.0],    # Serve
                [0.4,   0.4],    # Search
                [0.1,   0.6]],   # Clarify
        evidence=["HallucinationRisk"],
        evidence_card=[2],
        state_names={"OptimalRoute": ROUTES, "HallucinationRisk": STATES},
    )


# --------------------------------------------------------------------------
# 3. BUILD THE NETWORK
# --------------------------------------------------------------------------

def build_model():
    model = DiscreteBayesianNetwork()
    cpds = []

    for bottleneck, inputs in MODULES.items():
        names = list(inputs)
        model.add_edges_from((name, bottleneck) for name in names)
        cpds += [root_cpd(name, prior) for name, (prior, _, _) in inputs.items()]
        cpds.append(weighted_cpd(
            bottleneck, names, [w for _, w, _ in inputs.values()], BOTTLENECK_BASE))

    bottlenecks = list(CONVERGENCE_WEIGHTS)
    model.add_edges_from((b, "HallucinationRisk") for b in bottlenecks)
    cpds.append(weighted_cpd(
        "HallucinationRisk", bottlenecks, list(CONVERGENCE_WEIGHTS.values()), RISK_BASE))

    model.add_edge("HallucinationRisk", "OptimalRoute")
    cpds.append(route_cpd())

    model.add_cpds(*cpds)
    assert model.check_model(), "Model is not valid - check CPD/edge consistency."
    return model


model = build_model()
INFERENCE = VariableElimination(model)   # built once, reused per request

INPUT_NODES = [name for inputs in MODULES.values() for name in inputs]


# --------------------------------------------------------------------------
# 4. EVIDENCE EXTRACTION
#    Turns raw pipeline measurements into Low/High states.
#    Cut-offs are PLACEHOLDERS - set them from your own data distribution.
#    Any raw value that is missing (None / not supplied) is left out, and
#    Variable Elimination averages over it.
# --------------------------------------------------------------------------

OPEN_ENDED_TYPES = {"open_ended", "opinion", "compound", "how", "why"}

EVIDENCE_RULES = {
    # Module 1 - Query Profiling (spaCy / simple rules on the query text)
    "QueryLength":          ("word_count",            lambda v: v < 4),
    "QuestionType":         ("question_type",         lambda v: v in OPEN_ENDED_TYPES),
    "EntitySpecificity":    ("named_entity_count",    lambda v: v == 0),
    "TopicComplexity":      ("sub_question_count",    lambda v: v >= 2),
    # Module 2 - Retrieval Diagnostics (FAISS + cross-encoder reranker)
    "FAISSVectorDistance":  ("top1_faiss_distance",   lambda v: v > 0.8),
    "RerankerScore":        ("top1_rerank_score",     lambda v: v < 0.5),
    "ChunkDensity":         ("relevant_chunk_count",  lambda v: v < 2),
    "SourceAuthority":      ("mean_source_authority", lambda v: v < 0.5),
    # Module 3 - User Context (conversation history)
    "FollowUpStatus":       ("is_ambiguous_followup", lambda v: bool(v)),
    "QueryReformulation":   ("similarity_to_prev_query", lambda v: v > 0.8),
    "UnresolvedReferences": ("unresolved_reference_count", lambda v: v >= 1),
    # Module 4 - Knowledge Coverage (pre-generation stand-ins for "does the
    # model know this?")
    "EntityPopularity":     ("entity_monthly_views",  lambda v: v < 1000),
    "TemporalSensitivity":  ("is_time_sensitive",     lambda v: bool(v)),
    "DomainShift":          ("kb_domain_similarity",  lambda v: v < 0.3),
}


def extract_evidence(raw: dict) -> dict:
    """
    raw: measurements from the pipeline, e.g.
        {"word_count": 3, "top1_rerank_score": 0.21, "is_time_sensitive": True}
    Returns e.g. {"QueryLength": "High", "RerankerScore": "High", ...}
    """
    evidence = {}
    for node, (key, is_high) in EVIDENCE_RULES.items():
        value = raw.get(key)
        if value is not None:
            evidence[node] = "High" if is_high(value) else "Low"
    return evidence


# --------------------------------------------------------------------------
# 5. ROUTING (runtime exact inference)
# --------------------------------------------------------------------------

def route_query(evidence: dict) -> dict:
    """
    evidence: observed input states, e.g. {"RerankerScore": "High", ...}
              (use extract_evidence() to build it from raw measurements).
    Unobserved inputs are marginalised by Variable Elimination.

    Returns P(HallucinationRisk=High), the full route distribution, and the
    chosen route (argmax of P(OptimalRoute), ties going to "Search" to match
    the deck's inclusive 0.2 <= P(H) <= 0.6 band).
    """
    unknown = set(evidence) - set(INPUT_NODES)
    if unknown:
        raise ValueError(f"Unknown evidence variables: {sorted(unknown)}")

    risk = INFERENCE.query(["HallucinationRisk"], evidence=evidence, show_progress=False)
    p_high = float(risk.values[risk.state_names["HallucinationRisk"].index("High")])

    route_post = INFERENCE.query(["OptimalRoute"], evidence=evidence, show_progress=False)
    dist = {r: float(route_post.values[route_post.state_names["OptimalRoute"].index(r)])
            for r in ROUTES}
    best = max(dist.values())
    route = "Search" if abs(dist["Search"] - best) < 1e-9 else max(dist, key=dist.get)

    return {
        "p_hallucination_high": round(p_high, 4),
        "route": route,
        "route_distribution": {r: round(p, 4) for r, p in dist.items()},
    }


def threshold_route(p_high: float) -> str:
    """The deck's rule, kept for cross-checking route_query()."""
    if p_high < 0.2:
        return "Serve"
    if p_high <= 0.6:
        return "Search"
    return "Clarify"


# --------------------------------------------------------------------------
# 6. SANITY SCENARIOS
# --------------------------------------------------------------------------

def _all(state):
    return {name: state for name in INPUT_NODES}


SCENARIOS = [
    ("Everything looks fine", _all("Low"), "Serve"),
    ("Clear query, but retrieval is poor",
     {**_all("Low"), "FAISSVectorDistance": "High", "RerankerScore": "High",
      "ChunkDensity": "High"}, "Search"),
    ("Clear query about a recent / rare topic",
     {**_all("Low"), "TemporalSensitivity": "High", "EntityPopularity": "High"}, "Search"),
    ("Vague, open-ended query with no entities",
     {**_all("Low"), "QueryLength": "High", "QuestionType": "High",
      "EntitySpecificity": "High"}, "Clarify"),
    ("Follow-up full of unresolved pronouns",
     {**_all("Low"), "FollowUpStatus": "High", "UnresolvedReferences": "High",
      "QueryReformulation": "High"}, "Clarify"),
    ("Everything looks bad", _all("High"), "Clarify"),
]


if __name__ == "__main__":
    print(f"Model valid. Nodes: {len(model.nodes())}, Edges: {len(model.edges())}\n")

    for label, evidence, expected in SCENARIOS:
        result = route_query(evidence)
        assert result["route"] == threshold_route(result["p_hallucination_high"]), \
            "OptimalRoute CPD disagrees with the threshold rule"
        mark = "OK " if result["route"] == expected else "XX "
        print(f"{mark}{label:<42} P(H)={result['p_hallucination_high']:.3f}  "
              f"-> {result['route']:<8} (expected {expected})")

    print("\nFrom raw pipeline measurements:")
    raw = {
        "word_count": 9, "question_type": "factoid", "named_entity_count": 2,
        "sub_question_count": 1, "top1_faiss_distance": 0.35,
        "top1_rerank_score": 0.18, "relevant_chunk_count": 1,
        "mean_source_authority": 0.7, "is_ambiguous_followup": False,
        "unresolved_reference_count": 0, "entity_monthly_views": 250_000,
        "is_time_sensitive": True, "kb_domain_similarity": 0.6,
        # similarity_to_prev_query not available -> marginalised
    }
    ev = extract_evidence(raw)
    print("  evidence:", ev)
    print("  result:  ", route_query(ev))
