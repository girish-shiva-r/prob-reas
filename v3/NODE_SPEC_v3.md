# Bayesian Epistemic Routing v3: node specification

Generated from `ber_v3.py` by `make_spec.py`. All numbers are expert priors, to be learned.

## Causes (9, hidden, Low/High)

| Cause | Parents | Prior / noisy-OR | Fixed by | Training label |
|---|---|---|---|---|
| **ReferentAmbiguity**: the name refers to several things ('What genre is Pulse?') | - | P(High) = 0.12 | Clarify (which one?) | the question's name is shared by 2+ KB/Wikipedia pages (AmbigDocs, AmbigQA) |
| **ConstraintAmbiguity**: the entity is clear but the answer depends on an unstated time/place/scope | - | P(High) = 0.06 | Clarify (which time / version / place?) | AmbigQA multi-answer questions, SituatedQA |
| **ContextDependence**: the question cannot stand alone ('Where did she go?') | - | P(High) = 0.12 | Rewrite from history, else Clarify | QReCC / TopiOCQA turn whose gold rewrite differs from the raw question |
| **ParametricGap**: the LLM does not know the answer without context | - | P(High) = 0.4 | (decides whether bad evidence matters) | closed-book run of the LLM is wrong |
| **ReasoningLoad**: needs 2+ facts: bridge / comparison / composition | - | P(High) = 0.15 | Decompose | MuSiQue hops >= 2, 2WikiMultiHopQA type |
| **TemporalMismatch**: the true answer changed; KB or model is out of date | - | P(High) = 0.06 | Search (fresh web results) | FreshQA fast-changing questions |
| **EvidenceGap**: the retrieved passages do not contain the answer | ReferentAmbiguity, ConstraintAmbiguity, ContextDependence, ReasoningLoad | leak 0.2, ReferentAmbiguity 0.3, ConstraintAmbiguity 0.25, ContextDependence 0.45, ReasoningLoad 0.35 | Search (deeper / full pages) | no gold answer string / alias in the top-k passages |
| **ContextMismatch**: passages look relevant but are about a same-name, different entity | ReferentAmbiguity, ContextDependence | leak 0.05, ReferentAmbiguity 0.4, ContextDependence 0.2 | Search with entity + type ('Belly film') | top passage's page shares the name but is not the gold page |
| **EvidenceConflict**: passages disagree with each other or with the model's memory | ContextMismatch, TemporalMismatch | leak 0.05, ContextMismatch 0.45, TemporalMismatch 0.45 | Search for more support, or hedge | ClashEval perturbed passages; gold answer in context but model still wrong |

## Symptoms (22, observed)

| Tier | Symptom | States | Parents | How it is measured |
|---|---|---|---|---|
| 0 | **NameCollision** | Unique / Few / Many | ReferentAmbiguity | Wikipedia/KB pages sharing the question entity's base name (strip '(film)' etc.): 1 / 2-3 / 4+ |
| 0 | **Disambiguator** | Present / Absent | ReferentAmbiguity | query carries a type word or year next to the entity ('the film', '(1999)', 'the band') |
| 0 | **TemporalAnchoring** | Stable / VolatileAnchored / VolatileUnanchored | ConstraintAmbiguity, TemporalMismatch | relation changes over time (CEO, population, champion, record)? if so, is a time given? |
| 0 | **TimeWords** | Absent / Present | TemporalMismatch | 'current', 'latest', 'now', 'this year', or a year >= the model's knowledge cut-off |
| 0 | **ResidualReferences** | None / One / Several | ContextDependence | pronouns / ellipsis still unresolved inside the question itself |
| 0 | **FollowUpForm** | No / Yes | ContextDependence | elliptical / 'and what about...' / <= 4 words while a conversation is running |
| 0 | **MultiHopForm** | Single / Bridge / Comparison | ReasoningLoad | comparison pattern ('which ... first', 'both'), bridge relative clause, possessive chain |
| 1 | **RerankerLogit** | VeryHigh / High / Mid / Low | EvidenceGap, ContextMismatch | top-1 RAW cross-encoder logit: > 6 / 2..6 / -2..2 / < -2 (ms-marco-MiniLM scale; re-learn) |
| 1 | **RerankMargin** | Clear / Narrow / Tied | ReferentAmbiguity, ContextMismatch | top-1 logit minus best logit from a DIFFERENT page: > 3 / 1..3 / < 1 |
| 1 | **TitleDiversity** | One / Two / ThreePlus | ReferentAmbiguity, ContextMismatch | distinct pages among passages within 2 logits of the top passage |
| 1 | **EntityCoverage** | Full / Partial / None | EvidenceGap | share of question entities found (fuzzy) in the top-3 passages: all / some / none |
| 1 | **SubjectTypeMatch** | Match / Unknown / Mismatch | ContextMismatch | type implied by the relation (screenwriter -> film) vs the top page's type (title/first sentence/Wikidata) |
| 1 | **AnswerTypePresence** | Present / Absent | EvidenceGap | expected answer type (who->PERSON, when->DATE, where->GPE/LOC, how many->CARDINAL) in top-3 NER |
| 1 | **Answerability** | High / Mid / Low | EvidenceGap, ReasoningLoad | max has-answer probability of an extractive SQuAD2 model over the top-3: > .7 / .3-.7 / < .3 |
| 1 | **CandidateAgreement** | Agree / Single / Disagree | EvidenceConflict | extractive answers from the top-3 passages: same / only one passage answers / different |
| 1 | **SupportSpread** | Concentrated / Spread | ReasoningLoad | question entities split across different pages and no single passage covers them all |
| 1 | **EvidenceAge** | Fresh / Stale | TemporalMismatch | newest supporting passage's date (page revision / article date) within 90 days? |
| 2 | **EntityFamiliarity** | Familiar / Vague / Unfamiliar | ParametricGap | mean token log-prob of the entity name in 'Here is a fact about <entity>:' (terciles, learned) |
| 2 | **ClosedBookConfidence** | Confident / Unsure / IDK | ParametricGap | P(first token starts 'I don't know') prefilling the bare question: < .1 / .1-.5 / > .5 |
| 2 | **GroundedConfidence** | Confident / Unsure / IDK | EvidenceGap, ParametricGap | same, prefilling the full RAG prompt (its KV cache is reused when we Serve) |
| 3 | **ParametricAgreement** | Agree / NoParametric / Disagree | EvidenceConflict, ParametricGap | closed-book answer (<= 8 tokens) vs the extractive span from the passages |
| 4 | **AnswerGrounding** | Supported / Partial / Unsupported | Outcome | generated answer found in / entailed by the passages (alias match + small NLI model) |

Tier cost (utility units): tier 1: 0.005, tier 2: 0.015, tier 3: 0.04

## Context (3, observed)

| Node | States | Meaning |
|---|---|---|
| **HistoryCandidates** | None / One / Many | antecedents for a pronoun in the previous question AND answer |
| **ClarifyRounds** | 0 / 1 / 2+ | clarifying questions already asked for this question |
| **SearchRounds** | 0 / 1 / 2+ | searches already done for this question |

## Outcome (answer model)

States ['Correct', 'Abstain', 'Wrong']; parents ConstraintAmbiguity, EvidenceGap, ContextMismatch, EvidenceConflict, ParametricGap, ReasoningLoad, TemporalMismatch.
Multinomial logit, base (Abstain, Wrong) = (-2.6, -2.8); effect of each parent being High (dAbstain, dWrong):

- ConstraintAmbiguity: (0.2, 1.4)
- EvidenceGap: (1.5, 0.5)
- ContextMismatch: (0.3, 2.4)
- EvidenceConflict: (0.3, 1.8)
- ParametricGap: (0.9, 0.5)
- ReasoningLoad: (0.9, 1.2)
- TemporalMismatch: (0.4, 2.0)
- interaction EvidenceGap & ParametricGap: (1.6, 0.6)

## Decision (6 actions) as interventions

Every action except Serve/Abstain clears some question-side causes, then searches the web again, so the 3 evidence-side causes are drawn anew from the web tables below. The same Outcome table scores the result.

| Action | Cost | Clears (P success) |
|---|---|---|
| **Serve** | 0.0 | nothing (answer now) |
| **Rewrite** | 0.1 | ContextDependence {'None': 0.1, 'One': 0.85, 'Many': 0.3} |
| **Search** | 0.08 | TemporalMismatch 0.85, ReferentAmbiguity (only if the relation implies a type) 0.8 |
| **Decompose** | 0.15 | ReasoningLoad 0.8 |
| **Clarify** | 0.2 | ReferentAmbiguity 0.9, ConstraintAmbiguity 0.9, ContextDependence 0.9 |
| **Abstain** | 0.0 | - (outcome = Abstain) |

Clarify success is scaled by user fatigue {'0': 1.0, '1': 0.7, '2+': 0.4}; each earlier clarifying question costs 0.1 more.

Web retrieval (noisy-OR; leak by SearchRounds):

- EvidenceGap: leak {'0': 0.1, '1': 0.3, '2+': 0.55}, ReferentAmbiguity 0.3, ConstraintAmbiguity 0.25, ContextDependence 0.5, ReasoningLoad 0.45
- ContextMismatch: leak {'0': 0.03, '1': 0.05, '2+': 0.08}, ReferentAmbiguity 0.45, ContextDependence 0.25
- EvidenceConflict: leak {'0': 0.04, '1': 0.05, '2+': 0.06}, ContextMismatch 0.45, TemporalMismatch 0.45

## Utility (policy, not learned)

Correct +1, Abstain 0, Wrong -lambda; default lambda = 2.5. Serve beats Abstain when P(Wrong)/P(Correct) < 1/lambda, so lambda is a risk level: `learn_v3.choose_lambda` picks it from a target share of wrong answers among served ones on validation data. Hedged answer after generation: {'Correct': 0.6, 'Abstain': 0.0, 'Wrong': -0.5}.

## How the numbers are learned (learn_v3.py)

1. Cut-offs of real-valued symptoms: information gain on the training split only.
2. Weak cause labels make causes partly observed (each label wrong with probability 0.05).
3. The tables above are only a Dirichlet prior; its strength alpha is chosen by grouped cross-validation (alpha = 0 ignores them).
4. EM from the hand-set start and random starts; the spread between starts is reported.
5. Action effects: MAP fit on questions where that action was actually run.
6. recovery_test.py checks all of this on data from a known 'true' network.
