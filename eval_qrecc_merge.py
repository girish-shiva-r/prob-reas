"""
Evaluate the rule-based clarification merge against QReCC's human rewrites.
---------------------------------------------------------------------------
QReCC (Anantha et al., 2021; github.com/apple/ml-qrecc) gives, for each
follow-up question, a human-written standalone rewrite:
    Question: "When was he born?"   Rewrite: "When was Johann Sebastian Bach born?"

Our merge resolves a gap using the user's REPLY to a clarifying question,
so we simulate that reply from the human rewrite: the words the human
added ("Johann Sebastian Bach") become what the user says, wrapped in the
ways people actually answer ("I meant ...", "oh sorry, I was talking about
...", "it's ..."). Then we compare our merged query with the human rewrite.

Metrics (computed on content words, i.e. what a retriever actually uses):
    exact   - merged == human rewrite (after lower-casing, punctuation removed)
    recall  - share of the rewrite's content words that the merge contains
    prec    - share of the merge's content words that are in the rewrite
              (junk like "meant", "sorry", "talking" lowers this)
    F1      - harmonic mean of the two
Baselines: do nothing (send the question as is) / naive append of the raw reply.

Also measured:
    * detection: does unresolved_references() flag pronouns that needed resolving?
    * history auto-resolution: when the previous question names exactly one
      entity, substitute it directly (no clarifying question at all).
    * latency per merge on this CPU.

Run:   python eval_qrecc_merge.py data/qrecc_test.json [--limit 3000]
Writes eval_qrecc_failures.csv with the worst merges for inspection.
"""

import argparse
import csv
import difflib
import json
import random
import re
import statistics
import time
import warnings
from collections import defaultdict

warnings.filterwarnings("ignore")

from clarify_loop import merge_clarification, resolve_from_history
from feature_extractors import REFERRING_PRONOUNS, load_default_nlp, pronouns_needing_resolution

STOP = set("""a an the of to in on at for by with from and or but is are was were be been being do does did
has have had what which who whom whose when where why how this that these those it its they them their he him
his she her i me my you your we our there here as about into than then so not no can could would should will
shall may might must also any some all more most very just""".split())
REPLY_STYLES = {
    "bare":             "{x}",
    "I meant":          "I meant {x}",
    "oh sorry, talking about": "oh sorry, I was talking about {x}",
    "it's":             "it's {x}",
    "trailing sorry":   "{x}, sorry",
}


def tokens(s):
    return re.findall(r"[\w']+|[^\w\s]", s)


def content(s):
    words = re.findall(r"[a-z0-9]+", s.lower().replace("'s", " "))
    return [w for w in words if w not in STOP]


def norm(s):
    return " ".join(re.findall(r"[a-z0-9]+", s.lower()))


def prf(pred, gold):
    p, g = content(pred), content(gold)
    if not g:
        return 1.0, 1.0, 1.0
    gs, ps = set(g), set(p)
    rec = len(gs & ps) / len(gs)
    prec = len(gs & ps) / len(ps) if ps else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return prec, rec, f1


def classify(q, r):
    """Return (kind, pronoun, added_text) describing how the human rewrote q into r."""
    a, b = tokens(q), tokens(r)
    ops = [o for o in difflib.SequenceMatcher(None, [w.lower() for w in a], [w.lower() for w in b]).get_opcodes()
           if o[0] != "equal"]
    added = " ".join(" ".join(b[j1:j2]) for _, _, _, j1, j2 in ops if j2 > j1)
    added = re.sub(r"\s+'s\b", "'s", added).strip(" ,.?")
    if len(ops) == 1 and ops[0][0] == "replace" and ops[0][2] - ops[0][1] == 1 \
            and a[ops[0][1]].lower() in REFERRING_PRONOUNS:
        return "pronoun swap", a[ops[0][1]], re.sub(r"'s$", "", added)
    if len(ops) == 1 and ops[0][0] == "insert":
        return "insertion", None, added
    return "complex", None, added


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path", nargs="?", default="data/qrecc_test.json")
    ap.add_argument("--limit", type=int, default=0, help="sample this many follow-ups (0 = all)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    nlp = load_default_nlp()
    data = [x for x in json.load(open(args.path, encoding="utf-8"))
            if x["Turn_no"] > 1 and x["Question"].strip() != x["Rewrite"].strip() and x["Context"]]
    if args.limit:
        random.Random(args.seed).shuffle(data)
        data = data[:args.limit]

    scores = defaultdict(list)          # (kind, method) -> [(p, r, f1, exact)]
    latency_ms, failures = [], []
    detect = {"flagged": 0, "total": 0}
    auto = {"applicable": 0, "exact": 0, "f1": []}

    for x in data:
        q, gold = x["Question"].strip(), x["Rewrite"].strip()
        kind, pron, added = classify(q, gold)
        if not added:
            continue
        reason = "UnresolvedReferences" if kind == "pronoun swap" else "FollowUpStatus"
        details = {"unresolved_pronouns": [pron]} if pron else {}

        def score(method, pred):
            p, r, f = prf(pred, gold)
            scores[(kind, method)].append((p, r, f, norm(pred) == norm(gold)))

        score("do nothing", q)
        for style, tpl in REPLY_STYLES.items():
            reply = tpl.format(x=added)
            score("naive append", f"{q} ({reply})")
            t0 = time.perf_counter()
            merged = merge_clarification(q, reply, reason, details, nlp=nlp)
            latency_ms.append((time.perf_counter() - t0) * 1000)
            score("our merge", merged)
            scores[(kind, f"our merge | {style}")].append((*prf(merged, gold)[:3], norm(merged) == norm(gold)))
            if kind == "pronoun swap" and prf(merged, gold)[2] < 0.8:
                failures.append([kind, style, q, reply, merged, gold])

        if kind == "pronoun swap":
            # detection: does the query get flagged as having a pronoun with no antecedent?
            prev_q = x["Context"][-2] if len(x["Context"]) >= 2 else ""
            detect["total"] += 1
            detect["flagged"] += bool(pronouns_needing_resolution(nlp(q)))
            # history auto-resolution: previous (resolved) question names exactly one entity
            guess, changed = resolve_from_history(q, [prev_q], nlp)
            if changed:
                auto["applicable"] += 1
                auto["exact"] += norm(guess) == norm(gold)
                auto["f1"].append(prf(guess, gold)[2])

    # ---- report ---------------------------------------------------------------
    kinds = ["pronoun swap", "insertion", "complex"]
    n_by_kind = {k: len(scores[(k, "do nothing")]) for k in kinds}
    total = sum(n_by_kind.values())
    print(f"QReCC follow-ups evaluated: {total}  " +
          "  ".join(f"{k}: {n} ({n / total:.0%})" for k, n in n_by_kind.items()))
    print("\nContent-word scores vs the human rewrite (mean over 5 reply styles):")
    print(f"{'edit type':<14}{'method':<16}{'exact':>7}{'recall':>8}{'prec':>7}{'F1':>7}")
    for k in kinds:
        for m in ["do nothing", "naive append", "our merge"]:
            rows = scores[(k, m)]
            if rows:
                print(f"{k:<14}{m:<16}"
                      f"{statistics.mean(r[3] for r in rows):>7.1%}{statistics.mean(r[1] for r in rows):>8.2f}"
                      f"{statistics.mean(r[0] for r in rows):>7.2f}{statistics.mean(r[2] for r in rows):>7.2f}")
    print("\nOur merge on pronoun swaps, by how the user phrased the reply:")
    for style in REPLY_STYLES:
        rows = scores[("pronoun swap", f"our merge | {style}")]
        print(f"  {style:<26} exact {statistics.mean(r[3] for r in rows):6.1%}   F1 {statistics.mean(r[2] for r in rows):.2f}")
    if detect["total"]:
        print(f"\nDetection: pronoun with no antecedent flagged in {detect['flagged'] / detect['total']:.1%} "
              f"of {detect['total']} pronoun-swap turns")
    if auto["applicable"]:
        print(f"History auto-resolution (prev question names exactly 1 entity): used on "
              f"{auto['applicable']} turns ({auto['applicable'] / detect['total']:.0%}), exact {auto['exact'] / auto['applicable']:.1%}, "
              f"F1 {statistics.mean(auto['f1']):.2f}")
    lat = sorted(latency_ms)
    print(f"\nLatency per merge (spaCy small, this CPU): mean {statistics.mean(lat):.1f} ms, "
          f"p95 {lat[int(0.95 * len(lat))]:.1f} ms")

    with open("eval_qrecc_failures.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["edit_type", "reply_style", "question", "simulated_reply", "our_merge", "human_rewrite"])
        w.writerows(failures)
    print(f"Wrote {len(failures)} low-scoring pronoun-swap merges to eval_qrecc_failures.csv")


if __name__ == "__main__":
    main()
