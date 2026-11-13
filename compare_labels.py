"""
Compare two labelling runs of the same questions, e.g. Kaggle (NF4, no flash-attn)
vs a local GPU (NF4, flash-attn), to measure how much the setup changes the labels.

    python compare_labels.py labels/kaggle_noflash.jsonl labels/local_flash.jsonl

Reports, on the questions both runs answered:
  * identical replies (word for word, after normalising case/punctuation)
  * identical outcome (correct / abstained / hallucinated)
  * a 3x3 table of how outcomes moved between the two runs
  * the hallucination rate in each run (what the Bayesian network learns from)
"""

import json
import sys
from collections import Counter

from label_dataset import normalize


def outcome(r):
    if r.get("empty") or not r["reply"].strip():
        return "empty"
    return "correct" if r["correct"] else "abstained" if r["abstained"] else "hallucinated"


def load(path):
    with open(path, encoding="utf-8") as f:
        return {r["id"]: r for r in map(json.loads, f)}


def main(a_path, b_path):
    a, b = load(a_path), load(b_path)
    common = sorted(set(a) & set(b))
    if not common:
        sys.exit("No questions in common.")
    n = len(common)
    same_reply = sum(normalize(a[i]["reply"]) == normalize(b[i]["reply"]) for i in common)
    moves = Counter((outcome(a[i]), outcome(b[i])) for i in common)
    same_outcome = sum(v for (x, y), v in moves.items() if x == y)
    print(f"{n} questions in both runs\n  A = {a_path}\n  B = {b_path}\n")
    print(f"identical replies:  {same_reply}/{n} ({same_reply / n:.1%})")
    print(f"identical outcome:  {same_outcome}/{n} ({same_outcome / n:.1%})\n")
    kinds = ["correct", "abstained", "hallucinated", "empty"]
    corner = "A \\ B"
    print(f"{corner:<14}" + "".join(f"{k:>14}" for k in kinds))
    for x in kinds:
        print(f"{x:<14}" + "".join(f"{moves[(x, y)]:>14}" for y in kinds))
    for label, run in (("A", a), ("B", b)):
        rate = sum(run[i]["hallucinated"] for i in common) / n
        print(f"\nhallucination rate {label}: {rate:.1%}", end="")
    print("\n\nExamples where the outcome changed:")
    shown = 0
    for i in common:
        if outcome(a[i]) != outcome(b[i]) and shown < 8:
            print(f"  {a[i]['question']!r}\n     A: {a[i]['reply']!r} ({outcome(a[i])})"
                  f"\n     B: {b[i]['reply']!r} ({outcome(b[i])})")
            shown += 1


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    main(sys.argv[1], sys.argv[2])
