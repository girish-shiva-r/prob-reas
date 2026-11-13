"""
Re-grade an existing labels file with the current grade() - no LLM needed (replies are stored).
    python regrade_labels.py labels/Qwen_Qwen2.5-7B-Instruct-bfloat16.jsonl
Writes the file in place (a .bak copy is kept) and prints how many rows changed outcome.
"""

import json
import shutil
import sys
from collections import Counter

from label_dataset import grade, summarize


def outcome(r):
    return next(k for k in ("correct", "abstained", "empty", "hallucinated") if r.get(k))


def main(path):
    rows = [json.loads(l) for l in open(path, encoding="utf-8")]
    moves = Counter()
    for r in rows:
        before = outcome(r)
        r.update(grade(r["reply"], r["answers"]))
        moves[(before, outcome(r))] += 1
    shutil.copy(path, path + ".bak")
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    changed = {k: v for k, v in moves.items() if k[0] != k[1]}
    print(f"{len(rows)} rows re-graded; changed outcome: {sum(changed.values())}")
    for (a, b), n in sorted(changed.items(), key=lambda x: -x[1]):
        print(f"   {a:>12} -> {b:<12} {n}")
    summarize(rows)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    main(sys.argv[1])
