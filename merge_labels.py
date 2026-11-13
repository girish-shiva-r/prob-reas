"""
Merge the parts labelled on different Kaggle accounts into one file.
    python merge_labels.py labels/Qwen_Qwen2.5-7B-Instruct-bfloat16__kb_hybrid.part*of3.jsonl
Writes labels/<name without .partXofN>.jsonl (+ its .profile.json) and prints the summary.
"""

import glob
import json
import re
import shutil
import sys

from label_dataset import summarize


def main(paths):
    paths = sorted({p for a in paths for p in glob.glob(a) if p.endswith(".jsonl")})
    if not paths:
        sys.exit(__doc__)
    rows, seen = [], set()
    for p in paths:
        n = 0
        for line in open(p, encoding="utf-8"):
            r = json.loads(line)
            if r["id"] not in seen:
                seen.add(r["id"])
                rows.append(r)
                n += 1
        print(f"  {p}: {n} rows")
    out = re.sub(r"\.part\d+of\d+(?=\.jsonl$)", "", paths[0])
    with open(out, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    prof = paths[0].replace(".jsonl", ".profile.json")
    try:
        shutil.copy(prof, out.replace(".jsonl", ".profile.json"))
    except FileNotFoundError:
        pass
    print(f"-> {out}: {len(rows)} questions")
    for ds in sorted({r.get("dataset") for r in rows}):
        print(f"\n=== {ds}")
        summarize([r for r in rows if r.get("dataset") == ds])


if __name__ == "__main__":
    main(sys.argv[1:])
