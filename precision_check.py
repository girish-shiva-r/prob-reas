"""
Is the damaged output ("Pete Do ### Response:", "Tim Bur Burton") caused by
numerical precision? Re-answer the same questions with two precision settings
and count damaged replies.

    python precision_check.py labels/<labels file>.jsonl            # default: NF4 + noflash (T4)
    python precision_check.py labels/<file>.jsonl --llm stripedhyena-nf4   # flash-attn GPU

Settings compared (everything identical except the Hyena poles/residues):
    A  old   - poles/residues rounded to the load dtype (how the damaged run was made)
    B  fixed - poles/residues re-read from the checkpoint in float32
If B removes most of the damage, the rounding was the cause and relabelling is worth it.
"""

import argparse
import collections
import gc
import json
import re
import time

import torch

from feature_extractors import FeatureExtractor, KnowledgeBase
from label_dataset import grade
from llm_backends import build_prompt, make_backend

STUTTER = re.compile(r"\b(\w{1,6}) (\1\w+)")      # "B Bacigalupi", "Bur Burton"
GLUED = re.compile(r"[a-z][A-Z][a-z]")             # "HelpWanted", "SocialNetwork"


def damaged(reply: str, raw: str) -> bool:
    return (not reply or "###" in reply or bool(STUTTER.search(reply)) or bool(GLUED.search(reply))
            or reply.endswith((" is", " of", " was")))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("labels")
    ap.add_argument("--llm", default="stripedhyena-nf4-noflash")
    ap.add_argument("--n-damaged", type=int, default=14)
    ap.add_argument("--n-clean", type=int, default=4)
    ap.add_argument("--compute-dtype", default=None, help="float16 on T4 (the damaged run); default = best for the GPU")
    args = ap.parse_args()

    rows = [json.loads(l) for l in open(args.labels, encoding="utf-8")]
    bad = [r for r in rows if damaged(r["reply"], r.get("raw_reply") or "")][:args.n_damaged]
    good = [r for r in rows if r["correct"] and not damaged(r["reply"], "")][:args.n_clean]
    sample = bad + good
    print(f"{len(bad)} damaged + {len(good)} clean questions from {args.labels}\n")

    kb = KnowledgeBase.load("kb")
    fx = FeatureExtractor(kb, popularity=json.load(open("kb/popularity.json")))
    prompts = []
    for r in sample:
        res = fx.extract(r["question"])
        ranked = sorted(zip(res.details["rerank_scores"], res.details["chunks"]), key=lambda x: -x[0])
        prompts.append(build_prompt(r["question"], [c["text"] for _, c in ranked], 3))

    ap_dtype = {"compute_dtype": args.compute_dtype} if args.compute_dtype else {}
    settings = {"A old": dict(ap_dtype, keep_poles_fp32=False),
                "B fixed": dict(ap_dtype, keep_poles_fp32=True)}
    out = {}
    for label, kw in settings.items():
        print(f"== loading {label} {kw}")
        llm = make_backend(args.llm, **kw)
        print("   pole/residue dtypes:", dict(collections.Counter(
            str(p.dtype) for n, p in llm.model.named_parameters() if n.endswith((".poles", ".residues")))))
        t0 = time.time()
        out[label] = [llm.generate(p) for p in prompts]
        print(f"   {len(prompts)} answers in {time.time() - t0:.0f}s (decoding mode {llm._mode})\n")
        del llm
        gc.collect()
        torch.cuda.empty_cache()

    print(f"{'question':<46} {'gold':<18} {'A old':<34} {'B fixed':<34}")
    for r, a, b in zip(sample, out["A old"], out["B fixed"]):
        print(f"{r['question'][:45]:<46} {r['answers'][0][:17]:<18} {a[:33]!r:<34} {b[:33]!r:<34}")
    print()
    for label, replies in out.items():
        dmg = sum(damaged(x, "") for x in replies)
        cor = sum(grade(x, r["answers"])["correct"] for x, r in zip(replies, sample))
        print(f"{label}: damaged {dmg}/{len(replies)}   correct {cor}/{len(replies)}")


if __name__ == "__main__":
    main()
