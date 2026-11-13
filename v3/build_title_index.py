"""
Build the Wikipedia title index used by NameCollision (how many pages share a name).
Run once where there is internet (Kaggle is fine, ~10 min, ~3 GB RAM):

    python build_title_index.py --out title_index.tsv.gz

Source: https://dumps.wikimedia.org/enwiki/latest/enwiki-latest-all-titles-in-ns0.gz
(every article title in the main namespace; note it includes redirects, which slightly
inflates counts - the cut-offs are learned, so a constant inflation does not matter).
Output: base name -> number of pages, plus a flag when a '(disambiguation)' page exists.
Record the dump date in the paper (results depend on the snapshot).
"""

import argparse
import gzip
import os
import urllib.request

from signals_v3 import TitleIndex

URL = "https://dumps.wikimedia.org/enwiki/latest/enwiki-latest-all-titles-in-ns0.gz"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="title_index.tsv.gz")
    ap.add_argument("--dump", default="enwiki-latest-all-titles-in-ns0.gz")
    args = ap.parse_args()
    if not os.path.exists(args.dump):
        print(f"downloading {URL} ...")
        req = urllib.request.Request(URL, headers={"User-Agent": "BayesianEpistemicRouting/0.3 (research)"})
        with urllib.request.urlopen(req) as r, open(args.dump, "wb") as f:
            while chunk := r.read(1 << 20):
                f.write(chunk)

    def titles():
        with gzip.open(args.dump, "rt", encoding="utf-8", errors="ignore") as f:
            next(f, None)                                       # header line 'page_title'
            for line in f:
                yield line.rstrip("\n")

    idx = TitleIndex.from_titles(titles())
    idx.save(args.out)
    print(f"{len(idx.counts):,} names shared by 2+ pages, {len(idx.disambig):,} with a disambiguation page "
          f"-> {args.out}")
    for name in ("Pulse", "Belly", "Escape", "Paris", "Inception", "Christopher Nolan"):
        print(f"   {name:<20} {idx.collisions(name)} pages")


if __name__ == "__main__":
    main()
