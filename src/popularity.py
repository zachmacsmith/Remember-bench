"""Popularity covariate via the infini-gram API (exact n-gram counts in open pretraining corpora).

For each item, query three ~12-word windows of the ground truth (start, middle, end) and keep the max count.
log_freq = log(count + 1). Default index: Dolma v1.7 (OLMo's corpus), so for OLMo it is true exposure and for
other models it is a proxy.

  python -I src/popularity.py [--index v4_dolma-v1_7_llama]
Writes data/items/pilot_popularity.csv.
"""
import argparse
import csv
import math
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import requests

sys.path.insert(0, os.path.dirname(__file__))
from common import ROOT, ITEMS, read_jsonl, clean_quotes  # noqa: E402

API = "https://api.infini-gram.io/"


def count(index, q):
    for attempt in range(5):
        try:
            r = requests.post(API, json={"index": index, "query_type": "count", "query": q}, timeout=60)
            d = r.json()
            if "count" in d:
                return d["count"], d.get("approx", False)
        except Exception:
            pass
        time.sleep(2 ** attempt)
    return None, None


def windows(text, w=12):
    toks = clean_quotes(text).replace("\n", " ").split()
    if len(toks) <= w:
        return [" ".join(toks)]
    mid = (len(toks) - w) // 2
    return [" ".join(toks[i:i + w]) for i in sorted({0, mid, len(toks) - w})]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", default="v4_dolma-v1_7_llama")
    ap.add_argument("--items", default="data/items/pilot_ground_truth.jsonl")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    items = read_jsonl(ROOT / args.items)

    def one(it):
        best, best_q, approx = -1, None, False
        for q in windows(it["ground_truth"]):
            c, a = count(args.index, q)
            if c is not None and c > best:
                best, best_q, approx = c, q, a
        print(it["item_id"], best, flush=True)
        return dict(item_id=it["item_id"], index=args.index, max_count=best, approx=approx,
                    log_freq=round(math.log(best + 1), 4) if best >= 0 else None, query=best_q)

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        rows = list(ex.map(one, items))
    with open(ITEMS / "pilot_popularity.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


if __name__ == "__main__":
    main()
