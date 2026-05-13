"""Time the pandas reference on a curated benchmark query set."""
from __future__ import annotations
import json
import time
from pathlib import Path

import pandas as pd
import tquery as tq

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA = REPO_ROOT / "bench" / "dataset_large.csv"
OUT  = REPO_ROOT / "bench" / "results_python.jsonl"

QUERIES = [
    "K50",
    "K50*",
    "K50 and K51",
    "K50 or K51",
    "not K50",
    "min 3 of K50",
    "2-5 of K50",
    "1st K50",
    "last 2 of K50",
    "K50 before K51",
    "K50 after K51",
    "every K50 before K51",
    "K50 inside 30 days after K51",
    "K50 inside 30 to 90 days after K51",
    "K50 inside 100 days",
    "K50 outside 30 days after K51",
    "K50 inside -5 to 20 days around K51",
    "K50 inside 5 events after K51",
    "K50 inside last 5 events",
    "(K50 or K51) and K52",
]

def main() -> None:
    df = pd.read_csv(DATA, parse_dates=["start_date"])
    load_ms = 0  # set after first read
    print(f"loaded {len(df)} rows, {df['pid'].nunique()} persons\n")

    with OUT.open("w") as fh:
        print(f"{'query':<55s}  {'count':>7s}  {'time_ms':>10s}")
        for q in QUERIES:
            # Warm up the cache once (parser + evaluator do internal caching)
            tq.tquery(df, q, pid="pid", date="start_date", cols="icd")
            t0 = time.perf_counter()
            r = tq.tquery(df, q, pid="pid", date="start_date", cols="icd")
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            print(f"{q:<55s}  {r.count:>7d}  {elapsed_ms:>10.2f}")
            fh.write(json.dumps({"query": q, "count": int(r.count), "time_ms": elapsed_ms}) + "\n")


if __name__ == "__main__":
    main()
