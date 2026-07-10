"""Randomized differential testing across backends.

Generates many SMALL adversarial timelines (few persons, few codes, a
short date horizon so windows and same-date collisions occur constantly)
and checks that pandas, Polars and DuckDB return identical person sets
for a broad pool of query forms. The golden corpus pins agreed answers
on one dataset; this tool hunts for divergence on datasets nobody
hand-picked — the class of bug that let the asof-nearest window bug
survive parity testing.

Usage (from the repo root):

    python tools/fuzz_parity.py                 # 40 datasets
    python tools/fuzz_parity.py --iters 200
    python tools/fuzz_parity.py --seed 1234     # reproduce a run

On mismatch it prints the seed, the query, the per-backend person sets
and a CSV of the dataset, then exits non-zero.
"""

from __future__ import annotations

import argparse
import sys

import numpy as np
import pandas as pd

import tquery as tq

CODES = ["K50", "K50.1", "K51", "K52", "X"]

# Query pool: every DSL family, biased toward the historically fragile
# forms (lower-bounded windows, self-reference, around, outside, zero
# counts, quantifiers, aggregates).
QUERIES = [
    # atoms / logic / counts
    "K50", "K50*", "K50 and K51", "K50 or K52", "not K50",
    "min 2 of K50", "max 2 of K50", "exactly 1 of K51",
    "1-2 of K50", "0-2 of K50", "exactly 0 of K50",
    "1st K50", "-1st K50", "first 2 of K51", "last 2 of K50",
    # temporal defaults + quantifiers
    "K50 before K51", "K50 after K51", "K50 simultaneously K51",
    "any K50 before any K51", "K50 before any K51", "any K50 after K51",
    "every K50 before K51", "K50 after every K51", "every K50 before every K51",
    # day windows (incl. lower bounds, around, self-reference, outside)
    "K50 inside 10 days after K51",
    "K50 inside 3 to 15 days after K51",
    "K50 inside 3 to 15 days before K51",
    "K50 inside 2 to 8 days around K51",
    "K50 inside -8 to -2 days around K51",
    "K50 inside -5 to 10 days around K51",
    "K50 inside 10 days after K50",
    "K50 inside 1 to 10 days after K50",
    "K50* inside 5 days after K50",
    "K50 outside 10 days after K51",
    "K50 outside 3 to 15 days after K51",
    "K50 inside 30 days",
    "K50 inside 100 days after every K51",
    "every K50 inside 100 days after K51",
    # event windows
    "K50 inside 3 events after K51",
    "K50 inside 0 to 3 events after K51",
    "K50 inside 2 events around K51",
    "K50* inside -2 to 2 events around K50",
    "K50 outside 2 events after K51",
    # spans / bounds
    "K50 inside last 3 events",
    "K50 inside 1st K51 to -1st K51",
    "K50 between 1st K51 and -1st K51",
    "K50 outside 1st K51 to -1st K51",
    # shifted anchors
    "1st K51 before 1st K50 - 10 days",
    "K50 inside 10 days after 1st K51 + 5 days",
    # comparisons / aggregates
    "dose > 50", "dose < -0.5",
    "sum(dose) > 100", "mean(dose) > 40", "count(dose) > 2",
    "range(dose) > 30", "rise(dose) > 20", "fall(dose) > 20",
    "rise(dose) > 50%", "range(dose) > 80%",
    "sum(dose) > 60 inside 20 days after K51",
    "sum(dose) < 60 inside 5 to 20 days after K51",
    "count(dose) == 0 inside 5 to 20 days after K51",
    "sum(dose) > 60 inside 20 days after every K51",
    "mean(dose) < 99999 inside 20 days after every K51",
    "sum(dose) > 60 inside 20 days",
    "range(dose) > 30 inside 3 events",
    "sum(dose) > 60 inside 0 to 3 events after K51",
    "sum(dose) > 60 outside 20 days after K51",
    # composition
    "min 2 of K50 inside 15 days after K51",
    "(K50 or K51) and K52",
    "not (K50 before K51)",
    "0-0 of K50 and K51",
]


def make_dataset(rng: np.random.Generator) -> pd.DataFrame:
    n_persons = int(rng.integers(3, 9))
    rows = []
    for pid in range(1, n_persons + 1):
        n_events = int(rng.integers(1, 11))
        days = rng.integers(0, 120, n_events)
        for d in days:
            code = CODES[int(rng.integers(0, len(CODES)))]
            dose = float(rng.integers(1, 101)) if rng.random() > 0.3 else np.nan
            rows.append((pid, pd.Timestamp("2020-01-01") + pd.Timedelta(days=int(d)), code, dose))
    df = pd.DataFrame(rows, columns=["pid", "start_date", "icd", "dose"])
    return df.sort_values(["pid", "start_date"]).reset_index(drop=True)


def run_backends(df: pd.DataFrame, expr: str) -> dict:
    import polars as pl
    kw = dict(pid="pid", date="start_date", cols="icd")
    out = {}
    out["pandas"] = set(tq.tquery(df, expr, **kw).pids)
    out["polars"] = set(tq.tquery(pl.from_pandas(df), expr, **kw).pids)
    out["duckdb"] = set(tq.tquery(df, expr, backend="duckdb", **kw).pids)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=40, help="number of random datasets")
    ap.add_argument("--seed", type=int, default=None, help="run a single specific seed")
    args = ap.parse_args()

    seeds = [args.seed] if args.seed is not None else list(range(args.iters))
    n_checks = 0
    mismatches = 0

    for seed in seeds:
        rng = np.random.default_rng(seed)
        df = make_dataset(rng)
        for expr in QUERIES:
            try:
                results = run_backends(df, expr)
            except Exception as e:  # noqa: BLE001 — a crash in any backend is a finding
                mismatches += 1
                print(f"\nCRASH  seed={seed}  query={expr!r}\n  {type(e).__name__}: {e}")
                print(df.to_csv(index=False))
                continue
            n_checks += 1
            vals = list(results.values())
            if not all(v == vals[0] for v in vals):
                mismatches += 1
                print(f"\nMISMATCH  seed={seed}  query={expr!r}")
                for k, v in results.items():
                    print(f"  {k:7s} -> {sorted(v)}")
                print(df.to_csv(index=False))

    print(f"\n{n_checks} checks across {len(seeds)} datasets, {mismatches} mismatches")
    return 1 if mismatches else 0


if __name__ == "__main__":
    sys.exit(main())
