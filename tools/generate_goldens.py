"""Generate cross-language golden fixtures from the Python reference implementation.

Outputs into `spec/golden/`:

- `dataset.csv`        — a stable synthetic dataset (seed=42).
- `queries.jsonl`      — one record per query: query string, expected AST JSON,
                         expected count, sorted list of expected pids.

Any port (R / Polars / SQL) must reproduce these. Run from the repo root:

    python tools/generate_goldens.py
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from tquery import tquery
from tquery._ast_json import to_json
from tquery._parser import parse
from tquery._testdata import make_test_data

REPO_ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = REPO_ROOT / "spec" / "golden"

# A curated list of queries exercising every AST node type and major edge case.
# Add to this list when you add a feature to the DSL.
QUERIES: list[str] = [
    # Atoms
    "K50",
    "K50*",
    "K50, K51",
    "K50-K52",
    "event",
    "5th event",
    "last 3 events",
    # Logical
    "K50 and K51",
    "K50 or K51",
    "not K50",
    "never K50",
    # Quantifiers / prefixes
    "min 2 of K50",
    "max 3 of K50",
    "exactly 1 of K50",
    "2-5 of K50",
    "1st K50",
    "-1st K50",
    "first 3 of K50",
    "last 2 of K50",
    # Temporal
    "K50 before K51",
    "K50 after K51",
    "K50 simultaneously K51",
    # Universal quantifiers
    "every K50 before K51",
    "K50 after every K51",
    "every K50 before every K51",
    # Time windows
    "K50 inside 30 days after K51",
    "K50 inside 30 days before K51",
    "K50 inside 30 days around K51",
    "K50 inside 100 days after K51",
    "K50 inside 1000 days after K51",
    "K50 inside 30 to 90 days after K51",
    "K50 inside 30 to 90 days before K51",
    "K50 inside 0 to 365 days after K51",
    "K50 inside 100 days",
    "K50 inside 365 days",
    "K50* inside 30 days after K51",
    "K50 inside 30 days after K51*",
    "K50 outside 30 days after K51",
    "K50 inside -5 to 20 days around K51",
    "K50 inside -30 to 30 days around K51",
    # Universal quantifiers INSIDE windows (exercises every_left / every_right paths)
    "K50 inside 100 days after every K51",
    "every K50 inside 100 days after K51",
    "every K50 inside 100 days after every K51",
    # Event windows
    "K50 inside 5 events after K51",
    "K50 outside 3 events after K51",
    # Span / between
    "K50 inside last 5 events",
    "K50 inside 1st K51 and -1st K51",
    # Shifted anchors
    "1st K51 before 1st K50 - 100 days",
    # Composition
    "min 3 of K50 before K51",
    "K50 inside 30 days after 1st K51",
    "(K50 or K51) and K52",
    # Aggregate expressions (v0.2)
    "sum(dose) > 300",
    "sum(dose) > 1000",
    "mean(dose) > 50",
    "min(dose) < 5",
    "max(dose) > 95",
    "count(dose_k) > 3",
    "sum(dose_k) > 200",
    "sum(dose) > 300 inside 90 days after K51",
    "sum(dose_k) > 100 inside 365 days after K50",
    "sum(dose) > 300 inside 90 days",
    "sum(dose_k) > 200 inside 365 days",
    "K50 and sum(dose) > 500",
    # v0.2.1: range + event-window aggregates
    "range(dose) > 50",
    "range(dose) > 90",
    "range(dose) > 30 inside 5 events",
    "range(dose) > 30 inside 2 events",
    "range(dose) > 30 inside 5 events after K50",
    "sum(dose) > 200 inside 5 events",
    "max(dose) > 80 inside 3 events",
    # v0.2.2: signed range — rise / fall
    "rise(dose) > 30",
    "rise(dose) > 80",
    "fall(dose) > 30",
    "fall(dose) > 80",
    "rise(dose) > 30 inside 5 events",
    "fall(dose) > 30 inside 5 events",
    "rise(dose) > 30 inside 90 days",
    "rise(dose) > 30 inside 90 days after K50",
    # v0.2.3: relative thresholds — rise/fall as percentages
    "rise(dose) > 50%",
    "rise(dose) > 200%",
    "fall(dose) > 50%",
    "fall(dose) > 90%",
    "rise(dose) > 50% inside 5 events",
    "rise(dose) > 100% inside 90 days",
    # v0.2.4: relative range
    "range(dose) > 50%",
    "range(dose) > 100%",
    "range(dose_k) > 200%",
    "range(dose) > 50% inside 5 events",
    "range(dose) > 100% inside 90 days",
]


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Stable synthetic dataset
    df = make_test_data(n_persons=500, events_per_person=10, seed=42)
    csv_path = OUT_DIR / "dataset.csv"
    df.to_csv(csv_path, index=False)
    print(f"wrote {csv_path}  ({len(df)} rows, {df['pid'].nunique()} persons)")

    # 2. Per-query fixtures
    jsonl_path = OUT_DIR / "queries.jsonl"
    n_ok = 0
    n_failed: list[tuple[str, str]] = []
    with jsonl_path.open("w") as fh:
        for q in QUERIES:
            try:
                ast = parse(q)
                ast_json = to_json(ast)
                result = tquery(
                    df, q, pid="pid", date="start_date", cols="icd"
                )
                rec = {
                    "query": q,
                    "ast": ast_json,
                    "count": int(result.count),
                    "pids": sorted(int(p) for p in result.pids),
                }
                fh.write(json.dumps(rec) + "\n")
                n_ok += 1
            except Exception as e:  # noqa: BLE001 — log and continue
                n_failed.append((q, f"{type(e).__name__}: {e}"))

    print(f"wrote {jsonl_path}  ({n_ok}/{len(QUERIES)} queries)")
    if n_failed:
        print(f"\n{len(n_failed)} queries failed to generate fixtures:")
        for q, err in n_failed:
            print(f"  - {q!r}: {err}")


if __name__ == "__main__":
    main()
