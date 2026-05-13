"""Polars backend parity test: every golden fixture's pids+count from
the polars evaluator must match the recorded Python (pandas reference)
output exactly."""
from __future__ import annotations
import json
import sys
from pathlib import Path

import polars as pl

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from tquery._evaluator_polars import tquery_polars  # noqa: E402

DPATH = REPO_ROOT / "spec" / "golden" / "dataset.csv"
QPATH = REPO_ROOT / "spec" / "golden" / "queries.jsonl"


def main() -> int:
    df = pl.read_csv(DPATH, try_parse_dates=True)
    print(f"dataset: {df.height} rows, {df.get_column('pid').n_unique()} persons")
    fixtures = [json.loads(l) for l in QPATH.read_text().splitlines()]
    print(f"running {len(fixtures)} fixtures\n")

    ok = 0
    failures = []
    for f in fixtures:
        q = f["query"]
        py_pids = sorted(int(p) for p in f["pids"])
        py_count = int(f["count"])
        try:
            r = tquery_polars(df, q, pid="pid", date="start_date", cols="icd")
            r_pids = sorted(int(p) for p in r.pids)
            r_count = int(r.count)
            if r_count == py_count and r_pids == py_pids:
                ok += 1
                status = "OK "
            else:
                status = "BAD"
                failures.append((q, py_pids, r_pids))
            print(f"[{status}] {q[:60]:<60s}  R={r_count} py={py_count}")
        except Exception as e:  # noqa: BLE001
            status = "ERR"
            failures.append((q, f"{type(e).__name__}: {e}", None))
            print(f"[ERR] {q[:60]:<60s}  {type(e).__name__}: {e}")

    print(f"\n{ok}/{len(fixtures)} polars fixtures pass")
    if failures and len(failures) <= 10:
        for q, py_pids, r_pids in failures[:10]:
            print(f"\nquery: {q}")
            if r_pids is None:
                print(f"  err: {py_pids}")
            else:
                only_r = sorted(set(r_pids) - set(py_pids))
                only_py = sorted(set(py_pids) - set(r_pids))
                print(f"  only in polars  ({len(only_r)}): {only_r[:10]}")
                print(f"  only in pandas  ({len(only_py)}): {only_py[:10]}")
    return 0 if ok == len(fixtures) else 1


if __name__ == "__main__":
    sys.exit(main())
