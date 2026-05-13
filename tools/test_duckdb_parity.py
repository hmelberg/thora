"""DuckDB backend parity test against the goldens."""
from __future__ import annotations
import json
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from tquery._evaluator_duckdb import tquery_duckdb  # noqa: E402

DPATH = REPO_ROOT / "spec" / "golden" / "dataset.csv"
QPATH = REPO_ROOT / "spec" / "golden" / "queries.jsonl"


def main() -> int:
    df = pd.read_csv(DPATH, parse_dates=["start_date"])
    print(f"dataset: {len(df)} rows, {df.pid.nunique()} persons")
    fixtures = [json.loads(l) for l in QPATH.read_text().splitlines()]
    print(f"running {len(fixtures)} fixtures\n")

    ok = 0
    failures: list[tuple[str, str | None, list[int], list[int] | None]] = []
    for f in fixtures:
        q = f["query"]
        py_pids = sorted(int(p) for p in f["pids"])
        py_count = int(f["count"])
        try:
            r = tquery_duckdb(df, q, pid="pid", date="start_date", cols="icd")
            r_pids = sorted(int(p) for p in r.pids)
            r_count = int(r.count)
            if r_count == py_count and r_pids == py_pids:
                ok += 1
                status = "OK "
            else:
                status = "BAD"
                failures.append((q, None, py_pids, r_pids))
            print(f"[{status}] {q[:60]:<60s}  D={r_count} py={py_count}")
        except Exception as e:  # noqa: BLE001
            short = f"{type(e).__name__}: {str(e)[:80]}"
            failures.append((q, short, py_pids, None))
            print(f"[ERR] {q[:60]:<60s}  {short}")

    print(f"\n{ok}/{len(fixtures)} duckdb fixtures pass")
    if failures and len(failures) <= 12:
        for q, err, py_pids, r_pids in failures[:12]:
            print(f"\nquery: {q}")
            if err is not None:
                print(f"  err: {err}")
            else:
                only_r = sorted(set(r_pids) - set(py_pids))
                only_py = sorted(set(py_pids) - set(r_pids))
                print(f"  only in duckdb  ({len(only_r)}): {only_r[:10]}")
                print(f"  only in pandas  ({len(only_py)}): {only_py[:10]}")
    return 0 if ok == len(fixtures) else 1


if __name__ == "__main__":
    sys.exit(main())
