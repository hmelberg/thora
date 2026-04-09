"""Exhaustive cross-validation: run many expressions through both evaluators.

This is a diagnostic script, not a test suite. It prints mismatches and details.
"""

from __future__ import annotations

import pandas as pd

from tquery import tquery
from tquery._string_evaluator import cross_validate, string_query_auto


# ---------------------------------------------------------------------------
# Dataset 1: Drug prescriptions (2 persons, 3 drug types)
# ---------------------------------------------------------------------------

drug_df = pd.DataFrame({
    "pid": [1, 1, 1, 1, 2, 2, 2],
    "start_date": pd.to_datetime([
        "2020-01-01", "2020-04-01", "2020-07-01", "2020-10-01",
        "2020-01-01", "2020-04-01", "2020-07-01",
    ]),
    "atc": [
        "L04AB02", "L04AB02", "L04AB04", "L04AB02",
        "L04AB04", "L04AB02", "L04AB06",
    ],
}).sort_values(["pid", "start_date"]).reset_index(drop=True)

drug_codes = {"i": ["L04AB02"], "a": ["L04AB04"], "g": ["L04AB06"]}

# ---------------------------------------------------------------------------
# Dataset 2: ICD diagnoses (3 persons, richer temporal structure)
# ---------------------------------------------------------------------------

icd_df = pd.DataFrame({
    "pid": [1, 1, 1, 2, 2, 2, 3, 3, 3],
    "start_date": pd.to_datetime([
        "2020-01-01", "2020-02-01", "2020-03-01",
        "2020-01-01", "2020-01-15", "2020-03-01",
        "2020-06-01", "2020-07-01", "2020-08-01",
    ]),
    "icd": ["K50", "K51", "K52", "K51", "K50", "K52", "K50", "K50", "K51"],
}).sort_values(["pid", "start_date"]).reset_index(drop=True)

icd_codes = {"a": ["K50"], "b": ["K51"], "c": ["K52"]}

# ---------------------------------------------------------------------------
# Dataset 3: Larger, more complex (5 persons, varied patterns)
# ---------------------------------------------------------------------------

complex_df = pd.DataFrame({
    "pid": [
        1, 1, 1, 1, 1,
        2, 2, 2,
        3, 3, 3, 3,
        4, 4,
        5, 5, 5, 5, 5,
    ],
    "start_date": pd.to_datetime([
        "2020-01-01", "2020-02-01", "2020-03-01", "2020-06-01", "2020-09-01",
        "2020-01-01", "2020-01-01", "2020-06-01",
        "2020-01-01", "2020-04-01", "2020-04-01", "2020-07-01",
        "2020-01-01", "2020-12-01",
        "2020-01-01", "2020-02-01", "2020-03-01", "2020-04-01", "2020-05-01",
    ]),
    "code": [
        "A01", "B02", "A01", "C03", "B02",
        "A01", "B02", "A01",
        "B02", "A01", "C03", "B02",
        "C03", "C03",
        "A01", "A01", "A01", "B02", "A01",
    ],
}).sort_values(["pid", "start_date"]).reset_index(drop=True)

complex_codes = {"x": ["A01"], "y": ["B02"], "z": ["C03"]}

# ---------------------------------------------------------------------------
# Dataset 4: With wildcards in codes dict
# ---------------------------------------------------------------------------

wildcard_df = pd.DataFrame({
    "pid": [1, 1, 2, 2, 3, 3],
    "start_date": pd.to_datetime([
        "2020-01-01", "2020-06-01",
        "2020-01-01", "2020-06-01",
        "2020-01-01", "2020-06-01",
    ]),
    "icd": ["K500", "K510", "K501", "K520", "K510", "K500"],
}).sort_values(["pid", "start_date"]).reset_index(drop=True)

wildcard_codes = {"a": ["K50*"], "b": ["K51*"], "c": ["K52*"]}

# ---------------------------------------------------------------------------
# Dataset 5: Same-day events (simultaneous)
# ---------------------------------------------------------------------------

sameday_df = pd.DataFrame({
    "pid": [1, 1, 1, 2, 2, 3, 3, 3],
    "start_date": pd.to_datetime([
        "2020-01-01", "2020-01-01", "2020-06-01",
        "2020-01-01", "2020-06-01",
        "2020-03-01", "2020-03-01", "2020-03-01",
    ]),
    "code": ["A01", "B02", "C03", "A01", "B02", "A01", "B02", "C03"],
}).sort_values(["pid", "start_date"]).reset_index(drop=True)

sameday_codes = {"x": ["A01"], "y": ["B02"], "z": ["C03"]}

# ---------------------------------------------------------------------------
# Dataset 6: Single-event persons and persons with only one code type
# ---------------------------------------------------------------------------

sparse_df = pd.DataFrame({
    "pid": [1, 2, 2, 3, 4, 4, 4, 4],
    "start_date": pd.to_datetime([
        "2020-01-01",
        "2020-01-01", "2020-06-01",
        "2020-01-01",
        "2020-01-01", "2020-02-01", "2020-03-01", "2020-04-01",
    ]),
    "code": ["A01", "A01", "A01", "B02", "A01", "B02", "A01", "B02"],
}).sort_values(["pid", "start_date"]).reset_index(drop=True)

sparse_codes = {"x": ["A01"], "y": ["B02"]}


# ---------------------------------------------------------------------------
# Expression sets
# ---------------------------------------------------------------------------

def run_tests(df, codes, expressions, label, cols=None):
    """Run cross-validation for each expression. Print mismatches."""
    col_kw = {"cols": cols} if cols else {}
    results = []
    for expr in expressions:
        try:
            df_pids, str_pids, match = cross_validate(
                df, expr, codes, mode="order", **col_kw,
            )
            results.append({
                "expr": expr,
                "match": match,
                "df_pids": df_pids,
                "str_pids": str_pids,
                "error": None,
            })
        except Exception as e:
            results.append({
                "expr": expr,
                "match": None,
                "df_pids": None,
                "str_pids": None,
                "error": f"{type(e).__name__}: {e}",
            })

    passed = sum(1 for r in results if r["match"] is True)
    failed = sum(1 for r in results if r["match"] is False)
    errors = sum(1 for r in results if r["error"] is not None)
    total = len(results)

    print(f"\n{'='*70}")
    print(f"  {label}: {passed}/{total} passed, {failed} MISMATCHES, {errors} errors")
    print(f"{'='*70}")

    for r in results:
        if r["error"]:
            print(f"  ERROR  {r['expr']}")
            print(f"         {r['error']}")
        elif not r["match"]:
            print(f"  FAIL   {r['expr']}")
            print(f"         df_pids={r['df_pids']}")
            print(f"         str_pids={r['str_pids']}")
        else:
            print(f"  OK     {r['expr']}  ->  {r['df_pids']}")

    return results


# === SIMPLE CODE QUERIES ===
simple_exprs = [
    "L04AB02",
    "L04AB04",
    "L04AB06",
    "L04AB02, L04AB04",
    "L04AB02, L04AB06",
]

# === LOGICAL OPERATORS ===
logical_exprs = [
    "L04AB02 and L04AB04",
    "L04AB02 and L04AB06",
    "L04AB04 and L04AB06",
    "L04AB02 or L04AB04",
    "L04AB02 or L04AB06",
    "not L04AB02",
    "not L04AB04",
    "not L04AB06",
    "not (L04AB02 and L04AB04)",
    "not (L04AB02 or L04AB06)",
]

# === TEMPORAL OPERATORS ===
temporal_exprs = [
    "L04AB02 before L04AB04",
    "L04AB04 before L04AB02",
    "L04AB02 after L04AB04",
    "L04AB04 after L04AB02",
    "L04AB02 before L04AB06",
    "L04AB06 before L04AB02",
    "L04AB04 before L04AB06",
    "L04AB06 after L04AB04",
]

# === PREFIX / QUANTIFIER OPERATORS ===
prefix_exprs = [
    "min 1 of L04AB02",
    "min 2 of L04AB02",
    "min 3 of L04AB02",
    "min 4 of L04AB02",
    "max 1 of L04AB02",
    "max 2 of L04AB02",
    "max 3 of L04AB02",
    "exactly 1 of L04AB02",
    "exactly 2 of L04AB02",
    "exactly 3 of L04AB02",
    "exactly 1 of L04AB04",
    "exactly 1 of L04AB06",
    "1st of L04AB02",
    "2nd of L04AB02",
    "3rd of L04AB02",
    "first 1 of L04AB02",
    "first 2 of L04AB02",
    "last 1 of L04AB02",
    "last 2 of L04AB02",
    "1-2 of L04AB02",
    "2-3 of L04AB02",
    "1-3 of L04AB02",
]

# === COMPOUND EXPRESSIONS ===
compound_exprs = [
    "(L04AB02 or L04AB04) before L04AB06",
    "(L04AB02 and L04AB04) before L04AB06",
    "(min 2 of L04AB02) before L04AB04",
    "(min 2 of L04AB02) after L04AB04",
    "L04AB02 before L04AB04 and L04AB06",
    "L04AB02 before L04AB04 or L04AB06",
    "(exactly 1 of L04AB02) before L04AB04",
    "1st of L04AB02 before 1st of L04AB04",
    "1st of L04AB02 after 1st of L04AB04",
    "1st of L04AB02 before 1st of L04AB06",
    "not L04AB02 and L04AB04",
    "not (L04AB02 before L04AB04)",
    "(min 2 of L04AB02) and L04AB04",
    "(min 2 of L04AB02) and (min 1 of L04AB04)",
]

# === ICD EXPRESSIONS ===
icd_exprs = [
    "K50",
    "K51",
    "K52",
    "K50 and K51",
    "K50 and K52",
    "K51 and K52",
    "K50 or K51",
    "K50 or K52",
    "not K50",
    "not K52",
    "K50 before K51",
    "K50 after K51",
    "K51 before K50",
    "K51 after K50",
    "K50 before K52",
    "K51 before K52",
    "K52 after K50",
    "min 2 of K50",
    "min 3 of K50",
    "exactly 1 of K50",
    "exactly 2 of K50",
    "max 1 of K50",
    "K50 before K51 and K52",
    "(K50 or K51) before K52",
    "(min 2 of K50) before K51",
    "1st of K50 before 1st of K51",
    "2nd of K50 before K51",
    "not K50 and K51",
    "K50 and K51 and K52",
    "K50 or K51 or K52",
    "not (K50 before K51)",
    "(K50 and K51) before K52",
    "1st of K50 before 1st of K52",
]

# === COMPLEX DATASET EXPRESSIONS ===
complex_exprs = [
    "A01",
    "B02",
    "C03",
    "A01 and B02",
    "A01 and C03",
    "B02 and C03",
    "A01 or B02 or C03",
    "not A01",
    "not C03",
    "A01 before B02",
    "B02 before A01",
    "A01 after B02",
    "A01 before C03",
    "C03 before A01",
    "B02 before C03",
    "min 2 of A01",
    "min 3 of A01",
    "min 4 of A01",
    "exactly 1 of B02",
    "exactly 2 of B02",
    "max 1 of C03",
    "(min 2 of A01) before B02",
    "(min 2 of A01) before C03",
    "1st of A01 before 1st of B02",
    "1st of A01 before 1st of C03",
    "(A01 or C03) before B02",
    "A01 before B02 and C03",
    "not A01 and B02",
    "not (A01 before B02)",
    "A01 and B02 and C03",
    "2nd of A01 before 1st of B02",
    "last 1 of A01 before last 1 of B02",
]

# === WILDCARD EXPRESSIONS ===
wildcard_exprs = [
    "K50*",
    "K51*",
    "K52*",
    "K50* and K51*",
    "K50* or K51*",
    "not K50*",
    "K50* before K51*",
    "K51* before K50*",
    "K50* after K51*",
    "min 2 of K50*",
    "(K50* or K51*) before K52*",
    "1st of K50* before 1st of K51*",
]

# === SAME-DAY EVENTS ===
sameday_exprs = [
    "A01",
    "B02",
    "C03",
    "A01 and B02",
    "A01 before B02",
    "B02 before A01",
    "A01 after B02",
    "A01 before C03",
    "min 2 of A01",
    "(A01 or B02) before C03",
    "A01 and B02 and C03",
    "not C03",
    "1st of A01 before 1st of B02",
]

# === SPARSE/EDGE CASES ===
sparse_exprs = [
    "A01",
    "B02",
    "A01 and B02",
    "A01 or B02",
    "not A01",
    "not B02",
    "A01 before B02",
    "B02 before A01",
    "min 2 of A01",
    "min 3 of A01",
    "min 4 of A01",
    "exactly 1 of A01",
    "exactly 2 of A01",
    "max 1 of A01",
    "(min 2 of A01) before B02",
    "1st of A01 before 1st of B02",
    "not (A01 before B02)",
    "A01 before B02 and not (min 3 of A01)",
]


# ---------------------------------------------------------------------------
# RUN ALL
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    all_results = []

    all_results.extend(run_tests(
        drug_df, drug_codes,
        simple_exprs + logical_exprs + temporal_exprs + prefix_exprs + compound_exprs,
        "Drug dataset", cols="atc",
    ))

    all_results.extend(run_tests(
        icd_df, icd_codes, icd_exprs,
        "ICD dataset", cols="icd",
    ))

    all_results.extend(run_tests(
        complex_df, complex_codes, complex_exprs,
        "Complex dataset (5 persons)", cols="code",
    ))

    all_results.extend(run_tests(
        wildcard_df, wildcard_codes, wildcard_exprs,
        "Wildcard codes dataset", cols="icd",
    ))

    all_results.extend(run_tests(
        sameday_df, sameday_codes, sameday_exprs,
        "Same-day events", cols="code",
    ))

    all_results.extend(run_tests(
        sparse_df, sparse_codes, sparse_exprs,
        "Sparse/edge-case dataset", cols="code",
    ))

    # Summary
    total = len(all_results)
    passed = sum(1 for r in all_results if r["match"] is True)
    failed = sum(1 for r in all_results if r["match"] is False)
    errors = sum(1 for r in all_results if r["error"] is not None)

    print(f"\n{'='*70}")
    print(f"  GRAND TOTAL: {passed}/{total} passed, {failed} MISMATCHES, {errors} errors")
    print(f"{'='*70}")

    if failed > 0:
        print("\n  MISMATCHES:")
        for r in all_results:
            if r["match"] is False:
                print(f"    {r['expr']}")
                print(f"      df={r['df_pids']}  str={r['str_pids']}")

    if errors > 0:
        print("\n  ERRORS:")
        for r in all_results:
            if r["error"]:
                print(f"    {r['expr']}")
                print(f"      {r['error']}")
