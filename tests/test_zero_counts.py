"""Explicit-0 count predicates: zero-count persons are included exactly
when the user writes a literal 0.

`0-2 of K50` — at most 2 K50s INCLUDING none; `exactly 0 of K50` ≡
`max 0 of K50` ≡ `not K50`; `min 0 of K50` is the tautology (everyone).
`max 2 of K50` (no written 0) keeps the classic reading: has K50, at
most 2, and marks only the matching rows. Zero-count persons are marked
on their full timeline (the `not` convention).
"""

import pandas as pd
import pytest

import tquery as tq

KW = dict(pid="pid", date="start_date", cols=["icd"])


@pytest.fixture
def df():
    """P1: 2 K50 | P2: 3 K50 | P3: 0 K50 (X and K51 only)."""
    out = pd.DataFrame({
        "pid": [1, 1, 2, 2, 2, 3, 3],
        "start_date": pd.to_datetime([
            "2020-01-01", "2020-02-01",
            "2020-01-01", "2020-02-01", "2020-03-01",
            "2020-01-01", "2020-05-01",
        ]),
        "icd": ["K50", "K50", "K50", "K50", "K50", "X", "K51"],
    })
    return out.sort_values(["pid", "start_date"]).reset_index(drop=True)


class TestExplicitZero:
    def test_range_with_zero_lower_bound(self, df):
        assert tq.tquery(df, "0-2 of K50", **KW).pids == {1, 3}

    def test_range_without_zero_unchanged(self, df):
        assert tq.tquery(df, "1-2 of K50", **KW).pids == {1}

    def test_max_without_zero_unchanged(self, df):
        # `max 2 of K50` still requires at least one K50.
        assert tq.tquery(df, "max 2 of K50", **KW).pids == {1}

    def test_exactly_zero_equals_not(self, df):
        assert (
            tq.tquery(df, "exactly 0 of K50", **KW).pids
            == tq.tquery(df, "not K50", **KW).pids
            == {3}
        )

    def test_max_zero_equals_not(self, df):
        assert tq.tquery(df, "max 0 of K50", **KW).pids == {3}

    def test_zero_zero_range_equals_not(self, df):
        assert tq.tquery(df, "0-0 of K50", **KW).pids == {3}

    def test_min_zero_is_tautology(self, df):
        assert tq.tquery(df, "min 0 of K50", **KW).pids == {1, 2, 3}

    def test_row_semantics(self, df):
        # Persons with matches keep only their matching rows; zero-count
        # persons are marked on their full timeline.
        rows = tq.tquery(df, "0-2 of K50", **KW).filter()
        assert rows[rows.pid == 1].icd.tolist() == ["K50", "K50"]
        assert rows[rows.pid == 3].icd.tolist() == ["X", "K51"]

    def test_composition_with_and(self, df):
        # Person-level composition works: zero-K50 persons WITH a K51.
        assert tq.tquery(df, "0-0 of K50 and K51", **KW).pids == {3}


class TestExplicitZeroBackendAgreement:
    EXPRS = [
        "0-2 of K50",
        "1-2 of K50",
        "max 2 of K50",
        "exactly 0 of K50",
        "max 0 of K50",
        "min 0 of K50",
        "0-0 of K50",
    ]

    @pytest.mark.parametrize("expr", EXPRS)
    def test_duckdb_agrees(self, df, expr):
        pytest.importorskip("duckdb")
        assert (
            tq.tquery(df, expr, backend="duckdb", **KW).pids
            == tq.tquery(df, expr, **KW).pids
        )

    @pytest.mark.parametrize("expr", EXPRS)
    def test_polars_agrees(self, df, expr):
        pl = pytest.importorskip("polars")
        assert (
            tq.tquery(pl.from_pandas(df), expr, **KW).pids
            == tq.tquery(df, expr, **KW).pids
        )
