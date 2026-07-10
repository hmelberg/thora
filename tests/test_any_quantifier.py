"""Existential `any` quantifier on temporal comparison sides.

`any X before any Y` means "some X before some Y" (min(X) < max(Y));
mirrored for `after` (max(X) > min(Y)). The bare default stays
first-vs-first. One-sided `any` on the side where the default already
acts existentially is a documented no-op (`any X before Y`,
`X after any Y`). In window contexts and standalone positions `any`
remains elided (windows are already existential over their events).
"""

import pandas as pd
import pytest

import tquery as tq
from tquery._ast import Quantifier
from tquery._parser import parse

KW = dict(pid="pid", date="start_date", cols=["icd"])


@pytest.fixture
def df():
    """K51 (2010) → K50 (2012) → K51 (2015): a K50 precedes a K51, but the
    FIRST K50 is after the FIRST K51 — separates existential from default."""
    out = pd.DataFrame({
        "pid": [1, 1, 1],
        "start_date": pd.to_datetime(["2010-01-01", "2012-01-01", "2015-01-01"]),
        "icd": ["K51", "K50", "K51"],
    })
    return out.sort_values(["pid", "start_date"]).reset_index(drop=True)


class TestAnySemantics:
    def test_default_is_first_vs_first(self, df):
        assert tq.tquery(df, "K50 before K51", **KW).count == 0

    def test_any_any_is_existential(self, df):
        assert tq.tquery(df, "any K50 before any K51", **KW).count == 1
        assert tq.tquery(df, "any K51 after any K50", **KW).count == 1

    def test_one_sided_any_meaningful_side(self, df):
        # before: `any` matters on the RIGHT (first X before SOME Y)
        assert tq.tquery(df, "K50 before any K51", **KW).count == 1
        # after: `any` matters on the LEFT (SOME X after first Y)
        assert tq.tquery(df, "any K51 after K50", **KW).count == 1

    def test_one_sided_any_noop_side(self, df):
        # ∃x < first(Y) ⇔ first(X) < first(Y): same as the default
        assert tq.tquery(df, "any K50 before K51", **KW).count == 0
        assert tq.tquery(df, "K51 after any K50", **KW).count == 0

    def test_every_any_synonym_of_mixed_every(self, df):
        # `every X before Y` already means "every X before SOME Y";
        # spelling the `any` must not change it.
        a = tq.tquery(df, "every K50 before K51", **KW).count
        b = tq.tquery(df, "every K50 before any K51", **KW).count
        assert a == b == 1

    def test_simultaneously_any_noop(self):
        d = pd.DataFrame({
            "pid": [1, 1],
            "start_date": pd.to_datetime(["2020-01-01", "2020-01-01"]),
            "icd": ["K50", "K51"],
        })
        assert (
            tq.tquery(d, "any K50 simultaneously any K51", **KW).count
            == tq.tquery(d, "K50 simultaneously K51", **KW).count
            == 1
        )


class TestAnyParsing:
    def test_wrapped_on_temporal_sides(self):
        a = parse("any K50 before any K51")
        assert isinstance(a.left, Quantifier) and a.left.kind == "any"
        assert isinstance(a.right, Quantifier) and a.right.kind == "any"

    def test_standalone_any_elided(self):
        assert not isinstance(parse("any K50"), Quantifier)

    def test_window_any_elided(self):
        a = parse("K50 inside 30 days after any K51")
        assert not isinstance(a.ref, Quantifier)
        a = parse("any K50 inside 30 days after K51")
        assert not isinstance(a.child, Quantifier)


class TestAnyBackendAgreement:
    EXPRS = [
        "any K50 before any K51",
        "K50 before any K51",
        "any K50 before K51",
        "any K51 after any K50",
        "any K51 after K50",
        "every K50 before any K51",
    ]

    @pytest.mark.parametrize("expr", EXPRS)
    def test_duckdb_agrees(self, df, expr):
        pytest.importorskip("duckdb")
        assert (
            tq.tquery(df, expr, backend="duckdb", **KW).count
            == tq.tquery(df, expr, **KW).count
        )

    @pytest.mark.parametrize("expr", EXPRS)
    def test_polars_agrees(self, df, expr):
        pl = pytest.importorskip("polars")
        assert (
            tq.tquery(pl.from_pandas(df), expr, **KW).count
            == tq.tquery(df, expr, **KW).count
        )
