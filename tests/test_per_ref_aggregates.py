"""Universal-ref anchored aggregates: `AGG(col) OP x inside ... every REF`.

Semantics (spec/semantics.md): for EACH reference event, the aggregate is
computed over that ref's own day-window and tested against the threshold;
a person matches iff they have at least one ref (no vacuous truth) and
EVERY ref's window passes. Empty windows follow the standard aggregate
defaults — sum/count evaluate to 0 (and compare normally), everything
else is NA and fails. The anchor row is part of its own window.
"""

import pandas as pd
import pytest

import tquery as tq

KW = dict(pid="pid", date="start_date", cols=["icd"])


@pytest.fixture
def df():
    """P1: K51 anchor with dose 100 in-window, second K51 with empty window.
    P2: two K51 anchors, each followed by a dose (70, 80) within 30 days.
    P3: no K51 at all (must never match an every-K51 query)."""
    out = pd.DataFrame([
        (1, "2020-01-01", "K51", None),
        (1, "2020-01-05", "X", 100.0),
        (1, "2020-06-01", "K51", None),
        (2, "2020-01-01", "K51", None),
        (2, "2020-01-05", "X", 70.0),
        (2, "2020-06-01", "K51", None),
        (2, "2020-06-10", "X", 80.0),
        (3, "2020-01-01", "X", 500.0),
    ], columns=["pid", "start_date", "icd", "dose"])
    out["start_date"] = pd.to_datetime(out["start_date"])
    return out.sort_values(["pid", "start_date"]).reset_index(drop=True)


class TestPerRefSemantics:
    def test_every_ref_requires_all_windows_pass(self, df):
        r = tq.tquery(df, "sum(dose) > 50 inside 30 days after every K51", **KW)
        assert r.pids == {2}  # P1's second window is empty: sum 0 fails

    def test_existential_form_differs(self, df):
        r = tq.tquery(df, "sum(dose) > 50 inside 30 days after K51", **KW)
        assert r.pids == {1, 2}

    def test_empty_window_sum_is_zero(self, df):
        # 0 >= 0 passes, so P1's empty window is fine here.
        r = tq.tquery(df, "sum(dose) >= 0 inside 30 days after every K51", **KW)
        assert r.pids == {1, 2}

    def test_empty_window_mean_is_na_and_fails(self, df):
        # Even a trivially-true threshold fails on an NA aggregate.
        r = tq.tquery(df, "mean(dose) < 99999 inside 30 days after every K51", **KW)
        assert r.pids == {2}

    def test_count_per_window(self, df):
        r = tq.tquery(df, "count(dose) >= 1 inside 30 days after every K51", **KW)
        assert r.pids == {2}

    def test_no_refs_never_match(self, df):
        # P3 has dose 500 but no K51 → excluded (no vacuous truth).
        r = tq.tquery(df, "sum(dose) >= 0 inside 30 days after every K51", **KW)
        assert 3 not in r.pids

    def test_anchor_row_in_own_window(self):
        df = pd.DataFrame({
            "pid": [1],
            "start_date": pd.to_datetime(["2020-01-01"]),
            "icd": ["B01"],
            "dose": [100.0],
        })
        r = tq.tquery(df, "sum(dose) > 50 inside 90 days after every B01", **KW)
        assert r.count == 1

    def test_around_window_chronological_for_rise(self):
        # Values 10 (5d before ref) then 50 (5d after ref): rise across the
        # around-window is 40 only if slices concatenate chronologically.
        df = pd.DataFrame([
            (1, "2020-01-05", "X", 10.0),
            (1, "2020-01-10", "K51", None),
            (1, "2020-01-15", "X", 50.0),
        ], columns=["pid", "start_date", "icd", "dose"])
        df["start_date"] = pd.to_datetime(df["start_date"])
        r = tq.tquery(df, "rise(dose) > 30 inside 2 to 20 days around every K51", **KW)
        assert r.count == 1

    def test_outside_every_rejected(self, df):
        with pytest.raises(TypeError, match="outside"):
            tq.tquery(df, "sum(dose) > 50 outside 30 days after every K51", **KW)

    def test_event_window_every_rejected(self, df):
        with pytest.raises(TypeError, match="event-window"):
            tq.tquery(df, "range(dose) > 3 inside 5 events after every K51", **KW)


class TestPerRefBackendAgreement:
    EXPRS = [
        "sum(dose) > 50 inside 30 days after every K51",
        "sum(dose) >= 0 inside 30 days after every K51",
        "count(dose) >= 1 inside 30 days after every K51",
        "mean(dose) < 99999 inside 30 days after every K51",
        "max(dose) > 60 inside 30 days after every K51",
        "range(dose) >= 0 inside 90 days around every K51",
        "rise(dose) >= 0 inside 30 days after every K51",
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
