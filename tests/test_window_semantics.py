"""Window band-existence + self-exclusion semantics (all day-window forms).

These tests pin the two rules introduced by the band-existence rewrite of
`eval_within_days`:

1. **Band existence** — a child row matches iff ANY reference row falls in
   the day-band, not just the nearest one. Lower-bounded windows
   (`inside 30 to 90 days after Y`) must not miss a qualifying far ref
   because a nearer ref fails the lower bound (the old asof-nearest bug).

2. **Self-exclusion** — a row never serves as its own reference. Relevant
   only when child and ref patterns overlap (`X inside 5 days after X`):
   a lone X does not match itself, but a *different* same-date X row does.
   Anchored aggregates are the exception: the anchor row stays inside its
   own window.
"""

import pandas as pd
import pytest

import tquery as tq

KW = dict(pid="pid", date="start_date", cols=["icd"])


def make_df(rows):
    df = pd.DataFrame(rows, columns=["pid", "start_date", "icd"])
    df["start_date"] = pd.to_datetime(df["start_date"])
    return df.sort_values(["pid", "start_date"]).reset_index(drop=True)


def count(rows, expr):
    return tq.tquery(make_df(rows), expr, **KW).count


class TestBandExistence:
    """A nearer non-qualifying ref must not shadow a farther qualifying one."""

    def test_range_after_decoy_ref(self):
        # K51 at day 0 (diff 60: qualifies) and day 55 (diff 5: too close).
        rows = [
            (1, "2020-01-01", "K51"),
            (1, "2020-02-25", "K51"),
            (1, "2020-03-01", "K50"),
        ]
        assert count(rows, "K50 inside 30 to 90 days after K51") == 1

    def test_range_before_decoy_ref(self):
        rows = [
            (1, "2020-01-01", "K50"),
            (1, "2020-01-05", "K51"),   # diff 4: too close
            (1, "2020-03-01", "K51"),   # diff 60: qualifies
        ]
        assert count(rows, "K50 inside 30 to 90 days before K51") == 1

    def test_range_around_decoy_ref(self):
        rows = [
            (1, "2020-01-10", "K50"),
            (1, "2020-01-11", "K51"),   # |diff| 1: too close
            (1, "2020-01-15", "K51"),   # |diff| 5: qualifies
        ]
        assert count(rows, "K50 inside 3 to 10 days around K51") == 1

    def test_wholly_negative_around_decoy_ref(self):
        # Window [-10, -3]: K50 must be 3-10 days BEFORE a K51.
        rows = [
            (1, "2020-01-01", "K50"),
            (1, "2020-01-03", "K51"),   # diff -2: too close
            (1, "2020-01-09", "K51"),   # diff -8: qualifies
        ]
        assert count(rows, "K50 inside -10 to -3 days around K51") == 1

    def test_range_after_no_qualifying_ref(self):
        rows = [
            (1, "2020-02-25", "K51"),   # diff 5: too close — only ref
            (1, "2020-03-01", "K50"),
        ]
        assert count(rows, "K50 inside 30 to 90 days after K51") == 0

    def test_every_left_uses_all_refs(self):
        # Both K50s have SOME qualifying K51 (not necessarily the nearest).
        rows = [
            (1, "2020-01-01", "K51"),
            (1, "2020-02-25", "K51"),   # decoy for the day-60 K50
            (1, "2020-03-01", "K50"),   # 60d after first K51
            (1, "2020-04-10", "K50"),   # 45d after second K51
        ]
        assert count(rows, "every K50 inside 30 to 90 days after K51") == 1


class TestSelfExclusion:
    """A row is never its own reference; other same-date rows still count."""

    def test_lone_event_does_not_match_itself(self):
        assert count([(1, "2020-01-01", "X")], "X inside 5 days after X") == 0

    def test_second_event_within_window_matches(self):
        rows = [(1, "2020-01-01", "X"), (1, "2020-01-04", "X")]
        assert count(rows, "X inside 5 days after X") == 1

    def test_same_date_other_event_matches(self):
        rows = [(1, "2020-01-01", "X"), (1, "2020-01-01", "X")]
        assert count(rows, "X inside 5 days after X") == 1
        assert count(rows, "X inside 0 to 5 days after X") == 1

    def test_lower_bound_excludes_same_date_pair(self):
        rows = [(1, "2020-01-01", "X"), (1, "2020-01-01", "X")]
        assert count(rows, "X inside 1 to 5 days after X") == 0

    def test_lower_bound_finds_other_event(self):
        # Old asof bug: nearest ref was the row itself (diff 0), hiding
        # the genuine match 3 days back.
        rows = [(1, "2020-01-01", "X"), (1, "2020-01-04", "X")]
        assert count(rows, "X inside 1 to 5 days after X") == 1

    def test_overlapping_patterns(self):
        # The K50 row matches both `K50*` (child) and `K50` (ref): it must
        # not anchor itself, but the K50.1 row 2 days later qualifies.
        rows = [(1, "2020-01-01", "K50"), (1, "2020-01-03", "K50.1")]
        assert count(rows, "K50* inside 5 days after K50") == 1
        # A lone K50 has no OTHER K50*-row nearby.
        assert count([(1, "2020-01-01", "K50")], "K50* inside 5 days after K50") == 0

    def test_disjoint_patterns_unaffected(self):
        rows = [(1, "2020-01-01", "K51"), (1, "2020-01-15", "K50")]
        assert count(rows, "K50 inside 30 days after K51") == 1

    def test_first_event_window_keeps_self(self):
        # `inside N days` (no ref) anchors on the person's first-event
        # DATE, not a row — the first event itself is at distance 0.
        rows = [(1, "2020-01-01", "X"), (1, "2020-06-01", "Y")]
        assert count(rows, "X inside 100 days") == 1


class TestAnchoredAggregateKeepsAnchor:
    """exclude_self does NOT apply to anchored aggregate windows."""

    def test_anchor_row_value_counts(self):
        df = pd.DataFrame({
            "pid": [1],
            "start_date": pd.to_datetime(["2020-01-01"]),
            "icd": ["B01"],
            "dose": [100.0],
        })
        r = tq.tquery(df, "sum(dose) > 50 inside 90 days after B01", **KW)
        assert r.count == 1

    def test_range_window_aggregate_uses_all_refs(self):
        # dose row is 60d after the first B01; the second B01 (5d before
        # the dose row) must not shadow it.
        df = pd.DataFrame({
            "pid": [1, 1, 1],
            "start_date": pd.to_datetime(["2020-01-01", "2020-02-25", "2020-03-01"]),
            "icd": ["B01", "B01", "X"],
            "dose": [None, None, 100.0],
        })
        r = tq.tquery(df, "sum(dose) > 50 inside 30 to 90 days after B01", **KW)
        assert r.count == 1


class TestEventWindowSemantics:
    """Event-position windows: literal explicit ranges, bare-form
    shorthands, and self-exclusion (a row never matches a window anchored
    at itself — positions are unique, so offset 0 IS the anchor row)."""

    def test_parser_literal_and_shorthand_forms(self):
        from tquery._parser import parse
        a = parse("K50 inside 0 to 5 events after K51")
        assert (a.min_events, a.max_events) == (0, 5)  # literal, not rewritten
        a = parse("K50 inside 5 events after K51")
        assert (a.min_events, a.max_events) == (1, 5)  # bare shorthand
        a = parse("K50 inside 3 events around K51")
        assert (a.min_events, a.max_events) == (-3, 3)  # bare around: symmetric
        a = parse("K50 inside -3 to 5 events around K51")
        assert (a.min_events, a.max_events) == (-3, 5)

    def test_around_spanning_zero_no_self_match(self):
        # Regression: a lone X used to match itself at offset 0.
        assert count([(1, "2020-01-01", "X")], "X inside -3 to 5 events around X") == 0
        rows = [(1, "2020-01-01", "X"), (1, "2020-01-05", "X")]
        assert count(rows, "X inside -3 to 5 events around X") == 1

    def test_bare_around_is_symmetric(self):
        # K50 one position BEFORE the K51 anchor — previously the bare
        # around form was silently after-side only (0..N).
        rows = [(1, "2020-01-01", "K50"), (1, "2020-01-05", "K51")]
        assert count(rows, "K50 inside 2 events around K51") == 1

    def test_explicit_zero_after_self_excluded(self):
        # Position 0 = the anchor itself; only ANOTHER X can match.
        assert count([(1, "2020-01-01", "X")], "X inside 0 to 3 events after X") == 0
        rows = [(1, "2020-01-01", "X"), (1, "2020-01-05", "X")]
        assert count(rows, "X inside 0 to 3 events after X") == 1

    def test_overlapping_patterns_no_self_match(self):
        assert count([(1, "2020-01-01", "K50")],
                     "K50* inside -2 to 2 events around K50") == 0
        rows = [(1, "2020-01-01", "K50"), (1, "2020-01-05", "K50.1")]
        assert count(rows, "K50* inside -2 to 2 events around K50") == 1

    def test_disjoint_patterns_unchanged(self):
        rows = [(1, "2020-01-01", "K51"), (1, "2020-01-05", "K50")]
        assert count(rows, "K50 inside 5 events after K51") == 1

    def test_aggregate_keeps_anchor_row(self):
        df = make_df([(1, "2020-01-01", "B01")])
        df["dose"] = [100.0]
        assert tq.tquery(df, "sum(dose) > 50 inside 0 to 5 events after B01", **KW).count == 1
        # bare form is 1..5 — the anchor's own dose is outside the window
        assert tq.tquery(df, "sum(dose) > 50 inside 5 events after B01", **KW).count == 0

    EVENT_CASES = [
        ([(1, "2020-01-01", "X")], "X inside -3 to 5 events around X"),
        ([(1, "2020-01-01", "X"), (1, "2020-01-05", "X")], "X inside -3 to 5 events around X"),
        ([(1, "2020-01-01", "X"), (1, "2020-01-05", "X")], "X inside 0 to 3 events after X"),
        ([(1, "2020-01-01", "K50"), (1, "2020-01-05", "K51")], "K50 inside 2 events around K51"),
        ([(1, "2020-01-01", "K51"), (1, "2020-01-05", "K50")], "K50 outside 3 events after K51"),
    ]

    @pytest.mark.parametrize("rows,expr", EVENT_CASES)
    def test_duckdb_agrees(self, rows, expr):
        pytest.importorskip("duckdb")
        df = make_df(rows)
        assert (
            tq.tquery(df, expr, backend="duckdb", **KW).count
            == tq.tquery(df, expr, **KW).count
        )

    @pytest.mark.parametrize("rows,expr", EVENT_CASES)
    def test_polars_agrees(self, rows, expr):
        pl = pytest.importorskip("polars")
        df = make_df(rows)
        assert (
            tq.tquery(pl.from_pandas(df), expr, **KW).count
            == tq.tquery(df, expr, **KW).count
        )


class TestEmptyWindowAggregates:
    """Anchored aggregates over EMPTY windows follow the spec defaults:
    sum/count evaluate to 0 (and compare normally), all other aggregates
    are NA and fail. Persons without any ref stay excluded."""

    @pytest.fixture
    def adf(self):
        # P1: K51 anchor, nothing 30-60d after (empty range window).
        # P2: K51 anchor, dose 30 at day 40 (in window).
        # P3: no K51 (never evaluable).
        df = pd.DataFrame([
            (1, "2020-01-01", "K51", 1.0),
            (1, "2020-01-02", "X", 1.0),
            (2, "2020-01-01", "K51", None),
            (2, "2020-02-10", "X", 30.0),
            (3, "2020-01-01", "X", 500.0),
        ], columns=["pid", "start_date", "icd", "dose"])
        df["start_date"] = pd.to_datetime(df["start_date"])
        return df.sort_values(["pid", "start_date"]).reset_index(drop=True)

    def test_empty_window_sum_is_zero(self, adf):
        r = tq.tquery(adf, "sum(dose) < 50 inside 30 to 60 days after K51", **KW)
        assert r.pids == {1, 2}

    def test_count_zero_expressible(self, adf):
        # "Who had NO dose rows in the window?" — previously unanswerable.
        r = tq.tquery(adf, "count(dose) == 0 inside 30 to 60 days after K51", **KW)
        assert r.pids == {1}

    def test_empty_window_mean_fails(self, adf):
        r = tq.tquery(adf, "mean(dose) < 99999 inside 30 to 60 days after K51", **KW)
        assert r.pids == {2}

    def test_non_evaluable_person_excluded(self, adf):
        r = tq.tquery(adf, "sum(dose) < 50 inside 30 to 60 days after K51", **KW)
        assert 3 not in r.pids

    def test_greater_than_thresholds_unchanged(self, adf):
        r = tq.tquery(adf, "sum(dose) > 10 inside 30 to 60 days after K51", **KW)
        assert r.pids == {2}

    def test_empty_event_window(self):
        # K51 is the person's LAST event: `inside 5 events after` is empty.
        df = make_df([(1, "2020-01-01", "X"), (1, "2020-02-01", "K51")])
        df["dose"] = [10.0, None]
        r = tq.tquery(df, "sum(dose) < 5 inside 5 events after K51", **KW)
        assert r.count == 1

    def test_all_na_window_sum_is_zero(self):
        # Rows exist in the window but the column is all-NA — same as empty.
        df = make_df([(1, "2020-01-01", "K51"), (1, "2020-01-10", "X")])
        df["dose"] = [None, None]
        r = tq.tquery(df, "sum(dose) == 0 inside 30 days after K51", **KW)
        assert r.count == 1

    AGG_CASES = [
        "sum(dose) < 50 inside 30 to 60 days after K51",
        "count(dose) == 0 inside 30 to 60 days after K51",
        "mean(dose) < 99999 inside 30 to 60 days after K51",
        "sum(dose) > 10 inside 30 to 60 days after K51",
    ]

    @pytest.mark.parametrize("expr", AGG_CASES)
    def test_duckdb_agrees(self, adf, expr):
        pytest.importorskip("duckdb")
        assert (
            tq.tquery(adf, expr, backend="duckdb", **KW).pids
            == tq.tquery(adf, expr, **KW).pids
        )

    @pytest.mark.parametrize("expr", AGG_CASES)
    def test_polars_agrees(self, adf, expr):
        pl = pytest.importorskip("polars")
        assert (
            tq.tquery(pl.from_pandas(adf), expr, **KW).pids
            == tq.tquery(adf, expr, **KW).pids
        )


class TestBackendAgreement:
    """pandas, Polars and DuckDB must agree on every case above."""

    CASES = [
        ([(1, "2020-01-01", "K51"), (1, "2020-02-25", "K51"), (1, "2020-03-01", "K50")],
         "K50 inside 30 to 90 days after K51"),
        ([(1, "2020-01-01", "K50"), (1, "2020-01-03", "K51"), (1, "2020-01-09", "K51")],
         "K50 inside -10 to -3 days around K51"),
        ([(1, "2020-01-01", "X")], "X inside 5 days after X"),
        ([(1, "2020-01-01", "X"), (1, "2020-01-01", "X")], "X inside 5 days after X"),
        ([(1, "2020-01-01", "X"), (1, "2020-01-04", "X")], "X inside 1 to 5 days after X"),
        ([(1, "2020-01-01", "K51"), (1, "2020-03-15", "K50")], "K50 outside 30 days after K51"),
    ]

    @pytest.mark.parametrize("rows,expr", CASES)
    def test_duckdb_agrees(self, rows, expr):
        duckdb = pytest.importorskip("duckdb")  # noqa: F841
        df = make_df(rows)
        assert (
            tq.tquery(df, expr, backend="duckdb", **KW).count
            == tq.tquery(df, expr, **KW).count
        )

    @pytest.mark.parametrize("rows,expr", CASES)
    def test_polars_agrees(self, rows, expr):
        pl = pytest.importorskip("polars")
        df = make_df(rows)
        assert (
            tq.tquery(pl.from_pandas(df), expr, **KW).count
            == tq.tquery(df, expr, **KW).count
        )
