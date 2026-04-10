"""Tests for the boolean-mask API: df.tq.mask() and TQueryResult.mask()."""

import pandas as pd
import pytest

from tquery import tquery


def test_mask_simple_code(simple_df):
    m = simple_df.tq.mask("K50")
    assert m.dtype == bool
    assert m.index.equals(simple_df.index)
    # All rows whose icd is K50
    expected = simple_df["icd"] == "K50"
    assert m.equals(expected)


def test_mask_last_2_of_code(simple_df):
    # Person 1: one K50 (last) → 1 row
    # Person 2: one K50 (last) → 1 row
    # Person 3: two K50 → both are "last 2" → 2 rows
    m = simple_df.tq.mask("last 2 of K50")
    assert m.dtype == bool
    assert m.index.equals(simple_df.index)
    assert int(m.sum()) == 4
    # Every selected row must itself be a K50 row
    assert (simple_df.loc[m, "icd"] == "K50").all()


def test_mask_ordinal(simple_df):
    # 2nd of K50 per person.
    # Person 1: only one K50 → no 2nd → 0 rows
    # Person 2: only one K50 → 0 rows
    # Person 3: two K50 → 2nd one selected → 1 row
    m = simple_df.tq.mask("2nd of K50")
    assert int(m.sum()) == 1
    selected = simple_df[m]
    assert (selected["icd"] == "K50").all()
    assert selected["pid"].tolist() == [3]


def test_mask_temporal(simple_df):
    # K50 before K51 — row-level result is some Series; we just verify
    # it is bool, aligned, and that the matching persons line up with
    # what tquery() reports.
    m = simple_df.tq.mask("K50 before K51")
    assert m.dtype == bool
    assert m.index.equals(simple_df.index)

    expected_pids = tquery(simple_df, "K50 before K51").pids
    matching_pids = set(simple_df.loc[m, "pid"].unique())
    assert matching_pids.issubset(expected_pids)


def test_mask_persons_level(simple_df):
    m = simple_df.tq.mask("K50 before K51", level="persons")
    assert m.dtype == bool
    # Indexed by pid, not df.index
    assert set(m.index) == set(simple_df["pid"].unique())
    # Person 1 (K50 then K51) and Person 3 (K50 K50 K51) match;
    # Person 2 has K51 before K50.
    assert m.loc[1] is True or bool(m.loc[1]) is True
    assert bool(m.loc[2]) is False
    assert bool(m.loc[3]) is True


def test_mask_invalid_level(simple_df):
    with pytest.raises(ValueError, match="level must be"):
        simple_df.tq.mask("K50", level="banana")


def test_result_mask_subexpression(simple_df):
    r = simple_df.tq("K50 before K51")
    sub = r.mask("K50")
    assert sub.dtype == bool
    assert sub.index.equals(simple_df.index)
    # Same answer as the standalone accessor call
    assert sub.equals(simple_df.tq.mask("K50"))


def test_result_mask_reuses_cache(simple_df):
    # After running the parent query, the K50 sub-mask should already
    # live in the evaluator's cache, so r.mask("K50") must not grow it.
    r = simple_df.tq("K50 before K51")
    cache = r._evaluator._cache
    size_before = len(cache)
    r.mask("K50")
    size_after = len(cache)
    assert size_after == size_before, (
        f"K50 should be cached from parent query, "
        f"cache grew {size_before} -> {size_after}"
    )


def test_result_mask_persons_level(simple_df):
    r = simple_df.tq("K50 before K51")
    pm = r.mask("K50", level="persons")
    assert set(pm.index) == set(simple_df["pid"].unique())
    # Every person has at least one K50 in this fixture
    assert pm.all()


def test_result_mask_invalid_level(simple_df):
    r = simple_df.tq("K50")
    with pytest.raises(ValueError, match="level must be"):
        r.mask("K50", level="oops")


def test_mask_within_days(within_df):
    # K51 within 30 days after K50: Person 1 only (19 days)
    m = within_df.tq.mask("K51 within 30 days after K50")
    assert m.dtype == bool
    assert m.index.equals(within_df.index)
    # Standalone expectation via tquery()
    assert m.equals(tquery(within_df, "K51 within 30 days after K50").rows)
