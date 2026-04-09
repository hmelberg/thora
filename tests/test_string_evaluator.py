"""Unit tests for the string-based query evaluator."""

import pandas as pd
import pytest

from tquery._string_evaluator import (
    StringEvaluator,
    StringMatch,
    _build_reverse_map,
    _resolve_labels,
    string_query,
    string_query_auto,
)
from tquery._types import TQueryStringError


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def codes():
    return {"i": ["L04AB02"], "a": ["L04AB04"], "g": ["L04AB06"]}


@pytest.fixture
def order_strings():
    """Stringify_order output for 3 persons.

    P1: iiai  (i at 0,1,3; a at 2)
    P2: aig   (a at 0; i at 1; g at 2)
    P3: ii    (i at 0,1)
    """
    return pd.Series({"P1": "iiai", "P2": "aig", "P3": "ii"})


@pytest.fixture
def time_strings():
    """Unmerged stringify_time output (DataFrame with label columns).

    Step=90 days. P1 spans 4 slots, P2 spans 3 slots.
    """
    return pd.DataFrame({
        "i": {"P1": "i i ", "P2": " i  "},
        "a": {"P1": "  a ", "P2": "a   "},
        "g": {"P1": "    ", "P2": "  g "},
    })


@pytest.fixture
def drug_df():
    """Same drug_df as test_stringify.py for auto-stringify tests."""
    return pd.DataFrame({
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


# ---------------------------------------------------------------------------
# Reverse mapping tests
# ---------------------------------------------------------------------------

class TestReverseMapping:
    def test_exact_codes(self, codes):
        reverse, wildcards = _build_reverse_map(codes)
        assert reverse["L04AB02"] == "i"
        assert reverse["L04AB04"] == "a"
        assert reverse["L04AB06"] == "g"
        assert wildcards == []

    def test_wildcard_codes(self):
        codes = {"bio": ["L04AB*"]}
        reverse, wildcards = _build_reverse_map(codes)
        # Without all_codes, wildcard stays as pattern
        assert len(wildcards) == 1
        assert wildcards[0] == ("L04AB", "bio")

    def test_wildcard_with_all_codes(self):
        codes = {"bio": ["L04AB*"]}
        all_codes = ["L04AB02", "L04AB04", "L04AB06", "A01"]
        reverse, wildcards = _build_reverse_map(codes, all_codes=all_codes)
        assert reverse["L04AB02"] == "bio"
        assert reverse["L04AB04"] == "bio"
        assert reverse["L04AB06"] == "bio"
        assert "A01" not in reverse


class TestResolveLabels:
    def test_exact_match(self, codes):
        reverse, wp = _build_reverse_map(codes)
        labels = _resolve_labels(("L04AB02",), reverse, wp)
        assert labels == {"i"}

    def test_multiple_codes(self, codes):
        reverse, wp = _build_reverse_map(codes)
        labels = _resolve_labels(("L04AB02", "L04AB04"), reverse, wp)
        assert labels == {"i", "a"}

    def test_wildcard_in_expression(self, codes):
        reverse, wp = _build_reverse_map(codes)
        labels = _resolve_labels(("L04AB*",), reverse, wp)
        # All three codes start with L04AB
        assert labels == {"i", "a", "g"}

    def test_variable_resolution(self, codes):
        reverse, wp = _build_reverse_map(codes)
        labels = _resolve_labels(
            ("@drugs",), reverse, wp,
            variables={"drugs": ["L04AB02", "L04AB04"]},
        )
        assert labels == {"i", "a"}


# ---------------------------------------------------------------------------
# StringMatch tests
# ---------------------------------------------------------------------------

class TestStringMatch:
    def test_pids_with_positions(self):
        sm = StringMatch({"P1": frozenset({0, 1}), "P2": frozenset({2})})
        assert sm.pids == {"P1", "P2"}

    def test_pids_empty_positions(self):
        sm = StringMatch({"P1": frozenset(), "P2": frozenset({2})})
        assert sm.pids == {"P2"}

    def test_pids_all_empty(self):
        sm = StringMatch({"P1": frozenset()})
        assert sm.pids == set()


# ---------------------------------------------------------------------------
# CodeAtom evaluation
# ---------------------------------------------------------------------------

class TestCodeEval:
    def test_single_code(self, order_strings, codes):
        ev = StringEvaluator(order_strings, codes)
        from tquery._ast import CodeAtom
        result = ev.evaluate(CodeAtom(codes=("L04AB02",)))
        assert result.pids == {"P1", "P2", "P3"}

    def test_code_not_present(self, order_strings, codes):
        ev = StringEvaluator(order_strings, codes)
        from tquery._ast import CodeAtom
        result = ev.evaluate(CodeAtom(codes=("L04AB06",)))
        # Only P2 has 'g'
        assert result.pids == {"P2"}

    def test_positions_correct(self, order_strings, codes):
        ev = StringEvaluator(order_strings, codes)
        from tquery._ast import CodeAtom
        result = ev.evaluate(CodeAtom(codes=("L04AB02",)))
        assert result.positions["P1"] == frozenset({0, 1, 3})
        assert result.positions["P2"] == frozenset({1})
        assert result.positions["P3"] == frozenset({0, 1})


# ---------------------------------------------------------------------------
# Prefix evaluation
# ---------------------------------------------------------------------------

class TestPrefixEval:
    def test_min(self, order_strings, codes):
        result = string_query("min 2 of L04AB02", order_strings, codes)
        # P1: 3 i's, P2: 1 i, P3: 2 i's
        assert result == {"P1", "P3"}

    def test_max(self, order_strings, codes):
        result = string_query("max 1 of L04AB02", order_strings, codes)
        assert result == {"P2"}

    def test_exactly(self, order_strings, codes):
        result = string_query("exactly 2 of L04AB02", order_strings, codes)
        assert result == {"P3"}

    def test_ordinal(self, order_strings, codes):
        # 1st of L04AB02 — returns only the first position
        ev = StringEvaluator(order_strings, codes)
        from tquery._ast import CodeAtom, PrefixExpr
        result = ev.evaluate(
            PrefixExpr(kind="ordinal", n=1, child=CodeAtom(codes=("L04AB02",)))
        )
        assert result.positions["P1"] == frozenset({0})
        assert result.positions["P2"] == frozenset({1})

    def test_first_n(self, order_strings, codes):
        ev = StringEvaluator(order_strings, codes)
        from tquery._ast import CodeAtom, PrefixExpr
        result = ev.evaluate(
            PrefixExpr(kind="first", n=2, child=CodeAtom(codes=("L04AB02",)))
        )
        assert result.positions["P1"] == frozenset({0, 1})
        assert result.positions["P3"] == frozenset({0, 1})

    def test_last_n(self, order_strings, codes):
        ev = StringEvaluator(order_strings, codes)
        from tquery._ast import CodeAtom, PrefixExpr
        result = ev.evaluate(
            PrefixExpr(kind="last", n=1, child=CodeAtom(codes=("L04AB02",)))
        )
        assert result.positions["P1"] == frozenset({3})


class TestRangePrefixEval:
    def test_range(self, order_strings, codes):
        # 2-3 of L04AB02
        ev = StringEvaluator(order_strings, codes)
        from tquery._ast import CodeAtom, RangePrefixExpr
        result = ev.evaluate(
            RangePrefixExpr(min_n=2, max_n=3, child=CodeAtom(codes=("L04AB02",)))
        )
        assert result.pids == {"P1", "P3"}


# ---------------------------------------------------------------------------
# Temporal evaluation
# ---------------------------------------------------------------------------

class TestTemporalEval:
    def test_before(self, order_strings, codes):
        result = string_query("L04AB02 before L04AB04", order_strings, codes)
        # P1: first i at 0, first a at 2 → before ✓
        # P2: first i at 1, first a at 0 → not before
        # P3: no 'a' → not matched
        assert result == {"P1"}

    def test_after(self, order_strings, codes):
        result = string_query("L04AB02 after L04AB04", order_strings, codes)
        # P2: first i at 1, first a at 0 → after ✓
        assert result == {"P2"}

    def test_before_no_common(self, order_strings, codes):
        # L04AB06 (g) before L04AB04 (a)
        result = string_query("L04AB06 before L04AB04", order_strings, codes)
        # Only P2 has both: g at 2, a at 0 → g is after a, not before
        assert result == set()

    def test_simultaneously_order_raises(self, order_strings, codes):
        with pytest.raises(TQueryStringError, match="simultaneously"):
            string_query("L04AB02 simultaneously L04AB04", order_strings, codes)

    def test_simultaneously_time_mode(self, time_strings, codes):
        # P1: i at {0,2}, a at {2} → overlap at position 2
        # P2: i at {1}, a at {0} → no overlap
        result = string_query(
            "L04AB02 simultaneously L04AB04",
            time_strings, codes, mode="time",
        )
        assert result == {"P1"}


# ---------------------------------------------------------------------------
# Logical operators
# ---------------------------------------------------------------------------

class TestLogicalEval:
    def test_and(self, order_strings, codes):
        result = string_query("L04AB02 and L04AB04", order_strings, codes)
        # P1 has both i and a, P2 has both
        assert result == {"P1", "P2"}

    def test_or(self, order_strings, codes):
        result = string_query("L04AB02 or L04AB06", order_strings, codes)
        assert result == {"P1", "P2", "P3"}

    def test_not(self, order_strings, codes):
        result = string_query("not L04AB06", order_strings, codes)
        # P2 has g, P1 and P3 don't
        assert result == {"P1", "P3"}


# ---------------------------------------------------------------------------
# WithinExpr
# ---------------------------------------------------------------------------

class TestWithinEval:
    def test_within_order_raises(self, order_strings, codes):
        with pytest.raises(TQueryStringError, match="within"):
            string_query(
                "L04AB02 within 30 days after L04AB04",
                order_strings, codes,
            )

    def test_within_time_mode(self, time_strings, codes):
        # step=90, within 90 days after → max 1 position distance
        # P1: i at {0,2}, a at {2}. i within 1 pos after a: pos 3 (doesn't exist)
        # P2: i at {1}, a at {0}. i within 1 pos after a: pos 1 is 1 away → YES
        result = string_query(
            "L04AB02 within 90 days after L04AB04",
            time_strings, codes, mode="time", step=90,
        )
        assert result == {"P2"}


# ---------------------------------------------------------------------------
# InsideExpr
# ---------------------------------------------------------------------------

class TestInsideEval:
    def test_inside_order(self, order_strings, codes):
        # inside 2 events after L04AB04 — for order mode
        # P1: a at pos 2. inside 2 events after = pos 3,4. i at pos 3 → YES
        # P2: a at pos 0. inside 2 events after = pos 1,2. i at pos 1 → YES
        result = string_query(
            "L04AB02 inside 2 events after L04AB04",
            order_strings, codes,
        )
        assert result == {"P1", "P2"}

    def test_inside_time_raises(self, time_strings, codes):
        with pytest.raises(TQueryStringError, match="inside"):
            string_query(
                "L04AB02 inside 2 events after L04AB04",
                time_strings, codes, mode="time",
            )


# ---------------------------------------------------------------------------
# ComparisonAtom
# ---------------------------------------------------------------------------

class TestComparisonEval:
    def test_comparison_raises(self, order_strings, codes):
        with pytest.raises(TQueryStringError, match="Column comparison"):
            string_query("glucose > 8", order_strings, codes)


# ---------------------------------------------------------------------------
# Compound expressions
# ---------------------------------------------------------------------------

class TestCompoundEval:
    def test_prefix_before(self, order_strings, codes):
        result = string_query(
            "(min 2 of L04AB02) before L04AB04",
            order_strings, codes,
        )
        # P1: min 2 of i → yes (3 i's), first i at 0 before first a at 2 → YES
        # P2: min 2 of i → no (1 i)
        # P3: no a → no
        assert result == {"P1"}

    def test_and_before(self, order_strings, codes):
        result = string_query(
            "L04AB02 before L04AB04 and L04AB06",
            order_strings, codes,
        )
        # (L04AB02 before L04AB04) and L04AB06
        # L04AB02 before L04AB04: {P1}
        # L04AB06: {P2}
        # Intersection: empty
        assert result == set()

    def test_or_before(self, order_strings, codes):
        result = string_query(
            "(L04AB02 or L04AB06) before L04AB04",
            order_strings, codes,
        )
        # Combined: P1 has i or g → first at 0; a at 2 → before ✓
        # P2: i or g → first at 1; a at 0 → not before
        assert result == {"P1"}


# ---------------------------------------------------------------------------
# Auto-stringify (string_query_auto)
# ---------------------------------------------------------------------------

class TestStringQueryAuto:
    def test_auto_order(self, drug_df, codes):
        result = string_query_auto(
            drug_df, "L04AB02", codes, mode="order", cols="atc",
        )
        assert result == {1, 2}

    def test_auto_before(self, drug_df, codes):
        result = string_query_auto(
            drug_df, "L04AB02 before L04AB04", codes,
            mode="order", cols="atc",
        )
        # P1: iiai → i before a ✓
        # P2: aig → i after a
        assert result == {1}


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_empty_strings(self, codes):
        strings = pd.Series(dtype=str)
        result = string_query("L04AB02", strings, codes)
        assert result == set()

    def test_single_char_string(self, codes):
        strings = pd.Series({"P1": "i"})
        result = string_query("L04AB02", strings, codes)
        assert result == {"P1"}

    def test_no_matching_label(self, codes):
        strings = pd.Series({"P1": "xyz"})
        result = string_query("L04AB02", strings, codes)
        assert result == set()

    def test_time_mode_requires_dataframe(self, codes):
        strings = pd.Series({"P1": "i i "})
        with pytest.raises(TQueryStringError, match="merge=False"):
            string_query("L04AB02", strings, codes, mode="time")
