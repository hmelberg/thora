"""Tests for parameterized multi_query with ?[...] notation."""

import pandas as pd
import pytest

from tquery import multi_query
from tquery.__init__ import _parse_slot


@pytest.fixture
def df():
    return pd.DataFrame({
        "pid": [1, 1, 1, 2, 2, 2, 3, 3, 3],
        "start_date": pd.to_datetime([
            "2020-01-01", "2020-02-01", "2020-03-01",
            "2020-01-01", "2020-01-15", "2020-03-01",
            "2020-06-01", "2020-07-01", "2020-08-01",
        ]),
        "icd": ["K50", "K51", "K52", "K51", "K50", "K52", "K50", "K50", "K51"],
    })


class TestParseSlot:
    def test_explicit_list(self):
        assert _parse_slot("0,1,2") == ["0", "1", "2"]

    def test_multi_char_list(self):
        assert _parse_slot("K50,K51,K52") == ["K50", "K51", "K52"]

    def test_numeric_range(self):
        assert _parse_slot("0-3") == ["0", "1", "2", "3"]

    def test_multi_digit_range(self):
        assert _parse_slot("50-53") == ["50", "51", "52", "53"]

    def test_letter_range(self):
        assert _parse_slot("a-d") == ["a", "b", "c", "d"]

    def test_single_value(self):
        assert _parse_slot("K50") == ["K50"]

    def test_whitespace_stripped(self):
        assert _parse_slot(" K50 , K51 ") == ["K50", "K51"]


class TestMultiQuery:
    def test_single_slot(self, df):
        result = multi_query(df, "K5?[0,1,2] before K51")
        assert isinstance(result, pd.Series)
        assert "K50 before K51" in result.index
        assert "K52 before K51" in result.index
        assert result["K50 before K51"] == 2  # P1 and P3

    def test_range_slot(self, df):
        result = multi_query(df, "K5?[0-2] before K51")
        assert len(result) == 3
        assert result["K50 before K51"] == 2

    def test_multiple_slots(self, df):
        result = multi_query(df, "K5?[0,1] before K5?[1,2]")
        assert len(result) == 4  # 2x2
        assert "K50 before K51" in result.index
        assert "K50 before K52" in result.index

    def test_non_code_slot(self, df):
        result = multi_query(df, "min ?[1,2] of K50")
        assert len(result) == 2
        assert result["min 1 of K50"] == 3
        assert result["min 2 of K50"] == 1  # only P3

    def test_no_slots(self, df):
        result = multi_query(df, "K50 before K51")
        assert len(result) == 1
        assert result["K50 before K51"] == 2

    def test_max_combinations_exceeded(self, df):
        with pytest.raises(ValueError, match="combinations"):
            multi_query(df, "K?[0-9]?[0-9] before K?[0-9]?[0-9]",
                        max_combinations=100)

    def test_invalid_query_returns_zero(self, df):
        # ZZZ doesn't exist — should return 0, not crash
        result = multi_query(df, "?[K50,ZZZ] before K51")
        assert result["ZZZ before K51"] == 0

    def test_accessor(self, df):
        result = df.tq.multi("K5?[0,1] before K51")
        assert len(result) == 2

    def test_cache_reuse(self, df):
        """Shared evaluator means common sub-expressions are cached."""
        # This should run fast because K51 is evaluated once and reused
        result = multi_query(df, "K5?[0-3] before K51")
        assert len(result) == 4

    def test_full_code_substitution(self, df):
        result = multi_query(df, "?[K50,K51] before K52")
        assert len(result) == 2
        assert "K50 before K52" in result.index
        assert "K51 before K52" in result.index
