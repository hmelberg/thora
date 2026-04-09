"""Tests for prefix (quantifier) evaluation."""

import pandas as pd
import pytest

from tquery._prefix import eval_prefix


@pytest.fixture
def df():
    """Person 1: A, A, B; Person 2: A, B, B; Person 3: A, A, A"""
    return pd.DataFrame({
        "pid": [1, 1, 1, 2, 2, 2, 3, 3, 3],
        "code": ["A", "A", "B", "A", "B", "B", "A", "A", "A"],
    })


@pytest.fixture
def mask_a(df):
    return df["code"] == "A"


class TestMinMaxExactly:
    def test_min_2(self, df, mask_a):
        result = eval_prefix("min", 2, mask_a, df["pid"])
        # Person 1: 2 A's (yes), Person 2: 1 A (no), Person 3: 3 A's (yes)
        assert list(result) == [True, True, False, False, False, False, True, True, True]

    def test_min_3(self, df, mask_a):
        result = eval_prefix("min", 3, mask_a, df["pid"])
        # Only Person 3 has 3 A's
        assert list(result) == [False, False, False, False, False, False, True, True, True]

    def test_max_1(self, df, mask_a):
        result = eval_prefix("max", 1, mask_a, df["pid"])
        # Person 2 has exactly 1 A
        assert list(result) == [False, False, False, True, False, False, False, False, False]

    def test_exactly_2(self, df, mask_a):
        result = eval_prefix("exactly", 2, mask_a, df["pid"])
        # Person 1 has exactly 2 A's
        assert list(result) == [True, True, False, False, False, False, False, False, False]


class TestOrdinal:
    def test_1st(self, df, mask_a):
        result = eval_prefix("ordinal", 1, mask_a, df["pid"])
        # 1st A per person: idx 0 (P1), idx 3 (P2), idx 6 (P3)
        assert list(result) == [True, False, False, True, False, False, True, False, False]

    def test_2nd(self, df, mask_a):
        result = eval_prefix("ordinal", 2, mask_a, df["pid"])
        # 2nd A: idx 1 (P1), none (P2), idx 7 (P3)
        assert list(result) == [False, True, False, False, False, False, False, True, False]

    def test_3rd(self, df, mask_a):
        result = eval_prefix("ordinal", 3, mask_a, df["pid"])
        # Only Person 3 has a 3rd A at idx 8
        assert list(result) == [False, False, False, False, False, False, False, False, True]


class TestFirstLast:
    def test_first_2(self, df, mask_a):
        result = eval_prefix("first", 2, mask_a, df["pid"])
        # First 2 A's per person:
        # P1: idx 0, 1; P2: idx 3 (only 1); P3: idx 6, 7
        assert list(result) == [True, True, False, True, False, False, True, True, False]

    def test_last_1(self, df, mask_a):
        result = eval_prefix("last", 1, mask_a, df["pid"])
        # Last A per person: idx 1 (P1), idx 3 (P2), idx 8 (P3)
        assert list(result) == [False, True, False, True, False, False, False, False, True]

    def test_last_2(self, df, mask_a):
        result = eval_prefix("last", 2, mask_a, df["pid"])
        # Last 2 A's per person:
        # P1: idx 0, 1 (both - only 2 total); P2: idx 3 (only 1); P3: idx 7, 8
        assert list(result) == [True, True, False, True, False, False, False, True, True]
