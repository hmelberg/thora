"""Tests for code expansion and row matching."""

import numpy as np
import pandas as pd
import pytest

from tquery._codes import (
    collect_unique_codes,
    expand_all_codes,
    expand_codes,
    get_matching_rows,
)
from tquery._types import TQueryCodeError


class TestExpandCodes:
    def test_plain_code(self):
        assert expand_codes("K50") == ["K50"]

    def test_wildcard_with_code_list(self):
        all_codes = ["K50", "K50.1", "K50.2", "K51", "K52"]
        result = expand_codes("K50*", all_codes=all_codes)
        assert result == ["K50", "K50.1", "K50.2"]

    def test_wildcard_without_code_list(self):
        result = expand_codes("K50*")
        assert result == ["K50*"]  # returned as-is for startswith matching

    def test_wildcard_no_match(self):
        with pytest.raises(TQueryCodeError, match="matched no codes"):
            expand_codes("ZZZ*", all_codes=["K50", "K51"])

    def test_range(self):
        all_codes = ["K50", "K51", "K52", "K53", "K54"]
        result = expand_codes("K50-K53", all_codes=all_codes)
        assert result == ["K50", "K51", "K52", "K53"]

    def test_range_no_match(self):
        with pytest.raises(TQueryCodeError, match="matched no codes"):
            expand_codes("Z90-Z99", all_codes=["K50", "K51"])

    def test_variable_reference(self):
        variables = {"antibiotics": ["J01", "J02", "J03"]}
        result = expand_codes("@antibiotics", variables=variables)
        assert result == ["J01", "J02", "J03"]

    def test_variable_string(self):
        variables = {"code": "K50"}
        result = expand_codes("@code", variables=variables)
        assert result == ["K50"]

    def test_variable_not_found(self):
        with pytest.raises(TQueryCodeError, match="not found"):
            expand_codes("@missing", variables={})

    def test_expand_all_deduplicates(self):
        all_codes = ["K50", "K50.1", "K51"]
        result = expand_all_codes(("K50", "K50*"), all_codes=all_codes)
        assert result == ["K50", "K50.1"]  # K50 not duplicated


class TestGetMatchingRows:
    @pytest.fixture
    def df(self):
        return pd.DataFrame({
            "pid": [1, 1, 2, 2, 3],
            "icd": ["K50", "K51", "K50.1", "K52", "K53"],
            "atc": ["A01", "A02", "A01", "A03", "A04"],
        })

    def test_exact_match_single_col(self, df):
        mask = get_matching_rows(df, ["K50"], ["icd"])
        assert list(mask) == [True, False, False, False, False]

    def test_exact_match_multiple_codes(self, df):
        mask = get_matching_rows(df, ["K50", "K51"], ["icd"])
        assert list(mask) == [True, True, False, False, False]

    def test_wildcard_match(self, df):
        mask = get_matching_rows(df, ["K50*"], ["icd"])
        assert list(mask) == [True, False, True, False, False]

    def test_multiple_columns(self, df):
        mask = get_matching_rows(df, ["K50", "A02"], ["icd", "atc"])
        assert list(mask) == [True, True, False, False, False]

    def test_missing_column_ignored(self, df):
        mask = get_matching_rows(df, ["K50"], ["icd", "nonexistent"])
        assert list(mask) == [True, False, False, False, False]

    def test_no_match(self, df):
        mask = get_matching_rows(df, ["ZZZ"], ["icd"])
        assert not mask.any()

    def test_separator_multivalue(self):
        df = pd.DataFrame({
            "pid": [1, 2, 3],
            "codes": ["K50,K51", "K52", "K50,K53"],
        })
        mask = get_matching_rows(df, ["K50"], ["codes"], sep=",")
        assert list(mask) == [True, False, True]


class TestCollectUniqueCodes:
    def test_basic(self):
        df = pd.DataFrame({"icd": ["K50", "K51", "K50", "K52"]})
        result = collect_unique_codes(df, ["icd"])
        assert result == ["K50", "K51", "K52"]

    def test_with_separator(self):
        df = pd.DataFrame({"codes": ["K50,K51", "K52"]})
        result = collect_unique_codes(df, ["codes"], sep=",")
        assert result == ["K50", "K51", "K52"]
