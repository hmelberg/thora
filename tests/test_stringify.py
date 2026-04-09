"""Tests for stringify functions and extract_codes."""

import pandas as pd
import pytest

from tquery._codes import extract_codes
from tquery._stringify import stringify_durations, stringify_order, stringify_time


@pytest.fixture
def drug_df():
    """Drug prescription data for 2 persons.

    P1: i(Jan 1) → i(Apr 1) → a(Jul 1) → i(Oct 1)
    P2: a(Jan 1) → i(Apr 1) → g(Jul 1)

    With codes = {'i': ['L04AB02'], 'a': ['L04AB04'], 'g': ['L04AB06']}
    """
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


@pytest.fixture
def codes():
    return {"i": ["L04AB02"], "a": ["L04AB04"], "g": ["L04AB06"]}


# ---------------------------------------------------------------------------
# extract_codes tests
# ---------------------------------------------------------------------------

class TestExtractCodes:
    def test_exact_match(self, drug_df, codes):
        result = extract_codes(drug_df, codes, ["atc"])
        assert list(result) == ["i", "i", "a", "i", "a", "i", "g"]

    def test_wildcard(self, drug_df):
        codes = {"bio": ["L04AB*"]}
        result = extract_codes(drug_df, codes, ["atc"])
        assert all(result == "bio")

    def test_no_match(self):
        df = pd.DataFrame({"pid": [1], "atc": ["ZZZ"]})
        codes = {"i": ["L04AB02"]}
        result = extract_codes(df, codes, ["atc"])
        assert pd.isna(result.iloc[0])

    def test_string_value(self, drug_df):
        codes = {"i": "L04AB02", "a": "L04AB04"}  # str, not list
        result = extract_codes(drug_df, codes, ["atc"])
        assert result.iloc[0] == "i"
        assert result.iloc[2] == "a"


# ---------------------------------------------------------------------------
# stringify_order tests
# ---------------------------------------------------------------------------

class TestStringifyOrder:
    def test_basic(self, drug_df, codes):
        result = stringify_order(drug_df, codes, cols="atc")
        assert result[1] == "iiai"
        assert result[2] == "aig"

    def test_no_repeats(self, drug_df, codes):
        result = stringify_order(drug_df, codes, cols="atc", keep_repeats=False)
        assert result[1] == "iai"  # 'iiai' → 'iai'
        assert result[2] == "aig"  # no change

    def test_only_unique(self, drug_df, codes):
        result = stringify_order(drug_df, codes, cols="atc", only_unique=True)
        assert result[1] == "ia"   # first i, first a (no duplicates)
        assert result[2] == "aig"  # all unique already

    def test_single_code(self, drug_df):
        codes = {"i": ["L04AB02"]}
        result = stringify_order(drug_df, codes, cols="atc")
        assert result[1] == "iii"  # P1 has 3 infliximab events
        assert result[2] == "i"    # P2 has 1

    def test_first_date_column(self, drug_df, codes):
        drug_df["ref_date"] = pd.to_datetime("2020-03-01")
        result = stringify_order(
            drug_df, codes, cols="atc", first_date="ref_date"
        )
        # P1: only events from Apr onwards → i, a, i → "iai"
        assert result[1] == "iai"


# ---------------------------------------------------------------------------
# stringify_time tests
# ---------------------------------------------------------------------------

class TestStringifyTime:
    def test_basic_merge(self, drug_df, codes):
        result = stringify_time(
            drug_df, codes, cols="atc", step=91, merge=True
        )
        assert isinstance(result, pd.Series)
        assert 1 in result.index
        assert 2 in result.index
        # P1: i at pos 0, i at pos 1, a at pos 2, i at pos 3 (approx 91-day steps)
        p1 = result[1]
        assert "i" in p1
        assert "a" in p1

    def test_no_merge(self, drug_df, codes):
        result = stringify_time(
            drug_df, codes, cols="atc", step=91, merge=False
        )
        assert isinstance(result, pd.DataFrame)
        assert "i" in result.columns
        assert "a" in result.columns

    def test_step_size_affects_length(self, drug_df, codes):
        r1 = stringify_time(drug_df, codes, cols="atc", step=30, merge=False)
        r2 = stringify_time(drug_df, codes, cols="atc", step=180, merge=False)
        # Smaller step → longer strings
        assert r1["i"].str.len().max() > r2["i"].str.len().max()

    def test_empty_result(self):
        df = pd.DataFrame({
            "pid": [1], "start_date": pd.to_datetime(["2020-01-01"]),
            "atc": ["ZZZ"],
        })
        codes = {"i": ["L04AB02"]}
        result = stringify_time(df, codes, cols="atc")
        assert len(result) == 0


# ---------------------------------------------------------------------------
# stringify_durations tests
# ---------------------------------------------------------------------------

class TestStringifyDurations:
    @pytest.fixture
    def duration_df(self):
        """Events with durations."""
        return pd.DataFrame({
            "pid": [1, 1, 2],
            "start_date": pd.to_datetime([
                "2020-01-01", "2020-07-01", "2020-01-01",
            ]),
            "atc": ["L04AB02", "L04AB04", "L04AB02"],
            "days": [180, 90, 365],  # duration in days
        })

    def test_with_duration_column(self, duration_df, codes):
        result = stringify_durations(
            duration_df, codes, cols="atc",
            event_duration="days", step=90, merge=False,
        )
        assert isinstance(result, pd.DataFrame)
        # P1: i starts at pos 0, lasts 180 days = 2 steps → "ii"
        #     a starts at pos 2, lasts 90 days = 1 step → " a" (in 'a' column)
        p1_i = result.loc[1, "i"]
        assert p1_i.startswith("ii")

    def test_with_end_date(self, codes):
        df = pd.DataFrame({
            "pid": [1, 1],
            "start_date": pd.to_datetime(["2020-01-01", "2020-04-01"]),
            "end_date": pd.to_datetime(["2020-03-31", "2020-06-30"]),
            "atc": ["L04AB02", "L04AB04"],
        })
        result = stringify_durations(
            df, codes, cols="atc",
            event_end="end_date", step=30, merge=False,
        )
        assert isinstance(result, pd.DataFrame)
        # i: Jan 1 to Mar 31 = 90 days, 90//30 = 3, fills positions 0-3 = 4 chars
        p1_i = result.loc[1, "i"]
        assert p1_i.count("i") >= 3  # at least 3 periods filled

    def test_merge(self, duration_df, codes):
        result = stringify_durations(
            duration_df, codes, cols="atc",
            event_duration="days", step=90, merge=True,
        )
        assert isinstance(result, pd.Series)

    def test_missing_both_end_and_duration(self, duration_df, codes):
        with pytest.raises(ValueError, match="event_end or event_duration"):
            stringify_durations(duration_df, codes, cols="atc")
