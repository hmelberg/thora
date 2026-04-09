"""Cross-validation tests: compare tquery (DataFrame) vs string_query (strings).

For each expression, both evaluators should return the same set of person IDs.
This validates that the temporal query logic is consistent across the two
independent implementations.
"""

import pandas as pd
import pytest

from tquery import tquery
from tquery._string_evaluator import cross_validate, string_query_auto


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def drug_df():
    """Drug prescription data for 2 persons.

    P1: i(Jan 1) -> i(Apr 1) -> a(Jul 1) -> i(Oct 1)
    P2: a(Jan 1) -> i(Apr 1) -> g(Jul 1)
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
def drug_codes():
    return {"i": ["L04AB02"], "a": ["L04AB04"], "g": ["L04AB06"]}


@pytest.fixture
def icd_df():
    """ICD diagnosis data for 3 persons.

    P1: K50 (Jan 1) -> K51 (Feb 1) -> K52 (Mar 1)
    P2: K51 (Jan 1) -> K50 (Jan 15) -> K52 (Mar 1)
    P3: K50 (Jun 1) -> K50 (Jul 1) -> K51 (Aug 1)
    """
    return pd.DataFrame({
        "pid": [1, 1, 1, 2, 2, 2, 3, 3, 3],
        "start_date": pd.to_datetime([
            "2020-01-01", "2020-02-01", "2020-03-01",
            "2020-01-01", "2020-01-15", "2020-03-01",
            "2020-06-01", "2020-07-01", "2020-08-01",
        ]),
        "icd": ["K50", "K51", "K52", "K51", "K50", "K52", "K50", "K50", "K51"],
    }).sort_values(["pid", "start_date"]).reset_index(drop=True)


@pytest.fixture
def icd_codes():
    return {"a": ["K50"], "b": ["K51"], "c": ["K52"]}


# ---------------------------------------------------------------------------
# Cross-validation: drug data (order mode)
# ---------------------------------------------------------------------------

DRUG_EXPRESSIONS = [
    "L04AB02",
    "L04AB04",
    "L04AB06",
    "L04AB02, L04AB04",
    "L04AB02 and L04AB04",
    "L04AB02 or L04AB04",
    "L04AB02 or L04AB06",
    "not L04AB02",
    "not L04AB06",
    "min 2 of L04AB02",
    "min 3 of L04AB02",
    "max 1 of L04AB02",
    "max 1 of L04AB04",
    "exactly 1 of L04AB04",
    "exactly 3 of L04AB02",
    "L04AB02 before L04AB04",
    "L04AB04 before L04AB02",
    "L04AB02 after L04AB04",
    "L04AB04 after L04AB02",
    "1st of L04AB02 before 1st of L04AB04",
    "(min 2 of L04AB02) before L04AB04",
    "L04AB02 before L04AB04 and L04AB06",
    "(L04AB02 or L04AB06) before L04AB04",
    "L04AB02 and L04AB04 and L04AB06",
]


@pytest.mark.parametrize("expr", DRUG_EXPRESSIONS)
def test_cross_validate_drugs_order(drug_df, drug_codes, expr):
    df_pids, str_pids, match = cross_validate(
        drug_df, expr, drug_codes, mode="order", cols="atc",
    )
    assert match, (
        f"Mismatch for '{expr}': df_pids={df_pids}, str_pids={str_pids}"
    )


# ---------------------------------------------------------------------------
# Cross-validation: ICD data (order mode)
# ---------------------------------------------------------------------------

ICD_EXPRESSIONS = [
    "K50",
    "K51",
    "K52",
    "K50 and K51",
    "K50 or K52",
    "not K52",
    "K50 before K51",
    "K50 after K51",
    "K51 before K50",
    "min 2 of K50",
    "exactly 1 of K50",
    "K50 before K51 and K52",
    "(min 2 of K50) before K51",
    "1st of K50 before 1st of K51",
]


@pytest.mark.parametrize("expr", ICD_EXPRESSIONS)
def test_cross_validate_icd_order(icd_df, icd_codes, expr):
    df_pids, str_pids, match = cross_validate(
        icd_df, expr, icd_codes, mode="order", cols="icd",
    )
    assert match, (
        f"Mismatch for '{expr}': df_pids={df_pids}, str_pids={str_pids}"
    )


# ---------------------------------------------------------------------------
# Cross-validation: variables
# ---------------------------------------------------------------------------

def test_cross_validate_variable(icd_df, icd_codes):
    df_pids, str_pids, match = cross_validate(
        icd_df, "@crohns before K51", icd_codes, mode="order",
        cols="icd", variables={"crohns": ["K50"]},
    )
    assert match


# ---------------------------------------------------------------------------
# Direct comparison helper tests
# ---------------------------------------------------------------------------

class TestCrossValidateHelper:
    def test_returns_tuple(self, drug_df, drug_codes):
        result = cross_validate(
            drug_df, "L04AB02", drug_codes, mode="order", cols="atc",
        )
        assert isinstance(result, tuple)
        assert len(result) == 3
        df_pids, str_pids, match = result
        assert isinstance(df_pids, set)
        assert isinstance(str_pids, set)
        assert isinstance(match, bool)

    def test_match_is_true_for_simple(self, drug_df, drug_codes):
        _, _, match = cross_validate(
            drug_df, "L04AB02", drug_codes, mode="order", cols="atc",
        )
        assert match

    def test_accessor(self, drug_df, drug_codes):
        _, _, match = drug_df.tq.cross_validate(
            "L04AB02 before L04AB04", drug_codes,
            mode="order", cols="atc",
        )
        assert match
