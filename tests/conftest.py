"""Shared test fixtures for tquery tests."""

import pandas as pd
import pytest


@pytest.fixture
def simple_df():
    """3 persons with known temporal relationships.

    Person 1: K50 (Jan) → K51 (Feb) → K52 (Mar)
        - K50 before K51: YES
        - K51 before K52: YES
    Person 2: K51 (Jan) → K50 (Jan 15) → K52 (Mar)
        - K50 before K51: NO (K51 came first)
        - K51 before K50: YES
    Person 3: K50 (Jun) → K50 (Jul) → K51 (Aug)
        - K50 before K51: YES
        - min 2 of K50: YES
        - min 3 of K50: NO
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
def multi_col_df():
    """DataFrame with multiple code columns."""
    return pd.DataFrame({
        "pid": [1, 1, 2, 2, 3],
        "start_date": pd.to_datetime([
            "2020-01-01", "2020-06-01",
            "2020-01-01", "2020-06-01",
            "2020-01-01",
        ]),
        "icd": ["K50", "K51", "K50", "K52", "K53"],
        "atc": ["A01", "A02", "A03", "A01", "A02"],
    }).sort_values(["pid", "start_date"]).reset_index(drop=True)


@pytest.fixture
def within_df():
    """DataFrame for testing 'within N days' queries.

    Person 1: K50 (Jan 1) → K51 (Jan 20)  → K52 (Jun 1)
        - K51 within 30 days after K50: YES (19 days)
        - K52 within 30 days after K50: NO (152 days)
    Person 2: K50 (Jan 1) → K51 (Mar 1)
        - K51 within 30 days after K50: NO (60 days)
    Person 3: K51 (Jan 1) → K50 (Jan 10)
        - K50 within 30 days after K51: YES (9 days)
    """
    return pd.DataFrame({
        "pid": [1, 1, 1, 2, 2, 3, 3],
        "start_date": pd.to_datetime([
            "2020-01-01", "2020-01-20", "2020-06-01",
            "2020-01-01", "2020-03-01",
            "2020-01-01", "2020-01-10",
        ]),
        "icd": ["K50", "K51", "K52", "K50", "K51", "K51", "K50"],
    }).sort_values(["pid", "start_date"]).reset_index(drop=True)
