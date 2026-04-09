"""Performance benchmarks for tquery.

Run with: pytest tests/test_performance.py -v
Skip with: pytest -m "not slow"
"""

import time

import pandas as pd
import pytest

from tquery import tquery
from tquery._testdata import make_test_data


@pytest.fixture(scope="module")
def large_df():
    """100K rows: 10K persons, ~10 events each."""
    return make_test_data(n_persons=10_000, events_per_person=10)


@pytest.fixture(scope="module")
def million_df():
    """~1M rows: 100K persons, ~10 events each."""
    return make_test_data(n_persons=100_000, events_per_person=10)


@pytest.mark.slow
class TestPerformance:
    def test_simple_code_100k(self, large_df):
        start = time.perf_counter()
        result = tquery(large_df, "K50")
        elapsed = time.perf_counter() - start
        assert result.count > 0
        assert elapsed < 2.0, f"Simple code query took {elapsed:.2f}s (expected <2s)"

    def test_before_100k(self, large_df):
        start = time.perf_counter()
        result = tquery(large_df, "K50 before K51")
        elapsed = time.perf_counter() - start
        assert result.count > 0
        assert elapsed < 5.0, f"Before query took {elapsed:.2f}s (expected <5s)"

    def test_compound_100k(self, large_df):
        start = time.perf_counter()
        result = tquery(large_df, "(K50 or K51) before K52")
        elapsed = time.perf_counter() - start
        assert result.count > 0
        assert elapsed < 5.0, f"Compound query took {elapsed:.2f}s (expected <5s)"

    def test_min_prefix_100k(self, large_df):
        start = time.perf_counter()
        result = tquery(large_df, "min 2 of K50")
        elapsed = time.perf_counter() - start
        assert elapsed < 2.0, f"Min prefix took {elapsed:.2f}s (expected <2s)"

    def test_within_100k(self, large_df):
        start = time.perf_counter()
        result = tquery(large_df, "K50 within 365 days after K51")
        elapsed = time.perf_counter() - start
        assert elapsed < 5.0, f"Within query took {elapsed:.2f}s (expected <5s)"

    def test_wildcard_100k(self, large_df):
        start = time.perf_counter()
        result = tquery(large_df, "K50*")
        elapsed = time.perf_counter() - start
        assert result.count > 0
        assert elapsed < 2.0, f"Wildcard query took {elapsed:.2f}s (expected <2s)"

    def test_simple_code_1m(self, million_df):
        start = time.perf_counter()
        result = tquery(million_df, "K50")
        elapsed = time.perf_counter() - start
        assert result.count > 0
        assert elapsed < 5.0, f"1M simple query took {elapsed:.2f}s (expected <5s)"

    def test_before_1m(self, million_df):
        start = time.perf_counter()
        result = tquery(million_df, "K50 before K51")
        elapsed = time.perf_counter() - start
        assert result.count > 0
        assert elapsed < 10.0, f"1M before query took {elapsed:.2f}s (expected <10s)"
