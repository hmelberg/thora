"""End-to-end tests: query string → result, using the full pipeline."""

import pandas as pd
import pytest

from tquery import tquery, count_persons, TQuerySyntaxError, TQueryColumnError


@pytest.fixture
def df():
    """3 persons with known temporal relationships.

    P1: K50 (Jan 1) → K51 (Feb 1) → K52 (Mar 1)
    P2: K51 (Jan 1) → K50 (Jan 15) → K52 (Mar 1)
    P3: K50 (Jun 1) → K50 (Jul 1) → K51 (Aug 1)
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


class TestSimpleCodeQueries:
    def test_single_code(self, df):
        result = tquery(df, "K50")
        assert result.count == 3  # all 3 persons have K50

    def test_single_code_fewer(self, df):
        result = tquery(df, "K52")
        assert result.count == 2  # P1 and P2 have K52

    def test_comma_codes(self, df):
        result = tquery(df, "K50, K51")
        assert result.count == 3

    def test_code_with_column(self, df):
        result = tquery(df, "K50 in icd")
        assert result.count == 3


class TestLogicalOperators:
    def test_and(self, df):
        result = tquery(df, "K50 and K51")
        assert result.count == 3  # all have both K50 and K51

    def test_and_restrictive(self, df):
        result = tquery(df, "K50 and K52")
        # P1: has K50 and K52 → YES
        # P2: has K50 and K52 → YES
        # P3: has K50 but not K52 → NO
        assert result.count == 2

    def test_or(self, df):
        result = tquery(df, "K50 or K52")
        assert result.count == 3  # all have at least one

    def test_not(self, df):
        result = tquery(df, "not K52")
        # P3 doesn't have K52
        assert result.count == 1
        assert 3 in result.pids


class TestTemporalQueries:
    def test_before(self, df):
        result = tquery(df, "K50 before K51")
        # P1: YES, P2: NO, P3: YES
        assert result.count == 2
        assert result.pids == {1, 3}

    def test_after(self, df):
        result = tquery(df, "K50 after K51")
        # P2: K50 (Jan 15) after K51 (Jan 1) → YES
        assert result.count == 1
        assert result.pids == {2}

    def test_before_and(self, df):
        result = tquery(df, "K50 before K51 and K52")
        # (K50 before K51) and K52
        # K50 before K51: P1, P3
        # K52: P1, P2
        # Intersection: P1
        assert result.count == 1
        assert result.pids == {1}


class TestPrefixQueries:
    def test_min(self, df):
        result = tquery(df, "min 2 of K50")
        # P1: 1 K50, P2: 1 K50, P3: 2 K50's
        assert result.count == 1
        assert 3 in result.pids

    def test_exactly(self, df):
        result = tquery(df, "exactly 1 of K50")
        assert result.count == 2
        assert result.pids == {1, 2}

    def test_1st_before(self, df):
        result = tquery(df, "1st of K50 before 1st of K51")
        # P1: 1st K50 (Jan) before 1st K51 (Feb) → YES
        # P2: 1st K50 (Jan 15) but 1st K51 (Jan 1) came first → NO
        # P3: 1st K50 (Jun) before 1st K51 (Aug) → YES
        assert result.count == 2
        assert result.pids == {1, 3}


class TestWithinQueries:
    @pytest.fixture
    def wdf(self):
        return pd.DataFrame({
            "pid": [1, 1, 1, 2, 2, 3, 3],
            "start_date": pd.to_datetime([
                "2020-01-01", "2020-01-20", "2020-06-01",
                "2020-01-01", "2020-03-01",
                "2020-01-01", "2020-01-10",
            ]),
            "icd": ["K50", "K51", "K52", "K50", "K51", "K51", "K50"],
        }).sort_values(["pid", "start_date"]).reset_index(drop=True)

    def test_within_after(self, wdf):
        result = tquery(wdf, "K51 within 30 days after K50")
        # P1: K51 19d after K50 → YES
        # P2: K51 60d after K50 → NO
        # P3: K51 before K50, not after → NO
        assert result.count == 1
        assert 1 in result.pids

    def test_within_around(self, wdf):
        result = tquery(wdf, "K51 within 30 days around K50")
        # P1: 19d → YES, P2: 60d → NO, P3: 9d → YES
        assert result.count == 2
        assert result.pids == {1, 3}


class TestCompoundExpressions:
    def test_parenthesized_or_before(self, df):
        result = tquery(df, "(K50 or K52) before K51")
        # Persons where K50 or K52 occurs before K51
        # P1: K50 (Jan) before K51 (Feb) → YES
        # P2: K52 (Mar) is after K51 (Jan), K50 (Jan 15) is after K51 (Jan 1) → NO
        # P3: K50 (Jun) before K51 (Aug) → YES
        assert result.count == 2

    def test_nested_prefix(self, df):
        result = tquery(df, "(min 2 of K50) before K51")
        # Only P3 has min 2 K50's, and K50 is before K51
        assert result.count == 1
        assert 3 in result.pids


class TestVariables:
    def test_variable_reference(self, df):
        result = tquery(df, "@crohns before K51", variables={"crohns": ["K50"]})
        assert result.count == 2  # P1 and P3

    def test_variable_list(self, df):
        result = tquery(
            df, "@ibd",
            variables={"ibd": ["K50", "K51"]},
        )
        assert result.count == 3  # all have K50 or K51


class TestResultObject:
    def test_rows(self, df):
        result = tquery(df, "K50")
        assert result.rows.dtype == bool
        assert len(result.rows) == len(df)
        assert result.rows.sum() == 4  # 4 rows with K50

    def test_filter_persons(self, df):
        result = tquery(df, "K50 before K51")
        filtered = result.filter("persons")
        assert set(filtered["pid"]) == {1, 3}
        # All rows for P1 and P3
        assert len(filtered) == 6

    def test_filter_rows(self, df):
        result = tquery(df, "K50")
        filtered = result.filter("rows")
        assert len(filtered) == 4  # only K50 rows

    def test_repr(self, df):
        result = tquery(df, "K50")
        assert "count=3" in repr(result)

    def test_pids(self, df):
        result = tquery(df, "K52")
        assert result.pids == {1, 2}


class TestAccessor:
    def test_accessor_call(self, df):
        result = df.tq("K50 before K51")
        assert result.count == 2

    def test_accessor_count(self, df):
        n = df.tq.count("K50")
        assert n == 3


class TestCountPersons:
    def test_count_persons_function(self, df):
        assert count_persons(df, "K50") == 3
        assert count_persons(df, "K50 before K51") == 2


class TestErrorHandling:
    def test_syntax_error(self, df):
        with pytest.raises(TQuerySyntaxError):
            tquery(df, "K50 befroe K51")

    def test_missing_pid_column(self, df):
        with pytest.raises(TQueryColumnError, match="pid_wrong"):
            tquery(df, "K50", pid="pid_wrong")

    def test_missing_date_column(self, df):
        with pytest.raises(TQueryColumnError, match="date_wrong"):
            tquery(df, "K50", date="date_wrong")
