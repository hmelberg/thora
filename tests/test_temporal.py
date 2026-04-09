"""Tests for temporal operations: before/after, within, inside/outside."""

import pandas as pd
import pytest

from tquery._temporal import eval_before_after, eval_within_days, eval_inside_outside


@pytest.fixture
def df():
    """3 persons:
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


class TestBeforeAfter:
    def test_k50_before_k51(self, df):
        left = df["icd"] == "K50"
        right = df["icd"] == "K51"
        result = eval_before_after(df, left, right, "before", "pid", "start_date")
        # P1: K50 (Jan) before K51 (Feb) → YES
        # P2: K50 (Jan 15) but K51 (Jan 1) came first → NO
        # P3: K50 (Jun) before K51 (Aug) → YES
        person_result = result.groupby(df["pid"]).any()
        assert person_result[1] == True
        assert person_result[2] == False
        assert person_result[3] == True

    def test_k51_before_k50(self, df):
        left = df["icd"] == "K51"
        right = df["icd"] == "K50"
        result = eval_before_after(df, left, right, "before", "pid", "start_date")
        person_result = result.groupby(df["pid"]).any()
        # P1: K51 (Feb) after K50 (Jan) → NO
        # P2: K51 (Jan 1) before K50 (Jan 15) → YES
        # P3: K51 (Aug) after K50 (Jun) → NO
        assert person_result[1] == False
        assert person_result[2] == True
        assert person_result[3] == False

    def test_k50_after_k51(self, df):
        left = df["icd"] == "K50"
        right = df["icd"] == "K51"
        result = eval_before_after(df, left, right, "after", "pid", "start_date")
        person_result = result.groupby(df["pid"]).any()
        # P1: K50 first → NO
        # P2: K50 after K51 → YES
        # P3: K50 first → NO
        assert person_result[1] == False
        assert person_result[2] == True
        assert person_result[3] == False

    def test_empty_left(self, df):
        left = pd.Series(False, index=df.index)
        right = df["icd"] == "K51"
        result = eval_before_after(df, left, right, "before", "pid", "start_date")
        assert not result.any()

    def test_empty_right(self, df):
        left = df["icd"] == "K50"
        right = pd.Series(False, index=df.index)
        result = eval_before_after(df, left, right, "before", "pid", "start_date")
        assert not result.any()

    def test_no_overlap_persons(self, df):
        """Left only in P1, right only in P3 → no match."""
        left = (df["pid"] == 1) & (df["icd"] == "K50")
        right = (df["pid"] == 3) & (df["icd"] == "K51")
        result = eval_before_after(df, left, right, "before", "pid", "start_date")
        assert not result.any()


class TestWithinDays:
    @pytest.fixture
    def wdf(self):
        """
        P1: K50 (Jan 1), K51 (Jan 20), K52 (Jun 1)
        P2: K50 (Jan 1), K51 (Mar 1)
        P3: K51 (Jan 1), K50 (Jan 10)
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

    def test_within_30_days_after(self, wdf):
        child = wdf["icd"] == "K51"
        ref = wdf["icd"] == "K50"
        result = eval_within_days(
            wdf, child, ref, 30, "after", "pid", "start_date"
        )
        # P1: K51 (Jan 20) is 19 days after K50 (Jan 1) → YES
        # P2: K51 (Mar 1) is 60 days after K50 (Jan 1) → NO
        # P3: K51 (Jan 1) is before K50 (Jan 10), not after → NO
        k51_rows = wdf[wdf["icd"] == "K51"]
        for idx, row in k51_rows.iterrows():
            if row["pid"] == 1:
                assert result[idx] == True, f"P1 K51 should match"
            elif row["pid"] == 2:
                assert result[idx] == False, f"P2 K51 should not match"
            elif row["pid"] == 3:
                assert result[idx] == False, f"P3 K51 should not match"

    def test_within_30_days_before(self, wdf):
        child = wdf["icd"] == "K50"
        ref = wdf["icd"] == "K51"
        result = eval_within_days(
            wdf, child, ref, 30, "before", "pid", "start_date"
        )
        # P1: K50 (Jan 1) is 19 days before K51 (Jan 20) → YES
        # P2: K50 (Jan 1) is 60 days before K51 (Mar 1) → NO
        # P3: K50 (Jan 10) is after K51 (Jan 1), not before → NO
        k50_rows = wdf[wdf["icd"] == "K50"]
        for idx, row in k50_rows.iterrows():
            if row["pid"] == 1:
                assert result[idx] == True
            elif row["pid"] == 2:
                assert result[idx] == False
            elif row["pid"] == 3:
                assert result[idx] == False

    def test_within_nearest(self, wdf):
        child = wdf["icd"] == "K51"
        ref = wdf["icd"] == "K50"
        result = eval_within_days(
            wdf, child, ref, 30, None, "pid", "start_date"
        )
        # P1: K51 19 days from K50 → YES
        # P2: K51 60 days from K50 → NO
        # P3: K51 9 days from K50 → YES (nearest)
        k51_rows = wdf[wdf["icd"] == "K51"]
        for idx, row in k51_rows.iterrows():
            if row["pid"] == 1:
                assert result[idx] == True
            elif row["pid"] == 2:
                assert result[idx] == False
            elif row["pid"] == 3:
                assert result[idx] == True

    def test_within_no_ref(self, wdf):
        """Without ref, within N days of first event per person."""
        child = wdf["icd"].isin(["K50", "K51", "K52"])
        result = eval_within_days(
            wdf, child, None, 30, None, "pid", "start_date"
        )
        # P1: Jan 1 is first, Jan 20 is 19d (yes), Jun 1 is 152d (no)
        # P2: Jan 1 is first (yes), Mar 1 is 60d (no)
        # P3: Jan 1 is first (yes), Jan 10 is 9d (yes)
        assert result[0] == True   # P1 Jan 1
        assert result[1] == True   # P1 Jan 20
        assert result[2] == False  # P1 Jun 1

    def test_empty_masks(self, wdf):
        empty = pd.Series(False, index=wdf.index)
        child = wdf["icd"] == "K50"
        assert not eval_within_days(wdf, empty, child, 30, "after", "pid", "start_date").any()
        assert not eval_within_days(wdf, child, empty, 30, "after", "pid", "start_date").any()


class TestInsideOutside:
    @pytest.fixture
    def idf(self):
        """P1: A, B, C, D, E (5 events)"""
        return pd.DataFrame({
            "pid": [1, 1, 1, 1, 1],
            "start_date": pd.to_datetime([
                "2020-01-01", "2020-02-01", "2020-03-01",
                "2020-04-01", "2020-05-01",
            ]),
            "code": ["A", "B", "C", "D", "E"],
        })

    def test_inside_2_events_after(self, idf):
        child = idf["code"].isin(["B", "C", "D", "E"])
        ref = idf["code"] == "A"
        result = eval_inside_outside(
            idf, child, ref, True, 2, "after", "pid"
        )
        # A is at event 0. Inside 2 events after = events 1, 2 (B, C)
        assert list(result) == [False, True, True, False, False]

    def test_outside_2_events_after(self, idf):
        child = idf["code"].isin(["B", "C", "D", "E"])
        ref = idf["code"] == "A"
        result = eval_inside_outside(
            idf, child, ref, False, 2, "after", "pid"
        )
        # Outside = events 3, 4 (D, E)
        assert list(result) == [False, False, False, True, True]
