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
        result = tquery(wdf, "K51 inside 30 days after K50")
        # P1: K51 19d after K50 → YES
        # P2: K51 60d after K50 → NO
        # P3: K51 before K50, not after → NO
        assert result.count == 1
        assert 1 in result.pids

    def test_within_around(self, wdf):
        result = tquery(wdf, "K51 inside 30 days around K50")
        # P1: 19d → YES, P2: 60d → NO, P3: 9d → YES
        assert result.count == 2
        assert result.pids == {1, 3}

    def test_inside_days_range_after(self, wdf):
        # K51 inside 15 to 45 days after K50:
        # P1: K51 19d after K50 → in [15, 45] → YES
        # P2: K51 60d after K50 → > 45 → NO
        # P3: K51 before K50 (not after) → NO
        result = tquery(wdf, "K51 inside 15 to 45 days after K50")
        assert result.pids == {1}

    def test_inside_signed_around_asymmetric(self, wdf):
        # K51 inside -5 to 40 days around K50 (child is between 5d
        # before and 40d after K50):
        # P1: K51 Jan 20 vs K50 Jan 1 → diff = +19 days → in [-5, 40] → YES
        # P2: K51 Mar 1 vs K50 Jan 1 → diff = +60 → NO
        # P3: K51 Jan 1 vs K50 Jan 10 → diff = -9 → outside [-5, 40] → NO
        result = tquery(wdf, "K51 inside -5 to 40 days around K50")
        assert result.pids == {1}

    def test_inside_events_range(self, wdf):
        # wdf P1: K50 Jan 1 (pos 0), K51 Jan 20 (pos 1), K52 Jun 1 (pos 2)
        # K51 inside 1 to 1 events after K50:
        # P1: K51 at pos 1 vs K50 at pos 0 → offset +1, in [1, 1] → YES
        # P2: K51 at pos 1 vs K50 at pos 0 → offset +1 → YES
        # P3: K51 at pos 0, K50 at pos 1 → K51 is before K50, offset -1 → NO
        result = tquery(wdf, "K51 inside 1 to 1 events after K50")
        assert result.pids == {1, 2}

    def test_each_equivalent_to_every(self, wdf):
        r1 = tquery(wdf, "K51 inside 30 days after each K50")
        r2 = tquery(wdf, "K51 inside 30 days after every K50")
        assert r1.pids == r2.pids

    def test_always_equivalent_to_every(self, wdf):
        r1 = tquery(wdf, "always K51 inside 30 days after K50")
        r2 = tquery(wdf, "every K51 inside 30 days after K50")
        assert r1.pids == r2.pids

    def test_never_equivalent_to_not(self, wdf):
        r1 = tquery(wdf, "never K51 inside 30 days after K50")
        r2 = tquery(wdf, "not K51 inside 30 days after K50")
        assert r1.pids == r2.pids


class TestEventAtomQueries:
    def test_event_matches_all_rows(self, df):
        m = df.tq.mask("event")
        assert m.dtype == bool
        assert m.all()
        assert len(m) == len(df)

    def test_events_alias(self, df):
        # `event` and `events` should produce identical masks
        assert df.tq.mask("event").equals(df.tq.mask("events"))

    def test_nth_event(self, df):
        # df fixture has 3 rows per person (sorted by date).
        # 2nd event per person → exactly one row per pid (the middle row).
        m = df.tq.mask("2nd event")
        assert int(m.sum()) == 3
        per_person = df[m].groupby("pid").size()
        assert (per_person == 1).all()

    def test_last_n_events(self, df):
        # last 2 events per person → 6 rows total (2 per pid, 3 persons)
        m = df.tq.mask("last 2 of events")
        assert int(m.sum()) == 6
        per_person = df[m].groupby("pid").size()
        assert (per_person == 2).all()

    def test_first_n_events(self, df):
        m = df.tq.mask("first 1 of events")
        assert int(m.sum()) == 3  # one per person

    def test_min_n_event_universal(self, df):
        # Every person has 3 rows; `min 3 of event` should match everyone.
        result = tquery(df, "min 3 of event")
        assert result.count == 3
        # `min 4 of event` should match no one.
        assert tquery(df, "min 4 of event").count == 0

    def test_before_nth_event(self, df):
        # K50 before 3rd event per person.
        # P1: K50 is at row 0 (1st event); 3rd event at row 2. K50 before 3rd → YES
        # P2: K50 is at row 1 (2nd event); 3rd event at row 2. K50 before 3rd → YES
        # P3: K50s are 1st and 2nd events; 3rd event at row 2. K50 before 3rd → YES
        result = tquery(df, "K50 before 3rd event")
        assert result.count == 3

    def test_after_first_event(self, df):
        # Default `X after Y` semantics: min(X date) > min(Y date), i.e.
        # the first X comes after the first Y.
        # P1: first K50 = Jan 1 = 1st event. Jan 1 > Jan 1 → NO
        # P2: first K50 = Jan 15; 1st event = Jan 1 (K51). Jan 15 > Jan 1 → YES
        # P3: first K50 = Jun 1 = 1st event. Jun 1 > Jun 1 → NO
        result = tquery(df, "K50 after 1st event")
        assert result.pids == {2}


class TestBetweenPositional:
    @pytest.fixture
    def bdf(self):
        """Fixture for positional-bounds between.

        P1: K51 Jan, K50 Feb, K50 Mar, K51 Apr, K50 May, K51 Jun
            K51 ordinals: 1st=Jan, 2nd=Apr, 3rd=Jun
            K50 dates: Feb, Mar, May
        P2: K50 Jan (no K51 → bounds undefined)
        P3: K51 Jan, K51 Feb, K50 Mar, K51 Apr, K50 May, K51 Jun, K51 Jul
            K51 ordinals: 1st=Jan, 2nd=Feb, 3rd=Apr, 4th=Jun, 5th=Jul
            K50 dates: Mar, May
        """
        return pd.DataFrame({
            "pid": [1, 1, 1, 1, 1, 1,
                    2,
                    3, 3, 3, 3, 3, 3, 3],
            "start_date": pd.to_datetime([
                "2020-01-01", "2020-02-01", "2020-03-01",
                "2020-04-01", "2020-05-01", "2020-06-01",
                "2020-01-01",
                "2020-01-01", "2020-02-01", "2020-03-01",
                "2020-04-01", "2020-05-01", "2020-06-01", "2020-07-01",
            ]),
            "icd": ["K51", "K50", "K50", "K51", "K50", "K51",
                    "K50",
                    "K51", "K51", "K50", "K51", "K50", "K51", "K51"],
        }).sort_values(["pid", "start_date"]).reset_index(drop=True)

    def test_between_same_code_narrow(self, bdf):
        # K50 between 1st K51 and 2nd K51:
        # P1: window [Jan, Apr] → K50 Feb, K50 Mar match (2 rows)
        # P2: no K51 → no bounds → no matches
        # P3: window [Jan, Feb] → no K50 in that window (first K50 is Mar)
        m = bdf.tq.mask("K50 inside 1st K51 and 2nd K51")
        selected = bdf[m]
        assert (selected["icd"] == "K50").all()
        assert set(selected["pid"]) == {1}
        assert len(selected) == 2

    def test_between_same_code_wide(self, bdf):
        # K50 between 1st K51 and 3rd K51:
        # P1: window [Jan, Jun] → all three K50s match (Feb, Mar, May)
        # P3: window [Jan, Apr] → K50 Mar matches (K50 May is outside)
        m = bdf.tq.mask("K50 inside 1st K51 and 3rd K51")
        selected = bdf[m]
        per_person = selected.groupby("pid").size()
        assert per_person.get(1, 0) == 3
        assert per_person.get(3, 0) == 1
        assert 2 not in per_person.index

    def test_between_full_k51_range(self, bdf):
        # K50 between 1st K51 and 5th K51 (P3 has 5 K51s; P1 only 3)
        # P3: window [Jan, Jul] → K50 Mar and K50 May match
        # P1: only 3 K51s → no 5th K51 → no match for P1
        m = bdf.tq.mask("K50 inside 1st K51 and 5th K51")
        selected = bdf[m]
        assert set(selected["pid"]) == {3}
        assert len(selected) == 2

    def test_between_count(self, bdf):
        # count is person-level: persons with at least one matching K50.
        r = tquery(bdf, "K50 inside 1st K51 and 2nd K51")
        assert r.count == 1
        assert r.pids == {1}

    def test_between_missing_bound_excluded_from_evaluable(self, bdf):
        # P2 has no K51 → bounds undefined → not evaluable
        r = tquery(bdf, "K50 inside 1st K51 and 2nd K51")
        assert 2 not in r.evaluable_pids

    def test_range_days_works(self, bdf):
        # `inside N to M days` (replaces the old `between N and M days` form).
        r = tquery(bdf, "K50 inside 0 to 120 days after K51")
        # Just check it parses and returns a sensible result (no crash).
        assert isinstance(r.rows, pd.Series)

    def test_outside_bounds_complement(self, bdf):
        inside = bdf.tq.mask("K50 inside 1st K51 and 2nd K51")
        outside = bdf.tq.mask("K50 outside 1st K51 and 2nd K51")
        k50 = bdf.tq.mask("K50")
        # Inside and outside are disjoint.
        assert not (inside & outside).any()
        # Every K50 of an evaluable person is either inside or outside.
        r = tquery(bdf, "K50 inside 1st K51 and 2nd K51")
        evaluable_rows = bdf["pid"].isin(r.evaluable_pids)
        assert ((k50 & evaluable_rows) == (inside | outside)).all()


class TestWithinSpan:
    @pytest.fixture
    def sdf(self):
        """Same fixture as bdf but scoped to within-span tests.

        P1: K51 Jan, K50 Feb, K50 Mar, K51 Apr, K50 May, K51 Jun    (6 rows)
        P2: K50 Jan                                                   (1 row)
        P3: K51 Jan, K51 Feb, K50 Mar, K51 Apr, K50 May, K51 Jun, K51 Jul (7 rows)
        """
        return pd.DataFrame({
            "pid": [1, 1, 1, 1, 1, 1,
                    2,
                    3, 3, 3, 3, 3, 3, 3],
            "start_date": pd.to_datetime([
                "2020-01-01", "2020-02-01", "2020-03-01",
                "2020-04-01", "2020-05-01", "2020-06-01",
                "2020-01-01",
                "2020-01-01", "2020-02-01", "2020-03-01",
                "2020-04-01", "2020-05-01", "2020-06-01", "2020-07-01",
            ]),
            "icd": ["K51", "K50", "K50", "K51", "K50", "K51",
                    "K50",
                    "K51", "K51", "K50", "K51", "K50", "K51", "K51"],
        }).sort_values(["pid", "start_date"]).reset_index(drop=True)

    def test_within_first_n_of_k51(self, sdf):
        # K50 within first 2 of K51:
        # P1: K51s Jan, Apr (first 2) → span [Jan, Apr]. K50s Feb, Mar match.
        # P2: no K51 → not evaluable.
        # P3: K51s Jan, Feb → span [Jan, Feb]. No K50s match (first K50 is Mar).
        m = sdf.tq.mask("K50 inside first 2 of K51")
        selected = sdf[m]
        assert set(selected["pid"]) == {1}
        assert len(selected) == 2

    def test_within_last_3_of_k51(self, sdf):
        # K50 within last 3 of K51:
        # P1: last 3 K51 = all 3 → span [Jan, Jun]. All K50s match.
        # P2: no K51.
        # P3: last 3 K51 = Apr, Jun, Jul → span [Apr, Jul]. K50 May matches; K50 Mar does not.
        m = sdf.tq.mask("K50 inside last 3 of K51")
        per_person = sdf[m].groupby("pid").size()
        assert per_person.get(1, 0) == 3
        assert per_person.get(3, 0) == 1

    def test_within_last_n_events(self, sdf):
        # K50 within last 5 events (of the timeline itself):
        # P1: last 5 events Feb..Jun → K50s Feb, Mar, May match (3).
        # P2: 1 event (K50 Jan); span [Jan, Jan]; K50 matches itself (1).
        # P3: last 5 events Mar..Jul → K50s Mar, May match (2).
        m = sdf.tq.mask("K50 inside last 5 events")
        per_person = sdf[m].groupby("pid").size()
        assert per_person.get(1, 0) == 3
        assert per_person.get(2, 0) == 1
        assert per_person.get(3, 0) == 2

    def test_within_span_missing_ref_excluded_from_evaluable(self, sdf):
        r = tquery(sdf, "K50 inside first 2 of K51")
        assert 2 not in r.evaluable_pids

    def test_within_span_count(self, sdf):
        r = tquery(sdf, "K50 inside last 3 of K51")
        assert r.count == 2
        assert r.pids == {1, 3}

    def test_classic_within_days_still_works(self, sdf):
        # Regression: integer-prefix `inside N days` form still parses and works.
        r = tquery(sdf, "K50 inside 45 days after K51")
        assert isinstance(r.rows, pd.Series)

    def test_outside_span_complement(self, sdf):
        inside = sdf.tq.mask("K50 inside last 3 of K51")
        outside = sdf.tq.mask("K50 outside last 3 of K51")
        k50 = sdf.tq.mask("K50")
        assert not (inside & outside).any()
        r = tquery(sdf, "K50 inside last 3 of K51")
        evaluable_rows = sdf["pid"].isin(r.evaluable_pids)
        assert ((k50 & evaluable_rows) == (inside | outside)).all()


class TestShiftedAnchors:
    @pytest.fixture
    def shdf(self):
        """Fixture with known day gaps between K51 and K50.

        P1: K51 Jan 1, K50 May 15  (134 days apart → K51 is 134d before K50)
        P2: K51 Feb 1, K50 Feb 20  (19 days apart)
        P3: K51 Mar 1, K50 Mar 10  (9 days)
        P4: K50 Jan 10 (no K51)
        """
        return pd.DataFrame({
            "pid": [1, 1, 2, 2, 3, 3, 4],
            "start_date": pd.to_datetime([
                "2020-01-01", "2020-05-15",
                "2020-02-01", "2020-02-20",
                "2020-03-01", "2020-03-10",
                "2020-01-10",
            ]),
            "icd": ["K51", "K50", "K51", "K50", "K51", "K50", "K50"],
        }).sort_values(["pid", "start_date"]).reset_index(drop=True)

    def test_before_shifted_minus(self, shdf):
        # 1st K51 before 1st K50 - 100 days:
        # P1: K51 Jan 1 < (May 15 - 100d = Feb 5) → YES
        # P2: K51 Feb 1 < (Feb 20 - 100d = Nov 12 2019) → NO
        # P3: K51 Mar 1 < (Mar 10 - 100d = Dec 1 2019) → NO
        # P4: no K51 → NO
        r = tquery(shdf, "1st K51 before 1st K50 - 100 days")
        assert r.pids == {1}

    def test_before_shifted_plus(self, shdf):
        # 1st K50 before 1st K51 + 30 days — i.e., K50 occurs before
        # (first K51 + 30 days).
        # P1: K50 May 15 < (Jan 1 + 30d = Jan 31)? No.
        # P2: K50 Feb 20 < (Feb 1 + 30d = Mar 2)? Yes.
        # P3: K50 Mar 10 < (Mar 1 + 30d = Mar 31)? Yes.
        # P4: no K51 → NO
        r = tquery(shdf, "1st K50 before 1st K51 + 30 days")
        assert r.pids == {2, 3}

    def test_parens_equivalent(self, shdf):
        r1 = tquery(shdf, "1st K51 before 1st K50 - 100 days")
        r2 = tquery(shdf, "1st K51 before (1st K50 - 100 days)")
        assert r1.pids == r2.pids

    def test_within_shifted_ref(self, shdf):
        # K50 inside 20 days after 1st K51 + 20 days:
        # shifted ref per person is (1st K51 + 20 days).
        # Window = [shifted_ref, shifted_ref + 20 days].
        # P1: shifted ref = Jan 21. Window [Jan 21, Feb 10]. K50 May 15 → NO.
        # P2: shifted ref = Feb 21. Window [Feb 21, Mar 12]. K50 Feb 20 → NO (before window).
        # P3: shifted ref = Mar 21. Window [Mar 21, Apr 10]. K50 Mar 10 → NO.
        # P4: no K51 → not evaluable.
        # So no pid matches — fine; test that the query evaluates without error.
        r = tquery(shdf, "K50 inside 20 days after 1st K51 + 20 days")
        assert isinstance(r.rows, pd.Series)
        # Now shift backward so the window catches more K50s.
        # K50 inside 200 days after 1st K51 - 30 days:
        # P1: shifted ref = Dec 2 2019. Window [Dec 2 2019, Jun 19 2020]. K50 May 15 → YES.
        # P2: shifted ref = Jan 2. Window [Jan 2, Jul 20]. K50 Feb 20 → YES.
        # P3: shifted ref = Jan 31. Window [Jan 31, Aug 18]. K50 Mar 10 → YES.
        r2 = tquery(shdf, "K50 inside 200 days after 1st K51 - 30 days")
        assert r2.pids == {1, 2, 3}

    def test_shifted_in_bounds(self, shdf):
        # K50 inside 1st K51 - 10 days and 1st K51 + 200 days:
        # Window is [K51 date - 10, K51 date + 200] per person.
        # P1: window [Dec 22 2019, Jul 19 2020]. K50 May 15 → YES.
        # P2: window [Jan 22, Aug 19]. K50 Feb 20 → YES.
        # P3: window [Feb 20, Sep 17]. K50 Mar 10 → YES.
        r = tquery(shdf, "K50 inside 1st K51 - 10 days and 1st K51 + 200 days")
        assert r.pids == {1, 2, 3}

    def test_shift_chain(self, shdf):
        # 1st K51 before 1st K50 - 50 days - 50 days (= -100 days):
        # Equivalent to test_before_shifted_minus.
        r1 = tquery(shdf, "1st K51 before 1st K50 - 50 days - 50 days")
        r2 = tquery(shdf, "1st K51 before 1st K50 - 100 days")
        assert r1.pids == r2.pids

    def test_shift_in_event_window_rejected(self, shdf):
        with pytest.raises(TypeError, match="event-count window"):
            tquery(shdf, "K50 inside 3 events after 1st K51 + 7 days")

    def test_shift_in_span_rejected(self, shdf):
        # A single shifted date has no span.
        with pytest.raises(TypeError, match="no span"):
            tquery(shdf, "K50 inside 1st K51 + 7 days")

    def test_shift_on_left_of_temporal_rejected(self, shdf):
        with pytest.raises(TypeError, match="reference side"):
            tquery(shdf, "1st K51 + 10 days before K50")


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


class TestQuantifiers:
    """Tests for `every` / `any` quantifier modifiers.

    Fixture design (notation: K@day means K event on day N):

    P1: K51@10            ; K50@50
        — single K51 followed by single K50
    P2: K51@10, K51@200   ; K50@50
        — two K51s; only the first is followed by a K50; second has no K50
    P3: K51@10            ; K50@5, K50@50
        — single K51; one K50 before it (day 5), one after (day 50)
    P4: K51@10            ; (no K50)
    P5: (no K51)          ; K50@50
    """

    @pytest.fixture
    def qdf(self):
        return pd.DataFrame({
            "pid": [1, 1, 2, 2, 2, 3, 3, 3, 4, 5],
            "start_date": pd.to_datetime([
                "2020-01-11", "2020-02-20",                  # P1: K51@10, K50@50
                "2020-01-11", "2020-02-20", "2020-07-29",    # P2: K51@10, K50@50, K51@200
                "2020-01-06", "2020-01-11", "2020-02-20",    # P3: K50@5, K51@10, K50@50
                "2020-01-11",                                 # P4: K51@10
                "2020-02-20",                                 # P5: K50@50
            ]),
            "icd": [
                "K51", "K50",
                "K51", "K50", "K51",
                "K50", "K51", "K50",
                "K51",
                "K50",
            ],
        }).sort_values(["pid", "start_date"]).reset_index(drop=True)

    def test_any_is_default(self, qdf):
        # Explicit any should produce identical results to no quantifier
        baseline = tquery(qdf, "K50 after K51")
        with_any = tquery(qdf, "any K50 after any K51")
        assert baseline.pids == with_any.pids

    def test_default_after(self, qdf):
        # K50 after K51 (existing semantic = first K50 after first K51):
        # P1: K50@50 > K51@10 ✓
        # P2: K50@50 > K51@10 ✓
        # P3: first K50@5, first K51@10 → 5 > 10? NO
        # P4: no K50
        # P5: no K51
        result = tquery(qdf, "K50 after K51")
        assert result.pids == {1, 2}

    def test_after_every_right(self, qdf):
        # K50 after every K51: max(K50) > max(K51), and K51 non-empty
        # P1: max(K50)=50, max(K51)=10 → 50 > 10 ✓
        # P2: max(K50)=50, max(K51)=200 → 50 > 200 NO
        # P3: max(K50)=50, max(K51)=10 → ✓ (but does default fail? yes — universal RHS allows it)
        # P4: no K50
        # P5: no K51 (vacuous-truth excluded)
        result = tquery(qdf, "K50 after every K51")
        assert result.pids == {1, 3}

    def test_every_after_distinct_from_default(self, qdf):
        # `every K50 after K51` ≡ default by design (universal LHS = existing default).
        baseline = tquery(qdf, "K50 after K51")
        with_every = tquery(qdf, "every K50 after K51")
        assert baseline.pids == with_every.pids

    def test_after_every_both(self, qdf):
        # every K50 after every K51: min(K50) > max(K51)
        # P1: min(K50)=50, max(K51)=10 → 50 > 10 ✓
        # P2: min(K50)=50, max(K51)=200 → NO
        # P3: min(K50)=5,  max(K51)=10 → NO
        # P4, P5: missing one side
        result = tquery(qdf, "every K50 after every K51")
        assert result.pids == {1}

    def test_before_every_left(self, qdf):
        # every K50 before K51: max(K50) < max(K51), both non-empty
        # P1: max(K50)=50, max(K51)=10 → 50 < 10 NO
        # P2: max(K50)=50, max(K51)=200 → 50 < 200 ✓
        # P3: max(K50)=50, max(K51)=10 → NO
        result = tquery(qdf, "every K50 before K51")
        assert result.pids == {2}

    def test_before_every_both(self, qdf):
        # every K50 before every K51: max(K50) < min(K51)
        # P1: max(K50)=50, min(K51)=10 → NO
        # P2: max(K50)=50, min(K51)=10 → NO
        # P3: max(K50)=50, min(K51)=10 → NO
        result = tquery(qdf, "every K50 before every K51")
        assert result.pids == set()

    def test_every_no_vacuous_truth(self, qdf):
        # P5 has K50 but no K51 — must NOT match `every K51` queries.
        result = tquery(qdf, "K50 after every K51")
        assert 5 not in result.pids

    def test_within_every_right(self, qdf):
        # K50 within 100 days after every K51:
        # for each K51, there must be a K50 within 100 days after.
        # P1: K51@10 → K50@50 (40d after) ✓ — only one K51 → ✓
        # P2: K51@10 → K50@50 ✓; K51@200 → no K50 after → ✗
        # P3: K51@10 → K50@50 (40d) ✓ → ✓
        # P4: no K50; P5: no K51
        result = tquery(qdf, "K50 inside 100 days after every K51")
        assert result.pids == {1, 3}

    def test_within_every_left(self, qdf):
        # every K50 within 100 days after K51:
        # every K50 must have at least one K51 within 100 days before it.
        # P1: K50@50 → K51@10 (40d before) ✓ → ✓
        # P2: K50@50 → K51@10 ✓ → ✓ (only one K50)
        # P3: K50@5 → no K51 before → ✗ ; K50@50 → K51@10 ✓
        #     Not every K50 satisfies → ✗
        # P4, P5: missing one side
        result = tquery(qdf, "every K50 inside 100 days after K51")
        assert result.pids == {1, 2}

    def test_within_every_both(self, qdf):
        # every K50 within 100 days after every K51
        # P1: only one K50 and one K51, K50 is 40d after → ✓
        # P2: K51@200 has no K50 after → ✗
        # P3: K50@5 has no K51 before → ✗
        result = tquery(qdf, "every K50 inside 100 days after every K51")
        assert result.pids == {1}

    def test_simultaneously_every(self, qdf):
        # Sanity: simultaneously with every quantifier should not crash
        # and require the universal side to be subset of the other.
        # Build a small ad-hoc dataset
        sdf = pd.DataFrame({
            "pid": [1, 1, 2, 2, 2],
            "start_date": pd.to_datetime([
                "2020-01-01", "2020-01-01",  # P1: K50 and K51 same day
                "2020-01-01", "2020-01-01", "2020-02-01",  # P2: K50,K51 same day + extra K50
            ]),
            "icd": ["K50", "K51", "K50", "K51", "K50"],
        }).sort_values(["pid", "start_date"]).reset_index(drop=True)
        # every K50 simultaneously K51:
        #   P1: only K50@01-01, has matching K51 ✓
        #   P2: K50@01-01 has K51@01-01 ✓; K50@02-01 has no K51@02-01 ✗
        result = tquery(sdf, "every K50 simultaneously K51")
        assert result.pids == {1}

    def test_not_every(self, qdf):
        # `not (K50 after every K51)` should be the complement among all persons
        result = tquery(qdf, "not (K50 after every K51)")
        # `K50 after every K51` matched {1, 3}; complement is {2, 4, 5}
        assert result.pids == {2, 4, 5}


class TestProportions:
    """Tests for the count/evaluable/pct properties.

    Fixture:
        P1: K50@day1, K51@day10, K52@day20  (K50 before K51, has K52)
        P2: K51@day1, K50@day10              (K51 before K50, no K52)
        P3: K50@day1                          (only K50 — undefined for K50/K51 compare)
        P4: K51@day1                          (only K51 — undefined for K50/K51 compare)
        P5: K52@day1                          (neither K50 nor K51)
    """

    @pytest.fixture
    def pdf(self):
        return pd.DataFrame({
            "pid": [1, 1, 1, 2, 2, 3, 4, 5],
            "start_date": pd.to_datetime([
                "2020-01-01", "2020-01-10", "2020-01-20",  # P1: K50, K51, K52
                "2020-01-01", "2020-01-10",                # P2: K51, K50
                "2020-01-01",                               # P3: K50
                "2020-01-01",                               # P4: K51
                "2020-01-01",                               # P5: K52
            ]),
            "icd": ["K50", "K51", "K52", "K51", "K50", "K50", "K51", "K52"],
        }).sort_values(["pid", "start_date"]).reset_index(drop=True)

    def test_total(self, pdf):
        result = tquery(pdf, "K50")
        assert result.total == 5

    def test_evaluable_for_plain_code(self, pdf):
        # No comparative subexpression → everyone is evaluable
        result = tquery(pdf, "K50")
        assert result.evaluable == 5

    def test_pct_plain_code(self, pdf):
        # 3 of 5 persons have K50 (P1, P2, P3); evaluable = 5
        result = tquery(pdf, "K50")
        assert result.count == 3
        assert result.pct() == 60.0
        assert result.pct(dropna=False) == 60.0

    def test_evaluable_for_temporal(self, pdf):
        # K50 before K51 — defined only for persons with both K50 and K51
        # Persons with K50: P1, P2, P3. With K51: P1, P2, P4. Both: P1, P2.
        result = tquery(pdf, "K50 before K51")
        assert result.evaluable == 2
        assert result.evaluable_pids == {1, 2}

    def test_pct_temporal_conditional(self, pdf):
        # Default semantic: first K50 < first K51
        # P1: K50@1 < K51@10 ✓
        # P2: K50@10 < K51@1 ✗
        # 1 match out of 2 evaluable → 50%
        result = tquery(pdf, "K50 before K51")
        assert result.count == 1
        assert result.pct() == 50.0

    def test_pct_temporal_marginal(self, pdf):
        # Same query, but denominator = 5 (all persons) → 20%
        result = tquery(pdf, "K50 before K51")
        assert result.pct(dropna=False) == 20.0

    def test_compound_and_intersects_evaluable(self, pdf):
        # (K50 before K51) and K52
        # evaluable = (K50 ∩ K51) ∩ all = {1, 2}
        # K52 holders: {1, 5}
        # K50 before K51 holders: {1}
        # AND result: {1}
        result = tquery(pdf, "(K50 before K51) and K52")
        assert result.count == 1
        assert result.evaluable == 2  # not 5 — K52's universal definedness doesn't widen
        assert result.pct() == 50.0

    def test_compound_or_unions_evaluable(self, pdf):
        # (K50 before K51) or K52
        # evaluable = {1, 2} ∪ {all 5} = all 5 (Z is defined for everyone)
        # OR result: {1} ∪ {1, 5} = {1, 5}
        result = tquery(pdf, "(K50 before K51) or K52")
        assert result.count == 2
        assert result.evaluable == 5
        assert result.pct() == 40.0

    def test_within_window_evaluable(self, pdf):
        # K50 within 100 days after K51 — also requires both K50 and K51
        result = tquery(pdf, "K50 inside 100 days after K51")
        assert result.evaluable == 2

    def test_zero_evaluable_returns_zero(self, pdf):
        # ZZZ does not exist → no person is evaluable for `K50 before ZZZ`
        result = tquery(pdf, "K50 before ZZZ")
        assert result.count == 0
        assert result.evaluable == 0
        assert result.pct() == 0.0
        assert result.pct(dropna=False) == 0.0

    def test_repr_includes_pct(self, pdf):
        result = tquery(pdf, "K50 before K51")
        r = repr(result)
        assert "count=1" in r
        assert "evaluable=2" in r
        assert "total=5" in r
        assert "50.0%" in r

    def test_accessor_pct(self, pdf):
        # df.tq.pct shortcut
        assert pdf.tq.pct("K50 before K51") == 50.0
        assert pdf.tq.pct("K50 before K51", dropna=False) == 20.0

    def test_accessor_pct_plain(self, pdf):
        assert pdf.tq.pct("K50") == 60.0

    def test_not_widens_evaluable(self, pdf):
        # `not X` in the existing evaluator collapses undefined to True
        # (persons absent from X are included). So the negation is
        # well-defined for everyone and evaluable widens to all persons.
        # `K50 before K51` matches {P1}; `not (...)` matches {P2,P3,P4,P5}.
        result = tquery(pdf, "not (K50 before K51)")
        assert result.evaluable == 5
        assert result.count == 4
        assert result.pct() == 80.0

    def test_every_quantifier_evaluable(self, pdf):
        # `every K50 after K51` ≡ default `K50 after K51` semantically.
        # evaluable = (K50 ∩ K51) for both sides = {1, 2}
        # P1: min(K50)=1, min(K51)=10. 1 > 10? No.
        # P2: min(K50)=10, min(K51)=1. 10 > 1? Yes.
        result = tquery(pdf, "every K50 after K51")
        assert result.evaluable == 2
        assert result.count == 1
        assert result.pct() == 50.0


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
