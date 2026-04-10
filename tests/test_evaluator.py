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
        result = tquery(qdf, "K50 within 100 days after every K51")
        assert result.pids == {1, 3}

    def test_within_every_left(self, qdf):
        # every K50 within 100 days after K51:
        # every K50 must have at least one K51 within 100 days before it.
        # P1: K50@50 → K51@10 (40d before) ✓ → ✓
        # P2: K50@50 → K51@10 ✓ → ✓ (only one K50)
        # P3: K50@5 → no K51 before → ✗ ; K50@50 → K51@10 ✓
        #     Not every K50 satisfies → ✗
        # P4, P5: missing one side
        result = tquery(qdf, "every K50 within 100 days after K51")
        assert result.pids == {1, 2}

    def test_within_every_both(self, qdf):
        # every K50 within 100 days after every K51
        # P1: only one K50 and one K51, K50 is 40d after → ✓
        # P2: K51@200 has no K50 after → ✗
        # P3: K50@5 has no K51 before → ✗
        result = tquery(qdf, "every K50 within 100 days after every K51")
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
        result = tquery(pdf, "K50 within 100 days after K51")
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
