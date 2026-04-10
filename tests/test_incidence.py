"""Tests for the incidence calculation module."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import tquery as tq
from tquery import (
    TQueryConfig,
    fit_decay,
    incidence,
    raw_incidence,
    singles_pattern,
    washout_pattern,
)
from tquery._incidence import (
    _exponential,
    _first_year_under_threshold,
)

from tests._synthetic import make_cohort, make_data


# ---------------------------------------------------------------------------
# Hand-built fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def small_df():
    """Five persons with deterministic first events.

    P1: K50 in 2018 only (single event, first year 2018)
    P2: K50 in 2018, K50 in 2019  (first K50 = 2018, two K50 events)
    P3: K50 in 2019 only (single event, first year 2019)
    P4: K50 in 2019, K51 in 2020  (first K50 = 2019, only one K50)
    P5: K51 in 2020 only (no K50 at all)
    """
    return pd.DataFrame({
        "pid": [1, 2, 2, 3, 4, 4, 5],
        "start_date": pd.to_datetime([
            "2018-06-01",
            "2018-03-01", "2019-03-01",
            "2019-06-01",
            "2019-09-01", "2020-01-01",
            "2020-06-01",
        ]),
        "icd": ["K50", "K50", "K50", "K50", "K50", "K51", "K51"],
    }).sort_values(["pid", "start_date"]).reset_index(drop=True)


@pytest.fixture
def washout_df():
    """For washout pattern testing.

    P1: prior K50 in 2014, then K50 in 2018  → not truly new in 2018
    P2: K50 in 2018 only  → truly new in 2018
    P3: K50 in 2018 only  → truly new in 2018
    P4: K50 in 2014 only  → exists to extend the data window
    """
    return pd.DataFrame({
        "pid": [1, 1, 2, 3, 4],
        "start_date": pd.to_datetime([
            "2014-01-01",
            "2018-06-01",
            "2018-03-01",
            "2018-09-01",
            "2014-01-01",
        ]),
        "icd": ["K50", "K50", "K50", "K50", "K50"],
    }).sort_values(["pid", "start_date"]).reset_index(drop=True)


@pytest.fixture
def singles_df():
    """For singles pattern testing with required_events=2.

    P1: K50 in 2018 only — true singleton in 2018, never accumulates
    P2: K50 in 2018, K50 in 2020 — single in 2018, becomes >=2 with expansion
    P3: K50 in 2018 (Jan), K50 in 2018 (Dec) — already 2 events in 2018
    PA: K50 in 2015 — extends min_date so we can expand backward
    PB: K50 in 2022 — extends max_date so we can expand forward
    """
    return pd.DataFrame({
        "pid": [1, 2, 2, 3, 3, 100, 200],
        "start_date": pd.to_datetime([
            "2018-06-01",
            "2018-06-01", "2020-06-01",
            "2018-01-15", "2018-12-15",
            "2015-01-01",
            "2022-01-01",
        ]),
        "icd": ["K50"] * 7,
    }).sort_values(["pid", "start_date"]).reset_index(drop=True)


# ---------------------------------------------------------------------------
# raw_incidence
# ---------------------------------------------------------------------------

class TestRawIncidence:

    def test_default_required_events_one(self, small_df):
        """All persons count; first event year per person."""
        out = raw_incidence(small_df)
        # P1 first 2018, P2 first 2018, P3 first 2019, P4 first 2019, P5 first 2020
        assert dict(out) == {2018: 2, 2019: 2, 2020: 1}

    def test_with_expr_filter(self, small_df):
        """Filter rows by tquery expression first."""
        out = raw_incidence(small_df, "K50")
        # P1, P2 first K50 in 2018; P3, P4 first K50 in 2019; P5 has no K50
        assert dict(out) == {2018: 2, 2019: 2}

    def test_required_events_two(self, small_df):
        """Only persons with >=2 matching events count."""
        out = raw_incidence(small_df, "K50", required_events=2)
        # Only P2 has >= 2 K50 events. First K50 = 2018.
        assert dict(out) == {2018: 1}

    def test_empty_df(self):
        df = pd.DataFrame({"pid": [], "start_date": pd.to_datetime([]), "icd": []})
        out = raw_incidence(df)
        assert out.empty

    def test_returns_int_index_and_name(self, small_df):
        out = raw_incidence(small_df)
        assert out.name == "raw_incidence"
        assert all(isinstance(y, (int, np.integer)) for y in out.index)


# ---------------------------------------------------------------------------
# washout_pattern
# ---------------------------------------------------------------------------

class TestWashoutPattern:

    def test_single_year_decay(self, washout_df):
        """At max lookback, P1 (with prior 2014 K50) drops out."""
        out = washout_pattern(washout_df, year=2018, step_days=400)
        # 3 persons in 2018: P1, P2, P3
        assert out.iloc[0] == 3.0
        # Asymptote at full lookback should be 2 (P1 drops out)
        assert out.iloc[-1] == 2.0

    def test_pct_returns_fractions(self, washout_df):
        out = washout_pattern(washout_df, year=2018, step_days=400, pct=True)
        assert out.iloc[0] == pytest.approx(1.0)
        assert out.iloc[-1] == pytest.approx(2 / 3)

    def test_all_years_returns_dataframe(self, washout_df):
        out = washout_pattern(washout_df, step_days=400)
        # Earliest year (2014) is excluded — there's no lookback for it
        assert isinstance(out, pd.DataFrame)
        assert 2018 in out.columns
        assert 2014 not in out.columns

    def test_asymptote_matches_raw_incidence(self, washout_df):
        """At max lookback, washout count for year y == raw_incidence[y]."""
        wp = washout_pattern(washout_df, year=2018, step_days=400)
        raw = raw_incidence(washout_df)
        # Both should agree on 2018: 2 truly new persons
        assert wp.iloc[-1] == raw.loc[2018]


# ---------------------------------------------------------------------------
# singles_pattern
# ---------------------------------------------------------------------------

class TestSinglesPattern:

    def test_single_year_decay(self, singles_df):
        """P2 has events in 2018+2020; expanding 800 days captures both."""
        out = singles_pattern(
            singles_df, year=2018, required_events=2, step_days=400,
        )
        # Candidates in 2018 = persons with <2 events in 2018 = {P1, P2}
        # P3 has 2 events in 2018, not a candidate.
        assert out.iloc[0] == 2.0
        # Eventually only P1 remains a singleton (true singleton)
        assert out.iloc[-1] == 1.0

    def test_pct_mode(self, singles_df):
        out = singles_pattern(
            singles_df, year=2018, required_events=2, step_days=400, pct=True,
        )
        assert out.iloc[0] == pytest.approx(1.0)
        # Asymptote: 1 / 2 candidates remain
        assert out.iloc[-1] == pytest.approx(0.5)

    def test_all_years_returns_dataframe(self, singles_df):
        out = singles_pattern(
            singles_df, required_events=2, step_days=400,
        )
        assert isinstance(out, pd.DataFrame)


# ---------------------------------------------------------------------------
# fit_decay
# ---------------------------------------------------------------------------

class TestFitDecay:

    def test_recovers_known_exponential(self):
        """Fit a noiseless exponential and check parameters are recovered."""
        true_a = 0.3
        true_b = 1.5
        x = np.arange(0, 2000, 100, dtype=float)
        y = _exponential(x, true_a, true_b)
        pat = pd.Series(y, index=x)

        result = fit_decay(pat, model="exponential")
        assert result["model"] == "exponential"
        assert result["coeffs"][0] == pytest.approx(true_a, abs=0.01)
        assert result["coeffs"][1] == pytest.approx(true_b, abs=0.05)
        assert result["asymptote"] == pytest.approx(true_a, abs=0.01)

    def test_predict_callable(self):
        x = np.arange(0, 2000, 100, dtype=float)
        y = _exponential(x, 0.3, 1.5)
        pat = pd.Series(y, index=x)
        result = fit_decay(pat)
        pred = result["predict"](x)
        np.testing.assert_allclose(pred, y, atol=1e-6)

    def test_unknown_model_raises(self):
        pat = pd.Series([1.0, 0.8, 0.6], index=[0, 100, 200])
        with pytest.raises(ValueError, match="Unknown decay model"):
            fit_decay(pat, model="bogus")

    def test_returns_aic_and_r2(self):
        x = np.arange(0, 2000, 100, dtype=float)
        y = _exponential(x, 0.3, 1.5)
        pat = pd.Series(y, index=x)
        result = fit_decay(pat, model="exponential")
        assert "aic" in result
        assert "r2" in result
        # Noiseless data → near-perfect fit
        assert result["r2"] == pytest.approx(1.0, abs=1e-6)

    def test_all_mode_fits_every_model(self):
        x = np.arange(0, 2000, 100, dtype=float)
        y = _exponential(x, 0.3, 1.5)
        pat = pd.Series(y, index=x)
        results = fit_decay(pat, model="all")
        assert set(results) == {"exponential", "hyperbolic", "rational"}
        # Each entry has the standard fields
        for name, fit in results.items():
            assert fit["model"] == name
            assert "aic" in fit
            assert "r2" in fit
            assert "asymptote" in fit
            assert callable(fit["predict"])

    def test_all_mode_picks_truth_via_aic(self):
        """Exponential is the true generator → it should be the
        best-AIC model on noiseless exponential data (or tied)."""
        x = np.arange(0, 2000, 100, dtype=float)
        y = _exponential(x, 0.3, 1.5)
        pat = pd.Series(y, index=x)
        results = fit_decay(pat, model="all")
        aics = {name: fit["aic"] for name, fit in results.items()
                if "error" not in fit}
        best = min(aics, key=aics.get)
        # Either exponential wins outright, or it's within 2 of best
        # (the standard "no meaningful difference" AIC threshold).
        assert aics[best] - aics["exponential"] >= -2.0

    def test_dataframe_averages_columns(self):
        x = np.arange(0, 2000, 100, dtype=float)
        y1 = _exponential(x, 0.3, 1.5)
        y2 = _exponential(x, 0.3, 1.5) + 0.01  # tiny noise
        pat_df = pd.DataFrame({2010: y1, 2011: y2}, index=x)
        result = fit_decay(pat_df, model="exponential")
        assert result["asymptote"] == pytest.approx(0.305, abs=0.02)

    def test_empty_pattern_raises(self):
        with pytest.raises(ValueError, match="empty"):
            fit_decay(pd.Series(dtype=float))


# ---------------------------------------------------------------------------
# incidence (master function)
# ---------------------------------------------------------------------------

class TestIncidence:

    @pytest.fixture
    def synthetic(self):
        """A constant-true-incidence stream of 200 persons per year, then a
        clipped 2010-2019 observation window. Ground truth: 200/year."""
        df = make_data(
            n_per_cohort=200,
            start_year=2000,
            end_year=2020,
            cohort_duration=10,
            seed=42,
        )
        sample = df[
            (df["date"].dt.year >= 2010) & (df["date"].dt.year <= 2019)
        ].copy()
        return sample

    def test_raw_is_inflated_in_early_years(self, synthetic):
        raw = raw_incidence(synthetic, date="date")
        # The first observed year is biased high (no lookback at all)
        assert raw.loc[2010] > raw.loc[2019]

    def test_functional_washout_flattens(self, synthetic):
        raw = raw_incidence(synthetic, date="date")
        adj = incidence(
            synthetic, date="date",
            washout="functional", lookahead="none",
        )
        # Adjustment should bring early years closer to later years
        assert adj.loc[2010] < raw.loc[2010]
        # And reduce overall variance
        assert adj.std() < raw.std()

    def test_lookahead_auto_no_op_for_re1(self, synthetic):
        adj_none = incidence(
            synthetic, date="date",
            washout="none", lookahead="none",
            required_events=1,
        )
        adj_auto = incidence(
            synthetic, date="date",
            washout="none", lookahead="auto",
            required_events=1,
        )
        pd.testing.assert_series_equal(adj_none, adj_auto)

    def test_lookahead_corrects_late_year_undercount(self, synthetic):
        """With required_events=2, raw counts the late years sharply
        under because the second confirming event hasn't been observed
        yet. The functional lookahead correction operates on raw_re1
        (uncensored) and should produce a much higher estimate for the
        late years than the right-censored re=2 count.
        """
        adj_no = incidence(
            synthetic, date="date",
            washout="none", lookahead="none",
            required_events=2,
        )
        adj_la = incidence(
            synthetic, date="date",
            washout="none", lookahead="functional",
            required_events=2,
        )
        # The last year is the most right-censored — lookahead must help.
        last = adj_no.index.max()
        assert adj_la.loc[last] > adj_no.loc[last]

    def test_full_correction_recovers_constant_truth(self, synthetic):
        """washout+lookahead together on a constant-true-incidence
        cohort should give a fairly flat result across the window."""
        adj = incidence(
            synthetic, date="date",
            washout="functional", lookahead="functional",
            required_events=2,
        )
        raw = raw_incidence(synthetic, date="date", required_events=2)
        # The corrected series should have a much smaller relative
        # spread than the raw re=2 count, which is right-censored.
        raw_spread = (raw.max() - raw.min()) / raw.mean()
        adj_spread = (adj.max() - adj.min()) / adj.mean()
        assert adj_spread < raw_spread

    def test_invalid_washout_raises(self):
        with pytest.raises(ValueError, match="washout"):
            incidence(pd.DataFrame({"pid": [], "start_date": pd.to_datetime([])}),
                      washout="bogus")

    def test_invalid_lookahead_raises(self):
        with pytest.raises(ValueError, match="lookahead"):
            incidence(pd.DataFrame({"pid": [], "start_date": pd.to_datetime([])}),
                      lookahead="bogus")

    def test_empty_df(self):
        df = pd.DataFrame({"pid": [], "start_date": pd.to_datetime([])})
        out = incidence(df)
        assert out.empty


# ---------------------------------------------------------------------------
# Accessor integration
# ---------------------------------------------------------------------------

class TestAccessor:

    def test_raw_incidence(self, small_df):
        a = small_df.tq.raw_incidence("K50")
        b = raw_incidence(small_df, "K50")
        pd.testing.assert_series_equal(a, b)

    def test_incidence(self, small_df):
        a = small_df.tq.incidence("K50", washout="none")
        b = incidence(small_df, "K50", washout="none")
        pd.testing.assert_series_equal(a, b)

    def test_washout_pattern(self, washout_df):
        a = washout_df.tq.washout_pattern(year=2018, step_days=400)
        b = washout_pattern(washout_df, year=2018, step_days=400)
        pd.testing.assert_series_equal(a, b)

    def test_singles_pattern(self, singles_df):
        a = singles_df.tq.singles_pattern(
            year=2018, required_events=2, step_days=400,
        )
        b = singles_pattern(
            singles_df, year=2018, required_events=2, step_days=400,
        )
        pd.testing.assert_series_equal(a, b)


# ---------------------------------------------------------------------------
# Config integration
# ---------------------------------------------------------------------------

class TestConfigIntegration:

    def test_custom_pid_and_date_via_config(self):
        df = pd.DataFrame({
            "patient": [1, 2, 3],
            "event_date": pd.to_datetime(["2018-01-01", "2019-01-01", "2020-01-01"]),
            "icd": ["K50", "K50", "K50"],
        })
        cfg = TQueryConfig(pid="patient", date="event_date")
        out = raw_incidence(df, config=cfg)
        assert dict(out) == {2018: 1, 2019: 1, 2020: 1}

    def test_explicit_kwargs_override_config(self):
        df = pd.DataFrame({
            "p": [1, 2],
            "d": pd.to_datetime(["2018-01-01", "2019-01-01"]),
        })
        out = raw_incidence(df, pid="p", date="d")
        assert dict(out) == {2018: 1, 2019: 1}


# ---------------------------------------------------------------------------
# Internal helper sanity
# ---------------------------------------------------------------------------

class TestFirstYearUnderThreshold:

    def test_basic(self, small_df):
        # required_events=2 → persons with <2 events in their first year
        # P1: 1 event in first year (2018) → under  → contributes to 2018
        # P2: 1 event in first year (2018) → under  → contributes to 2018
        # P3: 1 event in first year (2019) → under  → contributes to 2019
        # P4: 1 event in first year (2019) → under  → contributes to 2019
        # P5: 1 event in first year (2020) → under  → contributes to 2020
        out = _first_year_under_threshold(small_df, "pid", "start_date", 2)
        assert dict(out) == {2018: 2, 2019: 2, 2020: 1}
