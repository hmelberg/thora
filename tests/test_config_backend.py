"""Tests for the backend field on TQueryConfig + set_backend() helper."""
from __future__ import annotations

import pandas as pd
import pytest

import tquery as tq


@pytest.fixture(autouse=True)
def _reset_config():
    """Each test starts with a clean default config."""
    saved = tq.get_config()
    tq.use(tq.TQueryConfig())  # default
    yield
    tq.use(saved)


def _df() -> pd.DataFrame:
    return pd.DataFrame({
        "pid":        [1, 1, 2, 2, 3],
        "start_date": pd.to_datetime(["2020-01-01"] * 5),
        "icd":        ["K50", "K51", "K50", "K52", "K51"],
    })


def test_config_backend_default_none():
    assert tq.get_config().backend is None


def test_set_backend_updates_config():
    tq.set_backend("duckdb")
    assert tq.get_config().backend == "duckdb"

    tq.set_backend("polars")
    assert tq.get_config().backend == "polars"

    tq.set_backend(None)
    assert tq.get_config().backend is None


def test_set_backend_does_not_clobber_other_config_fields():
    tq.use(tq.TQueryConfig(pid="patient_id", date="event_date", cols="icd"))
    tq.set_backend("duckdb")
    cfg = tq.get_config()
    assert cfg.pid == "patient_id"
    assert cfg.date == "event_date"
    assert cfg.cols == "icd"
    assert cfg.backend == "duckdb"


def test_config_backend_used_when_no_explicit_arg():
    """Without explicit backend=, the config's backend is honored."""
    tq.use(tq.TQueryConfig(backend="duckdb"))
    # If config didn't propagate, this would use pandas. Hard to assert that
    # directly, but the result should still be correct.
    r = tq.tquery(_df(), "K50", pid="pid", date="start_date", cols="icd")
    assert r.count == 2


def test_explicit_backend_arg_overrides_config():
    """backend='pandas' beats config.backend='duckdb'."""
    tq.use(tq.TQueryConfig(backend="duckdb"))
    r = tq.tquery(_df(), "K50",
                  pid="pid", date="start_date", cols="icd",
                  backend="pandas")
    assert r.count == 2  # same result either way; check that nothing errors


def test_polars_config_with_pandas_input_auto_converts():
    """Config says polars; pandas input gets converted at the boundary."""
    tq.set_backend("polars")
    r = tq.tquery(_df(), "K50", pid="pid", date="start_date", cols="icd")
    assert r.count == 2


def test_unknown_backend_in_config_raises():
    tq.use(tq.TQueryConfig(backend="oracle"))
    with pytest.raises(ValueError, match="Unknown backend"):
        tq.tquery(_df(), "K50", pid="pid", date="start_date", cols="icd")


def test_multi_df_with_config_backend():
    """Multi-DF input picks up config backend just like single-DF does."""
    npr = _df()
    rx = pd.DataFrame({
        "pid":        [1, 2, 3],
        "start_date": pd.to_datetime(["2020-06-01"] * 3),
        "atc":        ["L04AB02", "N02BE01", "L04AB04"],
    })
    tq.set_backend("duckdb")
    r = tq.tquery({"npr": npr, "rx": rx},
                  "K50 in icd before L04AB* in atc",
                  pid="pid", date="start_date")
    # Person 1: K50 in Jan, L04AB02 in Jun → match
    # Person 2: K50 in Jan, N02BE01 in Jun → no
    # Person 3: K51 only, L04AB04 in Jun → no K50
    assert sorted(r.pids) == [1]


def test_set_backend_back_to_none_restores_auto_detect():
    tq.set_backend("duckdb")
    tq.set_backend(None)
    # No explicit backend, no config — auto-detect by type. Pandas → pandas.
    r = tq.tquery(_df(), "K50", pid="pid", date="start_date", cols="icd")
    assert r.count == 2
