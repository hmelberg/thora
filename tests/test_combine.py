"""Tests for tquery.combine() — multi-DataFrame helper."""
from __future__ import annotations

import pandas as pd
import pytest

import tquery as tq


def _npr() -> pd.DataFrame:
    return pd.DataFrame({
        "pid":        [1, 1, 2, 2, 3],
        "start_date": pd.to_datetime(["2020-01-01", "2020-06-01",
                                       "2020-02-01", "2020-08-01",
                                       "2020-03-01"]),
        "icd":        ["K50", "K51", "K50", "K52", "K51"],
    })


def _rx() -> pd.DataFrame:
    return pd.DataFrame({
        "pid":        [1, 2, 2, 3],
        "start_date": pd.to_datetime(["2020-09-01", "2020-10-01",
                                       "2021-01-01", "2020-05-01"]),
        "atc":        ["L04AB02", "N02BE01", "L04AB04", "L04AB02"],
    })


def test_combine_dict_form_preserves_all_rows():
    out = tq.combine({"npr": _npr(), "rx": _rx()})
    assert len(out) == 5 + 4
    assert set(out["__source__"].unique()) == {"npr", "rx"}


def test_combine_sorts_by_pid_then_date():
    out = tq.combine({"npr": _npr(), "rx": _rx()})
    pid_date_pairs = list(zip(out["pid"], out["start_date"]))
    assert pid_date_pairs == sorted(pid_date_pairs)


def test_combine_list_form_auto_names():
    out = tq.combine([_npr(), _rx()])
    assert set(out["__source__"].unique()) == {"source_0", "source_1"}


def test_combine_tuple_with_names():
    out = tq.combine((_npr(), _rx()), names=["npr", "rx"])
    assert set(out["__source__"].unique()) == {"npr", "rx"}


def test_combine_per_atom_routing_via_in_column():
    out = tq.combine({"npr": _npr(), "rx": _rx()})
    r = tq.tquery(out, "K50 in icd before L04AB* in atc",
                  pid="pid", date="start_date")
    # pid 1: K50 in Jan before L04AB02 in Sep → match
    # pid 2: K50 in Feb before L04AB04 in 2021-Jan → match
    # pid 3: no K50 → no match
    assert sorted(r.pids) == [1, 2]


def test_combine_missing_pid_raises():
    with pytest.raises(tq.TQueryColumnError):
        bad = _npr().drop(columns=["pid"])
        tq.combine({"npr": bad, "rx": _rx()})


def test_combine_missing_date_raises():
    with pytest.raises(tq.TQueryColumnError):
        bad = _npr().drop(columns=["start_date"])
        tq.combine({"npr": bad, "rx": _rx()})


def test_combine_empty_input_raises():
    with pytest.raises(ValueError, match="empty"):
        tq.combine([])
    with pytest.raises(ValueError, match="empty"):
        tq.combine({})


def test_combine_names_length_mismatch_raises():
    with pytest.raises(ValueError, match="entries"):
        tq.combine([_npr(), _rx()], names=["only_one"])


def test_combine_warns_on_mismatched_pid_dtype():
    npr_str = _npr()
    npr_str["pid"] = npr_str["pid"].astype(str)
    with pytest.warns(UserWarning, match="differing dtypes"):
        tq.combine({"npr": npr_str, "rx": _rx()})


def test_combine_custom_source_col_name():
    out = tq.combine({"npr": _npr(), "rx": _rx()}, source_col="registry")
    assert "registry" in out.columns
    assert "__source__" not in out.columns
