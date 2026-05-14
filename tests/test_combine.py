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


# ---------- Smart-combine: tq.tquery accepts list/tuple/dict directly ----------


def test_smart_combine_tuple_input_matches_explicit_combine():
    """tq.tquery((df1, df2), expr) is equivalent to combine + tquery."""
    explicit = tq.combine({"npr": _npr(), "rx": _rx()})
    r_explicit = tq.tquery(explicit, "K50 in icd before L04AB* in atc",
                           pid="pid", date="start_date")
    r_implicit = tq.tquery((_npr(), _rx()), "K50 in icd before L04AB* in atc",
                           pid="pid", date="start_date")
    assert sorted(r_explicit.pids) == sorted(r_implicit.pids)


def test_smart_combine_dict_input_matches_explicit_combine():
    explicit = tq.combine({"npr": _npr(), "rx": _rx()})
    r_explicit = tq.tquery(explicit, "K50 in icd before L04AB* in atc",
                           pid="pid", date="start_date")
    r_implicit = tq.tquery({"npr": _npr(), "rx": _rx()},
                           "K50 in icd before L04AB* in atc",
                           pid="pid", date="start_date")
    assert sorted(r_explicit.pids) == sorted(r_implicit.pids)


def test_smart_combine_pre_filter_reduces_rows():
    """Safe AST → pre-filter drops non-matching rows before concat."""
    from tquery._smart_combine import smart_combine_for_query
    from tquery._parser import parse

    npr, rx = _npr(), _rx()
    ast = parse("K50 in icd")
    filtered = smart_combine_for_query(
        {"npr": npr, "rx": rx}, ast,
        combine_fn=tq.combine,
        pid="pid", date="start_date", cols=None, sep=None, variables=None,
    )
    full = tq.combine({"npr": npr, "rx": rx})
    assert len(filtered) < len(full)
    # Every kept row from NPR matches K50; RX rows drop entirely (no icd col).
    assert (filtered["icd"] == "K50").all()


def test_smart_combine_bail_out_not_expr_full_concat():
    """`not K50` requires the full pid universe — falls back to full combine."""
    from tquery._smart_combine import smart_combine_for_query
    from tquery._parser import parse

    npr, rx = _npr(), _rx()
    ast = parse("not K50")
    out = smart_combine_for_query(
        {"npr": npr, "rx": rx}, ast,
        combine_fn=tq.combine,
        pid="pid", date="start_date", cols=None, sep=None, variables=None,
    )
    full = tq.combine({"npr": npr, "rx": rx})
    assert len(out) == len(full)


def test_smart_combine_bail_out_aggregate_full_concat():
    """`sum(dose) > 100` needs all dose rows — falls back to full combine."""
    from tquery._smart_combine import smart_combine_for_query
    from tquery._parser import parse

    npr, rx = _npr(), _rx()
    npr["dose"] = [10, 20, 30, 40, 50]
    rx["dose"] = [100, 200, 150, 80]
    ast = parse("sum(dose) > 100")
    out = smart_combine_for_query(
        {"npr": npr, "rx": rx}, ast,
        combine_fn=tq.combine,
        pid="pid", date="start_date", cols=None, sep=None, variables=None,
    )
    full = tq.combine({"npr": npr, "rx": rx})
    assert len(out) == len(full)


def test_smart_combine_correct_under_pre_filter_for_temporal_query():
    """Pre-filter + window evaluator must produce the same pids as full combine."""
    npr, rx = _npr(), _rx()
    full = tq.combine({"npr": npr, "rx": rx})

    r_full = tq.tquery(full, "K50 in icd before L04AB* in atc",
                       pid="pid", date="start_date")
    r_smart = tq.tquery((npr, rx), "K50 in icd before L04AB* in atc",
                        pid="pid", date="start_date")
    assert sorted(r_full.pids) == sorted(r_smart.pids)


def test_smart_combine_correct_under_bail_out():
    """For bail-out queries, smart-combine must still produce correct results
    via full concat."""
    npr, rx = _npr(), _rx()
    npr["dose"] = [10, 20, 30, 40, 50]
    rx["dose"] = [100, 200, 150, 80]
    full = tq.combine({"npr": npr, "rx": rx})

    r_full = tq.tquery(full, "sum(dose) > 100", pid="pid", date="start_date")
    r_smart = tq.tquery({"npr": npr, "rx": rx}, "sum(dose) > 100",
                        pid="pid", date="start_date")
    assert sorted(r_full.pids) == sorted(r_smart.pids)


def test_smart_combine_duckdb_backend_consistency():
    """All backends produce the same result on multi-DF input."""
    npr, rx = _npr(), _rx()
    r_pandas = tq.tquery({"npr": npr, "rx": rx},
                         "K50 in icd before L04AB* in atc",
                         pid="pid", date="start_date")
    r_duckdb = tq.tquery({"npr": npr, "rx": rx},
                         "K50 in icd before L04AB* in atc",
                         pid="pid", date="start_date", backend="duckdb")
    assert sorted(r_pandas.pids) == sorted(r_duckdb.pids)


# ---------- DuckDB native multi-DF (UNION ALL BY NAME, no pandas concat) ----------


def test_duckdb_native_multi_df_dict_input():
    """DuckDB backend accepts dict input directly — no pandas concat."""
    r = tq.tquery({"npr": _npr(), "rx": _rx()},
                  "K50 in icd before L04AB* in atc",
                  pid="pid", date="start_date", backend="duckdb")
    assert sorted(r.pids) == [1, 2]


def test_duckdb_native_multi_df_tuple_input():
    r = tq.tquery((_npr(), _rx()),
                  "K50 in icd before L04AB* in atc",
                  pid="pid", date="start_date", backend="duckdb")
    assert sorted(r.pids) == [1, 2]


def test_duckdb_native_multi_df_list_input():
    r = tq.tquery([_npr(), _rx()],
                  "K50 in icd before L04AB* in atc",
                  pid="pid", date="start_date", backend="duckdb")
    assert sorted(r.pids) == [1, 2]


def test_duckdb_native_multi_df_three_sources():
    """3-way intersection across NPR / RX / procedure registry."""
    proc = pd.DataFrame({
        "pid":        [1, 2, 3],
        "start_date": pd.to_datetime(["2020-09-01", "2020-11-01", "2020-09-01"]),
        "ncsp":       ["JFB10", "FNG00", "JFB10"],
    })
    r = tq.tquery({"npr": _npr(), "rx": _rx(), "proc": proc},
                  "K50 in icd and L04AB* in atc and JFB10 in ncsp",
                  pid="pid", date="start_date", backend="duckdb")
    assert sorted(r.pids) == [1]


def test_duckdb_native_multi_df_handles_bail_out_queries():
    """Even queries that bail out of pre-filter work natively in DuckDB."""
    r = tq.tquery({"npr": _npr(), "rx": _rx()}, "not K50",
                  pid="pid", date="start_date", backend="duckdb")
    # Persons without K50: pid 3 (K51 only) and the pids only in rx (... but rx
    # pids all overlap with npr in our fixtures, so just 3).
    # The 'not' is at the pid level — anyone without K50 anywhere matches.
    assert 3 in r.pids
    assert 1 not in r.pids
    assert 2 not in r.pids


def test_duckdb_native_matches_pandas_path():
    """End-to-end parity: native DuckDB multi-DF vs. pandas combine + DuckDB."""
    npr, rx = _npr(), _rx()
    # Native path
    r_native = tq.tquery({"npr": npr, "rx": rx},
                         "K50 in icd before L04AB* in atc",
                         pid="pid", date="start_date", backend="duckdb")
    # Pre-combine + single-DF DuckDB
    combined = tq.combine({"npr": npr, "rx": rx})
    r_concat = tq.tquery(combined, "K50 in icd before L04AB* in atc",
                         pid="pid", date="start_date", backend="duckdb")
    assert sorted(r_native.pids) == sorted(r_concat.pids)


# ---------- Polars native multi-DF (pl.concat with diagonal_relaxed) ----------


def _npr_polars():
    import polars as pl
    from datetime import date
    return pl.DataFrame({
        "pid":        [1, 1, 2, 2, 3],
        "start_date": [date(2020,1,1), date(2020,6,1),
                       date(2020,2,1), date(2020,8,1), date(2020,3,1)],
        "icd":        ["K50", "K51", "K50", "K52", "K51"],
    })


def _rx_polars():
    import polars as pl
    from datetime import date
    return pl.DataFrame({
        "pid":        [1, 2, 2, 3],
        "start_date": [date(2020,9,1), date(2020,10,1),
                       date(2021,1,1), date(2020,5,1)],
        "atc":        ["L04AB02", "N02BE01", "L04AB04", "L04AB02"],
    })


def test_polars_native_multi_df_dict():
    r = tq.tquery({"npr": _npr_polars(), "rx": _rx_polars()},
                  "K50 in icd before L04AB* in atc",
                  pid="pid", date="start_date", backend="polars")
    assert sorted(r.pids) == [1, 2]


def test_polars_native_multi_df_tuple():
    r = tq.tquery((_npr_polars(), _rx_polars()),
                  "K50 in icd before L04AB* in atc",
                  pid="pid", date="start_date", backend="polars")
    assert sorted(r.pids) == [1, 2]


def test_polars_native_multi_df_mixed_pandas_polars():
    """Mixed pandas + polars input gets normalised at the boundary."""
    r = tq.tquery({"npr": _npr_polars().to_pandas(), "rx": _rx_polars()},
                  "K50 in icd before L04AB* in atc",
                  pid="pid", date="start_date", backend="polars")
    assert sorted(r.pids) == [1, 2]


def test_polars_native_multi_df_bail_out_query():
    """`not K50` falls back to full combine via _polars_combine."""
    r = tq.tquery((_npr_polars(), _rx_polars()), "not K50",
                  pid="pid", date="start_date", backend="polars")
    # pid 3 has only K51 → matches `not K50`.
    assert 3 in r.pids
    assert 1 not in r.pids
    assert 2 not in r.pids


def test_polars_native_matches_pandas_path():
    """Cross-backend parity on polars multi-DF input."""
    npr_pl, rx_pl = _npr_polars(), _rx_polars()
    r_polars = tq.tquery({"npr": npr_pl, "rx": rx_pl},
                         "K50 in icd before L04AB* in atc",
                         pid="pid", date="start_date", backend="polars")
    r_pandas = tq.tquery({"npr": npr_pl.to_pandas(), "rx": rx_pl.to_pandas()},
                         "K50 in icd before L04AB* in atc",
                         pid="pid", date="start_date")
    assert sorted(r_polars.pids) == sorted(r_pandas.pids)


def test_polars_native_config_backend_works():
    """Polars multi-DF works when backend is set via tq.set_backend."""
    tq.set_backend("polars")
    try:
        r = tq.tquery((_npr_polars(), _rx_polars()),
                      "K50 in icd before L04AB* in atc",
                      pid="pid", date="start_date")
        assert sorted(r.pids) == [1, 2]
    finally:
        tq.set_backend(None)
