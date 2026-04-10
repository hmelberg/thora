"""Incidence calculation with bias correction.

Provides annual incidence (new-case-per-year) counts from event-level
register data, with optional corrections for two systematic biases:

* **Left-censoring (washout bias)** — patients whose disease started
  before the data begins look like new cases in the early years of the
  observation window.
* **Right-censoring (forward bias)** — under a "≥N events" case
  definition, recent single-event patients haven't yet had time to
  accumulate confirming events.

The module follows tquery conventions: it resolves ``pid`` / ``date`` /
``cols`` / ``sep`` via the active :class:`TQueryConfig`, accepts an
optional ``expr`` to filter rows using the query language, and uses
:func:`tquery._codes.get_matching_rows` indirectly via the main
:func:`tquery` evaluator. ``scipy`` is only imported lazily inside
:func:`fit_decay` (and the ``functional`` adjustment paths).
"""

from __future__ import annotations

from typing import Any, Callable

import numpy as np
import pandas as pd

from tquery._types import TQueryConfig, _merge_kwargs

# A small constant year length, to convert "K years of lookback" to days
# without seasonal noise. Calendar-year boundaries are still respected
# elsewhere; this only matters when reading a fitted curve at a "1 year
# of lookback" point.
_DAYS_PER_YEAR = 365.25


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _filter_df(
    df: pd.DataFrame,
    expr: str | None,
    eval_kw: dict[str, Any],
) -> pd.DataFrame:
    """Apply an optional tquery expression to select matching rows.

    A None expression returns the DataFrame unchanged.
    """
    if expr is None:
        return df
    # Local import to avoid an import cycle (tquery.__init__ imports
    # this module).
    from tquery import tquery as _tquery
    result = _tquery(df, expr, **eval_kw)
    return df[result.rows]


def _resolve_kwargs(
    config: TQueryConfig | None,
    *,
    pid: str | None,
    date: str | None,
    cols: str | list[str] | None,
    sep: str | None,
    variables: dict[str, Any] | None,
) -> dict[str, Any]:
    """Resolve incidence kwargs against an optional config.

    Returns a dict containing ``pid`` and ``date`` always, plus any of
    ``cols``, ``sep``, ``variables`` that were set. Suitable for passing
    to :func:`tquery.tquery`.
    """
    return _merge_kwargs(
        config, pid=pid, date=date, cols=cols, sep=sep, variables=variables,
    )


def _first_event_year(
    df: pd.DataFrame,
    pid_col: str,
    date_col: str,
) -> pd.Series:
    """Year (int) of each person's first matching event."""
    first = df.groupby(pid_col)[date_col].min()
    return first.dt.year.astype(int)


def _restrict_to_qualifying(
    df: pd.DataFrame,
    pid_col: str,
    required_events: int,
) -> pd.DataFrame:
    """Drop persons with fewer than ``required_events`` rows in df."""
    if required_events <= 1:
        return df
    counts = df.groupby(pid_col).size()
    keep = counts.index[counts >= required_events]
    return df[df[pid_col].isin(keep)]


# ---------------------------------------------------------------------------
# Decay model registry (no scipy at import time)
# ---------------------------------------------------------------------------

def _exponential(x: np.ndarray, a: float, b: float) -> np.ndarray:
    """a + (1 - a) * exp(-b * x / 365.25). Decays from 1 to a."""
    return a + (1.0 - a) * np.exp(-b * x / _DAYS_PER_YEAR)


def _hyperbolic(x: np.ndarray, a: float, b: float, c: float) -> np.ndarray:
    """a + b / (1 + c * x). Decays from a + b to a."""
    return a + b / (1.0 + c * x)


def _rational(x: np.ndarray, a: float, b: float, c: float) -> np.ndarray:
    """(a*x + b) / (x + c). Asymptote a as x -> inf."""
    return (a * x + b) / (x + c)


def _exponential_asymptote(coeffs: np.ndarray) -> float:
    return float(coeffs[0])


def _hyperbolic_asymptote(coeffs: np.ndarray) -> float:
    return float(coeffs[0])


def _rational_asymptote(coeffs: np.ndarray) -> float:
    return float(coeffs[0])


_DECAY_MODELS: dict[str, dict[str, Any]] = {
    "exponential": {
        "func": _exponential,
        "asymptote": _exponential_asymptote,
        "p0": [0.5, 1.0],
    },
    "hyperbolic": {
        "func": _hyperbolic,
        "asymptote": _hyperbolic_asymptote,
        "p0": [0.5, 0.5, 0.001],
    },
    "rational": {
        "func": _rational,
        "asymptote": _rational_asymptote,
        "p0": [0.5, 1.0, 100.0],
    },
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def raw_incidence(
    df: pd.DataFrame,
    expr: str | None = None,
    *,
    required_events: int = 1,
    pid: str | None = None,
    date: str | None = None,
    cols: str | list[str] | None = None,
    sep: str | None = None,
    variables: dict[str, Any] | None = None,
    config: TQueryConfig | None = None,
) -> pd.Series:
    """Number of new (first-event) cases per calendar year.

    A person is counted in year *y* if their *first* matching event in
    the DataFrame falls in year *y*. With ``required_events > 1`` a
    person must additionally have at least that many matching events
    anywhere in the data to be counted at all.

    No bias correction is applied — see :func:`incidence` for the
    corrected version.

    Args:
        df: Event-level DataFrame.
        expr: Optional tquery expression filtering rows before counting.
            ``None`` (default) counts every row.
        required_events: Minimum number of matching events for a person
            to count as a case. Default ``1``.
        pid, date, cols, sep, variables, config: Same as
            :func:`tquery.tquery`.

    Returns:
        Series indexed by year (int), values are integer person counts,
        sorted by year.
    """
    kw = _resolve_kwargs(
        config, pid=pid, date=date, cols=cols, sep=sep, variables=variables,
    )
    pid_col = kw["pid"]
    date_col = kw["date"]

    eval_kw = {k: v for k, v in kw.items()
               if k in ("pid", "date", "cols", "sep", "variables")}

    sub = _filter_df(df, expr, eval_kw)
    sub = _restrict_to_qualifying(sub, pid_col, required_events)

    if sub.empty:
        return pd.Series(dtype=int, name="raw_incidence")

    first_year = _first_event_year(sub, pid_col, date_col)
    out = first_year.value_counts().sort_index()
    out.index = out.index.astype(int)
    out.name = "raw_incidence"
    return out


def washout_pattern(
    df: pd.DataFrame,
    expr: str | None = None,
    *,
    year: int | list[int] | None = None,
    step_days: int = 200,
    pct: bool = False,
    required_events: int = 1,
    pid: str | None = None,
    date: str | None = None,
    cols: str | list[str] | None = None,
    sep: str | None = None,
    variables: dict[str, Any] | None = None,
    config: TQueryConfig | None = None,
) -> pd.Series | pd.DataFrame:
    """Empirical lookback decay for one or all years.

    For a target ``year``, the persons "appearing as new cases" are all
    persons with at least one matching event that year (after the
    optional ``required_events`` filter). The function then expands the
    available lookback window in ``step_days`` increments and reports
    how many of those persons still appear new — i.e. have NO earlier
    matching events within the lookback window.

    The asymptote of this curve at maximum lookback equals the value
    that :func:`raw_incidence` would return for that year (only persons
    whose first event in the entire data is in ``year``).

    Args:
        df: Event-level DataFrame.
        expr: Optional tquery expression filtering rows.
        year: A single year or list of years. ``None`` (default) runs
            every year present in the data after the earliest year (the
            earliest year has no lookback at all).
        step_days: Increment used to grow the lookback window.
        pct: If True, return decay as a fraction of the year-0 count.
        required_events: Persons must have at least this many matching
            events overall to be considered.
        pid, date, cols, sep, variables, config: As elsewhere.

    Returns:
        For a single year, a Series indexed by lookback days. For
        multiple years, a DataFrame indexed by lookback days with one
        column per year.
    """
    kw = _resolve_kwargs(
        config, pid=pid, date=date, cols=cols, sep=sep, variables=variables,
    )
    pid_col = kw["pid"]
    date_col = kw["date"]
    eval_kw = {k: v for k, v in kw.items()
               if k in ("pid", "date", "cols", "sep", "variables")}

    sub = _filter_df(df, expr, eval_kw)
    sub = _restrict_to_qualifying(sub, pid_col, required_events)

    if sub.empty:
        return pd.Series(dtype=float, name="washout_pattern")

    sub_years = sub[date_col].dt.year

    if year is None:
        min_year = int(sub_years.min())
        years = sorted(int(y) for y in sub_years.unique() if y > min_year)
    elif isinstance(year, (list, tuple)):
        years = [int(y) for y in year]
    else:
        years = None  # single-year branch
        target_year = int(year)

    if years is not None:
        per_year = {
            y: _washout_for_year(
                sub, pid_col, date_col,
                target_year=y, step_days=step_days, pct=pct,
            )
            for y in years
        }
        df_out = pd.DataFrame(per_year)
        df_out.index.name = "lookback_days"
        return df_out

    s = _washout_for_year(
        sub, pid_col, date_col,
        target_year=target_year, step_days=step_days, pct=pct,
    )
    s.name = f"washout_{target_year}"
    return s


def _washout_for_year(
    df: pd.DataFrame,
    pid_col: str,
    date_col: str,
    *,
    target_year: int,
    step_days: int,
    pct: bool,
) -> pd.Series:
    """Empirical washout decay for a single year."""
    in_year = df[date_col].dt.year == target_year
    patient_pids = pd.unique(df.loc[in_year, pid_col])
    n_patients = patient_pids.size

    if n_patients == 0:
        return pd.Series(dtype=float)

    year_start = pd.Timestamp(year=target_year, month=1, day=1)
    earliest = df[date_col].min()
    max_lookback = max(0, (year_start - earliest).days)

    # Collect history rows for those patients only
    hist_mask = df[pid_col].isin(patient_pids) & (df[date_col] < year_start)
    hist = df.loc[hist_mask, [pid_col, date_col]].copy()
    hist["days_back"] = (year_start - hist[date_col]).dt.days

    out: dict[int, float] = {}
    out[0] = 1.0 if pct else float(n_patients)

    if max_lookback > 0:
        steps = list(range(step_days, max_lookback + 1, step_days))
        if not steps or steps[-1] != max_lookback:
            steps.append(max_lookback)
        hist_sorted = hist.sort_values("days_back")
        for d in steps:
            with_history = hist_sorted.loc[hist_sorted["days_back"] <= d, pid_col]
            n_with_history = with_history.nunique()
            n_still_new = n_patients - n_with_history
            out[d] = (n_still_new / n_patients) if pct else float(n_still_new)

    s = pd.Series(out, dtype=float)
    s.index.name = "lookback_days"
    return s


def singles_pattern(
    df: pd.DataFrame,
    expr: str | None = None,
    *,
    year: int | list[int] | None = None,
    step_days: int = 200,
    required_events: int = 2,
    pct: bool = False,
    pid: str | None = None,
    date: str | None = None,
    cols: str | list[str] | None = None,
    sep: str | None = None,
    variables: dict[str, Any] | None = None,
    config: TQueryConfig | None = None,
) -> pd.Series | pd.DataFrame:
    """Empirical decay of "still under-counted" persons.

    For a target year, persons with **fewer than** ``required_events``
    matching events that year are the candidates — they look like
    they may not satisfy the case definition. The function then expands
    the observation window symmetrically in ``step_days`` increments
    (forward AND backward) and reports how many of those candidates
    *still* have fewer than ``required_events`` events in the expanded
    window. The asymptote of this decay is the "true singleton" rate.

    Used to estimate the forward-bias correction in :func:`incidence`
    when ``required_events >= 2``.

    Args:
        df: Event-level DataFrame.
        expr: Optional tquery expression filtering rows.
        year: A single year, a list of years, or ``None`` for all years
            with sufficient surrounding data.
        step_days: Symmetric expansion increment (in days).
        required_events: The case-definition threshold. Persons with
            strictly fewer than this many events are candidates.
        pct: If True, return decay as a fraction of the year-0 count.
        pid, date, cols, sep, variables, config: As elsewhere.

    Returns:
        For a single year, a Series indexed by expansion days. For
        multiple years, a DataFrame indexed by expansion days with one
        column per year.
    """
    kw = _resolve_kwargs(
        config, pid=pid, date=date, cols=cols, sep=sep, variables=variables,
    )
    pid_col = kw["pid"]
    date_col = kw["date"]
    eval_kw = {k: v for k, v in kw.items()
               if k in ("pid", "date", "cols", "sep", "variables")}

    sub = _filter_df(df, expr, eval_kw)
    if sub.empty:
        return pd.Series(dtype=float, name="singles_pattern")

    sub_years = sub[date_col].dt.year

    if year is None:
        min_y = int(sub_years.min())
        max_y = int(sub_years.max())
        years = list(range(min_y + 1, max_y))
    elif isinstance(year, (list, tuple)):
        years = [int(y) for y in year]
    else:
        years = None
        target_year = int(year)

    if years is not None:
        per_year = {
            y: _singles_for_year(
                sub, pid_col, date_col,
                target_year=y, step_days=step_days,
                required_events=required_events, pct=pct,
            )
            for y in years
        }
        df_out = pd.DataFrame(per_year)
        df_out.index.name = "expand_days"
        return df_out

    s = _singles_for_year(
        sub, pid_col, date_col,
        target_year=target_year, step_days=step_days,
        required_events=required_events, pct=pct,
    )
    s.name = f"singles_{target_year}"
    return s


def _singles_for_year(
    df: pd.DataFrame,
    pid_col: str,
    date_col: str,
    *,
    target_year: int,
    step_days: int,
    required_events: int,
    pct: bool,
) -> pd.Series:
    """Empirical singles decay for a single year."""
    in_year = df[date_col].dt.year == target_year
    counts_in_year = df.loc[in_year].groupby(pid_col).size()
    candidates = counts_in_year.index[counts_in_year < required_events]
    n_candidates = int(candidates.size)

    if n_candidates == 0:
        return pd.Series(dtype=float)

    year_start = pd.Timestamp(year=target_year, month=1, day=1)
    year_end = pd.Timestamp(year=target_year, month=12, day=31)
    min_date = df[date_col].min()
    max_date = df[date_col].max()

    cand_set = set(candidates)
    cand_df = df[df[pid_col].isin(cand_set)][[pid_col, date_col]]

    out: dict[int, float] = {}
    out[0] = 1.0 if pct else float(n_candidates)

    expand = step_days
    while True:
        start_w = year_start - pd.Timedelta(days=expand)
        end_w = year_end + pd.Timedelta(days=expand)
        if start_w < min_date or end_w > max_date:
            break
        in_window = (cand_df[date_col] >= start_w) & (cand_df[date_col] <= end_w)
        counts_w = cand_df.loc[in_window].groupby(pid_col).size()
        # Persons in candidates with < required_events in the window.
        # Use reindex to count zero-event candidates as 0.
        counts_w = counts_w.reindex(list(cand_set), fill_value=0)
        still_n = int((counts_w < required_events).sum())
        out[expand] = (still_n / n_candidates) if pct else float(still_n)
        expand += step_days

    s = pd.Series(out, dtype=float)
    s.index.name = "expand_days"
    return s


# ---------------------------------------------------------------------------
# Curve fitting
# ---------------------------------------------------------------------------

def _fit_one(
    xdata: np.ndarray,
    yvals: np.ndarray,
    model: str,
    curve_fit: Callable,
) -> dict[str, Any]:
    """Fit one decay model and compute AIC + R² goodness-of-fit metrics."""
    spec = _DECAY_MODELS[model]
    func: Callable = spec["func"]

    coeffs, _ = curve_fit(func, xdata, yvals, p0=spec["p0"], maxfev=10000)
    asymptote = spec["asymptote"](coeffs)

    pred = func(xdata, *coeffs)
    residuals = yvals - pred
    rss = float(np.sum(residuals ** 2))
    n = int(yvals.size)
    k = int(coeffs.size)
    if rss > 0 and n > 0:
        aic = n * float(np.log(rss / n)) + 2 * k
    else:
        aic = float("-inf")
    tss = float(np.sum((yvals - yvals.mean()) ** 2))
    r2 = (1.0 - rss / tss) if tss > 0 else float("nan")

    def predict(x: np.ndarray | pd.Series | float) -> np.ndarray:
        x_arr = np.asarray(x, dtype=float)
        return func(x_arr, *coeffs)

    return {
        "model": model,
        "coeffs": coeffs,
        "asymptote": float(asymptote),
        "predict": predict,
        "aic": float(aic),
        "r2": float(r2),
    }


def fit_decay(
    pattern: pd.Series | pd.DataFrame,
    model: str = "exponential",
) -> dict[str, Any]:
    """Fit a parametric decay curve to an empirical pattern.

    Lazy-imports :mod:`scipy.optimize`. Install with
    ``pip install tquery[incidence]`` if missing.

    Args:
        pattern: Either a Series indexed by days, or a DataFrame indexed
            by days with one column per year — in the DataFrame case the
            mean across columns is fitted.
        model: One of ``"exponential"``, ``"hyperbolic"``, ``"rational"``,
            or ``"all"``. The ``"all"`` mode fits every model and
            returns a dict keyed by model name (useful for choosing
            between them via AIC).

    Returns:
        For a single model, a dict with keys:

        * ``model`` — the model name
        * ``coeffs`` — fitted parameter array
        * ``asymptote`` — long-x limit of the curve
        * ``predict`` — callable taking a days array and returning the
          fitted curve values
        * ``aic`` — Akaike information criterion (lower is better)
        * ``r2`` — coefficient of determination

        For ``model="all"``, a dict mapping each model name to the
        per-model dict above. If a model fails to fit, its entry is
        ``{"model": name, "error": <message>}``.
    """
    valid = set(_DECAY_MODELS) | {"all"}
    if model not in valid:
        raise ValueError(
            f"Unknown decay model '{model}'. "
            f"Choose from {sorted(_DECAY_MODELS)} or 'all'."
        )

    try:
        from scipy.optimize import curve_fit
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "fit_decay requires scipy. Install with "
            "`pip install tquery[incidence]` or `pip install scipy`."
        ) from exc

    if isinstance(pattern, pd.DataFrame):
        ydata = pattern.mean(axis=1).dropna()
    else:
        ydata = pattern.dropna()

    if ydata.empty:
        raise ValueError("Cannot fit a decay model to an empty pattern.")

    xdata = np.asarray(ydata.index, dtype=float)
    yvals = np.asarray(ydata.values, dtype=float)

    if model == "all":
        results: dict[str, Any] = {}
        for name in _DECAY_MODELS:
            try:
                results[name] = _fit_one(xdata, yvals, name, curve_fit)
            except Exception as exc:  # noqa: BLE001
                results[name] = {"model": name, "error": str(exc)}
        return results

    return _fit_one(xdata, yvals, model, curve_fit)


# ---------------------------------------------------------------------------
# Adjustment math
# ---------------------------------------------------------------------------

def _functional_washout_adjust(
    raw: pd.Series,
    decay: dict[str, Any],
) -> pd.Series:
    """Divide raw counts by the decay value at each year's lookback.

    Year ``y`` has ``(y - y0) * 365.25`` days of available lookback,
    where ``y0`` is the earliest year in ``raw``. The decay value at
    that lookback is the fraction of new cases that still appear new;
    dividing recovers an estimate biased by only the asymptote, then
    rescaling to the asymptote produces the corrected count.
    """
    if raw.empty:
        return raw.astype(float)
    years = raw.index.to_numpy()
    y0 = int(years.min())
    lookback_days = (years - y0) * _DAYS_PER_YEAR
    decay_at = decay["predict"](lookback_days)
    asymptote = decay["asymptote"]
    # Avoid division by zero — clip to a small positive value.
    decay_at = np.clip(decay_at, 1e-9, None)
    adjusted = raw.values * (asymptote / decay_at)
    out = pd.Series(adjusted, index=raw.index, dtype=float)
    out.name = "incidence"
    return out


def _historical_washout_adjust(
    raw: pd.Series,
    pattern: pd.DataFrame,
) -> pd.Series:
    """Apply the empirical mean decay year-by-year (no curve fit).

    Treats the year offset (year - earliest_year) as the index into
    the empirical decay, picking the closest available lookback step.
    """
    if raw.empty:
        return raw.astype(float)
    avg = pattern.mean(axis=1).dropna()
    if avg.empty:
        return raw.astype(float)
    years = raw.index.to_numpy()
    y0 = int(years.min())
    lookback_days = (years - y0) * _DAYS_PER_YEAR

    # Lookup nearest empirical decay value
    keys = np.asarray(avg.index, dtype=float)
    vals = avg.values
    decay_at = np.empty_like(lookback_days, dtype=float)
    for i, ld in enumerate(lookback_days):
        idx = int(np.argmin(np.abs(keys - ld)))
        decay_at[i] = vals[idx]
    asymptote = float(vals[-1])
    decay_at = np.clip(decay_at, 1e-9, None)
    adjusted = raw.values * (asymptote / decay_at)
    out = pd.Series(adjusted, index=raw.index, dtype=float)
    out.name = "incidence"
    return out


def _first_year_under_threshold(
    df: pd.DataFrame,
    pid_col: str,
    date_col: str,
    required_events: int,
) -> pd.Series:
    """Persons whose *first* event is in year y, with fewer than
    ``required_events`` events in year y itself.

    Indexed by year, integer counts. This definition is censoring-free
    (only depends on events in each person's first observed year), so
    it can be applied uniformly to early and late years for the forward
    bias correction.
    """
    first_year = df.groupby(pid_col)[date_col].min().dt.year.astype(int)
    yr = df[date_col].dt.year.astype(int)
    person_fy = df[pid_col].map(first_year)
    in_first_year = yr == person_fy
    counts_in_first = df[in_first_year].groupby(pid_col).size()
    under_pids = counts_in_first.index[counts_in_first < required_events]
    fy_of_under = first_year.loc[under_pids]
    if fy_of_under.empty:
        return pd.Series(dtype=int)
    out = fy_of_under.value_counts().sort_index().astype(int)
    out.index = out.index.astype(int)
    return out


def _functional_lookahead_adjust(
    raw_incidence_series: pd.Series,
    df: pd.DataFrame,
    pid_col: str,
    date_col: str,
    required_events: int,
    *,
    step_days: int,
    model: str,
) -> pd.Series:
    """Subtract estimated true-singleton contribution from raw incidence.

    The asymptote of the singles_pattern decay estimates the fraction of
    in-year singletons that NEVER accumulate enough events. Multiplied
    by the count of first-year-singletons per year, this gives an
    estimate of the persons who will remain under the case-definition
    threshold, which we subtract from the raw counts.
    """
    s_pat = singles_pattern(
        df,
        pct=True,
        step_days=step_days,
        required_events=required_events,
        pid=pid_col,
        date=date_col,
    )
    if isinstance(s_pat, pd.Series):
        s_pat_df = s_pat.to_frame()
    else:
        s_pat_df = s_pat
    if s_pat_df.empty:
        return raw_incidence_series.astype(float)
    decay = fit_decay(s_pat_df, model=model)
    asymptote = decay["asymptote"]

    n_under = _first_year_under_threshold(
        df, pid_col, date_col, required_events
    )
    n_under = n_under.reindex(raw_incidence_series.index, fill_value=0)
    estimated_singles = asymptote * n_under
    adjusted = raw_incidence_series.astype(float) - estimated_singles
    adjusted = adjusted.clip(lower=0)
    adjusted.name = "incidence"
    return adjusted


def _historical_lookahead_adjust(
    raw_incidence_series: pd.Series,
    df: pd.DataFrame,
    pid_col: str,
    date_col: str,
    required_events: int,
    *,
    step_days: int,
) -> pd.Series:
    """Empirical analogue: use the minimum observed singles fraction."""
    s_pat = singles_pattern(
        df,
        pct=True,
        step_days=step_days,
        required_events=required_events,
        pid=pid_col,
        date=date_col,
    )
    if isinstance(s_pat, pd.Series):
        s_pat_df = s_pat.to_frame()
    else:
        s_pat_df = s_pat
    if s_pat_df.empty:
        return raw_incidence_series.astype(float)
    asymptote = float(s_pat_df.mean(axis=1).min())
    n_under = _first_year_under_threshold(
        df, pid_col, date_col, required_events
    )
    n_under = n_under.reindex(raw_incidence_series.index, fill_value=0)
    estimated_singles = asymptote * n_under
    adjusted = raw_incidence_series.astype(float) - estimated_singles
    adjusted = adjusted.clip(lower=0)
    adjusted.name = "incidence"
    return adjusted


# ---------------------------------------------------------------------------
# Master function
# ---------------------------------------------------------------------------

def incidence(
    df: pd.DataFrame,
    expr: str | None = None,
    *,
    required_events: int = 1,
    washout: str = "functional",
    lookahead: str = "auto",
    model: str = "exponential",
    step_days: int = 365,
    pid: str | None = None,
    date: str | None = None,
    cols: str | list[str] | None = None,
    sep: str | None = None,
    variables: dict[str, Any] | None = None,
    config: TQueryConfig | None = None,
) -> pd.Series:
    """Bias-corrected annual incidence.

    Combines :func:`raw_incidence` with washout (left-censoring) and
    optional forward (right-censoring) corrections.

    Args:
        df: Event-level DataFrame.
        expr: Optional tquery expression filtering rows.
        required_events: Case-definition threshold. Default ``1``.
        washout: One of ``"none"``, ``"historical"``, ``"functional"``
            (default). ``"functional"`` fits a parametric decay curve to
            the empirical washout pattern; ``"historical"`` uses the
            empirical mean directly.
        lookahead: One of ``"none"``, ``"historical"``, ``"functional"``,
            ``"auto"`` (default). ``"auto"`` enables the functional
            forward correction iff ``required_events >= 2``.
        model: Curve model name passed to :func:`fit_decay`.
        step_days: Step used when computing the empirical patterns.
        pid, date, cols, sep, variables, config: As elsewhere.

    Returns:
        A float Series indexed by year, holding the corrected counts.
    """
    if washout not in {"none", "historical", "functional"}:
        raise ValueError(
            f"washout must be 'none', 'historical' or 'functional'; got {washout!r}"
        )
    if lookahead not in {"none", "historical", "functional", "auto"}:
        raise ValueError(
            f"lookahead must be 'none', 'historical', 'functional', or 'auto'; "
            f"got {lookahead!r}"
        )

    kw = _resolve_kwargs(
        config, pid=pid, date=date, cols=cols, sep=sep, variables=variables,
    )
    pid_col = kw["pid"]
    date_col = kw["date"]
    eval_kw = {k: v for k, v in kw.items()
               if k in ("pid", "date", "cols", "sep", "variables")}

    sub_full = _filter_df(df, expr, eval_kw)

    # Decide effective lookahead and whether we need the "count everyone,
    # subtract estimated singletons" semantics. With lookahead enabled and
    # required_events >= 2 we count first events of *all* persons (re=1)
    # so the result is uncensored, then subtract the estimated true
    # singletons. Without lookahead we use the standard re=N restriction.
    effective_lookahead = lookahead
    if lookahead == "auto":
        effective_lookahead = "functional" if required_events >= 2 else "none"
    use_lookahead = (effective_lookahead != "none") and (required_events >= 2)

    if use_lookahead:
        sub = sub_full
    else:
        sub = _restrict_to_qualifying(sub_full, pid_col, required_events)

    raw = raw_incidence(
        sub,
        required_events=1,
        pid=pid_col,
        date=date_col,
    ).astype(float)

    if raw.empty:
        return raw

    # ---- Washout (left-censoring) ----
    if washout == "none":
        result = raw.copy()
    else:
        pat = washout_pattern(
            sub, pct=True, step_days=step_days,
            pid=pid_col, date=date_col,
        )
        pat_df = pat.to_frame() if isinstance(pat, pd.Series) else pat
        if pat_df.empty:
            result = raw.copy()
        elif washout == "functional":
            decay = fit_decay(pat_df, model=model)
            result = _functional_washout_adjust(raw, decay)
        else:  # historical
            result = _historical_washout_adjust(raw, pat_df)

    # ---- Lookahead (right-censoring) ----
    if not use_lookahead:
        result.name = "incidence"
        return result

    if effective_lookahead == "functional":
        result = _functional_lookahead_adjust(
            result, sub_full, pid_col, date_col, required_events,
            step_days=step_days, model=model,
        )
    else:  # historical
        result = _historical_lookahead_adjust(
            result, sub_full, pid_col, date_col, required_events,
            step_days=step_days,
        )
    return result
