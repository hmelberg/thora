"""Prefix (quantifier) evaluation for tquery.

Handles: min N, max N, exactly N, ordinal (1st/2nd/...),
first N, last N.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def eval_prefix(
    kind: str,
    n: int,
    child_mask: pd.Series,
    pid_col: pd.Series,
) -> pd.Series:
    """Evaluate a prefix quantifier on a child row mask.

    For 'min', 'max', 'exactly': returns a person-level boolean broadcast
    back to row-level (all rows for matching persons are True).

    For 'ordinal', 'first', 'last': returns a row-level mask selecting
    only specific occurrences per person.

    Args:
        kind: One of 'min', 'max', 'exactly', 'ordinal', 'first', 'last'.
        n: The numeric argument (count or position).
        child_mask: Boolean Series marking matching rows.
        pid_col: Series of person IDs (same index as child_mask).

    Returns:
        Boolean pd.Series aligned to child_mask.index.
    """
    if kind in ("min", "max", "exactly"):
        return _eval_count_prefix(kind, n, child_mask, pid_col)
    elif kind == "ordinal":
        return _eval_ordinal(n, child_mask, pid_col)
    elif kind == "first":
        return _eval_first_n(n, child_mask, pid_col)
    elif kind == "last":
        return _eval_last_n(n, child_mask, pid_col)
    else:
        raise ValueError(f"Unknown prefix kind: {kind!r}")


def eval_range_prefix(
    min_n: int,
    max_n: int,
    child_mask: pd.Series,
    pid_col: pd.Series,
) -> pd.Series:
    """Count range: persons with between min_n and max_n events (inclusive).

    Explicit-0 rule: with a literal lower bound of 0 (`0-2 of K50`),
    persons with ZERO matching events are included — marked on their
    full timeline, the same convention as `not`. A lower bound >= 1
    keeps the classic behavior (matching rows of persons with matches).
    """
    counts = child_mask.groupby(pid_col).transform("sum")
    result = child_mask & (counts >= min_n) & (counts <= max_n)
    if min_n == 0:
        result = result | (counts == 0)
    return result


def _eval_count_prefix(
    kind: str,
    n: int,
    child_mask: pd.Series,
    pid_col: pd.Series,
) -> pd.Series:
    """min/max/exactly N — person-level count condition broadcast to rows.

    Explicit-0 rule: when the user writes a literal 0 (`min 0`, `max 0`,
    `exactly 0`), persons with ZERO matching events are included and
    marked on their full timeline (the `not` convention) — `exactly 0 of
    K50` ≡ `not K50`. With n >= 1 the classic behavior stands: `max 2 of
    K50` requires at least one K50 and marks only the matching rows.
    """
    counts = child_mask.groupby(pid_col).transform("sum")
    if kind == "min":
        result = child_mask & (counts >= n)
    elif kind == "max":
        result = child_mask & (counts <= n)
    else:  # exactly
        result = child_mask & (counts == n)
    if n == 0:
        result = result | (counts == 0)
    return result


def _eval_ordinal(
    n: int,
    child_mask: pd.Series,
    pid_col: pd.Series,
) -> pd.Series:
    """Select only the nth occurrence per person (1-based).

    Positive n counts from the start (1st, 2nd, ...); negative n counts
    from the end (-1st = last, -2nd = second-to-last, ...).
    """
    cumcount = child_mask.groupby(pid_col).cumsum()
    if n > 0:
        return child_mask & (cumcount == n)
    # Negative: reverse_pos = total − cumcount + 1
    # (1 = last match, 2 = 2nd-to-last, ...)
    total = child_mask.groupby(pid_col).transform("sum")
    reverse_pos = total - cumcount + 1
    return child_mask & (reverse_pos == abs(n))


def _eval_first_n(
    n: int,
    child_mask: pd.Series,
    pid_col: pd.Series,
) -> pd.Series:
    """Select the first N occurrences per person."""
    cumcount = child_mask.groupby(pid_col).cumsum()
    return child_mask & (cumcount <= n)


def _eval_last_n(
    n: int,
    child_mask: pd.Series,
    pid_col: pd.Series,
) -> pd.Series:
    """Select the last N occurrences per person."""
    # Reverse cumulative count: total per person minus cumcount gives
    # how many remain after this row. We want those where remaining < n.
    cumcount = child_mask.groupby(pid_col).cumsum()
    total = child_mask.groupby(pid_col).transform("sum")
    # The kth occurrence from the end has cumcount == total - k + 1
    # We want the last n, i.e., cumcount > total - n
    return child_mask & (cumcount > total - n)
