"""Temporal operations for tquery: before/after, within, inside/outside.

All functions operate on boolean masks (pd.Series) aligned to the
input DataFrame's index. They never mutate the DataFrame.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def eval_before_after(
    df: pd.DataFrame,
    left_mask: pd.Series,
    right_mask: pd.Series,
    op: str,
    pid_col: str,
    date_col: str,
) -> pd.Series:
    """Person-level temporal comparison: does left occur before/after right?

    Returns a row-level boolean mask where all rows for matching persons
    are marked True (for the left-hand side events).

    For 'before': persons where min date of left < min date of right.
    For 'after': persons where min date of left > min date of right.
    For 'simultaneously': persons where any left date == any right date.
    """
    if not left_mask.any() or not right_mask.any():
        return pd.Series(False, index=df.index)

    pid = df[pid_col]
    dates = df[date_col]

    if op == "simultaneously":
        # Persons who have both left and right events on the same date
        left_dates = dates[left_mask].groupby(pid[left_mask]).apply(set)
        right_dates = dates[right_mask].groupby(pid[right_mask]).apply(set)
        common = left_dates.index.intersection(right_dates.index)
        if common.empty:
            return pd.Series(False, index=df.index)
        matching_pids = set()
        for p in common:
            if left_dates[p] & right_dates[p]:
                matching_pids.add(p)
        return pid.isin(matching_pids) & left_mask
    else:
        # before / after: compare first occurrence dates
        left_first = dates[left_mask].groupby(pid[left_mask]).first()
        right_first = dates[right_mask].groupby(pid[right_mask]).first()

        # Align on common persons who have both events
        common_pids = left_first.index.intersection(right_first.index)
        if common_pids.empty:
            return pd.Series(False, index=df.index)

        left_first = left_first.reindex(common_pids)
        right_first = right_first.reindex(common_pids)

        if op == "before":
            matching = left_first < right_first
        else:  # after
            matching = left_first > right_first

        matching_pids = set(matching.index[matching])
        return pid.isin(matching_pids) & left_mask


def eval_within_days(
    df: pd.DataFrame,
    child_mask: pd.Series,
    ref_mask: pd.Series | None,
    days: int,
    direction: str | None,
    pid_col: str,
    date_col: str,
    min_days: int = 0,
) -> pd.Series:
    """Row-level: mark child rows within a day range of reference events.

    Args:
        days: Maximum days distance (upper bound).
        min_days: Minimum days distance (lower bound, for 'between M and N days').
        direction: 'before', 'after', 'around', or None.

    If ref_mask is None, this is a standalone time window relative to
    the first event per person.
    """
    if not child_mask.any():
        return pd.Series(False, index=df.index)

    pid = df[pid_col]
    dates = df[date_col]

    if ref_mask is None:
        # No reference: within day range of first event per person
        first_date = dates.groupby(pid).transform("first")
        diff = (dates - first_date).dt.days.abs()
        return child_mask & (diff >= min_days) & (diff <= days)

    if not ref_mask.any():
        return pd.Series(False, index=df.index)

    # Use merge_asof for efficient nearest-event matching (max distance)
    child_df = pd.DataFrame({
        "pid": pid[child_mask].values,
        "date": dates[child_mask].values,
        "orig_idx": child_mask.index[child_mask],
    }).sort_values("date")

    ref_df = pd.DataFrame({
        "pid": pid[ref_mask].values,
        "ref_date": dates[ref_mask].values,
    }).sort_values("ref_date")

    if direction == "after":
        merged = pd.merge_asof(
            child_df, ref_df,
            by="pid", left_on="date", right_on="ref_date",
            direction="backward",
            tolerance=pd.Timedelta(days=days),
        )
    elif direction == "before":
        merged = pd.merge_asof(
            child_df, ref_df,
            by="pid", left_on="date", right_on="ref_date",
            direction="forward",
            tolerance=pd.Timedelta(days=days),
        )
    else:
        merged = pd.merge_asof(
            child_df, ref_df,
            by="pid", left_on="date", right_on="ref_date",
            direction="nearest",
            tolerance=pd.Timedelta(days=days),
        )

    # Apply max distance filter (merge_asof handles this via tolerance)
    has_match = merged["ref_date"].notna()

    # Apply min distance filter (merge_asof can't do this)
    if min_days > 0 and has_match.any():
        day_diff = (merged["date"] - merged["ref_date"]).dt.days.abs()
        has_match = has_match & (day_diff >= min_days)

    matched_idx = merged.loc[has_match, "orig_idx"].values
    result = pd.Series(False, index=df.index)
    result.iloc[result.index.get_indexer(matched_idx)] = True
    return result


def eval_inside_outside(
    df: pd.DataFrame,
    child_mask: pd.Series,
    ref_mask: pd.Series,
    inside: bool,
    n_events: int,
    direction: str,
    pid_col: str,
) -> pd.Series:
    """Row-level: mark child rows inside/outside N events of reference events.

    'inside 5 events after K51' means: within the next 5 events (rows)
    after any K51 occurrence for that person.
    """
    if not child_mask.any() or not ref_mask.any():
        return pd.Series(False, index=df.index)

    pid = df[pid_col]
    result = pd.Series(False, index=df.index)

    # Compute event number per person
    event_num = pid.groupby(pid).cumcount()

    # For each person, find reference event positions
    ref_positions = event_num[ref_mask]
    ref_pids = pid[ref_mask]

    # For each person, check if child events are within n_events
    for p in ref_pids.unique():
        p_mask = pid == p
        p_event_num = event_num[p_mask]
        p_child = child_mask[p_mask]
        p_ref_positions = ref_positions[ref_pids == p].values

        in_window = pd.Series(False, index=p_event_num.index)
        for ref_pos in p_ref_positions:
            if direction == "after":
                in_window = in_window | (
                    (p_event_num > ref_pos) & (p_event_num <= ref_pos + n_events)
                )
            elif direction == "before":
                in_window = in_window | (
                    (p_event_num < ref_pos) & (p_event_num >= ref_pos - n_events)
                )
            else:  # around
                in_window = in_window | (
                    (p_event_num - ref_pos).abs() <= n_events
                )

        if inside:
            result[p_mask] = p_child & in_window
        else:
            result[p_mask] = p_child & ~in_window

    return result
