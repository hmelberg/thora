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
    every_left: bool = False,
    every_right: bool = False,
    left_offset_days: int = 0,
    right_offset_days: int = 0,
) -> pd.Series:
    """Person-level temporal comparison: does left occur before/after right?

    Returns a row-level boolean mask where all rows for matching persons
    are marked True (for the left-hand side events).

    Existential semantics (default):
        before: min(left) < max(right)  — some left precedes some right
        after:  max(left) > min(right)  — some left follows some right
        simultaneously: any left date == any right date

    Universal semantics (every_left / every_right):
        For 'after K51':
            every_right=True  → max(K51) < max(K50): every K51 is followed by some K50
            every_left=True   → min(K50) > min(K51): every K50 is preceded by some K51
            both              → min(K50) > max(K51): every K50 follows every K51

        'before' is symmetric (swap roles).

        'simultaneously' universal: every event on the quantified side has a
        same-date partner on the other side.

    Universal sides additionally require non-empty events for that side
    (no vacuous truth: a person with no K51 does NOT satisfy `every K51`).
    """
    if not left_mask.any() or not right_mask.any():
        return pd.Series(False, index=df.index)

    pid = df[pid_col]
    dates = df[date_col]
    left_shift = pd.Timedelta(days=left_offset_days)
    right_shift = pd.Timedelta(days=right_offset_days)

    if op == "simultaneously":
        # Build per-person date sets for both sides (offsets applied)
        left_dates = (dates[left_mask] + left_shift).groupby(pid[left_mask]).apply(set)
        right_dates = (dates[right_mask] + right_shift).groupby(pid[right_mask]).apply(set)
        common = left_dates.index.intersection(right_dates.index)
        if common.empty:
            return pd.Series(False, index=df.index)
        matching_pids: set = set()
        for p in common:
            l = left_dates[p]
            r = right_dates[p]
            if every_left and not l.issubset(r):
                continue
            if every_right and not r.issubset(l):
                continue
            if not every_left and not every_right and not (l & r):
                continue
            matching_pids.add(p)
        return pid.isin(matching_pids) & left_mask

    # before / after: per-person aggregates
    left_groups = dates[left_mask].groupby(pid[left_mask])
    right_groups = dates[right_mask].groupby(pid[right_mask])

    # Need both min and max for universal comparisons
    left_min = left_groups.min()
    left_max = left_groups.max()
    right_min = right_groups.min()
    right_max = right_groups.max()

    common_pids = left_min.index.intersection(right_min.index)
    if common_pids.empty:
        return pd.Series(False, index=df.index)

    left_min = left_min.reindex(common_pids) + left_shift
    left_max = left_max.reindex(common_pids) + left_shift
    right_min = right_min.reindex(common_pids) + right_shift
    right_max = right_max.reindex(common_pids) + right_shift

    # NOTE: the historical default for `K50 before K51` is "first K50 < first K51",
    # which mathematically corresponds to "∃ K50 (the earliest one) ∀ K51, k50 < k51"
    # — i.e. universal quantification over the right side. By symmetry, the historical
    # default for `K50 after K51` corresponds to universal LHS. We preserve those
    # semantics so adding `every` on the "implied" side is a no-op (documented).
    if op == "before":
        if every_left and every_right:
            matching = left_max < right_min   # every left strictly before every right
        elif every_left:
            matching = left_max < right_max   # every K50 has at least one K51 after it
        else:
            # default and `every_right` (same as default): first K50 before every K51
            matching = left_min < right_min
    else:  # after
        if every_left and every_right:
            matching = left_min > right_max
        elif every_right:
            matching = left_max > right_max   # every K51 has at least one K50 after it
        else:
            # default and `every_left` (same as default): every K50 has K51 before it
            matching = left_min > right_min

    matching_pids = set(matching.index[matching])
    return pid.isin(matching_pids) & left_mask


def _eval_around_signed(
    df: pd.DataFrame,
    child_mask: pd.Series,
    ref_mask: pd.Series,
    min_days: int,
    max_days: int,
    pid_col: str,
    date_col: str,
    ref_offset_days: int = 0,
) -> pd.Series:
    """Existential signed window for `around`: signed_diff in [min_days, max_days].

    Uses both backward and forward asofs so rows on either side of ref
    can match. Complete when the window spans zero (min_days <= 0 <=
    max_days); for wholly-negative or wholly-positive signed windows, one
    of the asof tolerances collapses and only the relevant side is used.
    """
    pid = df[pid_col]
    dates = df[date_col]

    child_df = pd.DataFrame({
        "pid": pid[child_mask].values,
        "date": dates[child_mask].values,
        "orig_idx": child_mask.index[child_mask],
    }).sort_values("date")

    ref_df = pd.DataFrame({
        "pid": pid[ref_mask].values,
        "ref_date": dates[ref_mask].values + pd.Timedelta(days=ref_offset_days),
    }).sort_values("ref_date")

    has_match = pd.Series(False, index=child_df.index)

    # Backward asof: finds the most recent ref at or before each child
    # (signed_diff >= 0). Tolerance = max_days (upper bound).
    if max_days >= 0:
        bw = pd.merge_asof(
            child_df, ref_df,
            by="pid", left_on="date", right_on="ref_date",
            direction="backward",
            tolerance=pd.Timedelta(days=max_days),
        )
        bw_diff = (bw["date"] - bw["ref_date"]).dt.days
        bw_match = bw["ref_date"].notna() & (bw_diff >= min_days) & (bw_diff <= max_days)
        has_match = has_match | bw_match.values

    # Forward asof: finds the earliest ref at or after each child
    # (signed_diff <= 0). Tolerance = -min_days (magnitude).
    if min_days <= 0:
        fw = pd.merge_asof(
            child_df, ref_df,
            by="pid", left_on="date", right_on="ref_date",
            direction="forward",
            tolerance=pd.Timedelta(days=-min_days),
        )
        fw_diff = (fw["date"] - fw["ref_date"]).dt.days
        fw_match = fw["ref_date"].notna() & (fw_diff >= min_days) & (fw_diff <= max_days)
        has_match = has_match | fw_match.values

    matched_idx = child_df.loc[has_match.values, "orig_idx"].values
    result = pd.Series(False, index=df.index)
    result.iloc[result.index.get_indexer(matched_idx)] = True
    return result


def eval_within_days(
    df: pd.DataFrame,
    child_mask: pd.Series,
    ref_mask: pd.Series | None,
    days: int,
    direction: str | None,
    pid_col: str,
    date_col: str,
    min_days: int = 0,
    every_left: bool = False,
    every_right: bool = False,
    ref_offset_days: int = 0,
) -> pd.Series:
    """Row-level: mark child rows within a day range of reference events.

    Args:
        days: Maximum days distance (upper bound).
        min_days: Minimum days distance (lower bound, for 'between M and N days').
        direction: 'before', 'after', 'around', or None.
        every_left: If True, every child event must have a qualifying ref event.
        every_right: If True, every ref event must have a qualifying child event.

    If ref_mask is None, this is a standalone time window relative to
    the first event per person.

    Universal modes return ALL child rows for matching persons (not just
    the matched ones), matching the convention of `eval_before_after`.
    """
    if not child_mask.any():
        return pd.Series(False, index=df.index)

    pid = df[pid_col]
    dates = df[date_col]

    if ref_mask is None:
        # No reference: within day range of first event per person.
        # (ref_offset_days ignored here — no ref to shift.)
        first_date = dates.groupby(pid).transform("first")
        diff = (dates - first_date).dt.days.abs()
        return child_mask & (diff >= min_days) & (diff <= days)

    if not ref_mask.any():
        return pd.Series(False, index=df.index)

    ref_shift = pd.Timedelta(days=ref_offset_days)

    # Signed-around: asymmetric window `inside min_days to days around Y`
    # where min_days can be negative. Existential-only; universal modes
    # fall through to the existing logic (which uses unsigned |diff|).
    if direction == "around" and min_days < 0 and not (every_left or every_right):
        return _eval_around_signed(
            df, child_mask, ref_mask, min_days, days, pid_col, date_col,
            ref_offset_days=ref_offset_days,
        )

    # ----- LHS-anchored matching: for each child, find nearest ref in window -----
    # (Existing existential logic; also used for the `every_left` check.)
    child_df = pd.DataFrame({
        "pid": pid[child_mask].values,
        "date": dates[child_mask].values,
        "orig_idx": child_mask.index[child_mask],
    }).sort_values("date")

    ref_df = pd.DataFrame({
        "pid": pid[ref_mask].values,
        "ref_date": dates[ref_mask].values + ref_shift,
    }).sort_values("ref_date")

    if direction == "after":
        # child happens after ref: from child, look backward in time for ref
        asof_dir_lhs = "backward"
    elif direction == "before":
        asof_dir_lhs = "forward"
    else:
        asof_dir_lhs = "nearest"

    merged = pd.merge_asof(
        child_df, ref_df,
        by="pid", left_on="date", right_on="ref_date",
        direction=asof_dir_lhs,
        tolerance=pd.Timedelta(days=days),
    )

    has_match = merged["ref_date"].notna()
    if min_days > 0 and has_match.any():
        day_diff = (merged["date"] - merged["ref_date"]).dt.days.abs()
        has_match = has_match & (day_diff >= min_days)

    if not (every_left or every_right):
        # Existential default: return matched child rows
        matched_idx = merged.loc[has_match, "orig_idx"].values
        result = pd.Series(False, index=df.index)
        result.iloc[result.index.get_indexer(matched_idx)] = True
        return result

    # ----- Universal mode: determine which persons satisfy the predicate -----
    # Persons must have both child and ref events present (non-empty rule).
    child_pids = set(child_df["pid"].unique())
    ref_pids = set(ref_df["pid"].unique())
    candidate_pids = child_pids & ref_pids

    matching_pids: set = set(candidate_pids)

    if every_left:
        # Every child event must have a qualifying ref. Per person:
        # number of matched child rows == total child rows.
        merged_with_match = merged.assign(_has=has_match)
        per_pid_total = merged_with_match.groupby("pid").size()
        per_pid_matched = merged_with_match.groupby("pid")["_has"].sum()
        full_coverage = per_pid_total[per_pid_total == per_pid_matched].index
        matching_pids &= set(full_coverage)

    if every_right:
        # Every ref event must have a qualifying child. Run merge_asof from
        # the ref side, looking the OPPOSITE direction.
        if direction == "after":
            asof_dir_rhs = "forward"   # from ref, look forward for a later child
        elif direction == "before":
            asof_dir_rhs = "backward"  # from ref, look backward for an earlier child
        else:
            asof_dir_rhs = "nearest"

        # merge_asof requires sorted left side
        ref_lookup = pd.DataFrame({
            "pid": pid[ref_mask].values,
            "ref_date": dates[ref_mask].values + ref_shift,
        }).sort_values("ref_date")
        child_lookup = pd.DataFrame({
            "pid": pid[child_mask].values,
            "child_date": dates[child_mask].values,
        }).sort_values("child_date")

        rhs_merged = pd.merge_asof(
            ref_lookup, child_lookup,
            by="pid", left_on="ref_date", right_on="child_date",
            direction=asof_dir_rhs,
            tolerance=pd.Timedelta(days=days),
        )
        rhs_has = rhs_merged["child_date"].notna()
        if min_days > 0 and rhs_has.any():
            day_diff = (rhs_merged["child_date"] - rhs_merged["ref_date"]).dt.days.abs()
            rhs_has = rhs_has & (day_diff >= min_days)

        rhs_with_match = rhs_merged.assign(_has=rhs_has)
        per_pid_total = rhs_with_match.groupby("pid").size()
        per_pid_matched = rhs_with_match.groupby("pid")["_has"].sum()
        full_coverage = per_pid_total[per_pid_total == per_pid_matched].index
        matching_pids &= set(full_coverage)

    # Return ALL child rows for matching persons
    return child_mask & pid.isin(matching_pids)


def eval_inside_outside(
    df: pd.DataFrame,
    child_mask: pd.Series,
    ref_mask: pd.Series,
    inside: bool,
    min_events: int,
    max_events: int,
    direction: str,
    pid_col: str,
) -> pd.Series:
    """Row-level: mark child rows inside/outside an event-count window.

    The window is the closed integer range ``[min_events, max_events]``
    in row offsets from each ref row. For example,
    ``inside 1 to 5 events after Y`` ⇒ min_events=1, max_events=5,
    direction="after"; `direction` of "before" mirrors the offsets
    (offset −k is |k| rows earlier); `around` uses signed offsets
    directly (so `inside -3 to 5 events around Y` is positions −3..+5).

    `inside=False` flips to the row-level complement.
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

    # For each person, check if child events fall in the window
    for p in ref_pids.unique():
        p_mask = pid == p
        p_event_num = event_num[p_mask]
        p_child = child_mask[p_mask]
        p_ref_positions = ref_positions[ref_pids == p].values

        in_window = pd.Series(False, index=p_event_num.index)
        for ref_pos in p_ref_positions:
            if direction == "after":
                # offsets +min_events..+max_events relative to ref
                lo = ref_pos + min_events
                hi = ref_pos + max_events
                in_window = in_window | (
                    (p_event_num >= lo) & (p_event_num <= hi)
                )
            elif direction == "before":
                # offsets −max_events..−min_events (i.e., positions
                # ref-max..ref-min going backwards)
                lo = ref_pos - max_events
                hi = ref_pos - min_events
                in_window = in_window | (
                    (p_event_num >= lo) & (p_event_num <= hi)
                )
            else:  # around: signed offsets
                lo = ref_pos + min_events
                hi = ref_pos + max_events
                in_window = in_window | (
                    (p_event_num >= lo) & (p_event_num <= hi)
                )

        if inside:
            matched = (p_child & in_window).values
        else:
            matched = (p_child & ~in_window).values
        result.loc[p_mask.values] = matched

    return result
