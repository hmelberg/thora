"""Stringify functions for converting event data into treatment pattern strings.

Three main functions:
- stringify_order: events in chronological order ('iiaga')
- stringify_time: events at time positions ('i  i a  ')
- stringify_durations: events filling their duration ('iii aa  ')
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from tquery._codes import (
    collect_unique_codes,
    expand_codes,
    extract_codes,
    get_matching_rows,
)
from tquery._stringops import interleave_strings
from tquery._types import TQueryColumnError


@dataclass
class _PreparedData:
    """Pre-processed data shared by all stringify functions."""
    df: pd.DataFrame
    labels: pd.Series
    pid_col: str
    date_col: str
    label_order: list[str]


def _resolve_cols(
    df: pd.DataFrame,
    cols: str | list[str] | None,
    pid: str,
    date: str,
) -> list[str]:
    """Resolve column specification to a concrete list."""
    if cols is None:
        return [c for c in df.columns if c not in (pid, date) and df[c].dtype == object]
    if isinstance(cols, str):
        return [cols]
    return list(cols)


def _filter_date_window(
    df: pd.DataFrame,
    event_start: str,
    pid: str,
    first_date: str | dict | None,
    last_date: str | dict | None,
) -> pd.DataFrame:
    """Filter rows to the specified date window."""
    if first_date is not None:
        if isinstance(first_date, str) and first_date in df.columns:
            df = df[df[event_start] >= df[first_date]]
        elif isinstance(first_date, dict):
            start_dates = df[pid].map(first_date)
            df = df[df[event_start] >= start_dates]
        elif isinstance(first_date, str):
            df = df[df[event_start] >= pd.to_datetime(first_date)]

    if last_date is not None:
        if isinstance(last_date, str) and last_date in df.columns:
            df = df[df[event_start] <= df[last_date]]
        elif isinstance(last_date, dict):
            end_dates = df[pid].map(last_date)
            df = df[df[event_start] <= end_dates]
        elif isinstance(last_date, str):
            df = df[df[event_start] <= pd.to_datetime(last_date)]

    return df


def _prepare(
    df: pd.DataFrame,
    codes: dict[str, str | list[str]],
    *,
    cols: str | list[str] | None = None,
    pid: str = "pid",
    event_start: str = "start_date",
    sep: str | None = None,
    first_date: str | dict | None = None,
    last_date: str | dict | None = None,
) -> _PreparedData:
    """Shared setup for all stringify functions."""
    if pid not in df.columns:
        raise TQueryColumnError(f"Person ID column '{pid}' not found")
    if event_start not in df.columns:
        raise TQueryColumnError(f"Date column '{event_start}' not found")

    resolved_cols = _resolve_cols(df, cols, pid, event_start)

    # Normalize codes dict: ensure values are lists
    norm_codes: dict[str, list[str]] = {}
    for label, patterns in codes.items():
        if isinstance(patterns, str):
            norm_codes[label] = [patterns]
        else:
            norm_codes[label] = list(patterns)

    # Collect all code patterns across all labels
    all_patterns: list[str] = []
    for patterns in norm_codes.values():
        all_patterns.extend(patterns)

    all_codes = collect_unique_codes(df, resolved_cols, sep)

    # Expand patterns for row matching
    expanded: list[str] = []
    for pat in all_patterns:
        expanded.extend(expand_codes(pat, all_codes=all_codes))

    # Filter to matching rows
    mask = get_matching_rows(df, expanded, resolved_cols, sep)
    subset = df[mask].copy()

    # Drop rows with missing pid or date
    subset = subset.dropna(subset=[pid, event_start])

    # Apply date window
    subset = _filter_date_window(subset, event_start, pid, first_date, last_date)

    # Assign labels
    labels = extract_codes(subset, norm_codes, resolved_cols, sep, all_codes)

    # Drop rows that didn't match any label
    has_label = labels.notna()
    subset = subset[has_label]
    labels = labels[has_label]

    # Sort and align indices
    sort_order = subset.sort_values([pid, event_start]).index
    subset = subset.loc[sort_order].reset_index(drop=True)
    labels = labels.loc[sort_order].reset_index(drop=True)

    label_order = list(norm_codes.keys())

    return _PreparedData(
        df=subset,
        labels=labels,
        pid_col=pid,
        date_col=event_start,
        label_order=label_order,
    )


def stringify_order(
    df: pd.DataFrame,
    codes: dict[str, str | list[str]],
    *,
    cols: str | list[str] | None = None,
    pid: str = "pid",
    event_start: str = "start_date",
    sep: str | None = None,
    first_date: str | dict | None = None,
    last_date: str | dict | None = None,
    keep_repeats: bool = True,
    only_unique: bool = False,
) -> pd.Series:
    """Create per-person strings showing events in chronological order.

    Args:
        df: Event-level DataFrame.
        codes: Dict mapping shorthand labels to code patterns.
               E.g., {'i': ['L04AB02'], 'a': ['L04AB04']}.
        cols: Column(s) to search for codes. None = auto-detect.
        pid: Person ID column name.
        event_start: Event date column name.
        sep: Separator for multi-value cells.
        first_date: Start date filter (column name, dict, or date string).
        last_date: End date filter (column name, dict, or date string).
        keep_repeats: If False, remove consecutive duplicate characters.
        only_unique: If True, keep only first occurrence of each label per person.

    Returns:
        Series indexed by pid, values are pattern strings like 'iiaga'.
    """
    prep = _prepare(
        df, codes, cols=cols, pid=pid, event_start=event_start,
        sep=sep, first_date=first_date, last_date=last_date,
    )

    result = prep.labels.groupby(prep.df[prep.pid_col]).agg("".join)
    result.index.name = pid

    if not keep_repeats:
        result = result.str.replace(r"(.)\1+", r"\1", regex=True)

    if only_unique:
        def _uniqify(text: str) -> str:
            seen: set[str] = set()
            out: list[str] = []
            for ch in text:
                if ch not in seen:
                    seen.add(ch)
                    out.append(ch)
            return "".join(out)
        result = result.apply(_uniqify)

    return result


def stringify_time(
    df: pd.DataFrame,
    codes: dict[str, str | list[str]],
    *,
    cols: str | list[str] | None = None,
    pid: str = "pid",
    event_start: str = "start_date",
    sep: str | None = None,
    step: int = 90,
    no_event: str = " ",
    first_date: str | dict | None = None,
    last_date: str | dict | None = None,
    merge: bool = True,
) -> pd.Series | pd.DataFrame:
    """Create per-person strings showing events at time positions.

    Each character position represents one step-day period.

    Args:
        df: Event-level DataFrame.
        codes: Dict mapping shorthand labels to code patterns.
        cols: Column(s) to search. None = auto-detect.
        pid: Person ID column name.
        event_start: Event date column name.
        sep: Separator for multi-value cells.
        step: Number of days per character position.
        no_event: Character for empty time slots.
        first_date: Start date filter.
        last_date: End date filter.
        merge: If True, interleave all tracks into one string.

    Returns:
        If merge=True: Series indexed by pid.
        If merge=False: DataFrame indexed by pid, columns = labels.
    """
    prep = _prepare(
        df, codes, cols=cols, pid=pid, event_start=event_start,
        sep=sep, first_date=first_date, last_date=last_date,
    )

    if prep.df.empty:
        if merge:
            return pd.Series(dtype=str)
        return pd.DataFrame(columns=prep.label_order)

    dates = prep.df[prep.date_col]
    pids = prep.df[prep.pid_col]

    # Compute per-person min date as reference
    min_dates = pids.map(dates.groupby(pids).min())
    positions = ((dates - min_dates).dt.days // step).astype(int)

    # Max position per person determines string length
    max_pos_per_person = positions.groupby(pids).max()

    # Build one string column per label
    all_pids = pids.unique()
    string_df = pd.DataFrame(index=all_pids)
    string_df.index.name = pid

    for label in prep.label_order:
        label_mask = prep.labels == label
        label_df = prep.df[label_mask].copy()
        label_df["_pos"] = positions[label_mask].values

        if not label_df.empty:
            def _make_string(group_data: pd.DataFrame, _label: str = label) -> str:
                p = group_data.name  # group key = pid
                length = max_pos_per_person.get(p, 0) + 1
                chars = list(no_event * length)
                for pos in group_data["_pos"].values:
                    if 0 <= pos < length:
                        chars[pos] = _label
                return "".join(chars)

            strings = label_df.groupby(prep.pid_col, sort=False).apply(
                _make_string, include_groups=False,
            )
            string_df[label] = strings
        else:
            string_df[label] = no_event

    # Fill NaN for persons who have some labels but not others
    for label in prep.label_order:
        if label in string_df.columns:
            string_df[label] = string_df[label].fillna(no_event)

    if merge:
        return interleave_strings(string_df, prep.label_order, no_event=no_event)
    return string_df


def stringify_durations(
    df: pd.DataFrame,
    codes: dict[str, str | list[str]],
    *,
    cols: str | list[str] | None = None,
    pid: str = "pid",
    event_start: str = "start_date",
    event_end: str | None = None,
    event_duration: str | None = None,
    sep: str | None = None,
    step: int = 120,
    no_event: str = " ",
    first_date: str | dict | None = None,
    last_date: str | dict | None = None,
    merge: bool = True,
) -> pd.Series | pd.DataFrame:
    """Create per-person strings filling in event durations.

    Like stringify_time but each event fills positions for its full
    duration, not just the start position.

    Args:
        df: Event-level DataFrame.
        codes: Dict mapping shorthand labels to code patterns.
        cols: Column(s) to search. None = auto-detect.
        pid: Person ID column name.
        event_start: Event date column name.
        event_end: Column name for event end date.
        event_duration: Column name for event duration in days.
            Exactly one of event_end or event_duration must be provided.
        sep: Separator for multi-value cells.
        step: Number of days per character position.
        no_event: Character for empty time slots.
        first_date: Start date filter.
        last_date: End date filter.
        merge: If True, interleave all tracks into one string.

    Returns:
        If merge=True: Series indexed by pid.
        If merge=False: DataFrame indexed by pid, columns = labels.
    """
    if event_end is None and event_duration is None:
        raise ValueError("Either event_end or event_duration must be provided")

    prep = _prepare(
        df, codes, cols=cols, pid=pid, event_start=event_start,
        sep=sep, first_date=first_date, last_date=last_date,
    )

    if prep.df.empty:
        if merge:
            return pd.Series(dtype=str)
        return pd.DataFrame(columns=prep.label_order)

    dates = prep.df[prep.date_col]
    pids = prep.df[prep.pid_col]

    # Compute per-person min date as reference
    min_dates = pids.map(dates.groupby(pids).min())
    start_positions = ((dates - min_dates).dt.days // step).astype(int)

    # Compute end positions
    if event_end is not None:
        if event_end not in prep.df.columns:
            raise TQueryColumnError(f"Event end column '{event_end}' not found")
        end_dates = prep.df[event_end]
        end_positions = ((end_dates - min_dates).dt.days // step).astype(int)
    else:
        if event_duration not in prep.df.columns:
            raise TQueryColumnError(f"Duration column '{event_duration}' not found")
        durations = prep.df[event_duration]
        end_positions = start_positions + (durations // step).astype(int)

    max_pos_per_person = end_positions.groupby(pids).max().clip(lower=0)

    all_pids = pids.unique()
    string_df = pd.DataFrame(index=all_pids)
    string_df.index.name = pid

    for label in prep.label_order:
        label_mask = prep.labels == label

        label_df = prep.df[label_mask].copy()
        label_df["_start"] = start_positions[label_mask].values
        label_df["_end"] = end_positions[label_mask].values

        if not label_df.empty:
            def _make_duration_string(
                group_data: pd.DataFrame, _label: str = label
            ) -> str:
                p = group_data.name  # group key = pid
                length = max_pos_per_person.get(p, 0) + 1
                chars = list(no_event * length)
                for start_pos, end_pos in zip(
                    group_data["_start"].values, group_data["_end"].values
                ):
                    for pos in range(max(0, start_pos), min(length, end_pos + 1)):
                        chars[pos] = _label
                return "".join(chars)

            strings = label_df.groupby(prep.pid_col, sort=False).apply(
                _make_duration_string, include_groups=False,
            )
            string_df[label] = strings
        else:
            string_df[label] = no_event

    for label in prep.label_order:
        if label in string_df.columns:
            string_df[label] = string_df[label].fillna(no_event)

    if merge:
        return interleave_strings(string_df, prep.label_order, no_event=no_event)
    return string_df
