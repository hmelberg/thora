"""String manipulation helpers for treatment pattern strings.

Pure string operations on pandas Series — no DataFrame/code logic.
"""

from __future__ import annotations

import re
from itertools import zip_longest

import pandas as pd


def interleave_strings(
    df: pd.DataFrame,
    cols: list[str] | None = None,
    *,
    time_sep: str = "|",
    no_event: str = " ",
) -> pd.Series:
    """Merge multiple code-track columns character-by-character.

    Each column contains a per-person event string (one char per time period).
    The result interleaves them with a separator between time periods.

    Example: columns 'i' = 'i  i' and 'a' = ' a  ' with time_sep='|'
             → 'i |a | |i '

    Args:
        df: DataFrame where each column is a string series.
        cols: Columns to interleave. None = all columns.
        time_sep: Separator between time periods.
        no_event: Character for empty slots.
    """
    if cols is None:
        cols = list(df.columns)

    def _interleave(row: pd.Series) -> str:
        tracks = [str(row[c]) if pd.notna(row[c]) else "" for c in cols]
        periods = zip_longest(*tracks, fillvalue=no_event)
        return time_sep.join("".join(chars) for chars in periods)

    return df.apply(_interleave, axis=1)


def overlay_strings(
    df: pd.DataFrame,
    cols: list[str] | None = None,
    *,
    collision: str = "x",
    no_event: str = " ",
) -> pd.Series:
    """Overlay multiple code tracks, marking simultaneous events.

    At each position, if only one track has an event, that character is used.
    If multiple tracks have events, the collision character is used.

    Args:
        df: DataFrame where each column is a string series.
        cols: Columns to overlay. None = all columns.
        collision: Character for positions with multiple events.
        no_event: Character for empty positions.
    """
    if cols is None:
        cols = list(df.columns)

    def _overlay(row: pd.Series) -> str:
        tracks = [str(row[c]) if pd.notna(row[c]) else "" for c in cols]
        max_len = max((len(t) for t in tracks), default=0)
        result: list[str] = []
        for i in range(max_len):
            chars = {t[i] for t in tracks if i < len(t)} - {no_event}
            if len(chars) == 0:
                result.append(no_event)
            elif len(chars) == 1:
                result.append(chars.pop())
            else:
                result.append(collision)
        return "".join(result)

    return df.apply(_overlay, axis=1)


def left_justify(s: pd.Series, *, fill: str = " ") -> pd.Series:
    """Right-pad all strings in a Series to uniform length.

    After stringify, ensures all event strings have the same length so
    positions align across persons.
    """
    max_len = s.str.len().max()
    return s.str.pad(width=max_len, side="right", fillchar=fill)


def shorten(s: pd.Series, *, agg: int = 3, no_event: str = " ") -> pd.Series:
    """Reduce time resolution by aggregating consecutive positions.

    Groups every `agg` consecutive characters. If the group contains any
    event character, uses that character; otherwise uses no_event.

    Args:
        s: Series of event strings.
        agg: Number of positions to merge into one.
        no_event: Character representing no event.
    """
    empty = no_event * agg

    def _shorten_one(text: str) -> str:
        units = [text[i:i + agg] for i in range(0, len(text), agg)]
        result: list[str] = []
        for unit in units:
            if unit == empty or set(unit) == {no_event}:
                result.append(no_event)
            else:
                # Use first non-empty character
                for ch in unit:
                    if ch != no_event:
                        result.append(ch)
                        break
        return "".join(result)

    return s.apply(_shorten_one)


def del_repeats(s: pd.Series) -> pd.Series:
    """Remove consecutively repeated characters from each string.

    'aaabbc' → 'abc', 'iiiai' → 'iai'
    """
    return s.str.replace(r"(.)\1+", r"\1", regex=True)


def del_singles(s: pd.Series) -> pd.Series:
    """Remove isolated single characters from each string.

    Keeps a character only if it is the same as its neighbor.
    'aabca' → 'aab' (c and final a are isolated singles).
    """
    def _del_singles_one(text: str) -> str:
        if len(text) < 2:
            return ""
        result: list[str] = []
        for i, ch in enumerate(text):
            prev_same = i > 0 and text[i - 1] == ch
            next_same = i < len(text) - 1 and text[i + 1] == ch
            if prev_same or next_same:
                result.append(ch)
        return "".join(result)

    return s.apply(_del_singles_one)
