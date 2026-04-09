"""Result types, configuration, and custom exceptions for tquery."""

from __future__ import annotations

from copy import copy
from dataclasses import dataclass, field, fields
from functools import cached_property
from typing import Any, TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:
    pass


# ---------------------------------------------------------------------------
# Configuration / Profiles
# ---------------------------------------------------------------------------

@dataclass
class TQueryConfig:
    """Default parameters for tquery functions.

    Create named profiles for different datasets and switch between them.

    Examples:
        # Norwegian Prescription Registry
        npr = TQueryConfig(
            pid='patient_id',
            date='dispensing_date',
            cols='atc',
            name='NPR',
        )

        # Hospital admissions
        hospital = TQueryConfig(
            pid='pasient_id',
            date='inn_dato',
            cols=['hoved', 'bi1', 'bi2'],
            name='Hospital',
        )

        # Set as active default
        tquery.use(npr)
        df.tq('K50 before K51')  # uses pid='patient_id', etc.

        # Switch
        tquery.use(hospital)

        # Or pass directly
        df.tq('K50 before K51', config=hospital)
    """
    pid: str = "pid"
    date: str = "start_date"
    event_end: str | None = None
    event_duration: str | None = None
    cols: str | list[str] | None = None
    sep: str | None = None
    variables: dict[str, Any] | None = None
    codebooks_dir: str | None = None
    name: str = "default"

    def as_kwargs(self) -> dict[str, Any]:
        """Return config values as a keyword argument dict.

        Only includes non-None values (except pid and date which are
        always included).
        """
        result: dict[str, Any] = {"pid": self.pid, "date": self.date}
        if self.event_end is not None:
            result["event_end"] = self.event_end
        if self.event_duration is not None:
            result["event_duration"] = self.event_duration
        if self.cols is not None:
            result["cols"] = self.cols
        if self.sep is not None:
            result["sep"] = self.sep
        if self.variables is not None:
            result["variables"] = self.variables
        return result

    def __repr__(self) -> str:
        parts = [f"pid={self.pid!r}", f"date={self.date!r}"]
        if self.event_end is not None:
            parts.append(f"event_end={self.event_end!r}")
        if self.event_duration is not None:
            parts.append(f"event_duration={self.event_duration!r}")
        if self.cols is not None:
            parts.append(f"cols={self.cols!r}")
        if self.sep is not None:
            parts.append(f"sep={self.sep!r}")
        return f"TQueryConfig({', '.join(parts)}, name={self.name!r})"


# Global active config
_active_config = TQueryConfig()


def get_config() -> TQueryConfig:
    """Return the currently active TQueryConfig."""
    return _active_config


def use(config: TQueryConfig) -> None:
    """Set the active TQueryConfig globally."""
    global _active_config
    _active_config = config


def _merge_kwargs(config: TQueryConfig | None, **explicit: Any) -> dict[str, Any]:
    """Merge explicit kwargs over config defaults.

    Explicit non-None kwargs take precedence over config values.
    """
    cfg = config if config is not None else _active_config
    merged = cfg.as_kwargs()
    for key, val in explicit.items():
        if val is not None or key not in merged:
            merged[key] = val
    return merged


class TQueryResult:
    """Result of evaluating a temporal query expression.

    Provides multiple views of the result:
    - rows: boolean Series at the row level (same index as input df)
    - persons: boolean Series at the person level (one entry per pid)
    - pids: set of person IDs matching the query
    - count: number of persons matching
    - filter(): returns a filtered DataFrame
    """

    def __init__(self, row_mask: pd.Series, df: pd.DataFrame, pid: str) -> None:
        self._row_mask = row_mask
        self._df = df
        self._pid = pid

    @property
    def rows(self) -> pd.Series:
        """Boolean Series: True for each row matching the query."""
        return self._row_mask

    @cached_property
    def persons(self) -> pd.Series:
        """Boolean Series: True for each person with at least one matching row."""
        return self._row_mask.groupby(self._df[self._pid]).any()

    @cached_property
    def pids(self) -> set:
        """Set of person IDs matching the query."""
        s = self.persons
        return set(s.index[s])

    @cached_property
    def count(self) -> int:
        """Number of persons matching the query."""
        return int(self.persons.sum())

    @cached_property
    def event_counts(self) -> pd.Series:
        """Count of matching events per person."""
        return self._row_mask.groupby(self._df[self._pid]).sum().astype(int)

    def filter(self, level: str = "persons") -> pd.DataFrame:
        """Return filtered DataFrame.

        Args:
            level: 'rows' for matching rows only, 'persons' for all rows of
                   matching persons (default).
        """
        if level == "rows":
            return self._df[self._row_mask]
        mask = self._df[self._pid].isin(self.pids)
        return self._df[mask]

    def __repr__(self) -> str:
        return f"TQueryResult(count={self.count}, matching_rows={int(self._row_mask.sum())})"


class TQuerySyntaxError(Exception):
    """Raised when a query expression has invalid syntax."""

    def __init__(self, message: str, expr: str = "", pos: int = -1) -> None:
        self.expr = expr
        self.pos = pos
        if expr and pos >= 0:
            pointer = " " * pos + "^"
            message = f"{message}\n  {expr}\n  {pointer}"
        super().__init__(message)


class TQueryColumnError(Exception):
    """Raised when a referenced column does not exist in the DataFrame."""


class TQueryCodeError(Exception):
    """Raised when a code pattern expands to zero matching codes."""
