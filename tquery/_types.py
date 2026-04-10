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
    - evaluable: number of persons for whom the query is well-defined
    - total: total number of persons in the DataFrame
    - pct: count / evaluable * 100 (excludes "missing" persons)
    - pct_total: count / total * 100 (includes "missing" persons)
    - filter(): returns a filtered DataFrame
    """

    def __init__(
        self,
        row_mask: pd.Series,
        df: pd.DataFrame,
        pid: str,
        *,
        ast: Any = None,
        evaluator: Any = None,
    ) -> None:
        self._row_mask = row_mask
        self._df = df
        self._pid = pid
        self._ast = ast
        self._evaluator = evaluator

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
    def total(self) -> int:
        """Total number of distinct persons in the DataFrame."""
        return int(self._df[self._pid].nunique())

    @cached_property
    def evaluable_pids(self) -> set:
        """Set of person IDs for whom the query is well-defined.

        A person is excluded ("missing") if the query is undefined for
        them — i.e. they lack one of the events being compared in a
        temporal/within/inside subexpression. For queries with no such
        comparison, every person is evaluable.
        """
        if self._ast is None or self._evaluator is None:
            # Backward-compat path: no AST attached, assume all defined.
            return set(self._df[self._pid].unique())
        return self._evaluator.evaluable_pids(self._ast)

    @cached_property
    def evaluable(self) -> int:
        """Number of persons for whom the query is well-defined."""
        return len(self.evaluable_pids)

    @cached_property
    def event_counts(self) -> pd.Series:
        """Count of matching events per person."""
        return self._row_mask.groupby(self._df[self._pid]).sum().astype(int)

    def pct(self, dropna: bool = True) -> float:
        """Percentage of persons matching the query, in 0..100.

        Args:
            dropna: If True (default), the denominator is `evaluable` —
                persons for whom the query is well-defined. This gives
                the *conditional* percentage. If False, the denominator
                is `total` (all persons), giving the *marginal* percentage.

        Examples:
            >>> r = df.tq("K50 before K51")
            >>> r.pct()             # 11.8 — of those with both K50 and K51
            >>> r.pct(dropna=False) # 2.0 — of all 100 persons
        """
        denom = self.evaluable if dropna else self.total
        if denom == 0:
            return 0.0
        return self.count / denom * 100.0

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
        if self._ast is not None and self._evaluator is not None:
            return (
                f"TQueryResult(count={self.count}, evaluable={self.evaluable}, "
                f"total={self.total}, pct={self.pct():.1f}%)"
            )
        return (
            f"TQueryResult(count={self.count}, "
            f"matching_rows={int(self._row_mask.sum())})"
        )


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


class TQueryStringError(Exception):
    """Raised when a string-based query cannot be evaluated against
    stringify output (e.g. unsupported AST node, label not in codes
    dict, or pattern that requires DataFrame-level information that
    is unavailable in string form)."""
