"""tquery: Temporal query language for event-level health data in pandas."""

from __future__ import annotations

import re
import warnings
from itertools import product
from typing import Any

import pandas as pd

from tquery._codebook import (
    Codebook,
    count_codes,
    get_codebook,
    get_label,
    search_codes,
)
from tquery._codes import extract_codes
from tquery._evaluator import Evaluator
from tquery._parser import parse
from tquery._stringify import stringify_durations, stringify_order, stringify_time
from tquery._types import (
    TQueryCodeError,
    TQueryColumnError,
    TQueryConfig,
    TQueryResult,
    TQuerySyntaxError,
    _merge_kwargs,
    get_config,
    use,
)

__all__ = [
    "tquery",
    "count_persons",
    "event_counts",
    "multi_query",
    "stringify_order",
    "stringify_time",
    "stringify_durations",
    "extract_codes",
    "count_codes",
    "search_codes",
    "get_label",
    "get_codebook",
    "TQueryConfig",
    "TQueryResult",
    "TQuerySyntaxError",
    "TQueryColumnError",
    "TQueryCodeError",
    "use",
    "get_config",
]

__version__ = "0.1.0"


def tquery(
    df: pd.DataFrame,
    expr: str,
    *,
    pid: str | None = None,
    date: str | None = None,
    cols: str | list[str] | None = None,
    sep: str | None = None,
    variables: dict[str, Any] | None = None,
    config: TQueryConfig | None = None,
) -> TQueryResult:
    """Evaluate a temporal query expression against a DataFrame.

    Args:
        df: Event-level DataFrame, ideally sorted by (pid, date).
        expr: A tquery expression string, e.g. 'K50 before K51'.
        pid: Column name for person ID. Falls back to active config.
        date: Column name for event date. Falls back to active config.
        cols: Column(s) to search for codes. Falls back to active config.
        sep: If cells contain multiple codes, the separator string.
        variables: Dict mapping @variable names to code lists.
        config: A TQueryConfig to use instead of the global default.
                Explicit keyword args still override the config.

    Returns:
        A TQueryResult with .count, .pids, .rows, .persons, .filter()
    """
    kw = _merge_kwargs(
        config, pid=pid, date=date, cols=cols, sep=sep, variables=variables,
    )
    # Filter to only Evaluator-accepted kwargs
    eval_kw = {k: v for k, v in kw.items()
               if k in ("pid", "date", "cols", "sep", "variables")}
    ast = parse(expr)
    evaluator = Evaluator(df, **eval_kw)
    row_mask = evaluator.evaluate(ast)
    return TQueryResult(row_mask, df, kw["pid"])


def count_persons(
    df: pd.DataFrame,
    expr: str,
    **kwargs: Any,
) -> int:
    """Count persons matching a temporal query expression.

    Shorthand for tquery(df, expr, **kwargs).count.
    """
    return tquery(df, expr, **kwargs).count


def event_counts(
    df: pd.DataFrame,
    expr: str,
    **kwargs: Any,
) -> pd.Series:
    """Count matching events per person.

    Returns a Series indexed by pid with the number of matching
    events for each person.

    Shorthand for tquery(df, expr, **kwargs).event_counts.
    """
    return tquery(df, expr, **kwargs).event_counts


# ---------------------------------------------------------------------------
# Parameterized queries: ?[...] notation
# ---------------------------------------------------------------------------

_SLOT_RE = re.compile(r"\?\[([^\]]+)\]")


def _parse_slot(content: str) -> list[str]:
    """Parse the contents of a ?[...] slot into a list of substitution values.

    Supports:
        ?[0,1,2]      → ['0', '1', '2']           (explicit list)
        ?[K50,K51]    → ['K50', 'K51']             (multi-char list)
        ?[0-5]        → ['0','1','2','3','4','5']   (single-char numeric range)
        ?[a-d]        → ['a','b','c','d']           (single-char letter range)
        ?[50-53]      → ['50','51','52','53']        (multi-digit numeric range)
    """
    content = content.strip()

    # Check for range: digits-digits (e.g., 0-9, 50-53)
    range_match = re.match(r"^(\d+)-(\d+)$", content)
    if range_match:
        start, end = int(range_match.group(1)), int(range_match.group(2))
        return [str(i) for i in range(start, end + 1)]

    # Check for single-char letter range: a-z
    letter_match = re.match(r"^([a-zA-Z])-([a-zA-Z])$", content)
    if letter_match:
        start, end = ord(letter_match.group(1)), ord(letter_match.group(2))
        return [chr(c) for c in range(start, end + 1)]

    # Comma-separated list
    return [v.strip() for v in content.split(",")]


def multi_query(
    df: pd.DataFrame,
    template: str,
    *,
    max_combinations: int = 1000,
    pid: str | None = None,
    date: str | None = None,
    cols: str | list[str] | None = None,
    sep: str | None = None,
    variables: dict[str, Any] | None = None,
    config: TQueryConfig | None = None,
) -> pd.Series:
    """Run parameterized queries for all ?[...] variants.

    Substitutes each ?[...] slot with its possible values, generates
    all combinations, and runs each as a separate query. Uses a single
    shared Evaluator so that common sub-expressions are cached.

    Args:
        df: Event-level DataFrame.
        template: Query template with ?[...] slots.
                  E.g., 'K5?[0,1,2] before K51'
        max_combinations: Maximum allowed combinations before raising
                          an error. Default 1000.
        pid, date, cols, sep, variables, config: Same as tquery().

    Returns:
        Series indexed by the concrete query string, values are person counts.

    Examples:
        >>> multi_query(df, 'K5?[0,1,2] before K51')
        K50 before K51    234
        K51 before K51      0
        K52 before K51     89
        dtype: int64

        >>> multi_query(df, 'K50 within ?[30,60,90] days after K51')
        K50 within 30 days after K51    12
        K50 within 60 days after K51    34
        K50 within 90 days after K51    56
        dtype: int64
    """
    # Find all ?[...] slots
    slots = _SLOT_RE.findall(template)

    if not slots:
        # No slots — just run the query directly
        return pd.Series({template: tquery(df, template, pid=pid, date=date,
                          cols=cols, sep=sep, variables=variables,
                          config=config).count})

    # Parse each slot into its list of values
    slot_values = [_parse_slot(s) for s in slots]

    # Check combinatorial size
    n_combos = 1
    for sv in slot_values:
        n_combos *= len(sv)

    if n_combos > max_combinations:
        raise ValueError(
            f"Template generates {n_combos} combinations "
            f"(limit is {max_combinations}). Reduce the ?[...] ranges "
            f"or increase max_combinations."
        )

    if n_combos > 100:
        warnings.warn(
            f"Running {n_combos} query combinations. This may take a moment.",
            stacklevel=2,
        )

    # Build one shared Evaluator for cache reuse across all queries
    kw = _merge_kwargs(
        config, pid=pid, date=date, cols=cols, sep=sep, variables=variables,
    )
    eval_kw = {k: v for k, v in kw.items()
               if k in ("pid", "date", "cols", "sep", "variables")}
    evaluator = Evaluator(df, **eval_kw)
    pid_col = kw["pid"]

    # Generate all combinations and evaluate
    results: dict[str, int] = {}

    for combo in product(*slot_values):
        # Substitute values into template (left-to-right, one per slot)
        query = template
        for val in combo:
            query = _SLOT_RE.sub(val, query, count=1)

        # Parse and evaluate using the shared evaluator (shared cache!)
        try:
            ast = parse(query)
            row_mask = evaluator.evaluate(ast)
            count = int(row_mask.groupby(df[pid_col]).any().sum())
        except (TQuerySyntaxError, TQueryCodeError):
            count = 0

        results[query] = count

    return pd.Series(results, dtype=int)


@pd.api.extensions.register_dataframe_accessor("tq")
class TQueryAccessor:
    """Pandas DataFrame accessor for tquery.

    Uses the active TQueryConfig for default parameters.

    Examples:
        df.tq('K50 before K51').count
        df.tq.count('K50')
        df.tq.stringify_order(codes, cols='atc')

        # With explicit config override:
        df.tq('K50 before K51', config=my_config)
    """

    def __init__(self, df: pd.DataFrame) -> None:
        self._df = df

    def period(self, start: str | None = None, end: str | None = None) -> TQueryAccessor:
        """Return accessor for a date-filtered slice of the DataFrame.

        Args:
            start: Inclusive start date (string, e.g., '2020-01-01').
            end: Inclusive end date (string, e.g., '2020-12-31').

        Examples:
            df.tq.period('2020-01-01', '2020-12-31').count('K50')
            df.tq.period('2020-01-01').count('K50 before K51')
            df.tq.period(end='2019-12-31').count('K50')
        """
        cfg = get_config()
        date_col = cfg.date
        mask = pd.Series(True, index=self._df.index)
        if start is not None:
            mask = mask & (self._df[date_col] >= pd.to_datetime(start))
        if end is not None:
            mask = mask & (self._df[date_col] <= pd.to_datetime(end))
        return self._df[mask].tq

    def year(self, y: int) -> TQueryAccessor:
        """Return accessor filtered to a specific year.

        Example:
            df.tq.year(2020).count('K50 before K51')
        """
        cfg = get_config()
        date_col = cfg.date
        return self._df[self._df[date_col].dt.year == y].tq

    def __call__(self, expr: str, **kwargs: Any) -> TQueryResult:
        return tquery(self._df, expr, **kwargs)

    def count(self, expr: str, **kwargs: Any) -> int:
        return count_persons(self._df, expr, **kwargs)

    def multi(self, template: str, **kwargs: Any) -> pd.Series:
        return multi_query(self._df, template, **kwargs)

    def event_counts(self, expr: str, **kwargs: Any) -> pd.Series:
        return event_counts(self._df, expr, **kwargs)

    def labels(self, cols: str | list[str] | None = None) -> pd.DataFrame:
        """Add label columns next to code columns in the DataFrame.

        For each code column, adds a '{col}_label' column with the
        human-readable label from the codebooks.

        Example:
            df.tq.labels(cols='icd')
            # → DataFrame with added 'icd_label' column
        """
        cfg = get_config()
        extra = [cfg.codebooks_dir] if cfg.codebooks_dir else None
        cb = get_codebook(extra)

        if cols is None:
            cols = [c for c in self._df.columns
                    if c not in (cfg.pid, cfg.date) and self._df[c].dtype == object]
        elif isinstance(cols, str):
            cols = [cols]

        result = self._df.copy()
        for col in cols:
            if col in result.columns:
                result[f"{col}_label"] = cb.labels(result[col])
        return result

    def count_codes(
        self,
        cols: str | list[str] | None = None,
        *,
        per_person: bool = False,
        pattern: str | None = None,
    ) -> pd.Series:
        """Count code frequencies across columns.

        Args:
            cols: Columns to count. None = auto-detect.
            per_person: If True, count unique persons per code.
            pattern: Filter to codes matching pattern (e.g., 'K50*').

        Returns:
            Series indexed by code, values are counts, sorted descending.
        """
        cfg = get_config()
        return count_codes(
            self._df, cols, pid=cfg.pid, date=cfg.date,
            per_person=per_person, pattern=pattern, sep=cfg.sep,
        )

    def search_codes(self, keyword: str, system: str | None = None) -> pd.DataFrame:
        """Search codebooks by keyword in labels.

        Example:
            df.tq.search_codes('diabetes')
        """
        cfg = get_config()
        extra = [cfg.codebooks_dir] if cfg.codebooks_dir else None
        return get_codebook(extra).search(keyword, system)

    def stringify_order(self, codes: dict, **kwargs: Any) -> pd.Series:
        kw = _merge_kwargs(kwargs.pop("config", None), **kwargs)
        # Remap 'date' → 'event_start' for stringify API
        kw.setdefault("event_start", kw.pop("date", "start_date"))
        kw.pop("variables", None)
        return stringify_order(self._df, codes, **kw)

    def stringify_time(self, codes: dict, **kwargs: Any) -> pd.Series | pd.DataFrame:
        kw = _merge_kwargs(kwargs.pop("config", None), **kwargs)
        kw.setdefault("event_start", kw.pop("date", "start_date"))
        kw.pop("variables", None)
        return stringify_time(self._df, codes, **kw)

    def stringify_durations(self, codes: dict, **kwargs: Any) -> pd.Series | pd.DataFrame:
        kw = _merge_kwargs(kwargs.pop("config", None), **kwargs)
        kw.setdefault("event_start", kw.pop("date", "start_date"))
        kw.pop("variables", None)
        return stringify_durations(self._df, codes, **kw)
