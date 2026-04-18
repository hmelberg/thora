"""String-based query evaluator for cross-validation of tquery expressions.

Evaluates tquery AST nodes against stringify output strings instead of
DataFrame rows. This provides an independent evaluation path for
verifying that temporal query logic is correct.

Three public functions:
- string_query: evaluate expression against pre-computed strings
- string_query_auto: auto-stringify then evaluate
- cross_validate: compare DataFrame and string evaluators
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

from tquery._ast import (
    ASTNode,
    BetweenExpr,
    BinaryLogical,
    CodeAtom,
    ComparisonAtom,
    EventAtom,
    InsideExpr,
    NotExpr,
    PrefixExpr,
    RangePrefixExpr,
    ShiftExpr,
    TemporalExpr,
    WithinExpr,
    WithinSpanExpr,
)
from tquery._codes import collect_unique_codes, expand_codes
from tquery._parser import parse
from tquery._stringify import (
    _prepare,
    stringify_durations,
    stringify_order,
    stringify_time,
)
from tquery._types import TQueryStringError, _merge_kwargs


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------

@dataclass
class StringMatch:
    """Per-person matching positions from evaluating an AST node on strings.

    positions maps pid -> frozenset of matching string positions.
    An empty frozenset means the person exists but has no matches.
    """
    positions: dict[Any, frozenset[int]]

    @property
    def pids(self) -> set:
        """Person IDs with at least one matching position."""
        return {pid for pid, pos in self.positions.items() if pos}


# ---------------------------------------------------------------------------
# Reverse mapping: codes dict -> {code: label}
# ---------------------------------------------------------------------------

def _build_reverse_map(
    codes: dict[str, str | list[str]],
    all_codes: list[str] | None = None,
) -> tuple[dict[str, str], list[tuple[str, str]]]:
    """Build reverse mapping from concrete codes to labels.

    Returns:
        (reverse, wildcard_prefixes) where reverse maps concrete codes
        to their label, and wildcard_prefixes is a list of
        (prefix, label) for patterns that couldn't be fully expanded.
    """
    reverse: dict[str, str] = {}
    wildcard_prefixes: list[tuple[str, str]] = []

    for label, patterns in codes.items():
        if isinstance(patterns, str):
            patterns = [patterns]
        for pattern in patterns:
            expanded = expand_codes(pattern, all_codes=all_codes)
            for code in expanded:
                if code.endswith("*"):
                    wildcard_prefixes.append((code[:-1], label))
                elif code not in reverse:
                    reverse[code] = label

    return reverse, wildcard_prefixes


def _resolve_labels(
    code_patterns: tuple[str, ...],
    reverse: dict[str, str],
    wildcard_prefixes: list[tuple[str, str]],
    all_codes: list[str] | None = None,
    variables: dict[str, Any] | None = None,
) -> set[str]:
    """Map expression code patterns to stringify label characters."""
    labels: set[str] = set()

    for pattern in code_patterns:
        expanded = expand_codes(pattern, all_codes=all_codes, variables=variables)
        for code in expanded:
            if code in reverse:
                labels.add(reverse[code])
            elif code.endswith("*"):
                # Wildcard in expression: check if any label pattern overlaps
                prefix = code[:-1]
                # Check concrete codes in reverse map
                for rc, lbl in reverse.items():
                    if rc.startswith(prefix):
                        labels.add(lbl)
                # Check wildcard prefixes
                for wp, lbl in wildcard_prefixes:
                    if wp.startswith(prefix) or prefix.startswith(wp):
                        labels.add(lbl)
            else:
                # Try wildcard prefix matching
                for wp, lbl in wildcard_prefixes:
                    if code.startswith(wp):
                        labels.add(lbl)

    return labels


# ---------------------------------------------------------------------------
# Pre-compute label positions from strings
# ---------------------------------------------------------------------------

def _index_order_strings(
    strings: pd.Series,
    labels: set[str],
    no_event: str,
) -> dict[str, dict[Any, frozenset[int]]]:
    """Build label -> {pid -> positions} from stringify_order output."""
    label_positions: dict[str, dict[Any, frozenset[int]]] = {
        lbl: {} for lbl in labels
    }
    for pid, s in strings.items():
        if not isinstance(s, str):
            continue
        for i, ch in enumerate(s):
            if ch != no_event and ch in label_positions:
                if pid not in label_positions[ch]:
                    label_positions[ch][pid] = frozenset()
                label_positions[ch][pid] = label_positions[ch][pid] | {i}
    return label_positions


def _index_time_strings(
    strings: pd.DataFrame,
    labels: set[str],
    no_event: str,
) -> dict[str, dict[Any, frozenset[int]]]:
    """Build label -> {pid -> positions} from unmerged stringify_time/durations."""
    label_positions: dict[str, dict[Any, frozenset[int]]] = {
        lbl: {} for lbl in labels
    }
    for lbl in labels:
        if lbl not in strings.columns:
            continue
        for pid, s in strings[lbl].items():
            if not isinstance(s, str):
                continue
            positions = frozenset(
                i for i, ch in enumerate(s) if ch != no_event
            )
            if positions:
                label_positions[lbl][pid] = positions
    return label_positions


# ---------------------------------------------------------------------------
# StringEvaluator
# ---------------------------------------------------------------------------

class StringEvaluator:
    """Evaluates a parsed AST against stringify output strings.

    Each AST node is dispatched to a handler that returns a StringMatch.
    Results are cached by AST node identity (frozen dataclass = hashable).
    """

    def __init__(
        self,
        strings: pd.Series | pd.DataFrame,
        codes: dict[str, str | list[str]],
        mode: str = "order",
        *,
        step: int = 90,
        no_event: str = " ",
        variables: dict[str, Any] | None = None,
        all_codes: list[str] | None = None,
        position_dates: dict[Any, tuple] | None = None,
    ) -> None:
        self._strings = strings
        self._codes = codes
        self._mode = mode
        self._step = step
        self._no_event = no_event
        self._variables = variables or {}
        self._all_codes = all_codes
        self._cache: dict[ASTNode, StringMatch] = {}
        self._position_dates = position_dates

        # Build reverse mapping
        self._reverse, self._wildcard_prefixes = _build_reverse_map(
            codes, all_codes=all_codes,
        )

        # All labels from the codes dict
        self._all_labels = set(codes.keys())

        # All person IDs from the strings index
        if isinstance(strings, pd.DataFrame):
            self._all_pids: set = set(strings.index)
        else:
            self._all_pids = set(strings.index)

        # Pre-compute label positions
        if mode == "order":
            self._label_positions = _index_order_strings(
                strings, self._all_labels, no_event,
            )
        else:
            if not isinstance(strings, pd.DataFrame):
                raise TQueryStringError(
                    f"Mode '{mode}' requires unmerged DataFrame "
                    f"(stringify with merge=False), got Series"
                )
            self._label_positions = _index_time_strings(
                strings, self._all_labels, no_event,
            )

    def evaluate(self, node: ASTNode) -> StringMatch:
        """Evaluate an AST node, returning a StringMatch."""
        cached = self._cache.get(node)
        if cached is not None:
            return cached
        result = self._dispatch(node)
        self._cache[node] = result
        return result

    def _dispatch(self, node: ASTNode) -> StringMatch:
        if isinstance(node, CodeAtom):
            return self._eval_code(node)
        elif isinstance(node, EventAtom):
            raise TQueryStringError(
                "`event`/`events` atom is not supported by the string evaluator"
            )
        elif isinstance(node, ComparisonAtom):
            return self._eval_comparison(node)
        elif isinstance(node, PrefixExpr):
            return self._eval_prefix(node)
        elif isinstance(node, RangePrefixExpr):
            return self._eval_range_prefix(node)
        elif isinstance(node, NotExpr):
            return self._eval_not(node)
        elif isinstance(node, BinaryLogical):
            return self._eval_logical(node)
        elif isinstance(node, TemporalExpr):
            return self._eval_temporal(node)
        elif isinstance(node, WithinExpr):
            return self._eval_within(node)
        elif isinstance(node, InsideExpr):
            return self._eval_inside(node)
        elif isinstance(node, BetweenExpr):
            raise TQueryStringError(
                "`between EXPR and EXPR` is not supported by the string evaluator"
            )
        elif isinstance(node, WithinSpanExpr):
            raise TQueryStringError(
                "`within EXPR` (positional span) is not supported by the string evaluator"
            )
        elif isinstance(node, ShiftExpr):
            raise TQueryStringError(
                "Shifted anchor dates (`± N days`) are not supported by the string evaluator"
            )
        else:
            raise TypeError(f"Unknown AST node type: {type(node)}")

    def _eval_code(self, node: CodeAtom) -> StringMatch:
        labels = _resolve_labels(
            node.codes,
            self._reverse,
            self._wildcard_prefixes,
            all_codes=self._all_codes,
            variables=self._variables,
        )
        # Merge positions from all matching labels
        positions: dict[Any, frozenset[int]] = {}
        for lbl in labels:
            lbl_pos = self._label_positions.get(lbl, {})
            for pid, pos in lbl_pos.items():
                if pid in positions:
                    positions[pid] = positions[pid] | pos
                else:
                    positions[pid] = pos
        return StringMatch(positions)

    def _eval_comparison(self, node: ComparisonAtom) -> StringMatch:
        raise TQueryStringError(
            f"Column comparisons ('{node.column} {node.op} {node.value}') "
            f"are not supported in string mode"
        )

    def _eval_prefix(self, node: PrefixExpr) -> StringMatch:
        child = self.evaluate(node.child)
        positions: dict[Any, frozenset[int]] = {}

        for pid, pos in child.positions.items():
            if not pos:
                continue
            count = len(pos)
            sorted_pos = sorted(pos)

            if node.kind == "min":
                if count >= node.n:
                    positions[pid] = pos
            elif node.kind == "max":
                if count <= node.n:
                    positions[pid] = pos
            elif node.kind == "exactly":
                if count == node.n:
                    positions[pid] = pos
            elif node.kind == "ordinal":
                if node.n > 0 and count >= node.n:
                    positions[pid] = frozenset({sorted_pos[node.n - 1]})
                elif node.n < 0 and count >= abs(node.n):
                    # -1 = last, -2 = 2nd-to-last
                    positions[pid] = frozenset({sorted_pos[node.n]})
            elif node.kind == "first":
                if count > 0:
                    positions[pid] = frozenset(sorted_pos[:node.n])
            elif node.kind == "last":
                if count > 0:
                    positions[pid] = frozenset(sorted_pos[-node.n:])

        return StringMatch(positions)

    def _eval_range_prefix(self, node: RangePrefixExpr) -> StringMatch:
        child = self.evaluate(node.child)
        positions: dict[Any, frozenset[int]] = {}
        for pid, pos in child.positions.items():
            count = len(pos)
            if node.min_n <= count <= node.max_n:
                positions[pid] = pos
        return StringMatch(positions)

    def _eval_not(self, node: NotExpr) -> StringMatch:
        child = self.evaluate(node.child)
        child_pids = child.pids
        positions: dict[Any, frozenset[int]] = {}
        for pid in self._all_pids:
            if pid not in child_pids:
                # Return all positions for this person
                positions[pid] = self._all_positions_for(pid)
        return StringMatch(positions)

    def _eval_logical(self, node: BinaryLogical) -> StringMatch:
        left = self.evaluate(node.left)
        right = self.evaluate(node.right)

        left_pids = left.pids
        right_pids = right.pids

        if node.op == "and":
            matching_pids = left_pids & right_pids
        else:  # or
            matching_pids = left_pids | right_pids

        positions: dict[Any, frozenset[int]] = {}
        for pid in matching_pids:
            lp = left.positions.get(pid, frozenset())
            rp = right.positions.get(pid, frozenset())
            positions[pid] = lp | rp
        return StringMatch(positions)

    def _eval_temporal(self, node: TemporalExpr) -> StringMatch:
        left = self.evaluate(node.left)
        right = self.evaluate(node.right)

        if node.op == "simultaneously":
            if self._mode == "order":
                raise TQueryStringError(
                    "'simultaneously' is not supported for stringify_order "
                    "(no time information). Use stringify_time or "
                    "stringify_durations instead."
                )
            positions: dict[Any, frozenset[int]] = {}
            for pid in left.pids & right.pids:
                overlap = left.positions[pid] & right.positions[pid]
                if overlap:
                    positions[pid] = left.positions[pid]
            return StringMatch(positions)

        # before / after: compare first occurrence positions (or dates)
        positions = {}
        for pid in left.pids & right.pids:
            left_first_pos = min(left.positions[pid])
            right_first_pos = min(right.positions[pid])

            # When position_dates are available, compare actual dates
            # (strict < / >) to correctly handle same-day events.
            # Without dates, fall back to position comparison which
            # cannot distinguish same-day events from sequential ones.
            if self._position_dates is not None and pid in self._position_dates:
                dates = self._position_dates[pid]
                left_date = dates[left_first_pos]
                right_date = dates[right_first_pos]
                if node.op == "before" and left_date < right_date:
                    positions[pid] = left.positions[pid]
                elif node.op == "after" and left_date > right_date:
                    positions[pid] = left.positions[pid]
            else:
                if node.op == "before" and left_first_pos < right_first_pos:
                    positions[pid] = left.positions[pid]
                elif node.op == "after" and left_first_pos > right_first_pos:
                    positions[pid] = left.positions[pid]
        return StringMatch(positions)

    def _eval_within(self, node: WithinExpr) -> StringMatch:
        if self._mode == "order":
            raise TQueryStringError(
                "'within N days' is not supported for stringify_order "
                "(no time information). Use stringify_time or "
                "stringify_durations instead."
            )

        child = self.evaluate(node.child)
        max_pos_dist = node.days // self._step
        min_pos_dist = node.min_days // self._step

        if node.ref is None:
            # Within N days of first event per person
            positions: dict[Any, frozenset[int]] = {}
            for pid, child_pos in child.positions.items():
                if not child_pos:
                    continue
                all_pos = self._all_positions_for(pid)
                if not all_pos:
                    continue
                first_pos = min(all_pos)
                matched = frozenset(
                    p for p in child_pos
                    if min_pos_dist <= abs(p - first_pos) <= max_pos_dist
                )
                if matched:
                    positions[pid] = matched
            return StringMatch(positions)

        ref = self.evaluate(node.ref)
        positions = {}
        for pid in child.pids & ref.pids:
            child_pos = child.positions[pid]
            ref_pos = ref.positions[pid]
            matched: set[int] = set()
            for cp in child_pos:
                for rp in ref_pos:
                    dist = cp - rp
                    abs_dist = abs(dist)
                    if abs_dist < min_pos_dist or abs_dist > max_pos_dist:
                        continue
                    if node.direction == "after" and dist > 0:
                        matched.add(cp)
                        break
                    elif node.direction == "before" and dist < 0:
                        matched.add(cp)
                        break
                    elif node.direction in ("around", None):
                        matched.add(cp)
                        break
            if matched:
                positions[pid] = frozenset(matched)
        return StringMatch(positions)

    def _eval_inside(self, node: InsideExpr) -> StringMatch:
        if self._mode != "order":
            raise TQueryStringError(
                "'inside/outside N events' is only supported for "
                "stringify_order (where position = event number). "
                f"Got mode '{self._mode}'."
            )

        child = self.evaluate(node.child)
        ref = self.evaluate(node.ref)
        positions: dict[Any, frozenset[int]] = {}

        for pid in child.pids & ref.pids:
            child_pos = child.positions[pid]
            ref_pos = ref.positions[pid]
            matched: set[int] = set()

            for cp in child_pos:
                is_inside = False
                for rp in ref_pos:
                    dist = cp - rp
                    if node.direction == "after":
                        if node.min_events <= dist <= node.max_events:
                            is_inside = True
                            break
                    elif node.direction == "before":
                        if -node.max_events <= dist <= -node.min_events:
                            is_inside = True
                            break
                    else:  # around: signed offsets
                        if node.min_events <= dist <= node.max_events:
                            is_inside = True
                            break

                if node.inside and is_inside:
                    matched.add(cp)
                elif not node.inside and not is_inside:
                    matched.add(cp)

            if matched:
                positions[pid] = frozenset(matched)
        return StringMatch(positions)

    def _all_positions_for(self, pid: Any) -> frozenset[int]:
        """Get all positions (any label) for a person."""
        all_pos: set[int] = set()
        for lbl_pos in self._label_positions.values():
            if pid in lbl_pos:
                all_pos |= lbl_pos[pid]
        return frozenset(all_pos)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def string_query(
    expr: str,
    strings: pd.Series | pd.DataFrame,
    codes: dict[str, str | list[str]],
    mode: str = "order",
    *,
    step: int = 90,
    no_event: str = " ",
    variables: dict[str, Any] | None = None,
    all_codes: list[str] | None = None,
    position_dates: dict[Any, tuple] | None = None,
) -> set:
    """Evaluate a tquery expression against stringify output strings.

    Returns the set of matching person IDs, analogous to
    tquery(df, expr).pids.

    Args:
        expr: A tquery expression string, e.g. 'K50 before K51'.
        strings: Output of stringify_order (Series) or
                 stringify_time/durations with merge=False (DataFrame).
        codes: The same codes dict used to produce the strings.
        mode: "order", "time", or "durations".
        step: Days per character position (for time/durations mode).
        no_event: Empty-slot character (for time/durations mode).
        variables: For @variable references in expressions.
        all_codes: All unique codes from the original dataset
                   (needed for wildcard expansion in expressions).
        position_dates: Optional per-person date tuples for order mode.
            Maps pid to a tuple of dates aligned with string positions.
            When provided, 'before'/'after' compare actual dates
            (strict </>), correctly handling same-day events.
            When omitted, positions are compared directly, which may
            give incorrect results for events on the same date
            (stringify_order assigns different positions to same-day
            events based on row order).

    Returns:
        Set of person IDs matching the expression.

    Raises:
        TQueryStringError: If the expression uses features not
            supported by the chosen stringify mode.
    """
    ast = parse(expr)
    evaluator = StringEvaluator(
        strings, codes, mode,
        step=step, no_event=no_event,
        variables=variables, all_codes=all_codes,
        position_dates=position_dates,
    )
    result = evaluator.evaluate(ast)
    return result.pids


def string_query_auto(
    df: pd.DataFrame,
    expr: str,
    codes: dict[str, str | list[str]],
    mode: str = "order",
    *,
    pid: str = "pid",
    date: str = "start_date",
    cols: str | list[str] | None = None,
    sep: str | None = None,
    step: int = 90,
    no_event: str = " ",
    variables: dict[str, Any] | None = None,
    config: "TQueryConfig | None" = None,
    **stringify_kwargs: Any,
) -> set:
    """Auto-stringify then evaluate expression against the strings.

    Convenience wrapper that calls the appropriate stringify function
    and then evaluates the expression against the resulting strings.

    Args:
        df: Event-level DataFrame.
        expr: A tquery expression string.
        codes: Dict mapping labels to code patterns.
        mode: "order", "time", or "durations".
        pid, date, cols, sep: DataFrame column configuration.
        step: Days per position (for time/durations).
        no_event: Empty-slot character.
        variables: For @variable references.
        config: Optional TQueryConfig.
        **stringify_kwargs: Extra kwargs passed to the stringify function
            (e.g., event_end, event_duration for durations mode).

    Returns:
        Set of matching person IDs.
    """
    kw = _merge_kwargs(config, pid=pid, date=date, cols=cols, sep=sep)
    pid_col = kw["pid"]
    date_col = kw["date"]
    stringify_cols = kw.get("cols")
    stringify_sep = kw.get("sep")

    # Resolve cols for collect_unique_codes
    if stringify_cols is None:
        resolved_cols = [
            c for c in df.columns
            if c not in (pid_col, date_col) and df[c].dtype == object
        ]
    elif isinstance(stringify_cols, str):
        resolved_cols = [stringify_cols]
    else:
        resolved_cols = list(stringify_cols)

    all_codes = collect_unique_codes(df, resolved_cols, stringify_sep)

    shared = dict(
        cols=stringify_cols, pid=pid_col,
        event_start=date_col, sep=stringify_sep,
    )

    position_dates = None

    if mode == "order":
        strings = stringify_order(df, codes, **shared)
        # Build position_dates for accurate before/after with same-day events
        prep = _prepare(df, codes, **shared)
        position_dates = {}
        for pid_val, group in prep.df.groupby(prep.pid_col):
            position_dates[pid_val] = tuple(group[prep.date_col].values)
    elif mode == "time":
        strings = stringify_time(
            df, codes, **shared,
            step=step, no_event=no_event, merge=False,
        )
    elif mode == "durations":
        strings = stringify_durations(
            df, codes, **shared,
            step=step, no_event=no_event, merge=False,
            **stringify_kwargs,
        )
    else:
        raise ValueError(f"Unknown mode: {mode!r}. Use 'order', 'time', or 'durations'.")

    return string_query(
        expr, strings, codes, mode,
        step=step, no_event=no_event,
        variables=variables, all_codes=all_codes,
        position_dates=position_dates,
    )


def cross_validate(
    df: pd.DataFrame,
    expr: str,
    codes: dict[str, str | list[str]],
    mode: str = "order",
    *,
    pid: str = "pid",
    date: str = "start_date",
    cols: str | list[str] | None = None,
    sep: str | None = None,
    step: int = 90,
    no_event: str = " ",
    variables: dict[str, Any] | None = None,
    config: "TQueryConfig | None" = None,
    **stringify_kwargs: Any,
) -> tuple[set, set, bool]:
    """Run both DataFrame and string evaluators, compare results.

    Returns:
        (df_pids, string_pids, match) where match is True if identical.
    """
    # Avoid circular import
    from tquery import tquery as _tquery

    tq_kw: dict[str, Any] = {}
    if pid != "pid":
        tq_kw["pid"] = pid
    if date != "start_date":
        tq_kw["date"] = date
    if cols is not None:
        tq_kw["cols"] = cols
    if sep is not None:
        tq_kw["sep"] = sep
    if variables is not None:
        tq_kw["variables"] = variables
    if config is not None:
        tq_kw["config"] = config

    df_pids = _tquery(df, expr, **tq_kw).pids
    str_pids = string_query_auto(
        df, expr, codes, mode,
        pid=pid, date=date, cols=cols, sep=sep,
        step=step, no_event=no_event,
        variables=variables, config=config,
        **stringify_kwargs,
    )
    return df_pids, str_pids, df_pids == str_pids
