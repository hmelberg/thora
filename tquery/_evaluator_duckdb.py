"""DuckDB backend for tquery.

Compiles the parsed AST into a single SQL query and executes it via
DuckDB. Same parser, same AST, same goldens — but a structurally
different evaluation model: row-by-row computation in pandas/polars/R
is replaced by set-based SQL.

Public entry point: ``tquery.tquery_duckdb(df, expr, ...)``.
"""

from __future__ import annotations

from typing import Any

import duckdb

from tquery._ast import (
    AggregateExpr,
    ASTNode,
    BetweenExpr,
    BinaryLogical,
    CodeAtom,
    ComparisonAtom,
    EventAtom,
    InsideExpr,
    NotExpr,
    PrefixExpr,
    Quantifier,
    RangePrefixExpr,
    ShiftExpr,
    TemporalExpr,
    WithinExpr,
    WithinSpanExpr,
)
from tquery._codes import expand_all_codes, resolve_columns

# --- helpers --------------------------------------------------------------


def _sql_str(s: str) -> str:
    """Escape a Python string for SQL single-quoted literal."""
    return "'" + s.replace("'", "''") + "'"


def _sql_in_list(values: list[str]) -> str:
    return "(" + ", ".join(_sql_str(v) for v in values) + ")"


def _unwrap_quantifier(node: ASTNode) -> tuple[ASTNode, bool]:
    if isinstance(node, Quantifier):
        return node.child, node.kind == "every"
    return node, False


def _unwrap_shift(node: ASTNode) -> tuple[ASTNode, int]:
    offset = 0
    while isinstance(node, ShiftExpr):
        offset += node.offset_days
        node = node.child
    return node, offset


_AGG_FUNC_SQL = {
    "sum": "SUM",
    "mean": "AVG",
    "avg": "AVG",
    "min": "MIN",
    "max": "MAX",
    "median": "MEDIAN",
    "sd": "STDDEV_SAMP",
    "var": "VAR_SAMP",
    "count": "COUNT",
    "n": "COUNT",
}


# --- compiler -------------------------------------------------------------


class DuckDBCompiler:
    """Compile AST → SQL. `compile(node)` returns a SQL fragment that
    SELECTs from the registered events view all rows belonging to
    matching persons (rows whose pid appears in `pids` is the
    person-level result).
    """

    def __init__(
        self,
        events_view: str,
        pid: str,
        date: str,
        cols: list[str],
        all_codes: list[str],
        variables: dict[str, Any],
    ) -> None:
        self.events = events_view
        self.pid = pid
        self.date = date
        self.cols = cols
        self.all_codes = all_codes
        self.variables = variables

    # ---- public --------

    def compile(self, node: ASTNode) -> str:
        return self._dispatch(node)

    def matching_pids_sql(self, node: ASTNode) -> str:
        rows_sql = self.compile(node)
        return f"SELECT DISTINCT {self.pid} AS pid FROM ({rows_sql}) _t"

    # ---- dispatch --------

    def _dispatch(self, node: ASTNode) -> str:
        method = {
            CodeAtom:        self._code_atom,
            EventAtom:       self._event_atom,
            ComparisonAtom:  self._comparison_atom,
            AggregateExpr:   self._aggregate_standalone,
            PrefixExpr:      self._prefix,
            RangePrefixExpr: self._range_prefix,
            NotExpr:         self._not_expr,
            BinaryLogical:   self._logical,
            TemporalExpr:    self._temporal,
            WithinExpr:      self._within,
            WithinSpanExpr:  self._within_span,
            InsideExpr:      self._inside,
            BetweenExpr:     self._between,
        }.get(type(node))
        if method is None:
            raise TypeError(f"Unsupported AST node in DuckDB backend: {type(node).__name__}")
        return method(node)

    # ---- atoms --------

    def _code_atom(self, node: CodeAtom) -> str:
        codes = expand_all_codes(node.codes, all_codes=self.all_codes, variables=self.variables)
        cols = (
            resolve_columns(list(node.columns), self.cols + [self.pid, self.date])
            if node.columns else self.cols
        )
        exact = [c for c in codes if not c.endswith("*")]
        wildcards = [c[:-1] for c in codes if c.endswith("*")]
        parts = []
        for col in cols:
            if exact:
                parts.append(f"{col} IN {_sql_in_list(exact)}")
            for w in wildcards:
                parts.append(f"{col} LIKE {_sql_str(w + '%')}")
        if not parts:
            return f"SELECT * FROM {self.events} WHERE FALSE"
        return f"SELECT * FROM {self.events} WHERE " + " OR ".join(parts)

    def _event_atom(self, node: EventAtom) -> str:
        return f"SELECT * FROM {self.events}"

    def _comparison_atom(self, node: ComparisonAtom) -> str:
        return (
            f"SELECT * FROM {self.events} "
            f"WHERE {node.column} IS NOT NULL "
            f"AND {node.column} {node.op} {node.value}"
        )

    # ---- aggregate (standalone) --------

    def _aggregate_standalone(self, node: AggregateExpr, row_filter_sql: str | None = None) -> str:
        return self._aggregate_with_filter(node, row_filter_sql=row_filter_sql)

    def _aggregate_with_filter(
        self, node: AggregateExpr, row_filter_sql: str | None,
    ) -> str:
        """Compute per-pid aggregate, threshold, return rows of matching pids."""
        col = node.column
        per_pid_sql = self._per_pid_agg_sql(node, row_filter_sql)
        return (
            f"SELECT * FROM {self.events} "
            f"WHERE {self.pid} IN (SELECT {self.pid} FROM ({per_pid_sql}) _a "
            f"WHERE _agg IS NOT NULL AND _agg {node.op} {node.value})"
        )

    def _per_pid_agg_sql(
        self, node: AggregateExpr, row_filter_sql: str | None = None,
    ) -> str:
        """Returns SQL: SELECT pid, _agg FROM (...) — per-pid scalar agg."""
        source = self.events if row_filter_sql is None else f"({row_filter_sql})"

        if node.func == "range":
            expr = f"MAX({node.column}) - MIN({node.column})"
            return (
                f"SELECT {self.pid}, {expr} AS _agg "
                f"FROM {source} GROUP BY {self.pid}"
            )
        if node.func == "rise":
            return self._rise_fall_per_pid_sql(node, source, kind="rise")
        if node.func == "fall":
            return self._rise_fall_per_pid_sql(node, source, kind="fall")
        sql_fn = _AGG_FUNC_SQL[node.func]
        col_expr = node.column
        return (
            f"SELECT {self.pid}, {sql_fn}({col_expr}) AS _agg "
            f"FROM {source} GROUP BY {self.pid}"
        )

    def _rise_fall_per_pid_sql(self, node, source: str, kind: str) -> str:
        """Per-pid max drawup (rise) or max drawdown (fall) via a
        two-pass structure: window function in the inner SELECT
        (DuckDB rejects aggregate-of-window in one pass), then MAX
        over the result grouped by pid.
        """
        col = node.column
        if kind == "rise":
            inner_expr = (
                f"({col} - MIN({col}) OVER (PARTITION BY {self.pid} ORDER BY {self.date}, __rid__))"
            )
        else:
            inner_expr = (
                f"(MAX({col}) OVER (PARTITION BY {self.pid} ORDER BY {self.date}, __rid__) - {col})"
            )
        inner = (
            f"SELECT {self.pid}, {inner_expr} AS _delta "
            f"FROM {source} "
            f"WHERE {col} IS NOT NULL"
        )
        return (
            f"SELECT {self.pid}, MAX(_delta) AS _agg FROM ({inner}) _inner "
            f"GROUP BY {self.pid}"
        )

    # ---- prefix --------

    def _prefix(self, node: PrefixExpr) -> str:
        child_sql = self.compile(node.child)
        kind = node.kind; n = node.n
        if kind in ("min", "max", "exactly"):
            op = {"min": ">=", "max": "<=", "exactly": "="}[kind]
            return (
                f"SELECT * FROM ({child_sql}) _c "
                f"WHERE {self.pid} IN ("
                f"SELECT {self.pid} FROM ({child_sql}) _c2 "
                f"GROUP BY {self.pid} HAVING COUNT(*) {op} {n})"
            )
        # ordinal / first / last — use row_number within each pid.
        if kind == "first":
            return (
                f"SELECT _c.* FROM (SELECT *, ROW_NUMBER() OVER (PARTITION BY {self.pid} "
                f"ORDER BY {self.date}, __rid__) AS _rn FROM ({child_sql}) _i) _c "
                f"WHERE _c._rn <= {n}"
            )
        if kind == "last":
            return (
                f"SELECT _c.* FROM (SELECT *, "
                f"ROW_NUMBER() OVER (PARTITION BY {self.pid} ORDER BY {self.date} DESC, __rid__ DESC) AS _rn "
                f"FROM ({child_sql}) _i) _c WHERE _c._rn <= {n}"
            )
        if kind == "ordinal":
            if n > 0:
                return (
                    f"SELECT _c.* FROM (SELECT *, ROW_NUMBER() OVER (PARTITION BY {self.pid} "
                    f"ORDER BY {self.date}, __rid__) AS _rn FROM ({child_sql}) _i) _c "
                    f"WHERE _c._rn = {n}"
                )
            # negative ordinal: nth from end
            return (
                f"SELECT _c.* FROM (SELECT *, ROW_NUMBER() OVER (PARTITION BY {self.pid} "
                f"ORDER BY {self.date} DESC, __rid__ DESC) AS _rn FROM ({child_sql}) _i) _c "
                f"WHERE _c._rn = {abs(n)}"
            )
        raise ValueError(f"Unknown prefix kind: {kind}")

    def _range_prefix(self, node: RangePrefixExpr) -> str:
        child_sql = self.compile(node.child)
        return (
            f"SELECT * FROM ({child_sql}) _c "
            f"WHERE {self.pid} IN ("
            f"SELECT {self.pid} FROM ({child_sql}) _c2 "
            f"GROUP BY {self.pid} "
            f"HAVING COUNT(*) BETWEEN {node.min_n} AND {node.max_n})"
        )

    # ---- logical / not --------

    def _not_expr(self, node: NotExpr) -> str:
        child_sql = self.compile(node.child)
        return (
            f"SELECT * FROM {self.events} "
            f"WHERE {self.pid} NOT IN (SELECT DISTINCT {self.pid} FROM ({child_sql}) _c)"
        )

    def _logical(self, node: BinaryLogical) -> str:
        left_sql = self.compile(node.left)
        right_sql = self.compile(node.right)
        op = "INTERSECT" if node.op == "and" else "UNION"
        # Person-level: pids matching both/either, then return rows from BOTH sides
        # that belong to those persons. Semantics: existing evaluator returns
        # (mask_left | mask_right) restricted to matching persons.
        return (
            f"SELECT * FROM {self.events} _e "
            f"WHERE _e.{self.pid} IN ("
            f"SELECT {self.pid} FROM ({left_sql}) _l "
            f"{op} "
            f"SELECT {self.pid} FROM ({right_sql}) _r) "
            f"AND (_e.__rid__ IN (SELECT _l.__rid__ FROM ({left_sql}) _l) OR "
            f"_e.__rid__ IN (SELECT _r.__rid__ FROM ({right_sql}) _r))"
        )

    # ---- temporal --------

    def _temporal(self, node: TemporalExpr) -> str:
        left_inner, every_left = _unwrap_quantifier(node.left)
        if isinstance(left_inner, ShiftExpr):
            raise TypeError("Shifted anchors only valid on RHS of temporal ops")
        right_inner, every_right = _unwrap_quantifier(node.right)
        right_inner, ref_offset = _unwrap_shift(right_inner)

        left_sql = self.compile(left_inner)
        right_sql = self.compile(right_inner)
        right_date_expr = f"{self.date} + INTERVAL '{ref_offset}' DAY" if ref_offset else self.date

        if node.op == "simultaneously":
            l_dates = f"SELECT DISTINCT {self.pid}, {self.date} AS d FROM ({left_sql}) _l"
            r_dates = f"SELECT DISTINCT {self.pid}, {right_date_expr} AS d FROM ({right_sql}) _r"
            if every_left and every_right:
                # dates_left == dates_right (as sets)
                clause = (
                    f"SELECT _ll.{self.pid} FROM ({l_dates}) _ll "
                    f"GROUP BY _ll.{self.pid} HAVING "
                    f"COUNT(*) = (SELECT COUNT(*) FROM ({r_dates}) _rr WHERE _rr.{self.pid} = _ll.{self.pid}) "
                    f"AND NOT EXISTS (SELECT 1 FROM ({l_dates}) _l2 WHERE _l2.{self.pid} = _ll.{self.pid} "
                    f"AND _l2.d NOT IN (SELECT d FROM ({r_dates}) _r2 WHERE _r2.{self.pid} = _ll.{self.pid}))"
                )
            elif every_left:
                clause = (
                    f"SELECT _ll.{self.pid} FROM ({l_dates}) _ll "
                    f"WHERE NOT EXISTS (SELECT 1 FROM ({l_dates}) _l2 WHERE _l2.{self.pid} = _ll.{self.pid} "
                    f"AND _l2.d NOT IN (SELECT d FROM ({r_dates}) _r2 WHERE _r2.{self.pid} = _ll.{self.pid})) "
                    f"GROUP BY _ll.{self.pid}"
                )
            elif every_right:
                clause = (
                    f"SELECT _rr.{self.pid} FROM ({r_dates}) _rr "
                    f"WHERE NOT EXISTS (SELECT 1 FROM ({r_dates}) _r2 WHERE _r2.{self.pid} = _rr.{self.pid} "
                    f"AND _r2.d NOT IN (SELECT d FROM ({l_dates}) _l2 WHERE _l2.{self.pid} = _rr.{self.pid})) "
                    f"GROUP BY _rr.{self.pid}"
                )
            else:
                clause = (
                    f"SELECT _ll.{self.pid} FROM ({l_dates}) _ll "
                    f"JOIN ({r_dates}) _rr ON _ll.{self.pid} = _rr.{self.pid} AND _ll.d = _rr.d "
                    f"GROUP BY _ll.{self.pid}"
                )
            return (
                f"SELECT * FROM ({left_sql}) _l "
                f"WHERE {self.pid} IN ({clause})"
            )

        # before / after — per-pid min/max comparisons
        l_agg = (
            f"SELECT {self.pid}, MIN({self.date}) AS l_min, MAX({self.date}) AS l_max "
            f"FROM ({left_sql}) _l GROUP BY {self.pid}"
        )
        r_agg = (
            f"SELECT {self.pid}, MIN({right_date_expr}) AS r_min, "
            f"MAX({right_date_expr}) AS r_max "
            f"FROM ({right_sql}) _r GROUP BY {self.pid}"
        )
        if node.op == "before":
            if every_left and every_right:
                pred = "l_max < r_min"
            elif every_left:
                pred = "l_max < r_max"
            else:
                pred = "l_min < r_min"
        else:
            if every_left and every_right:
                pred = "l_min > r_max"
            elif every_right:
                pred = "l_max > r_max"
            else:
                pred = "l_min > r_min"
        matching = (
            f"SELECT _l.{self.pid} FROM ({l_agg}) _l "
            f"JOIN ({r_agg}) _r USING ({self.pid}) WHERE {pred}"
        )
        return (
            f"SELECT * FROM ({left_sql}) _ll WHERE {self.pid} IN ({matching})"
        )

    # ---- within (days) --------

    def _within(self, node: WithinExpr) -> str:
        if isinstance(node.child, AggregateExpr):
            return self._within_aggregate(node)
        child_inner, every_left = _unwrap_quantifier(node.child)
        if isinstance(child_inner, ShiftExpr):
            raise TypeError("Shifted anchor cannot be LHS of a window")
        ref_offset = 0
        every_right = False
        if node.ref is not None:
            ref_inner, every_right = _unwrap_quantifier(node.ref)
            ref_inner, ref_offset = _unwrap_shift(ref_inner)
            ref_sql = self.compile(ref_inner)
        else:
            ref_sql = None

        child_sql = self.compile(child_inner)
        if ref_sql is None:
            # First-event anchored
            first_sql = (
                f"SELECT {self.pid}, MIN({self.date}) AS _first FROM {self.events} "
                f"GROUP BY {self.pid}"
            )
            return (
                f"SELECT _c.* FROM ({child_sql}) _c "
                f"JOIN ({first_sql}) _f USING ({self.pid}) "
                f"WHERE ABS(DATE_DIFF('day', _c.{self.date}, _f._first)) "
                f"BETWEEN {node.min_days} AND {node.days}"
            )

        in_window_sql = self._rows_in_window_sql(
            child_sql, ref_sql, node.days, node.min_days,
            node.direction, ref_offset,
        )
        if every_left or every_right:
            matching_pids = self._universal_pids_sql(
                child_sql, ref_sql, in_window_sql,
                every_left, every_right,
                node.days, node.min_days, node.direction, ref_offset,
            )
            return (
                f"SELECT _c.* FROM ({child_sql}) _c "
                f"WHERE _c.{self.pid} IN ({matching_pids})"
            )
        if not node.outside:
            return in_window_sql
        # Row-level complement restricted to evaluable persons
        evaluable = f"SELECT DISTINCT {self.pid} FROM ({ref_sql}) _r"
        return (
            f"SELECT _c.* FROM ({child_sql}) _c "
            f"WHERE _c.{self.pid} IN ({evaluable}) "
            f"AND _c.__rid__ NOT IN (SELECT _w.__rid__ FROM ({in_window_sql}) _w)"
        )

    def _rows_in_window_sql(
        self, child_sql: str, ref_sql: str,
        days: int, min_days: int, direction: str | None, ref_offset_days: int,
    ) -> str:
        """Child rows that have at least one ref within [min_days, days]
        (signed if direction='around' + min_days<0)."""
        signed_around = direction == "around" and min_days < 0
        ref_date_expr = (
            f"_r.{self.date} + INTERVAL '{ref_offset_days}' DAY"
            if ref_offset_days else f"_r.{self.date}"
        )
        # Build the date-difference predicate.
        if signed_around:
            # signed delta = child - ref, must be in [min_days, days]
            pred = (
                f"DATE_DIFF('day', {ref_date_expr}, _c.{self.date}) "
                f"BETWEEN {min_days} AND {days}"
            )
        elif direction == "after":
            pred = (
                f"DATE_DIFF('day', {ref_date_expr}, _c.{self.date}) "
                f"BETWEEN {min_days} AND {days}"
            )
        elif direction == "before":
            pred = (
                f"DATE_DIFF('day', _c.{self.date}, {ref_date_expr}) "
                f"BETWEEN {min_days} AND {days}"
            )
        else:  # around / None
            pred = (
                f"ABS(DATE_DIFF('day', _c.{self.date}, {ref_date_expr})) "
                f"BETWEEN {min_days} AND {days}"
            )
        return (
            f"SELECT DISTINCT _c.* FROM ({child_sql}) _c "
            f"JOIN ({ref_sql}) _r ON _c.{self.pid} = _r.{self.pid} "
            f"WHERE {pred}"
        )

    def _universal_pids_sql(
        self, child_sql, ref_sql, in_window_sql,
        every_left, every_right,
        days, min_days, direction, ref_offset_days,
    ) -> str:
        """SQL returning pids satisfying every_left / every_right universal modes."""
        candidate = (
            f"SELECT DISTINCT {self.pid} FROM ({child_sql}) _c "
            f"INTERSECT SELECT DISTINCT {self.pid} FROM ({ref_sql}) _r"
        )
        clauses: list[str] = [f"SELECT {self.pid} FROM ({candidate}) _cand"]
        if every_left:
            # Every child row must be in window.
            clauses.append(
                f"SELECT {self.pid} FROM (SELECT {self.pid}, COUNT(*) AS _t "
                f"FROM ({child_sql}) _c GROUP BY {self.pid}) _ct "
                f"WHERE _ct._t = (SELECT COUNT(DISTINCT _w.__rid__) FROM ({in_window_sql}) _w "
                f"WHERE _w.{self.pid} = _ct.{self.pid})"
            )
        if every_right:
            opposite = (
                "before" if direction == "after"
                else "after" if direction == "before"
                else direction
            )
            rhs_in_window = self._rows_in_window_sql(
                ref_sql, child_sql, days, min_days, opposite, -ref_offset_days,
            )
            clauses.append(
                f"SELECT {self.pid} FROM (SELECT {self.pid}, COUNT(*) AS _t "
                f"FROM ({ref_sql}) _r GROUP BY {self.pid}) _rt "
                f"WHERE _rt._t = (SELECT COUNT(DISTINCT _w.__rid__) FROM ({rhs_in_window}) _w "
                f"WHERE _w.{self.pid} = _rt.{self.pid})"
            )
        return " INTERSECT ".join(f"({c})" for c in clauses)

    def _within_aggregate(self, node: WithinExpr) -> str:
        agg_node: AggregateExpr = node.child  # type: ignore[assignment]
        sliding = node.direction is None and node.ref is None
        if sliding:
            if node.outside:
                raise TypeError("`outside` over a sliding aggregate is not supported")
            return self._sliding_days_aggregate_sql(agg_node, node.days)
        # Anchored: build a row-mask of rows in the window of any ref,
        # then aggregate.
        ref_inner, _ = _unwrap_quantifier(node.ref)
        ref_inner, ref_offset = _unwrap_shift(ref_inner)
        ref_sql = self.compile(ref_inner)
        all_rows = f"SELECT * FROM {self.events}"
        in_window = self._rows_in_window_sql(
            all_rows, ref_sql, node.days, node.min_days,
            node.direction, ref_offset,
        )
        if node.outside:
            evaluable = f"SELECT DISTINCT {self.pid} FROM ({ref_sql}) _r"
            in_window = (
                f"SELECT _e.* FROM {self.events} _e "
                f"WHERE _e.{self.pid} IN ({evaluable}) "
                f"AND _e.__rid__ NOT IN (SELECT _w.__rid__ FROM ({in_window}) _w)"
            )
        return self._aggregate_with_filter(agg_node, row_filter_sql=in_window)

    def _sliding_days_aggregate_sql(self, node: AggregateExpr, days: int) -> str:
        """Sliding right-anchored day-window aggregate. For each row, the
        aggregate is computed over rows in [date - days, date]. Person
        matches if any row's window passes the threshold."""
        col = node.column
        # RANGE frames in DuckDB require a single ORDER BY column; no
        # tie-breaker on __rid__ here. For day-window aggregates this
        # is safe — same-date rows all share the same window.
        partition = f"PARTITION BY {self.pid} ORDER BY {self.date}"
        frame = f"RANGE BETWEEN INTERVAL {days} DAY PRECEDING AND CURRENT ROW"
        if node.func in ("sum", "mean", "avg", "min", "max", "median", "sd", "var", "count", "n"):
            fn = _AGG_FUNC_SQL[node.func]
            expr = f"{fn}({col}) OVER ({partition} {frame})"
        elif node.func == "range":
            expr = f"(MAX({col}) OVER ({partition} {frame}) - MIN({col}) OVER ({partition} {frame}))"
        elif node.func == "rise":
            # max drawup over the window is MAX(col - cummin_within_window).
            # Implement as a correlated subquery — heavy but correct.
            return self._sliding_days_drawup_sql(node, days, kind="rise")
        elif node.func == "fall":
            return self._sliding_days_drawup_sql(node, days, kind="fall")
        else:
            raise ValueError(f"Unsupported sliding aggregate func: {node.func}")

        rolled = (
            f"SELECT {self.pid}, {expr} AS _agg "
            f"FROM {self.events} "
            f"WHERE {col} IS NOT NULL"
        )
        return (
            f"SELECT * FROM {self.events} "
            f"WHERE {self.pid} IN ("
            f"SELECT DISTINCT {self.pid} FROM ({rolled}) _r "
            f"WHERE _r._agg IS NOT NULL AND _r._agg {node.op} {node.value})"
        )

    def _sliding_days_drawup_sql(self, node: AggregateExpr, days: int, kind: str) -> str:
        """Sliding window rise/fall implemented as a correlated lateral
        join per row. Slow but matches semantics."""
        col = node.column
        if kind == "rise":
            agg_expr = f"MAX(t.{col} - inner_min._mn)"
            inner = (
                f"(SELECT MIN(e2.{col}) AS _mn FROM {self.events} e2 "
                f"WHERE e2.{self.pid} = t.{self.pid} "
                f"AND e2.{self.date} BETWEEN t.{self.date} - INTERVAL {days} DAY AND t.{self.date}) inner_min"
            )
            # For drawup within window we need per-row rolling min, then per-row (val - rolling_min),
            # then max over rows. Use a single window function.
            # RANGE frame requires a single ORDER BY column.
            inner_expr = (
                f"({col} - MIN({col}) OVER (PARTITION BY {self.pid} ORDER BY {self.date} "
                f"RANGE BETWEEN INTERVAL {days} DAY PRECEDING AND CURRENT ROW))"
            )
        else:
            inner_expr = (
                f"(MAX({col}) OVER (PARTITION BY {self.pid} ORDER BY {self.date} "
                f"RANGE BETWEEN INTERVAL {days} DAY PRECEDING AND CURRENT ROW) - {col})"
            )
        rolled = (
            f"SELECT {self.pid}, {inner_expr} AS _delta "
            f"FROM {self.events} WHERE {col} IS NOT NULL"
        )
        return (
            f"SELECT * FROM {self.events} "
            f"WHERE {self.pid} IN ("
            f"SELECT DISTINCT {self.pid} FROM ({rolled}) _r "
            f"WHERE _r._delta IS NOT NULL AND _r._delta {node.op} {node.value})"
        )

    # ---- inside (events) --------

    def _inside(self, node: InsideExpr) -> str:
        if isinstance(node.child, AggregateExpr):
            return self._inside_aggregate(node)
        if isinstance(node.ref, ShiftExpr):
            raise TypeError("Shifted anchors cannot be the ref of an event-window")
        child_sql = self.compile(node.child)
        ref_sql = self.compile(node.ref)
        return self._event_window_sql(
            child_sql, ref_sql, node.inside,
            node.min_events, node.max_events, node.direction,
        )

    def _event_window_sql(
        self, child_sql, ref_sql, inside, min_events, max_events, direction,
    ) -> str:
        # Assign 0-based event positions per pid in input order.
        positions = (
            f"SELECT *, (ROW_NUMBER() OVER (PARTITION BY {self.pid} "
            f"ORDER BY {self.date}, __rid__) - 1) AS _pos FROM {self.events}"
        )
        # Map ref rows to their positions
        ref_pos = (
            f"SELECT _p.{self.pid}, _p._pos AS _ref_pos FROM ({positions}) _p "
            f"WHERE _p.__rid__ IN (SELECT _r.__rid__ FROM ({ref_sql}) _r)"
        )
        # Child rows + their positions, then check if there's a ref pos
        # within [min_events, max_events] offset in the right direction.
        if direction == "after":
            offset_pred = "(_c._pos - _rp._ref_pos) BETWEEN {lo} AND {hi}".format(
                lo=min_events, hi=max_events
            )
        elif direction == "before":
            offset_pred = "(_rp._ref_pos - _c._pos) BETWEEN {lo} AND {hi}".format(
                lo=min_events, hi=max_events
            )
        else:  # around
            offset_pred = "(_c._pos - _rp._ref_pos) BETWEEN {lo} AND {hi}".format(
                lo=min_events, hi=max_events
            )
        child_pos = (
            f"SELECT _p.* FROM ({positions}) _p "
            f"WHERE _p.__rid__ IN (SELECT _ch.__rid__ FROM ({child_sql}) _ch)"
        )
        in_window = (
            f"SELECT DISTINCT _c.* FROM ({child_pos}) _c "
            f"JOIN ({ref_pos}) _rp ON _c.{self.pid} = _rp.{self.pid} "
            f"WHERE {offset_pred}"
        )
        if inside:
            return f"SELECT * EXCLUDE (_pos) FROM ({in_window}) _x"
        # outside: child rows NOT in window, restricted to evaluable
        evaluable = f"SELECT DISTINCT {self.pid} FROM ({ref_sql}) _r"
        return (
            f"SELECT _c.* FROM ({child_sql}) _c "
            f"WHERE _c.{self.pid} IN ({evaluable}) "
            f"AND _c.__rid__ NOT IN (SELECT _w.__rid__ FROM ({in_window}) _w)"
        )

    def _inside_aggregate(self, node: InsideExpr) -> str:
        agg_node: AggregateExpr = node.child  # type: ignore[assignment]
        sliding = node.direction is None and node.ref is None
        if sliding:
            return self._sliding_events_aggregate_sql(agg_node, node.max_events)
        ref_sql = self.compile(node.ref)
        all_rows = f"SELECT * FROM {self.events}"
        in_window = self._event_window_sql(
            all_rows, ref_sql, True,
            node.min_events, node.max_events, node.direction,
        )
        if not node.inside:
            evaluable = f"SELECT DISTINCT {self.pid} FROM ({ref_sql}) _r"
            in_window = (
                f"SELECT _e.* FROM {self.events} _e "
                f"WHERE _e.{self.pid} IN ({evaluable}) "
                f"AND _e.__rid__ NOT IN (SELECT _w.__rid__ FROM ({in_window}) _w)"
            )
        return self._aggregate_with_filter(agg_node, row_filter_sql=in_window)

    def _sliding_events_aggregate_sql(self, node: AggregateExpr, window_size: int) -> str:
        col = node.column
        partition = f"PARTITION BY {self.pid} ORDER BY {self.date}, __rid__"
        frame = f"ROWS BETWEEN {window_size - 1} PRECEDING AND CURRENT ROW"
        if node.func in ("sum", "mean", "avg", "min", "max", "median", "sd", "var", "count", "n"):
            fn = _AGG_FUNC_SQL[node.func]
            expr = f"{fn}({col}) OVER ({partition} {frame})"
        elif node.func == "range":
            expr = f"(MAX({col}) OVER ({partition} {frame}) - MIN({col}) OVER ({partition} {frame}))"
        elif node.func == "rise":
            inner_expr = (
                f"({col} - MIN({col}) OVER ({partition} {frame}))"
            )
            rolled = (
                f"SELECT {self.pid}, {inner_expr} AS _delta "
                f"FROM {self.events} WHERE {col} IS NOT NULL"
            )
            return (
                f"SELECT * FROM {self.events} "
                f"WHERE {self.pid} IN ("
                f"SELECT DISTINCT {self.pid} FROM ({rolled}) _r "
                f"WHERE _r._delta IS NOT NULL AND _r._delta {node.op} {node.value})"
            )
        elif node.func == "fall":
            inner_expr = (
                f"(MAX({col}) OVER ({partition} {frame}) - {col})"
            )
            rolled = (
                f"SELECT {self.pid}, {inner_expr} AS _delta "
                f"FROM {self.events} WHERE {col} IS NOT NULL"
            )
            return (
                f"SELECT * FROM {self.events} "
                f"WHERE {self.pid} IN ("
                f"SELECT DISTINCT {self.pid} FROM ({rolled}) _r "
                f"WHERE _r._delta IS NOT NULL AND _r._delta {node.op} {node.value})"
            )
        else:
            raise ValueError(f"Unsupported sliding event aggregate: {node.func}")
        rolled = f"SELECT {self.pid}, {expr} AS _agg FROM {self.events}"
        return (
            f"SELECT * FROM {self.events} "
            f"WHERE {self.pid} IN ("
            f"SELECT DISTINCT {self.pid} FROM ({rolled}) _r "
            f"WHERE _r._agg IS NOT NULL AND _r._agg {node.op} {node.value})"
        )

    # ---- span / between --------

    def _within_span(self, node: WithinSpanExpr) -> str:
        child_sql = self.compile(node.child)
        ref_sql = self.compile(node.ref)
        return self._span_sql(child_sql, ref_sql, ref_sql, outside=node.outside)

    def _between(self, node: BetweenExpr) -> str:
        child_sql = self.compile(node.child)
        start_node, _ = _unwrap_shift(node.bound_start)
        end_node, _ = _unwrap_shift(node.bound_end)
        start_sql = self.compile(start_node)
        end_sql = self.compile(end_node)
        return self._span_sql(child_sql, start_sql, end_sql, outside=node.outside)

    def _span_sql(self, child_sql, start_sql, end_sql, outside) -> str:
        bounds = (
            f"SELECT _s.{self.pid}, MIN(_s.{self.date}) AS _smin, "
            f"MAX(_e.{self.date}) AS _emax "
            f"FROM ({start_sql}) _s "
            f"JOIN ({end_sql}) _e ON _s.{self.pid} = _e.{self.pid} "
            f"GROUP BY _s.{self.pid}"
        )
        positive = (
            f"SELECT _c.* FROM ({child_sql}) _c "
            f"JOIN ({bounds}) _b USING ({self.pid}) "
            f"WHERE _c.{self.date} BETWEEN _b._smin AND _b._emax"
        )
        if not outside:
            return positive
        evaluable = (
            f"SELECT DISTINCT {self.pid} FROM ({start_sql}) _s "
            f"UNION SELECT DISTINCT {self.pid} FROM ({end_sql}) _e"
        )
        return (
            f"SELECT _c.* FROM ({child_sql}) _c "
            f"WHERE _c.{self.pid} IN ({evaluable}) "
            f"AND _c.__rid__ NOT IN (SELECT _p.__rid__ FROM ({positive}) _p)"
        )


# --- public entry ---------------------------------------------------------

class _TQueryDuckDBResult:
    """Minimal result object — same shape (count, pids) as the other
    backends' TQueryResult for parity testing."""
    def __init__(self, pids: list[Any]) -> None:
        self.pids = sorted(pids)
        self.count = len(self.pids)


def tquery_duckdb(
    df,
    expr: str,
    *,
    pid: str = "pid",
    date: str = "start_date",
    cols: str | list[str] | None = None,
    sep: str | None = None,
    variables: dict[str, Any] | None = None,
):
    """Evaluate a tquery expression via DuckDB SQL compilation.

    `df` can be a pandas DataFrame or polars DataFrame — DuckDB can
    register both via its Python API. The date column is coerced to
    DATE; the input table is registered as `events` for the session.
    """
    from tquery._parser import parse

    if isinstance(cols, str):
        cols = [cols]
    elif cols is None:
        # Auto-detect string columns (excluding pid/date) — same convention
        # as pandas evaluator.
        try:
            schema = {c: df[c].dtype for c in df.columns}
        except Exception:
            schema = {}
        cols = [
            c for c in df.columns
            if c not in (pid, date) and str(schema.get(c, "")).startswith(("object", "string", "str"))
        ] or [c for c in df.columns if c not in (pid, date)]

    conn = duckdb.connect()
    try:
        conn.register("events_raw", df)
        # Cast the date column to DATE and add a synthetic row id —
        # DuckDB views don't expose ROWID, so we need our own.
        conn.execute(
            f"CREATE OR REPLACE VIEW events AS "
            f"SELECT * REPLACE (CAST({date} AS DATE) AS {date}), "
            f"ROW_NUMBER() OVER () AS __rid__ "
            f"FROM events_raw"
        )

        # Collect unique codes for wildcard/range expansion
        all_codes_sql = " UNION ".join(
            f"SELECT DISTINCT {c} AS code FROM events" for c in cols if c in df.columns
        )
        all_codes = (
            [r[0] for r in conn.execute(all_codes_sql).fetchall() if r[0] is not None]
            if all_codes_sql else []
        )
        all_codes = sorted(set(all_codes))

        ast = parse(expr)
        compiler = DuckDBCompiler(
            "events", pid=pid, date=date, cols=cols,
            all_codes=all_codes, variables=variables or {},
        )
        pid_sql = compiler.matching_pids_sql(ast)
        rows = conn.execute(pid_sql).fetchall()
        return _TQueryDuckDBResult([r[0] for r in rows])
    finally:
        conn.close()
