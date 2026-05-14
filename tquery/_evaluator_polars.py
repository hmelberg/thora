"""Polars backend for tquery.

Mirrors `_evaluator.py` (pandas) node-for-node, but operates on a
`pl.DataFrame`. Same parser, same AST, same goldens — only the
evaluation substrate differs.

Design notes:
- We materialize intermediate masks as `pl.Series` of booleans, length
  == nrow. Same shape as the pandas evaluator's `pd.Series` masks.
- For per-pid aggregates and group operations we use native polars
  expressions where they work cleanly (`sum`/`mean`/`min`/`max`/etc.).
- For per-window operations we lean on `rolling_*_by` (time) and
  `rolling_*` (events). Custom funcs (`rise`/`fall`) fall back to a
  per-pid python loop.
- For date arithmetic we use `.dt.offset_by(...)` / numpy timedeltas.
"""

from __future__ import annotations

import datetime
import operator
from collections import defaultdict
from typing import Any

import numpy as np
import polars as pl

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
from tquery._types import TQueryColumnError, TQueryResult

_OPS = {
    ">":  operator.gt,
    "<":  operator.lt,
    ">=": operator.ge,
    "<=": operator.le,
    "==": operator.eq,
    "!=": operator.ne,
}


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


def _agg_scalar(v: np.ndarray, func: str, relative: bool = False) -> float:
    """Per-cohort scalar aggregate over a 1-D numpy array. NA-skipping.

    ``relative=True`` (v0.2.3) selects the percentage variant for
    ``rise`` and ``fall``: gain ÷ earlier value (rise), or
    drop ÷ earlier value (fall). Pairs where the denominator is ≤ 0
    are excluded.
    """
    v = v[~np.isnan(v)]
    if v.size == 0:
        return 0.0 if func in ("sum", "count", "n") else float("nan")
    if v.size == 1 and func in ("sd", "var"):
        return float("nan")
    if func == "sum":
        return float(v.sum())
    if func in ("mean", "avg"):
        return float(v.mean())
    if func == "min":
        return float(v.min())
    if func == "max":
        return float(v.max())
    if func == "median":
        return float(np.median(v))
    if func == "sd":
        return float(v.std(ddof=1))
    if func == "var":
        return float(v.var(ddof=1))
    if func in ("count", "n"):
        return float(v.size)
    if func == "range":
        if v.size == 1:
            return 0.0 if not relative else (0.0 if v[0] > 0 else float("nan"))
        spread = float(v.max() - v.min())
        if not relative:
            return spread
        mn = float(v.min())
        return float("nan") if mn <= 0 else spread / mn
    if func == "rise":
        if v.size == 1:
            return 0.0
        if not relative:
            return float((v - np.minimum.accumulate(v)).max())
        cm = np.minimum.accumulate(v)
        safe = cm > 0
        if not safe.any():
            return 0.0
        ratio = np.where(safe, (v - cm) / np.where(safe, cm, 1.0), 0.0)
        return float(ratio.max())
    if func == "fall":
        if v.size == 1:
            return 0.0
        if not relative:
            return float((np.maximum.accumulate(v) - v).max())
        cm = np.maximum.accumulate(v)
        safe = cm > 0
        if not safe.any():
            return 0.0
        ratio = np.where(safe, (cm - v) / np.where(safe, cm, 1.0), 0.0)
        return float(ratio.max())
    raise ValueError(f"Unknown aggregate function: {func!r}")


def _shift_dates(dates: pl.Series, days: int) -> pl.Series:
    """Shift a date Series by an integer number of days via numpy."""
    if days == 0:
        return dates
    arr = dates.to_numpy().astype("datetime64[D]")
    arr = arr + np.timedelta64(days, "D")
    return pl.Series(arr).cast(pl.Date)


class PolarsEvaluator:
    """AST → row-level boolean pl.Series. Mirrors the pandas Evaluator."""

    def __init__(
        self,
        df: pl.DataFrame,
        pid: str = "pid",
        date: str = "start_date",
        cols: str | list[str] | None = None,
        sep: str | None = None,
        variables: dict[str, Any] | None = None,
    ) -> None:
        if pid not in df.columns:
            raise TQueryColumnError(f"Person ID column '{pid}' not found in DataFrame")
        if date not in df.columns:
            raise TQueryColumnError(f"Date column '{date}' not found in DataFrame")
        if df.schema[date] != pl.Date:
            df = df.with_columns(pl.col(date).cast(pl.Date))
        df = df.sort([pid, date])

        self.df = df
        self.pid = pid
        self.date = date
        self.sep = sep
        self.variables = variables or {}
        self.nrow = df.height

        if cols is None:
            self._default_cols = [
                c for c in df.columns
                if c not in (pid, date)
                and df.schema[c] in (pl.Utf8, pl.String, pl.Categorical)
            ]
        elif isinstance(cols, str):
            self._default_cols = [cols]
        else:
            self._default_cols = list(cols)

        self._all_codes = self._collect_unique_codes()

    def _collect_unique_codes(self) -> list[str]:
        if not self._default_cols:
            return []
        if self.sep is None:
            vals: list[str] = []
            for c in self._default_cols:
                if c in self.df.columns:
                    vals.extend(self.df.get_column(c).drop_nulls().to_list())
            return sorted(set(vals))
        result: set[str] = set()
        for c in self._default_cols:
            if c not in self.df.columns:
                continue
            for cell in self.df.get_column(c).drop_nulls().to_list():
                result.update(p.strip() for p in cell.split(self.sep))
        return sorted(result)

    def _false_mask(self) -> pl.Series:
        return pl.Series([False] * self.nrow, dtype=pl.Boolean)

    def _true_mask(self) -> pl.Series:
        return pl.Series([True] * self.nrow, dtype=pl.Boolean)

    # ---- dispatch ----

    def evaluate(self, node: ASTNode) -> pl.Series:
        if isinstance(node, CodeAtom):       return self._eval_code(node)
        if isinstance(node, EventAtom):      return self._true_mask()
        if isinstance(node, ComparisonAtom): return self._eval_comparison(node)
        if isinstance(node, AggregateExpr):  return self._eval_aggregate(node, row_mask=None)
        if isinstance(node, PrefixExpr):     return self._eval_prefix(node)
        if isinstance(node, RangePrefixExpr): return self._eval_range_prefix(node)
        if isinstance(node, NotExpr):        return self._eval_not(node)
        if isinstance(node, BinaryLogical):  return self._eval_logical(node)
        if isinstance(node, TemporalExpr):   return self._eval_temporal(node)
        if isinstance(node, WithinExpr):     return self._eval_within(node)
        if isinstance(node, WithinSpanExpr): return self._eval_within_span(node)
        if isinstance(node, InsideExpr):     return self._eval_inside(node)
        if isinstance(node, BetweenExpr):    return self._eval_between(node)
        if isinstance(node, ShiftExpr):
            raise TypeError("Shifted anchors only valid in temporal/within ref positions")
        if isinstance(node, Quantifier):
            raise TypeError(f"'{node.kind}' must appear inside a temporal/within context")
        raise TypeError(f"Unknown AST node type: {type(node)}")

    # ---- atoms ----

    def _eval_code(self, node: CodeAtom) -> pl.Series:
        codes = expand_all_codes(node.codes, all_codes=self._all_codes, variables=self.variables)
        if node.columns:
            cols = resolve_columns(list(node.columns), list(self.df.columns))
        else:
            cols = self._default_cols
        exact = [c for c in codes if not c.endswith("*")]
        wildcards = [c[:-1] for c in codes if c.endswith("*")]
        mask = self._false_mask()
        for col in cols:
            if col not in self.df.columns:
                continue
            vals = self.df.get_column(col)
            if exact:
                mask = mask | vals.is_in(exact)
            for w in wildcards:
                mask = mask | vals.str.starts_with(w).fill_null(False)
        return mask.fill_null(False)

    def _eval_comparison(self, node: ComparisonAtom) -> pl.Series:
        if node.column not in self.df.columns:
            raise TQueryColumnError(f"Column '{node.column}' not found in DataFrame")
        col = self.df.get_column(node.column)
        cmp = _OPS[node.op](col, node.value)
        return cmp.fill_null(False)

    # ---- prefix / quantifiers ----

    def _eval_prefix(self, node: PrefixExpr) -> pl.Series:
        child = self.evaluate(node.child)
        return self._dispatch_prefix(node.kind, node.n, child)

    def _dispatch_prefix(self, kind: str, n: int, child: pl.Series) -> pl.Series:
        pid = self.df.get_column(self.pid)
        child_int = child.cast(pl.Int64)
        local = pl.DataFrame({"_p": pid, "_m": child_int})
        if kind in ("min", "max", "exactly"):
            total = local.with_columns(_t=pl.col("_m").sum().over("_p")).get_column("_t")
            if kind == "min": return child & (total >= n)
            if kind == "max": return child & (total <= n)
            return child & (total == n)
        # cumulative count of TRUEs per pid, in input order
        local = local.with_columns(_cs=pl.col("_m").cum_sum().over("_p"))
        cs = local.get_column("_cs")
        if kind == "ordinal":
            if n > 0:
                return child & (cs == n)
            total = local.with_columns(_t=pl.col("_m").sum().over("_p")).get_column("_t")
            return child & ((total - cs + 1) == abs(n))
        if kind == "first":
            return child & (cs <= n)
        if kind == "last":
            total = local.with_columns(_t=pl.col("_m").sum().over("_p")).get_column("_t")
            return child & (cs > (total - n))
        raise ValueError(f"Unknown prefix kind: {kind}")

    def _eval_range_prefix(self, node: RangePrefixExpr) -> pl.Series:
        child = self.evaluate(node.child)
        pid = self.df.get_column(self.pid)
        total = (
            pl.DataFrame({"_p": pid, "_m": child.cast(pl.Int64)})
            .with_columns(_t=pl.col("_m").sum().over("_p"))
            .get_column("_t")
        )
        return child & (total >= node.min_n) & (total <= node.max_n)

    # ---- logical / not ----

    def _eval_not(self, node: NotExpr) -> pl.Series:
        child = self.evaluate(node.child)
        pid = self.df.get_column(self.pid)
        any_per_pid = (
            pl.DataFrame({"_p": pid, "_m": child})
            .with_columns(_any=pl.col("_m").any().over("_p"))
            .get_column("_any")
        )
        return ~any_per_pid.fill_null(False)

    def _eval_logical(self, node: BinaryLogical) -> pl.Series:
        left = self.evaluate(node.left)
        right = self.evaluate(node.right)
        pid = self.df.get_column(self.pid)
        loc = pl.DataFrame({"_p": pid, "_l": left, "_r": right}).with_columns(
            _la=pl.col("_l").any().over("_p"),
            _ra=pl.col("_r").any().over("_p"),
        )
        la = loc.get_column("_la"); ra = loc.get_column("_ra")
        per_pid = (la & ra) if node.op == "and" else (la | ra)
        return per_pid & (left | right)

    # ---- temporal (before/after/simultaneously) ----

    def _eval_temporal(self, node: TemporalExpr) -> pl.Series:
        left_inner, every_left = _unwrap_quantifier(node.left)
        if isinstance(left_inner, ShiftExpr):
            raise TypeError("Shifted anchors only valid on RHS")
        right_inner, every_right = _unwrap_quantifier(node.right)
        right_inner, ref_offset = _unwrap_shift(right_inner)

        left_mask = self.evaluate(left_inner)
        right_mask = self.evaluate(right_inner)
        if not bool(left_mask.any()) or not bool(right_mask.any()):
            return self._false_mask()

        pid = self.df.get_column(self.pid)
        date_col = self.df.get_column(self.date)

        l_pid = pid.filter(left_mask).to_numpy()
        l_date = date_col.filter(left_mask).to_numpy().astype("datetime64[D]")
        r_pid = pid.filter(right_mask).to_numpy()
        r_date = date_col.filter(right_mask).to_numpy().astype("datetime64[D]")
        if ref_offset:
            r_date = r_date + np.timedelta64(ref_offset, "D")

        if node.op == "simultaneously":
            left_dates_by_pid: dict[Any, set] = defaultdict(set)
            for p, d in zip(l_pid, l_date):
                left_dates_by_pid[p].add(d)
            right_dates_by_pid: dict[Any, set] = defaultdict(set)
            for p, d in zip(r_pid, r_date):
                right_dates_by_pid[p].add(d)
            common = set(left_dates_by_pid) & set(right_dates_by_pid)
            matching: list[Any] = []
            for p in common:
                l = left_dates_by_pid[p]; r = right_dates_by_pid[p]
                if every_left and not l.issubset(r): continue
                if every_right and not r.issubset(l): continue
                if not every_left and not every_right and not (l & r): continue
                matching.append(p)
            return pid.is_in(matching) & left_mask

        # before / after — per-person min/max aggregates (vectorised)
        l_df = pl.DataFrame({"_p": l_pid, "_d": l_date})
        r_df = pl.DataFrame({"_p": r_pid, "_d": r_date})
        lagg = l_df.group_by("_p").agg(
            l_min=pl.col("_d").min(), l_max=pl.col("_d").max()
        )
        ragg = r_df.group_by("_p").agg(
            r_min=pl.col("_d").min(), r_max=pl.col("_d").max()
        )
        agg = lagg.join(ragg, on="_p", how="inner")
        if node.op == "before":
            if every_left and every_right:
                ok = agg.get_column("l_max") < agg.get_column("r_min")
            elif every_left:
                ok = agg.get_column("l_max") < agg.get_column("r_max")
            else:
                ok = agg.get_column("l_min") < agg.get_column("r_min")
        else:
            if every_left and every_right:
                ok = agg.get_column("l_min") > agg.get_column("r_max")
            elif every_right:
                ok = agg.get_column("l_max") > agg.get_column("r_max")
            else:
                ok = agg.get_column("l_min") > agg.get_column("r_min")
        matching_pids = agg.filter(ok).get_column("_p").to_list()
        return pid.is_in(matching_pids) & left_mask

    # ---- WithinExpr (day-based) ----

    def _eval_within(self, node: WithinExpr) -> pl.Series:
        if isinstance(node.child, AggregateExpr):
            return self._eval_within_aggregate(node)
        child_inner, every_left = _unwrap_quantifier(node.child)
        if isinstance(child_inner, ShiftExpr):
            raise TypeError("Shifted anchor cannot be the LHS of a window")
        child_mask = self.evaluate(child_inner)
        if not bool(child_mask.any()):
            return self._false_mask()

        ref_offset = 0
        every_right = False
        if node.ref is not None:
            ref_inner, every_right = _unwrap_quantifier(node.ref)
            ref_inner, ref_offset = _unwrap_shift(ref_inner)
            ref_mask = self.evaluate(ref_inner)
            if not bool(ref_mask.any()):
                return self._false_mask()
        else:
            ref_mask = None

        pid = self.df.get_column(self.pid)
        date_col = self.df.get_column(self.date)

        if ref_mask is None:
            # first-event-anchored
            d_np = date_col.to_numpy().astype("datetime64[D]")
            p_np = pid.to_numpy()
            first_by_pid: dict[Any, Any] = {}
            for i, p in enumerate(p_np):
                if p not in first_by_pid or d_np[i] < first_by_pid[p]:
                    first_by_pid[p] = d_np[i]
            firsts = np.array([first_by_pid[p] for p in p_np], dtype="datetime64[D]")
            diff = np.abs((d_np - firsts).astype("timedelta64[D]").astype(int))
            return child_mask & pl.Series(
                (diff >= node.min_days) & (diff <= node.days)
            )

        in_window = self._rows_in_window(
            child_mask, ref_mask, node.days, node.min_days,
            node.direction, ref_offset,
        )
        if not (every_left or every_right):
            if not node.outside:
                return in_window
            evaluable = pid.filter(ref_mask).unique().to_list()
            return child_mask & ~in_window & pid.is_in(evaluable)

        # Universal modes
        matching_pids = self._universal_pids(
            child_mask, ref_mask, in_window,
            every_left, every_right,
            node.days, node.min_days, node.direction, ref_offset,
        )
        return child_mask & pid.is_in(matching_pids)

    def _rows_in_window(
        self, child_mask, ref_mask, days, min_days, direction, ref_offset_days,
    ) -> pl.Series:
        """Row mask of child rows whose date falls in the day-window of
        at least one ref row. Uses polars `join_asof` for speed; falls
        back to a per-pid two-pointer scan only for signed-around (which
        needs both directions) or unrestricted (direction == None).
        """
        pid = self.df.get_column(self.pid)
        date_col = self.df.get_column(self.date)

        signed_around = direction == "around" and min_days < 0
        # Build child + ref small frames for the join.
        idx_series = pl.Series("_idx", np.arange(self.nrow), dtype=pl.UInt32)
        child_df = pl.DataFrame({
            self.pid: pid.filter(child_mask),
            self.date: date_col.filter(child_mask),
            "_idx": idx_series.filter(child_mask),
        }).sort([self.pid, self.date])

        ref_date_shifted = (
            _shift_dates(date_col.filter(ref_mask), ref_offset_days)
            if ref_offset_days else date_col.filter(ref_mask)
        )
        ref_df = pl.DataFrame({
            self.pid: pid.filter(ref_mask),
            "_ref_date": ref_date_shifted,
        }).sort([self.pid, "_ref_date"])

        result = np.zeros(self.nrow, dtype=bool)

        def _apply_asof(strategy: str, tolerance: int) -> None:
            """One backward/forward join; mark child rows whose matched
            ref produces a delta in the requested range."""
            try:
                joined = child_df.join_asof(
                    ref_df, left_on=self.date, right_on="_ref_date",
                    by=self.pid, strategy=strategy,
                    tolerance=f"{tolerance}d",
                )
            except Exception:
                return
            if "_ref_date" not in joined.columns:
                return
            ref_dates = joined.get_column("_ref_date")
            delta = (
                joined.get_column(self.date).to_numpy().astype("datetime64[D]")
                - ref_dates.fill_null(joined.get_column(self.date)).to_numpy().astype("datetime64[D]")
            ).astype("timedelta64[D]").astype(int)
            matched_flag = ~ref_dates.is_null().to_numpy()
            if signed_around:
                ok = matched_flag & (delta >= min_days) & (delta <= days)
            elif direction == "after":
                ok = matched_flag & (delta >= min_days) & (delta <= days)
            elif direction == "before":
                neg = -delta
                ok = matched_flag & (neg >= min_days) & (neg <= days)
            else:
                ad = np.abs(delta)
                ok = matched_flag & (ad >= min_days) & (ad <= days)
            idxs = joined.get_column("_idx").to_numpy()
            result[idxs[ok]] = True

        if direction == "after":
            _apply_asof("backward", days)
        elif direction == "before":
            _apply_asof("forward", days)
        elif signed_around:
            # backward looks for ref ≤ child within `days`; forward for
            # ref > child within |min_days|.
            _apply_asof("backward", days)
            _apply_asof("forward", abs(min_days))
        else:
            # `around` (unsigned) or no direction — try both sides.
            _apply_asof("backward", days)
            _apply_asof("forward", days)

        return pl.Series(result, dtype=pl.Boolean)

    def _universal_pids(
        self, child_mask, ref_mask, in_window,
        every_left, every_right,
        days, min_days, direction, ref_offset_days,
    ) -> list[Any]:
        pid = self.df.get_column(self.pid)
        candidate = set(pid.filter(child_mask).to_list()) & set(pid.filter(ref_mask).to_list())
        matching = set(candidate)
        if every_left:
            full = self._every_pids_one_sided(
                pid, child_mask, child_mask & in_window
            )
            matching &= full
        if every_right:
            opposite = (
                "before" if direction == "after"
                else "after" if direction == "before"
                else direction
            )
            rhs_in_window = self._rows_in_window(
                ref_mask, child_mask, days, min_days, opposite, -ref_offset_days
            )
            full = self._every_pids_one_sided(
                pid, ref_mask, ref_mask & rhs_in_window
            )
            matching &= full
        return list(matching)

    @staticmethod
    def _every_pids_one_sided(pid, total_mask, hit_mask) -> set:
        """pids where count(total_mask) == count(hit_mask)."""
        p_total: dict[Any, int] = defaultdict(int)
        p_hit: dict[Any, int] = defaultdict(int)
        for p, t, h in zip(pid.to_list(), total_mask.to_list(), hit_mask.to_list()):
            if t:
                p_total[p] += 1
                if h:
                    p_hit[p] += 1
        return {p for p, t in p_total.items() if p_hit[p] == t}

    # ---- InsideExpr (event-position window) ----

    def _eval_inside(self, node: InsideExpr) -> pl.Series:
        if isinstance(node.child, AggregateExpr):
            return self._eval_inside_aggregate(node)
        if isinstance(node.ref, ShiftExpr):
            raise TypeError("Shifted anchors cannot be the ref of an event-window")
        child_mask = self.evaluate(node.child)
        ref_mask = self.evaluate(node.ref)
        return self._eval_event_window(
            child_mask, ref_mask, node.inside,
            node.min_events, node.max_events, node.direction,
        )

    def _eval_event_window(
        self, child_mask, ref_mask, inside, min_events, max_events, direction,
    ) -> pl.Series:
        if not bool(child_mask.any()) or not bool(ref_mask.any()):
            return self._false_mask()
        pid = self.df.get_column(self.pid)
        # event positions per pid (0-based), input order
        loc = pl.DataFrame({"_p": pid}).with_columns(
            _pos=(pl.col("_p").cum_count().over("_p") - 1)
        )
        pos = loc.get_column("_pos").to_numpy()
        p_np = pid.to_numpy()
        child_np = child_mask.to_numpy()
        ref_np = ref_mask.to_numpy()

        per_pid_refs: dict[Any, list[int]] = defaultdict(list)
        for i, r in enumerate(ref_np):
            if r:
                per_pid_refs[p_np[i]].append(int(pos[i]))

        result = np.zeros(self.nrow, dtype=bool)
        for p, ref_list in per_pid_refs.items():
            rows = np.where(p_np == p)[0]
            p_pos = pos[rows]; p_child = child_np[rows]
            in_win = np.zeros(rows.size, dtype=bool)
            for rp in ref_list:
                if direction == "after":
                    lo, hi = rp + min_events, rp + max_events
                elif direction == "before":
                    lo, hi = rp - max_events, rp - min_events
                else:
                    lo, hi = rp + min_events, rp + max_events
                in_win |= (p_pos >= lo) & (p_pos <= hi)
            sel = (p_child & in_win) if inside else (p_child & ~in_win)
            result[rows] = sel
        return pl.Series(result, dtype=pl.Boolean)

    # ---- Span / Between ----

    def _eval_within_span(self, node: WithinSpanExpr) -> pl.Series:
        child_mask = self.evaluate(node.child)
        ref_mask = self.evaluate(node.ref)
        return self._eval_span(child_mask, ref_mask, ref_mask, outside=node.outside)

    def _eval_between(self, node: BetweenExpr) -> pl.Series:
        child_mask = self.evaluate(node.child)
        start_node, _ = _unwrap_shift(node.bound_start)
        end_node, _ = _unwrap_shift(node.bound_end)
        start_mask = self.evaluate(start_node)
        end_mask = self.evaluate(end_node)
        return self._eval_span(child_mask, start_mask, end_mask, outside=node.outside)

    def _eval_span(self, child_mask, start_mask, end_mask, outside) -> pl.Series:
        if (not bool(child_mask.any())
                or not bool(start_mask.any())
                or not bool(end_mask.any())):
            return self._false_mask()
        pid = self.df.get_column(self.pid)
        date_col = self.df.get_column(self.date)
        p_np = pid.to_numpy()
        d_np = date_col.to_numpy().astype("datetime64[D]")
        s_np = start_mask.to_numpy()
        e_np = end_mask.to_numpy()
        c_np = child_mask.to_numpy()

        s_min: dict[Any, Any] = {}
        for i in np.where(s_np)[0]:
            p = p_np[i]; d = d_np[i]
            if p not in s_min or d < s_min[p]:
                s_min[p] = d
        e_max: dict[Any, Any] = {}
        for i in np.where(e_np)[0]:
            p = p_np[i]; d = d_np[i]
            if p not in e_max or d > e_max[p]:
                e_max[p] = d

        hits = np.zeros(self.nrow, dtype=bool)
        for i in np.where(c_np)[0]:
            p = p_np[i]
            if p in s_min and p in e_max:
                if s_min[p] <= d_np[i] <= e_max[p]:
                    hits[i] = True
        positive = hits & c_np
        if not outside:
            return pl.Series(positive, dtype=pl.Boolean)
        evaluable = set(p_np[s_np | e_np].tolist())
        out = c_np & ~hits & np.array([p in evaluable for p in p_np])
        return pl.Series(out, dtype=pl.Boolean)

    # ---- Aggregate evaluation ----

    def _eval_aggregate(self, node: AggregateExpr, row_mask: pl.Series | None) -> pl.Series:
        if node.column not in self.df.columns:
            raise TQueryColumnError(f"Column '{node.column}' not found in DataFrame")
        col = self.df.get_column(node.column)
        pid = self.df.get_column(self.pid)
        if row_mask is not None:
            col = col.filter(row_mask)
            sub_pid = pid.filter(row_mask)
        else:
            sub_pid = pid

        # Try fast polars-native paths for standard funcs. Relative
        # range/rise/fall fall through to the per-pid Python loop below,
        # which routes via _agg_scalar(relative=True).
        op = _OPS[node.op]
        relative = getattr(node, "relative", False)
        if node.func in ("sum", "mean", "avg", "min", "max", "median",
                         "sd", "var", "count", "n", "range") and not (
                             relative and node.func == "range"):
            df_local = pl.DataFrame({"_p": sub_pid, "_v": col})
            if node.func == "sum":
                agg = df_local.group_by("_p").agg(_agg=pl.col("_v").sum())
            elif node.func in ("mean", "avg"):
                agg = df_local.group_by("_p").agg(_agg=pl.col("_v").mean())
            elif node.func == "min":
                agg = df_local.group_by("_p").agg(_agg=pl.col("_v").min())
            elif node.func == "max":
                agg = df_local.group_by("_p").agg(_agg=pl.col("_v").max())
            elif node.func == "median":
                agg = df_local.group_by("_p").agg(_agg=pl.col("_v").median())
            elif node.func == "sd":
                agg = df_local.group_by("_p").agg(_agg=pl.col("_v").std())
            elif node.func == "var":
                agg = df_local.group_by("_p").agg(_agg=pl.col("_v").var())
            elif node.func in ("count", "n"):
                agg = df_local.group_by("_p").agg(_agg=pl.col("_v").count().cast(pl.Float64))
            elif node.func == "range":
                agg = df_local.group_by("_p").agg(
                    _agg=pl.col("_v").max() - pl.col("_v").min()
                )
            ok = op(agg.get_column("_agg"), node.value).fill_null(False)
            matching_pids = agg.filter(ok).get_column("_p").to_list()
            return pid.is_in(matching_pids)

        # Custom funcs (rise / fall): per-pid python loop on numpy values.
        v_np = col.to_numpy().astype(float)
        p_np = sub_pid.to_numpy()
        per_pid_vals: dict[Any, list[float]] = defaultdict(list)
        for p, v in zip(p_np, v_np):
            per_pid_vals[p].append(v)
        matching = []
        for p, vs in per_pid_vals.items():
            agg_v = _agg_scalar(np.array(vs), node.func, relative=getattr(node, "relative", False))
            if op(agg_v, node.value) if not np.isnan(agg_v) else False:
                matching.append(p)
        return pid.is_in(matching)

    def _eval_within_aggregate(self, node: WithinExpr) -> pl.Series:
        agg_node: AggregateExpr = node.child  # type: ignore[assignment]
        sliding = node.direction is None and node.ref is None
        if sliding:
            if node.outside:
                raise TypeError("`outside` over a sliding aggregate is not supported")
            return self._sliding_days_aggregate(agg_node, node.days)
        ref_inner, _ = _unwrap_quantifier(node.ref)
        ref_inner, ref_offset = _unwrap_shift(ref_inner)
        ref_mask = self.evaluate(ref_inner)
        all_rows = self._true_mask()
        in_window = self._rows_in_window(
            all_rows, ref_mask, node.days, node.min_days,
            node.direction, ref_offset,
        )
        if node.outside:
            pid = self.df.get_column(self.pid)
            evaluable = pid.filter(ref_mask).unique().to_list()
            in_window = pid.is_in(evaluable) & ~in_window
        return self._eval_aggregate(agg_node, row_mask=in_window)

    def _eval_inside_aggregate(self, node: InsideExpr) -> pl.Series:
        agg_node: AggregateExpr = node.child  # type: ignore[assignment]
        sliding = node.direction is None and node.ref is None
        if sliding:
            return self._sliding_events_aggregate(agg_node, node.max_events)
        ref_mask = self.evaluate(node.ref)
        all_rows = self._true_mask()
        in_window = self._eval_event_window(
            all_rows, ref_mask, True,
            node.min_events, node.max_events, node.direction,
        )
        if not node.inside:
            pid = self.df.get_column(self.pid)
            evaluable = pid.filter(ref_mask).unique().to_list()
            in_window = pid.is_in(evaluable) & ~in_window
        return self._eval_aggregate(agg_node, row_mask=in_window)

    def _sliding_days_aggregate(self, node: AggregateExpr, days: int) -> pl.Series:
        if node.column not in self.df.columns:
            raise TQueryColumnError(f"Column '{node.column}' not found in DataFrame")
        op = _OPS[node.op]
        threshold = node.value
        period = f"{days}d"

        try:
            if node.func == "sum":
                rolled = self.df.with_columns(
                    _agg=pl.col(node.column).rolling_sum_by(self.date, period).over(self.pid)
                )
            elif node.func in ("mean", "avg"):
                rolled = self.df.with_columns(
                    _agg=pl.col(node.column).rolling_mean_by(self.date, period).over(self.pid)
                )
            elif node.func == "min":
                rolled = self.df.with_columns(
                    _agg=pl.col(node.column).rolling_min_by(self.date, period).over(self.pid)
                )
            elif node.func == "max":
                rolled = self.df.with_columns(
                    _agg=pl.col(node.column).rolling_max_by(self.date, period).over(self.pid)
                )
            elif node.func == "range":
                if getattr(node, "relative", False):
                    raise TypeError("range% routes to generic")
                rolled = self.df.with_columns(
                    _hi=pl.col(node.column).rolling_max_by(self.date, period).over(self.pid),
                    _lo=pl.col(node.column).rolling_min_by(self.date, period).over(self.pid),
                ).with_columns(_agg=pl.col("_hi") - pl.col("_lo"))
            else:
                raise TypeError("fallback")
            agg = rolled.get_column("_agg")
            match = op(agg, threshold).fill_null(False)
            matching = rolled.filter(match).get_column(self.pid).unique().to_list()
            return self.df.get_column(self.pid).is_in(matching)
        except TypeError:
            pass

        # Slow per-pid fallback for funcs polars doesn't natively support
        # over date windows: median, sd, var, count, rise, fall, range%.
        return self._sliding_days_aggregate_generic(node, days)

    def _sliding_days_aggregate_generic(self, node, days):
        pid = self.df.get_column(self.pid)
        op = _OPS[node.op]
        threshold = node.value
        date_arr = self.df.get_column(self.date).to_numpy().astype("datetime64[D]")
        val_arr = self.df.get_column(node.column).to_numpy().astype(float)
        pid_arr = pid.to_numpy()
        matching: set = set()
        for p in np.unique(pid_arr):
            idx = np.where(pid_arr == p)[0]
            dates = date_arr[idx]; vals = val_arr[idx]
            order = np.argsort(dates)
            dates = dates[order]; vals = vals[order]
            n = len(dates)
            start = 0
            for r in range(n):
                lo = dates[r] - np.timedelta64(days, "D")
                while dates[start] < lo:
                    start += 1
                window = vals[start:r + 1]
                a = _agg_scalar(window, node.func, relative=getattr(node, "relative", False))
                if not np.isnan(a) and op(a, threshold):
                    matching.add(p)
                    break
        return pid.is_in(list(matching))

    def _sliding_events_aggregate(self, node: AggregateExpr, window_size: int) -> pl.Series:
        op = _OPS[node.op]
        threshold = node.value
        try:
            if node.func == "sum":
                rolled = self.df.with_columns(
                    _agg=pl.col(node.column).rolling_sum(window_size=window_size, min_samples=1).over(self.pid)
                )
            elif node.func in ("mean", "avg"):
                rolled = self.df.with_columns(
                    _agg=pl.col(node.column).rolling_mean(window_size=window_size, min_samples=1).over(self.pid)
                )
            elif node.func == "min":
                rolled = self.df.with_columns(
                    _agg=pl.col(node.column).rolling_min(window_size=window_size, min_samples=1).over(self.pid)
                )
            elif node.func == "max":
                rolled = self.df.with_columns(
                    _agg=pl.col(node.column).rolling_max(window_size=window_size, min_samples=1).over(self.pid)
                )
            elif node.func == "range":
                if getattr(node, "relative", False):
                    raise TypeError("range% routes to generic")
                rolled = self.df.with_columns(
                    _hi=pl.col(node.column).rolling_max(window_size=window_size, min_samples=1).over(self.pid),
                    _lo=pl.col(node.column).rolling_min(window_size=window_size, min_samples=1).over(self.pid),
                ).with_columns(_agg=pl.col("_hi") - pl.col("_lo"))
            else:
                raise TypeError("fallback")
            agg = rolled.get_column("_agg")
            match = op(agg, threshold).fill_null(False)
            matching = rolled.filter(match).get_column(self.pid).unique().to_list()
            return self.df.get_column(self.pid).is_in(matching)
        except TypeError:
            pass

        return self._sliding_events_aggregate_generic(node, window_size)

    def _sliding_events_aggregate_generic(self, node, window_size):
        pid = self.df.get_column(self.pid)
        op = _OPS[node.op]
        threshold = node.value
        pid_arr = pid.to_numpy()
        date_arr = self.df.get_column(self.date).to_numpy().astype("datetime64[D]")
        val_arr = self.df.get_column(node.column).to_numpy().astype(float)
        matching: set = set()
        for p in np.unique(pid_arr):
            idx = np.where(pid_arr == p)[0]
            dates = date_arr[idx]; vals = val_arr[idx]
            order = np.argsort(dates)
            vals = vals[order]
            n = len(vals)
            for r in range(n):
                lo = max(0, r - (window_size - 1))
                window = vals[lo:r + 1]
                a = _agg_scalar(window, node.func, relative=getattr(node, "relative", False))
                if not np.isnan(a) and op(a, threshold):
                    matching.add(p)
                    break
        return pid.is_in(list(matching))


def tquery_polars(
    df: pl.DataFrame,
    expr: str,
    *,
    pid: str = "pid",
    date: str = "start_date",
    cols: str | list[str] | None = None,
    sep: str | None = None,
    variables: dict[str, Any] | None = None,
) -> TQueryResult:
    """Evaluate a tquery expression against a polars DataFrame."""
    from tquery._parser import parse
    import pandas as pd
    ast = parse(expr)
    ev = PolarsEvaluator(df, pid=pid, date=date, cols=cols, sep=sep, variables=variables)
    mask = ev.evaluate(ast)
    mask_pd = pd.Series(mask.to_numpy(), name="match")
    df_pd = ev.df.to_pandas()
    return TQueryResult(mask_pd, df_pd, pid)
