"""AST evaluator for tquery.

Walks the AST tree produced by the parser and evaluates each node against
a pandas DataFrame, returning boolean Series results.
"""

from __future__ import annotations

import operator
from typing import Any

import numpy as np
import pandas as pd

from tquery._ast import (
    ASTNode,
    BinaryLogical,
    CodeAtom,
    ComparisonAtom,
    InsideExpr,
    NotExpr,
    PrefixExpr,
    RangePrefixExpr,
    TemporalExpr,
    WithinExpr,
)
from tquery._cache import EvalCache
from tquery._codes import (
    collect_unique_codes,
    expand_all_codes,
    get_matching_rows,
    resolve_columns,
)
from tquery._prefix import eval_prefix, eval_range_prefix
from tquery._temporal import eval_before_after, eval_inside_outside, eval_within_days
from tquery._types import TQueryColumnError


_OPS = {
    ">": operator.gt,
    "<": operator.lt,
    ">=": operator.ge,
    "<=": operator.le,
    "==": operator.eq,
    "!=": operator.ne,
}


class Evaluator:
    """Evaluates a parsed AST against a DataFrame.

    Each AST node is dispatched to a handler that returns a row-level
    boolean pd.Series (aligned to df.index). Results are cached by
    AST node identity (frozen dataclass = hashable).
    """

    def __init__(
        self,
        df: pd.DataFrame,
        pid: str = "pid",
        date: str = "start_date",
        cols: str | list[str] | None = None,
        sep: str | None = None,
        variables: dict[str, Any] | None = None,
    ) -> None:
        # Validate required columns
        if pid not in df.columns:
            raise TQueryColumnError(f"Person ID column '{pid}' not found in DataFrame")
        if date not in df.columns:
            raise TQueryColumnError(f"Date column '{date}' not found in DataFrame")

        self.df = df
        self.pid = pid
        self.date = date
        self.sep = sep
        self.variables = variables or {}
        self._cache = EvalCache()

        # Resolve default columns for code matching
        if cols is None:
            # Auto-detect: all object/string columns except pid and date
            self._default_cols = [
                c for c in df.columns
                if c not in (pid, date) and df[c].dtype == object
            ]
        elif isinstance(cols, str):
            self._default_cols = [cols]
        else:
            self._default_cols = list(cols)

        # Pre-collect all unique codes for wildcard/range expansion
        self._all_codes = collect_unique_codes(df, self._default_cols, sep)

    def evaluate(self, node: ASTNode) -> pd.Series:
        """Evaluate an AST node, returning a row-level boolean Series."""
        cached = self._cache.get(node)
        if cached is not None:
            return cached

        result = self._dispatch(node)
        self._cache.put(node, result)
        return result

    def _dispatch(self, node: ASTNode) -> pd.Series:
        if isinstance(node, CodeAtom):
            return self._eval_code(node)
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
        else:
            raise TypeError(f"Unknown AST node type: {type(node)}")

    def _eval_code(self, node: CodeAtom) -> pd.Series:
        if node.columns:
            # Resolve column patterns (icd*, icd1-icd10) against actual columns
            cols = resolve_columns(list(node.columns), list(self.df.columns))
        else:
            cols = self._default_cols
        codes = expand_all_codes(
            node.codes,
            all_codes=self._all_codes,
            variables=self.variables,
        )
        return get_matching_rows(self.df, codes, cols, self.sep)

    def _eval_comparison(self, node: ComparisonAtom) -> pd.Series:
        if node.column not in self.df.columns:
            raise TQueryColumnError(
                f"Column '{node.column}' not found in DataFrame"
            )
        op_func = _OPS[node.op]
        return op_func(self.df[node.column], node.value)

    def _eval_prefix(self, node: PrefixExpr) -> pd.Series:
        child_mask = self.evaluate(node.child)
        return eval_prefix(
            node.kind, node.n, child_mask, self.df[self.pid]
        )

    def _eval_range_prefix(self, node: RangePrefixExpr) -> pd.Series:
        child_mask = self.evaluate(node.child)
        return eval_range_prefix(
            node.min_n, node.max_n, child_mask, self.df[self.pid]
        )

    def _eval_not(self, node: NotExpr) -> pd.Series:
        child = self.evaluate(node.child)
        # NOT at person level: negate which persons match, then broadcast
        person_match = child.groupby(self.df[self.pid]).any()
        negated_pids = set(person_match.index[~person_match])
        # Also include persons not present at all in child
        all_pids = set(self.df[self.pid].unique())
        child_pids = set(person_match.index)
        absent_pids = all_pids - child_pids
        not_pids = negated_pids | absent_pids
        return self.df[self.pid].isin(not_pids)

    def _eval_logical(self, node: BinaryLogical) -> pd.Series:
        left = self.evaluate(node.left)
        right = self.evaluate(node.right)

        # Person-level semantics: AND/OR operate on person-level membership,
        # then broadcast back to row level.
        pid = self.df[self.pid]

        left_persons = left.groupby(pid).any()
        right_persons = right.groupby(pid).any()

        # Align on all persons
        all_pids_idx = pd.Index(self.df[self.pid].unique())
        left_persons = left_persons.reindex(all_pids_idx, fill_value=False)
        right_persons = right_persons.reindex(all_pids_idx, fill_value=False)

        if node.op == "and":
            matching = left_persons & right_persons
        else:  # or
            matching = left_persons | right_persons

        matching_pids = set(matching.index[matching])
        # Return row-level: rows from either left or right for matching persons
        return pid.isin(matching_pids) & (left | right)

    def _eval_temporal(self, node: TemporalExpr) -> pd.Series:
        left_mask = self.evaluate(node.left)
        right_mask = self.evaluate(node.right)
        return eval_before_after(
            self.df, left_mask, right_mask, node.op, self.pid, self.date
        )

    def _eval_within(self, node: WithinExpr) -> pd.Series:
        child_mask = self.evaluate(node.child)
        ref_mask = self.evaluate(node.ref) if node.ref is not None else None
        return eval_within_days(
            self.df, child_mask, ref_mask, node.days, node.direction,
            self.pid, self.date, min_days=node.min_days,
        )

    def _eval_inside(self, node: InsideExpr) -> pd.Series:
        child_mask = self.evaluate(node.child)
        ref_mask = self.evaluate(node.ref)
        return eval_inside_outside(
            self.df, child_mask, ref_mask, node.inside, node.n_events,
            node.direction, self.pid
        )
