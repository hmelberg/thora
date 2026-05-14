"""Smart multi-DataFrame combine with optional AST-driven row pre-filter.

Public entry: :func:`smart_combine_for_query`. Used by ``tquery.tquery``
when the input is a list / tuple / dict of DataFrames.

The basic idea:

1. Walk the parsed AST. If any node is a bail-out (``NotExpr``,
   ``AggregateExpr``, ``ComparisonAtom``, ``EventAtom``, or any window
   with ``outside``), we can't safely pre-filter — fall back to
   :func:`tquery.combine` which stacks everything.
2. Otherwise, collect every ``CodeAtom`` in the AST. For each input
   DataFrame, build a row mask = "rows matching at least one applicable
   code atom" (an atom is applicable to a DataFrame iff at least one of
   its referenced columns exists in that DataFrame). Drop everything
   else.
3. Concatenate the pre-filtered survivors via ``combine``.

Correctness:
  - The bail-out set covers every operator that needs the full row /
    person universe to produce its result.
  - For the operators that pass the safety check, every row required by
    the evaluator is preserved because at least one ``CodeAtom`` will
    have flagged it.

Performance:
  - For sparse code patterns over big registries, pre-filter can shrink
    each source by 100-1000× before concat.
  - For queries that hit the bail-out, we incur a single AST walk (cheap)
    and then full combine — no regression vs. the non-smart path.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

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


# ---------------------------------------------------------------------------
# AST analysis
# ---------------------------------------------------------------------------


def _analyze_ast(node: ASTNode) -> tuple[bool, list[CodeAtom]]:
    """Recursively walk ``node``.

    Returns ``(safe, atoms)``:

    - ``safe``: ``False`` if any descendant forces a full concat (i.e.,
      operators that need every row of every input). ``True`` otherwise.
    - ``atoms``: every ``CodeAtom`` encountered, regardless of safety.
      Useful even when ``safe=False`` for diagnostics; consumers ignore
      the list when ``safe`` is ``False``.
    """
    # Bail-out leaves — needs the full row universe.
    if isinstance(node, (EventAtom, ComparisonAtom, AggregateExpr, NotExpr)):
        return False, []

    # Pure leaf — collectable.
    if isinstance(node, CodeAtom):
        return True, [node]

    # Composite nodes with an ``outside`` flag.
    if isinstance(node, (WithinExpr, WithinSpanExpr, BetweenExpr)):
        if getattr(node, "outside", False):
            return False, []
    if isinstance(node, InsideExpr) and not node.inside:
        return False, []

    # Recurse into children.
    children = _children(node)
    all_safe = True
    all_atoms: list[CodeAtom] = []
    for child in children:
        if child is None:
            continue
        safe, atoms = _analyze_ast(child)
        all_atoms.extend(atoms)
        if not safe:
            all_safe = False
    return all_safe, all_atoms


def _children(node: ASTNode) -> list[ASTNode | None]:
    """Return direct AST children of ``node``."""
    if isinstance(node, (Quantifier, PrefixExpr, RangePrefixExpr, ShiftExpr)):
        return [node.child]
    if isinstance(node, BinaryLogical):
        return [node.left, node.right]
    if isinstance(node, TemporalExpr):
        return [node.left, node.right]
    if isinstance(node, WithinExpr):
        return [node.child, node.ref]
    if isinstance(node, WithinSpanExpr):
        return [node.child, node.ref]
    if isinstance(node, InsideExpr):
        return [node.child, node.ref]
    if isinstance(node, BetweenExpr):
        return [node.child, node.bound_start, node.bound_end]
    return []


# ---------------------------------------------------------------------------
# Per-DataFrame pre-filter
# ---------------------------------------------------------------------------


def _atom_mask_for_df(
    df: pd.DataFrame,
    atom: CodeAtom,
    default_cols: list[str],
    variables: dict[str, Any],
) -> pd.Series | None:
    """Row mask for rows in ``df`` matching ``atom``.

    Returns ``None`` if the atom can't be safely pre-filtered — e.g., a
    variable reference whose contents aren't available. The caller
    should treat that as a bail-out for this DataFrame.
    """
    cols = list(atom.columns) if atom.columns else default_cols
    applicable = [c for c in cols if c in df.columns]
    if not applicable:
        # This atom doesn't apply to this DataFrame at all — contributes
        # zero rows. Return an all-False mask.
        return pd.Series(False, index=df.index)

    mask = pd.Series(False, index=df.index)
    for code in atom.codes:
        if code.startswith("@"):
            # Variable reference — expand against the variables dict.
            varname = code[1:]
            if varname not in variables:
                # Variable missing; can't pre-filter safely. Defer.
                return None
            val = variables[varname]
            patterns = [val] if isinstance(val, str) else list(val)
            for p in patterns:
                m = _single_pattern_mask(df, p, applicable)
                if m is None:
                    return None
                mask |= m
        else:
            m = _single_pattern_mask(df, code, applicable)
            if m is None:
                return None
            mask |= m
    return mask


def _single_pattern_mask(
    df: pd.DataFrame, pattern: str, cols: list[str],
) -> pd.Series | None:
    """Row mask for a single pattern (plain code, wildcard, or range)."""
    mask = pd.Series(False, index=df.index)
    if pattern.endswith("*"):
        prefix = pattern[:-1]
        for col in cols:
            vals = df[col].astype(str)
            mask |= vals.str.startswith(prefix, na=False).fillna(False)
        return mask
    if "-" in pattern and not pattern.startswith("-"):
        parts = pattern.split("-", 1)
        if len(parts) == 2 and parts[0] and parts[1]:
            lo, hi = parts
            for col in cols:
                vals = df[col].astype(str)
                mask |= ((vals >= lo) & (vals <= hi)).fillna(False)
            return mask
    # Plain code — exact match.
    for col in cols:
        mask |= (df[col] == pattern).fillna(False)
    return mask


# ---------------------------------------------------------------------------
# Public entry
# ---------------------------------------------------------------------------


def smart_combine_for_query(
    dfs_input: "list | tuple | dict[str, pd.DataFrame]",
    ast: ASTNode,
    *,
    combine_fn,  # avoids circular import; passed in by caller
    pid: str,
    date: str,
    cols: "str | list[str] | None",
    sep: "str | None",
    variables: "dict[str, Any] | None",
    source_col: str = "__source__",
) -> pd.DataFrame:
    """Combine multiple input DataFrames into one, pre-filtering rows
    where the AST allows it.

    The bail-out behaviour: if pre-filter isn't safe (the AST contains
    operators that need the full row universe), this function falls
    back to a plain ``combine_fn(...)`` call — same result as if the
    user had stacked everything manually. No regression.

    ``combine_fn`` is the public :func:`tquery.combine`, passed in to
    sidestep the circular import with :mod:`tquery.__init__`.
    """
    if isinstance(dfs_input, dict):
        names_list = list(dfs_input.keys())
        df_list = list(dfs_input.values())
    else:
        df_list = list(dfs_input)
        names_list = [f"source_{i}" for i in range(len(df_list))]

    if not df_list:
        raise ValueError("`dfs` is empty — nothing to combine")

    variables = variables or {}
    if isinstance(cols, str):
        default_cols_global = [cols]
    elif cols is not None:
        default_cols_global = list(cols)
    else:
        default_cols_global = None  # autodetect per DataFrame

    safe, atoms = _analyze_ast(ast)
    if not safe or not atoms:
        # Bail out: no usable atoms, or AST forces full universe.
        # Either way we just stack everything.
        if isinstance(dfs_input, dict):
            return combine_fn(
                dfs_input, pid=pid, date=date, source_col=source_col,
            )
        return combine_fn(
            df_list, names=names_list, pid=pid, date=date, source_col=source_col,
        )

    filtered: list[pd.DataFrame] = []
    bail_after_atom_check = False
    for df in df_list:
        if pid not in df.columns or date not in df.columns:
            # Defer the validation to combine_fn — it has the proper
            # error message and source-name attribution.
            bail_after_atom_check = True
            break

        if default_cols_global is not None:
            default_cols = default_cols_global
        else:
            default_cols = [
                c for c in df.columns
                if c not in (pid, date) and pd.api.types.is_string_dtype(df[c])
            ]

        df_mask = pd.Series(False, index=df.index)
        unsafe_atom = False
        for atom in atoms:
            m = _atom_mask_for_df(df, atom, default_cols, variables)
            if m is None:
                unsafe_atom = True
                break
            df_mask |= m
        if unsafe_atom:
            bail_after_atom_check = True
            break
        filtered.append(df.loc[df_mask])

    if bail_after_atom_check:
        if isinstance(dfs_input, dict):
            return combine_fn(
                dfs_input, pid=pid, date=date, source_col=source_col,
            )
        return combine_fn(
            df_list, names=names_list, pid=pid, date=date, source_col=source_col,
        )

    if isinstance(dfs_input, dict):
        filtered_input: "dict[str, pd.DataFrame] | list[pd.DataFrame]" = dict(
            zip(names_list, filtered)
        )
    else:
        filtered_input = filtered

    return combine_fn(
        filtered_input,
        names=names_list if not isinstance(filtered_input, dict) else None,
        pid=pid, date=date, source_col=source_col,
    )
