"""Code expansion, column resolution, and row matching for medical code patterns.

Supports wildcards (K50*), ranges (K50-K53), and external variable
references (@antibiotics). Also supports column name patterns
(icd*, icd1-icd10).
"""

from __future__ import annotations

import re
from typing import Any

import numpy as np
import pandas as pd

from tquery._types import TQueryCodeError


def resolve_columns(
    patterns: list[str],
    all_columns: list[str],
) -> list[str]:
    """Resolve column name patterns against actual DataFrame columns.

    Supports:
    - Plain names: 'icd' → ['icd']
    - Wildcards: 'icd*' → ['icd1', 'icd2', ..., 'icdmain']
    - Ranges: 'icd1-icd10' → ['icd1', 'icd2', ..., 'icd10'] (alphabetic)
    - Slices: 'icd1:icd10' → columns from icd1 to icd10 in DataFrame order
              (pandas-style positional slice)

    Args:
        patterns: List of column name patterns.
        all_columns: List of actual column names in the DataFrame.

    Returns:
        Deduplicated list of matching column names.
    """
    seen: set[str] = set()
    result: list[str] = []

    for pat in patterns:
        if pat.endswith("*"):
            # Wildcard: icd*
            prefix = pat[:-1]
            for col in all_columns:
                if col.startswith(prefix) and col not in seen:
                    seen.add(col)
                    result.append(col)
        elif ":" in pat:
            # Slice: icd1:icd10 — positional slice in DataFrame column order
            parts = pat.split(":", 1)
            if len(parts) == 2 and parts[0] and parts[1]:
                start, end = parts
                if start in all_columns and end in all_columns:
                    si = all_columns.index(start)
                    ei = all_columns.index(end)
                    for col in all_columns[si:ei + 1]:
                        if col not in seen:
                            seen.add(col)
                            result.append(col)
        elif "-" in pat and not pat.startswith("-"):
            # Range: icd1-icd10 — alphabetic range
            parts = pat.split("-", 1)
            if len(parts) == 2 and parts[0] and parts[1]:
                start, end = parts
                for col in sorted(all_columns):
                    if start <= col <= end and col not in seen:
                        seen.add(col)
                        result.append(col)
        else:
            # Plain name
            if pat in all_columns and pat not in seen:
                seen.add(pat)
                result.append(pat)

    return result


def expand_codes(
    pattern: str,
    all_codes: list[str] | None = None,
    variables: dict[str, Any] | None = None,
) -> list[str]:
    """Expand a single code pattern into a list of concrete codes.

    Args:
        pattern: A code pattern — plain code ('K50'), wildcard ('K50*'),
                 range ('K50-K53'), or variable reference ('@antibiotics').
        all_codes: Sorted list of all codes in the dataset, needed for
                   wildcard and range expansion.
        variables: Dict mapping variable names to code lists, for @ references.

    Returns:
        List of concrete code strings.
    """
    # Variable reference
    if pattern.startswith("@"):
        varname = pattern[1:]
        if variables is None or varname not in variables:
            raise TQueryCodeError(f"Variable '{varname}' not found")
        val = variables[varname]
        if isinstance(val, str):
            return [val]
        return list(val)

    # Wildcard: K50*
    if pattern.endswith("*"):
        prefix = pattern[:-1]
        if all_codes is None:
            # Without a code list, return the pattern as-is for startswith matching
            return [pattern]
        matched = [c for c in all_codes if c.startswith(prefix)]
        if not matched:
            raise TQueryCodeError(
                f"Wildcard '{pattern}' matched no codes in the dataset"
            )
        return matched

    # Range: K50-K53
    if "-" in pattern and not pattern.startswith("-"):
        parts = pattern.split("-", 1)
        if len(parts) == 2 and parts[0] and parts[1]:
            start, end = parts
            if all_codes is None:
                return [pattern]
            matched = [c for c in all_codes if start <= c <= end]
            if not matched:
                raise TQueryCodeError(
                    f"Range '{pattern}' matched no codes in the dataset"
                )
            return matched

    # Plain code
    return [pattern]


def expand_all_codes(
    patterns: tuple[str, ...],
    all_codes: list[str] | None = None,
    variables: dict[str, Any] | None = None,
) -> list[str]:
    """Expand multiple code patterns, deduplicating results."""
    seen: set[str] = set()
    result: list[str] = []
    for pat in patterns:
        for code in expand_codes(pat, all_codes, variables):
            if code not in seen:
                seen.add(code)
                result.append(code)
    return result


def get_matching_rows(
    df: pd.DataFrame,
    codes: list[str],
    cols: list[str],
    sep: str | None = None,
) -> pd.Series:
    """Return a boolean Series marking rows that contain any of the given codes.

    Args:
        df: The input DataFrame.
        codes: List of concrete codes to match.
        cols: Column names to search in.
        sep: If set, cells contain multiple codes separated by this string
             (e.g., ',') and all sub-values are checked.

    Returns:
        Boolean pd.Series aligned to df.index.
    """
    mask = pd.Series(False, index=df.index)

    # Separate wildcard patterns from exact codes
    wildcards = [c for c in codes if c.endswith("*")]
    exact = [c for c in codes if not c.endswith("*")]

    for col in cols:
        if col not in df.columns:
            continue

        if sep is not None:
            # Multi-value cells: use regex matching
            all_patterns = []
            for c in exact:
                all_patterns.append(re.escape(c))
            for w in wildcards:
                all_patterns.append(re.escape(w[:-1]) + r"\S*")
            if all_patterns:
                regex = "|".join(all_patterns)
                mask = mask | df[col].astype(str).str.contains(
                    regex, regex=True, na=False
                )
        else:
            # Single-value cells
            if exact:
                code_set = set(exact)
                mask = mask | df[col].isin(code_set)

            for w in wildcards:
                prefix = w[:-1]
                vals = df[col].values
                if vals.dtype == object:
                    # Use numpy char operations for speed
                    str_vals = np.asarray(vals, dtype=str)
                    mask = mask | pd.Series(
                        np.char.startswith(str_vals, prefix),
                        index=df.index,
                    )
                else:
                    mask = mask | df[col].astype(str).str.startswith(prefix)

    return mask


def extract_codes(
    df: pd.DataFrame,
    codes: dict[str, str | list[str]],
    cols: list[str],
    sep: str | None = None,
    all_codes: list[str] | None = None,
) -> pd.Series:
    """Map each row to its shorthand label from the codes dict.

    Args:
        df: The input DataFrame.
        codes: Dict mapping shorthand labels to code patterns.
               Values can be a single code string or a list of patterns.
               Patterns support wildcards ('L04AB*') and ranges ('K50-K53').
               E.g., {'i': ['L04AB02', '4AB02'], 'a': ['L04AB04']}.
        cols: Column names to search in.
        sep: If cells contain multiple codes, the separator string.
        all_codes: Pre-computed sorted list of all codes in the dataset.

    Returns:
        Series (same index as df) with the label for the first matching
        code group, or NaN for non-matching rows.
    """
    if all_codes is None:
        all_codes = collect_unique_codes(df, cols, sep)

    # Build reverse lookup: {concrete_code: label}
    reverse: dict[str, str] = {}
    # Track wildcard prefixes for startswith matching
    wildcard_prefixes: list[tuple[str, str]] = []  # (prefix, label)

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

    # Vectorized lookup across columns
    result = pd.Series(np.nan, index=df.index, dtype=object)

    for col in cols:
        if col not in df.columns:
            continue
        unfilled = result.isna()
        if not unfilled.any():
            break

        if sep is not None:
            # Multi-value cells: check each sub-value
            for idx in unfilled[unfilled].index:
                val = df.at[idx, col]
                if pd.isna(val):
                    continue
                for sub in str(val).split(sep):
                    sub = sub.strip()
                    if sub in reverse:
                        result.at[idx] = reverse[sub]
                        break
                    for prefix, label in wildcard_prefixes:
                        if sub.startswith(prefix):
                            result.at[idx] = label
                            break
        else:
            vals = df.loc[unfilled, col]
            # Exact match via map
            mapped = vals.map(reverse)
            matched = mapped.notna()
            result.loc[unfilled & matched.reindex(df.index, fill_value=False)] = (
                mapped[matched].values
            )

            # Wildcard fallback for remaining
            still_unfilled = result.isna() & unfilled
            if still_unfilled.any() and wildcard_prefixes:
                remaining_vals = df.loc[still_unfilled, col].astype(str)
                for prefix, label in wildcard_prefixes:
                    str_vals = np.asarray(remaining_vals.values, dtype=str)
                    matches = np.char.startswith(str_vals, prefix)
                    match_idx = remaining_vals.index[matches]
                    result.loc[match_idx] = label
                    remaining_vals = remaining_vals[~matches]
                    if remaining_vals.empty:
                        break

    return result


def collect_unique_codes(
    df: pd.DataFrame,
    cols: list[str],
    sep: str | None = None,
) -> list[str]:
    """Collect all unique codes from the specified columns, sorted."""
    codes: set[str] = set()
    for col in cols:
        if col not in df.columns:
            continue
        vals = df[col].dropna()
        if sep is not None:
            for v in vals:
                codes.update(str(v).split(sep))
        else:
            codes.update(vals.astype(str).unique())
    return sorted(codes)
