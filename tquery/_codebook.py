"""Codebook system for mapping medical codes to human-readable labels.

Loads CSV files from the codebooks directory. Each CSV must have
'code' and 'label' columns. May optionally have 'level' and 'system'.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import pandas as pd


_DEFAULT_DIR = Path(__file__).parent / "codebooks"

# Singleton cache
_codebook: Codebook | None = None


def _infer_system(code: str) -> str:
    """Guess the coding system from a code's format."""
    code = code.strip()
    # ATC: single letter + digits/letters, 1-7 chars (A, A10, A10BA02)
    if re.match(r"^[A-Z]\d{2}[A-Z]{0,2}\d{0,2}$", code):
        return "atc"
    # ICD-10: letter + 2-3 digits, optional dot + digits (K50, K50.1)
    if re.match(r"^[A-Z]\d{2,3}(\.\d+)?$", code):
        return "icd10"
    # ICD-10 chapter range: A00-B99
    if re.match(r"^[A-Z]\d{2}-[A-Z]\d{2}$", code):
        return "icd10"
    return "unknown"


class Codebook:
    """Registry of code-to-label mappings loaded from CSV files."""

    def __init__(self, dirs: list[Path | str] | None = None) -> None:
        if dirs is None:
            dirs = [_DEFAULT_DIR]
        self._dirs = [Path(d) for d in dirs]
        self._data: pd.DataFrame | None = None
        self._lookup: dict[str, str] = {}
        self._ranges: list[tuple[str, str, str]] = []  # (start, end, label)
        self._load()

    def _load(self) -> None:
        frames: list[pd.DataFrame] = []
        for d in self._dirs:
            if not d.is_dir():
                continue
            for f in sorted(d.glob("*.csv")):
                try:
                    df = pd.read_csv(f, dtype=str).fillna("")
                except Exception:
                    continue
                if "code" not in df.columns or "label" not in df.columns:
                    continue
                if "system" not in df.columns:
                    df["system"] = df["code"].apply(_infer_system)
                if "level" not in df.columns:
                    df["level"] = ""
                df["_source"] = f.name
                frames.append(df[["code", "label", "level", "system", "_source"]])

        if frames:
            self._data = pd.concat(frames, ignore_index=True)
        else:
            self._data = pd.DataFrame(
                columns=["code", "label", "level", "system", "_source"]
            )

        # Build lookup structures
        self._lookup = {}
        self._ranges = []
        for _, row in self._data.iterrows():
            code = row["code"]
            label = row["label"]
            if "-" in code and re.match(r"^[A-Z]\d{2}-[A-Z]\d{2}$", code):
                # Range entry like A00-B99
                start, end = code.split("-")
                self._ranges.append((start, end, label))
            else:
                self._lookup[code] = label

    def label(self, code: str) -> str | None:
        """Look up label for a single code, with hierarchical fallback."""
        # Exact match
        if code in self._lookup:
            return self._lookup[code]

        # Hierarchical fallback: try progressively shorter prefixes
        # L04AB02 → L04AB → L04A → L04 → L
        c = code.replace(".", "")
        while len(c) > 1:
            c = c[:-1]
            if c in self._lookup:
                return self._lookup[c]

        # Range match for ICD-10 chapters (A00-B99 style)
        for start, end, range_label in self._ranges:
            if len(code) >= 3 and start <= code[:3] <= end:
                return range_label

        return None

    def labels(self, codes: pd.Series) -> pd.Series:
        """Map a Series of codes to labels. Vectorized where possible."""
        # Fast path: direct map for exact matches
        result = codes.map(self._lookup)

        # Fill remaining via hierarchical fallback
        missing = result.isna() & codes.notna()
        if missing.any():
            result[missing] = codes[missing].apply(
                lambda c: self.label(c) if pd.notna(c) else None
            )

        return result

    def search(
        self, keyword: str, system: str | None = None
    ) -> pd.DataFrame:
        """Search codebook by keyword in labels (case-insensitive).

        Args:
            keyword: Search term (matched against label text).
            system: Filter to a specific system ('icd10', 'atc').

        Returns:
            DataFrame with code, label, system columns.
        """
        mask = self._data["label"].str.contains(keyword, case=False, na=False)
        if system:
            mask = mask & (self._data["system"] == system)
        return (
            self._data.loc[mask, ["code", "label", "system"]]
            .drop_duplicates(subset=["code"])
            .reset_index(drop=True)
        )

    def get(self, pattern: str) -> pd.DataFrame:
        """Get codes matching a pattern (K50*, K50-K53) with labels.

        Returns:
            DataFrame with code, label columns.
        """
        if pattern.endswith("*"):
            prefix = pattern[:-1]
            mask = self._data["code"].str.startswith(prefix)
        elif "-" in pattern:
            start, end = pattern.split("-", 1)
            mask = (self._data["code"] >= start) & (self._data["code"] <= end)
        else:
            mask = self._data["code"] == pattern
        return (
            self._data.loc[mask, ["code", "label", "system"]]
            .drop_duplicates(subset=["code"])
            .reset_index(drop=True)
        )

    def __repr__(self) -> str:
        n = len(self._data)
        sources = self._data["_source"].nunique() if n > 0 else 0
        return f"Codebook({n} entries from {sources} files)"


def get_codebook(extra_dirs: list[Path | str] | None = None) -> Codebook:
    """Get the global Codebook instance, creating it if needed."""
    global _codebook
    if _codebook is None or extra_dirs is not None:
        dirs = [_DEFAULT_DIR]
        if extra_dirs:
            dirs.extend(Path(d) for d in extra_dirs)
        _codebook = Codebook(dirs)
    return _codebook


def reset_codebook() -> None:
    """Clear the cached codebook (forces reload on next access)."""
    global _codebook
    _codebook = None


def search_codes(keyword: str, system: str | None = None) -> pd.DataFrame:
    """Search codebooks by keyword in labels (case-insensitive).

    Standalone function — works without a DataFrame.

    >>> search_codes('diabetes')
        code  label                          system
    0   E10   Type 1 diabetes mellitus       icd10
    1   E11   Type 2 diabetes mellitus       icd10
    2   A10   Drugs used in diabetes         atc
    """
    return get_codebook().search(keyword, system)


def get_label(code: str) -> str | None:
    """Look up the label for a single code, with hierarchical fallback.

    >>> get_label('K50')
    'Diseases of the digestive system'
    >>> get_label('L04AB02')
    'Infliximab'
    """
    return get_codebook().label(code)


def count_codes(
    df: pd.DataFrame,
    cols: str | list[str] | None = None,
    *,
    pid: str = "pid",
    date: str = "start_date",
    per_person: bool = False,
    pattern: str | None = None,
    sep: str | None = None,
) -> pd.Series:
    """Count code frequencies across columns.

    Args:
        df: The DataFrame.
        cols: Columns to count. None = auto-detect object columns.
        pid: Person ID column (used for per_person counting).
        date: Date column (excluded from auto-detection).
        per_person: If True, count unique persons per code instead of events.
        pattern: Filter to codes matching this pattern (e.g., 'K50*').
        sep: Separator for multi-value cells.

    Returns:
        Series indexed by code, values are counts, sorted descending.
    """
    if cols is None:
        cols = [c for c in df.columns if c not in (pid, date) and df[c].dtype == object]
    elif isinstance(cols, str):
        cols = [cols]

    all_codes: list[pd.Series] = []
    all_pids: list[pd.Series] = []

    for col in cols:
        if col not in df.columns:
            continue
        vals = df[[pid, col]].dropna(subset=[col])
        if sep:
            exploded = vals[col].str.split(sep).explode().str.strip()
            pids = vals[pid].reindex(exploded.index)
            all_codes.append(exploded)
            all_pids.append(pids)
        else:
            all_codes.append(vals[col].astype(str))
            all_pids.append(vals[pid])

    if not all_codes:
        return pd.Series(dtype=int)

    combined_codes = pd.concat(all_codes, ignore_index=True)
    combined_pids = pd.concat(all_pids, ignore_index=True)

    if pattern:
        if pattern.endswith("*"):
            prefix = pattern[:-1]
            mask = combined_codes.str.startswith(prefix)
        else:
            mask = combined_codes == pattern
        combined_codes = combined_codes[mask]
        combined_pids = combined_pids[mask]

    if per_person:
        # Count unique persons per code
        pair = pd.DataFrame({"pid": combined_pids, "code": combined_codes})
        result = pair.drop_duplicates().groupby("code").size()
    else:
        result = combined_codes.value_counts()

    return result.sort_values(ascending=False).astype(int)
