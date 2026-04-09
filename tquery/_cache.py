"""Expression-level caching for the tquery evaluator.

AST nodes are frozen dataclasses and thus hashable, so they can serve
directly as dict keys.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from tquery._ast import ASTNode


class EvalCache:
    """Simple dict-backed cache mapping AST nodes to evaluated results."""

    __slots__ = ("_store",)

    def __init__(self) -> None:
        self._store: dict[ASTNode, pd.Series] = {}

    def get(self, node: ASTNode) -> pd.Series | None:
        return self._store.get(node)

    def put(self, node: ASTNode, result: pd.Series) -> None:
        self._store[node] = result

    def clear(self) -> None:
        self._store.clear()

    def __len__(self) -> int:
        return len(self._store)
