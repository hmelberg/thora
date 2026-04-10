"""AST node definitions for the tquery temporal query language.

All nodes are frozen (immutable) dataclasses with __slots__ for memory
efficiency. Being frozen makes them hashable, which enables using them
as cache keys in the evaluator.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Union


@dataclass(frozen=True, slots=True)
class CodeAtom:
    """A reference to one or more medical codes, optionally in specific columns."""
    codes: tuple[str, ...]
    columns: tuple[str, ...] | None = None


@dataclass(frozen=True, slots=True)
class ComparisonAtom:
    """A column comparison like 'glucose > 8'."""
    column: str
    op: str  # '>', '<', '>=', '<=', '==', '!='
    value: float


@dataclass(frozen=True, slots=True)
class PrefixExpr:
    """A quantifier prefix applied to a child expression.

    kind: 'min', 'max', 'exactly' for counts;
          'ordinal' for positional (1st, 2nd, ...);
          'first', 'last' for first/last N occurrences.
    """
    kind: str
    n: int
    child: ASTNode


@dataclass(frozen=True, slots=True)
class RangePrefixExpr:
    """Count range: persons with between min_n and max_n events."""
    min_n: int
    max_n: int
    child: ASTNode


@dataclass(frozen=True, slots=True)
class NotExpr:
    """Logical negation of a child expression."""
    child: ASTNode


@dataclass(frozen=True, slots=True)
class BinaryLogical:
    """Logical AND/OR combining two expressions (person-level semantics)."""
    op: str  # 'and' | 'or'
    left: ASTNode
    right: ASTNode


@dataclass(frozen=True, slots=True)
class TemporalExpr:
    """Temporal ordering: left occurs before/after/simultaneously with right."""
    op: str  # 'before' | 'after' | 'simultaneously'
    left: ASTNode
    right: ASTNode


@dataclass(frozen=True, slots=True)
class WithinExpr:
    """Time-window constraint: child events within a day range of ref events.

    For 'within N days': min_days=0, days=N.
    For 'between M and N days': min_days=M, days=N.
    """
    child: ASTNode
    days: int
    min_days: int = 0
    direction: str | None = None  # 'before', 'after', 'around', or None
    ref: ASTNode | None = None


@dataclass(frozen=True, slots=True)
class InsideExpr:
    """Event-window constraint: child events inside/outside N events of ref."""
    child: ASTNode
    inside: bool  # True = inside, False = outside
    n_events: int
    direction: str  # 'before', 'after', 'around'
    ref: ASTNode


@dataclass(frozen=True, slots=True)
class Quantifier:
    """Universal or existential quantifier over an atom in a temporal context.

    kind='any' is the default existential and is semantically a no-op — the
    parser elides it. kind='every' is universal: the surrounding temporal
    or within predicate must hold for ALL events of the child atom, AND the
    child must be non-empty for the person (no vacuous truth).
    """
    kind: str  # 'any' | 'every'
    child: ASTNode  # always a CodeAtom (enforced at parse time)


# Union type for all AST nodes
ASTNode = Union[
    CodeAtom,
    ComparisonAtom,
    PrefixExpr,
    RangePrefixExpr,
    NotExpr,
    BinaryLogical,
    TemporalExpr,
    WithinExpr,
    InsideExpr,
    Quantifier,
]
