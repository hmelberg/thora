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
class EventAtom:
    """Matches every row in the timeline.

    Spelled `event` or `events` in the DSL. Lets positional selectors
    (`5th event`, `last 5 events`) and temporal operators (`before 5th
    event`) attach to the timeline itself rather than a code pattern.
    """


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

    For 'inside N days': min_days=0, days=N.
    For 'inside M to N days': min_days=M, days=N.
    `outside=True` flips to the row-level complement restricted to
    evaluable persons.
    """
    child: ASTNode
    days: int
    min_days: int = 0
    direction: str | None = None  # 'before', 'after', 'around', or None
    ref: ASTNode | None = None
    outside: bool = False


@dataclass(frozen=True, slots=True)
class InsideExpr:
    """Event-window constraint: child events inside/outside an event
    window relative to a reference.

    The window is the closed integer range ``[min_events, max_events]``
    in row-positions offset from each ref row (positive = after, negative
    = before). Shorthand `inside N events after Y` stores
    ``min_events=1, max_events=N``.
    """
    child: ASTNode
    inside: bool  # True = inside, False = outside
    min_events: int
    max_events: int
    direction: str  # 'before', 'after', 'around'
    ref: ASTNode


@dataclass(frozen=True, slots=True)
class BetweenExpr:
    """Positional-bounds window.

    Child rows whose date falls in the per-person closed interval
    ``[min(bound_start dates), max(bound_end dates)]``. Spelled in the DSL
    as ``CHILD inside EXPR and EXPR`` (e.g., ``K50 inside 1st K51 and
    5th K51``). `outside=True` gives the row-level complement.
    """
    child: ASTNode
    bound_start: ASTNode
    bound_end: ASTNode
    outside: bool = False


@dataclass(frozen=True, slots=True)
class WithinSpanExpr:
    """Positional-span window.

    Child rows whose date falls in the per-person span
    ``[min(ref dates), max(ref dates)]`` — i.e., the date range covered
    by a single multi-row selector. Spelled as ``CHILD inside EXPR``
    (e.g., ``K50 inside last 5 events``). `outside=True` gives the
    row-level complement.
    """
    child: ASTNode
    ref: ASTNode
    outside: bool = False


@dataclass(frozen=True, slots=True)
class ShiftExpr:
    """Per-person synthetic date: child's date shifted by offset_days.

    ``child`` must be a single-date expression (ordinal selector or another
    ShiftExpr). ``offset_days`` is signed: negative = earlier, positive =
    later. Only meaningful as a reference in temporal / window / bounds
    positions; evaluation in other positions raises.
    """
    child: ASTNode
    offset_days: int


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
    EventAtom,
    ComparisonAtom,
    PrefixExpr,
    RangePrefixExpr,
    NotExpr,
    BinaryLogical,
    TemporalExpr,
    WithinExpr,
    WithinSpanExpr,
    InsideExpr,
    BetweenExpr,
    ShiftExpr,
    Quantifier,
]
