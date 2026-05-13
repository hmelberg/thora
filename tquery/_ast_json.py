"""Cross-language JSON serialisation for tquery ASTs.

The format is documented in `spec/ast.md`. Each node becomes a JSON object
with a `_node` discriminator and one key per field. Tuples become arrays;
None becomes JSON null. Recursive children nest as objects.

This module is intentionally trivial and free of pandas/numpy imports so
it can run during spec generation without pulling in the full evaluator.
"""

from __future__ import annotations

from dataclasses import fields
from typing import Any

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

_NODE_TYPES: tuple[type, ...] = (
    CodeAtom, EventAtom, ComparisonAtom, AggregateExpr,
    PrefixExpr, RangePrefixExpr,
    NotExpr, BinaryLogical,
    TemporalExpr, WithinExpr, WithinSpanExpr, InsideExpr, BetweenExpr,
    ShiftExpr, Quantifier,
)

_NAME_TO_TYPE = {t.__name__: t for t in _NODE_TYPES}


def to_json(node: ASTNode) -> dict[str, Any]:
    """Serialise an AST node to a JSON-compatible dict.

    EventAtom has no fields and serialises as `{"_node": "EventAtom"}`.
    """
    obj: dict[str, Any] = {"_node": type(node).__name__}
    for f in fields(node):
        obj[f.name] = _encode(getattr(node, f.name))
    return obj


def from_json(obj: Any) -> ASTNode:
    """Deserialise a JSON-compatible dict back to an AST node."""
    if not isinstance(obj, dict) or "_node" not in obj:
        raise ValueError(f"Not an AST JSON object: {obj!r}")
    name = obj["_node"]
    cls = _NAME_TO_TYPE.get(name)
    if cls is None:
        raise ValueError(f"Unknown AST node type: {name!r}")
    kwargs: dict[str, Any] = {}
    for f in fields(cls):
        if f.name not in obj:
            # Optional fields with defaults — let dataclass apply default
            continue
        kwargs[f.name] = _decode(obj[f.name], f.type)
    return cls(**kwargs)


def _encode(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, _NODE_TYPES):
        return to_json(value)
    if isinstance(value, tuple):
        return [_encode(v) for v in value]
    if isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError(f"Cannot encode value of type {type(value).__name__}: {value!r}")


def _decode(value: Any, type_hint: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, dict) and "_node" in value:
        return from_json(value)
    if isinstance(value, list):
        # All tuple-valued fields in the AST are tuple[str, ...] today.
        # If that ever changes, this needs a smarter type-hint reader.
        return tuple(value)
    return value
