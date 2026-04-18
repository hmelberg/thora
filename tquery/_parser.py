"""Tokenizer and recursive descent parser for the tquery DSL.

Converts a query string like 'min 3 of K50* before K51' into an AST.
No external dependencies — hand-written for zero-dep installation and
precise error messages.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum, auto
from typing import Any

from tquery._ast import (
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
from tquery._types import TQuerySyntaxError

# ---------------------------------------------------------------------------
# Tokenizer
# ---------------------------------------------------------------------------

class TokenType(Enum):
    CODE = auto()        # K50, 4AB02, K50.1
    STAR_CODE = auto()   # K50*
    RANGE_CODE = auto()  # K50-K53
    SLICE_CODE = auto()  # icd1:icd10 (column slice, pandas-style)
    INT_RANGE = auto()   # 2-5 (count range for prefixes)
    INT = auto()         # 3, 100
    FLOAT = auto()       # 8.5
    OP = auto()          # >, <, >=, <=, ==, !=
    LPAREN = auto()
    RPAREN = auto()
    COMMA = auto()
    AT = auto()          # @
    KEYWORD = auto()     # before, after, and, or, ...
    ORDINAL = auto()     # 1st, 2nd, 3rd, 4th, ...
    IDENT = auto()       # column names (when not a keyword or code)
    EOF = auto()


KEYWORDS = frozenset({
    "before", "after", "simultaneously",
    "inside", "outside",
    "and", "or", "not", "never",
    "min", "max", "exactly",
    "of", "in", "to",
    "first", "last",
    "days", "event", "events",
    "around",
    "every", "any", "each", "always",
})

# Keywords that are also valid as direction modifiers inside within/inside clauses
DIRECTION_KEYWORDS = frozenset({"before", "after", "around"})


@dataclass(slots=True)
class Token:
    type: TokenType
    value: Any
    pos: int  # position in original string


# Regex patterns for tokenization (order matters — more specific first)
_TOKEN_PATTERNS: list[tuple[str, TokenType | None]] = [
    (r"\s+", None),                                          # skip whitespace
    (r"\(", TokenType.LPAREN),
    (r"\)", TokenType.RPAREN),
    (r",", TokenType.COMMA),
    (r"@", TokenType.AT),
    (r">=|<=|!=|==|[><+\-]", TokenType.OP),
    (r"\d+(st|nd|rd|th)\b", TokenType.ORDINAL),              # 1st, 2nd, 3rd, 4th
    (r"\d+\.\d+", TokenType.FLOAT),                          # 8.5
    (r"\d+-\d+", TokenType.INT_RANGE),                       # 2-5 (count range)
    (r"\d+", TokenType.INT),                                  # 100
    # Code with wildcard: K50* (must come before plain code)
    (r"[A-Za-z_]\w*(?:\.\w+)?\*", TokenType.STAR_CODE),
    # Column slice: icd1:icd10 (colon-separated, for pandas-style column ranges)
    (r"[A-Za-z_]\w*:[A-Za-z_]\w*", TokenType.SLICE_CODE),
    # Code range: K50-K53 (letter-starting, hyphen, letter-starting)
    (r"[A-Za-z]\w*(?:\.\w+)?-[A-Za-z]\w*(?:\.\w+)?", TokenType.RANGE_CODE),
    # Plain code or identifier: K50, glucose, icd1
    (r"[A-Za-z_]\w*(?:\.\w+)?", TokenType.CODE),
]

_TOKEN_RE = re.compile(
    "|".join(f"(?P<g{i}>{pat})" for i, (pat, _) in enumerate(_TOKEN_PATTERNS))
)


def tokenize(expr: str) -> list[Token]:
    """Tokenize a tquery expression string into a list of Tokens."""
    tokens: list[Token] = []
    pos = 0
    for m in _TOKEN_RE.finditer(expr):
        if m.start() != pos:
            bad = expr[pos:m.start()]
            raise TQuerySyntaxError(
                f"Unexpected character(s): {bad!r}", expr, pos
            )
        pos = m.end()

        # Find which group matched
        for i, (_, ttype) in enumerate(_TOKEN_PATTERNS):
            if m.group(f"g{i}") is not None:
                if ttype is None:
                    break  # skip whitespace
                raw = m.group(f"g{i}")
                value: Any = raw

                if ttype == TokenType.INT:
                    value = int(raw)
                elif ttype == TokenType.FLOAT:
                    value = float(raw)
                elif ttype == TokenType.ORDINAL:
                    value = int(re.match(r"\d+", raw).group())  # type: ignore[union-attr]
                elif ttype == TokenType.CODE:
                    # Check if it's a keyword
                    if raw.lower() in KEYWORDS:
                        ttype = TokenType.KEYWORD
                        value = raw.lower()

                tokens.append(Token(ttype, value, m.start()))
                break
        else:
            raise TQuerySyntaxError(
                f"Internal tokenizer error at position {m.start()}", expr, m.start()
            )

    if pos != len(expr):
        raise TQuerySyntaxError(
            f"Unexpected character(s): {expr[pos:]!r}", expr, pos
        )

    tokens.append(Token(TokenType.EOF, None, len(expr)))
    return tokens


# ---------------------------------------------------------------------------
# Recursive Descent Parser
# ---------------------------------------------------------------------------

class Parser:
    """Recursive descent parser for tquery expressions.

    Grammar (precedence low → high):
        query         ::= or_expr
        or_expr       ::= and_expr ('or' and_expr)*
        and_expr      ::= temporal_expr ('and' temporal_expr)*
        temporal_expr ::= not_expr (('before'|'after'|'simultaneously') not_expr)?
        not_expr      ::= 'not'? prefix_expr
        prefix_expr   ::= prefix_core shift_suffix*
        shift_suffix  ::= ('+' | '-') INT 'days'
        prefix_core   ::= prefix? inside_expr
        inside_expr   ::= atom (('inside'|'outside') inside_tail)?
        inside_tail   ::= INT ('to' INT)? ('days'|'events') [dir ['of'] ref]
                        | EXPR ('and' EXPR)?
        prefix        ::= ('min'|'max'|'exactly') INT 'of'
                        | ORDINAL 'of'
                        | ('first'|'last') INT 'of'
        atom          ::= '(' query ')'
                        | 'event' | 'events'
                        | code_expr column_spec?
                        | column_comparison
        code_expr     ::= code_item (',' code_item)*
        code_item     ::= '@' IDENT | STAR_CODE | RANGE_CODE | CODE
        column_spec   ::= 'in' IDENT (',' IDENT)*
        column_comp   ::= IDENT OP NUMBER

    Precedence note: 'inside'/'outside' bind tighter than prefixes so that
    'min 2 of X inside 50 days after Y' means 'min 2 of (X inside 50 days
    after Y)' — i.e., the window filters rows first, then the count is
    applied.
    """

    def __init__(self, tokens: list[Token], expr: str) -> None:
        self.tokens = tokens
        self.expr = expr
        self.pos = 0

    def _peek(self) -> Token:
        return self.tokens[self.pos]

    def _advance(self) -> Token:
        tok = self.tokens[self.pos]
        self.pos += 1
        return tok

    def _expect_keyword(self, *values: str) -> Token:
        tok = self._peek()
        if tok.type == TokenType.KEYWORD and tok.value in values:
            return self._advance()
        expected = " or ".join(f"'{v}'" for v in values)
        raise TQuerySyntaxError(
            f"Expected {expected}, got {tok.value!r}", self.expr, tok.pos
        )

    def _at_keyword(self, *values: str) -> bool:
        tok = self._peek()
        return tok.type == TokenType.KEYWORD and tok.value in values

    def _at_type(self, *types: TokenType) -> bool:
        return self._peek().type in types

    def _error(self, msg: str) -> TQuerySyntaxError:
        return TQuerySyntaxError(msg, self.expr, self._peek().pos)

    # --- Grammar rules ---

    def parse(self) -> ASTNode:
        node = self._parse_or()
        if self._peek().type != TokenType.EOF:
            raise self._error(f"Unexpected token: {self._peek().value!r}")
        return node

    def _parse_or(self) -> ASTNode:
        left = self._parse_and()
        while self._at_keyword("or"):
            self._advance()
            right = self._parse_and()
            left = BinaryLogical("or", left, right)
        return left

    def _parse_and(self) -> ASTNode:
        left = self._parse_temporal()
        while self._at_keyword("and"):
            self._advance()
            right = self._parse_temporal()
            left = BinaryLogical("and", left, right)
        return left

    def _parse_temporal(self) -> ASTNode:
        lhs_q = self._try_quantified()
        left = lhs_q if lhs_q is not None else self._parse_not()
        if self._at_keyword("before", "after", "simultaneously"):
            op = self._advance().value
            rhs_q = self._try_quantified()
            right = rhs_q if rhs_q is not None else self._parse_not()
            return TemporalExpr(op, left, right)
        if isinstance(left, Quantifier):
            raise self._error(
                f"'{left.kind}' requires a temporal operator (before/after/simultaneously)"
            )
        return left

    def _try_quantified(self) -> ASTNode | None:
        """Try to parse a quantifier (`every`, `any`, `each`, `always`) followed by a code atom.

        `every`, `each`, and `always` are synonyms for universal
        quantification — `each` reads more naturally on a window's ref
        side, `always` on the subject side. `any` is the existential
        default and is elided as a no-op. All four are restricted to
        bare code expressions (same restrictions as `every`).

        Returns:
            - `Quantifier(every, CodeAtom)` for `every/each/always X`
            - A window expression if one follows
            - `X` for `any X` (the `any` is elided as a no-op)
            - `None` if no quantifier keyword is present (caller falls back)
        """
        if not self._at_keyword("every", "any", "each", "always"):
            return None
        kind_keyword = self._advance().value
        # `each`/`always` map onto `every` at the AST level.
        is_any = kind_keyword == "any"

        tok = self._peek()
        if tok.type == TokenType.LPAREN:
            raise self._error(
                f"'{kind_keyword}' cannot be applied to a parenthesized group"
            )
        if tok.type == TokenType.KEYWORD and tok.value in (
            "min", "max", "exactly", "first", "last", "not", "never",
        ):
            raise self._error(
                f"'{kind_keyword}' cannot be combined with '{tok.value}'"
            )
        if tok.type == TokenType.INT_RANGE or tok.type == TokenType.ORDINAL:
            raise self._error(
                f"'{kind_keyword}' cannot be combined with a count prefix"
            )
        atom = self._parse_code_expr()
        wrapped = atom if is_any else Quantifier(kind="every", child=atom)
        # If an inside/outside window follows, attach it to the quantified atom.
        if self._at_keyword("inside", "outside"):
            return self._parse_within(child=wrapped)
        return wrapped

    def _parse_not(self) -> ASTNode:
        if self._at_keyword("not", "never"):
            self._advance()
            child = self._parse_prefix()
            return NotExpr(child)
        return self._parse_prefix()

    def _parse_prefix(self) -> ASTNode:
        expr = self._parse_prefix_core()
        return self._maybe_attach_shift(expr)

    def _maybe_attach_shift(self, expr: ASTNode) -> ASTNode:
        """Consume any trailing `('+'|'-') INT 'days'` and wrap in ShiftExpr.

        Chains multiple shifts (`1st Y - 30 days - 7 days`). Validates at
        each step that the child is a single-date expression; raises
        `TQuerySyntaxError` otherwise.
        """
        while self._lookahead_is_shift():
            sign = 1 if self._advance().value == "+" else -1
            n = self._advance().value
            self._advance()  # 'days'
            if not _is_single_date_expr(expr):
                raise self._error(
                    "'± N days' requires a single-date anchor "
                    "(an ordinal like `1st K51` or `-1st event`)"
                )
            expr = ShiftExpr(expr, sign * n)
        return expr

    def _lookahead_is_shift(self) -> bool:
        if not (self._at_type(TokenType.OP) and self._peek().value in ("+", "-")):
            return False
        if self.pos + 2 >= len(self.tokens):
            return False
        t1 = self.tokens[self.pos + 1]
        t2 = self.tokens[self.pos + 2]
        return (
            t1.type == TokenType.INT
            and t2.type == TokenType.KEYWORD
            and t2.value == "days"
        )

    def _parse_prefix_core(self) -> ASTNode:
        # Count range prefix: 2-5 of K50
        if self._at_type(TokenType.INT_RANGE):
            raw = self._advance().value  # "2-5"
            parts = raw.split("-")
            min_n, max_n = int(parts[0]), int(parts[1])
            if self._at_keyword("of"):
                self._advance()
            child = self._parse_within()
            return RangePrefixExpr(min_n, max_n, child)

        # Check for quantifier prefixes
        if self._at_keyword("min", "max", "exactly"):
            kind = self._advance().value
            tok = self._peek()
            if tok.type != TokenType.INT:
                raise self._error(f"Expected integer after '{kind}', got {tok.value!r}")
            n = self._advance().value
            if self._at_keyword("of"):
                self._advance()
            child = self._parse_within()
            return PrefixExpr(kind, n, child)

        if self._at_type(TokenType.ORDINAL):
            n = self._advance().value
            if self._at_keyword("of"):
                self._advance()
            child = self._parse_within()
            return PrefixExpr("ordinal", n, child)

        # Negative ordinal: `-Nth X` means the Nth-from-last match per person.
        if (self._at_type(TokenType.OP) and self._peek().value == "-"
            and self.pos + 1 < len(self.tokens)
            and self.tokens[self.pos + 1].type == TokenType.ORDINAL):
            self._advance()  # consume '-'
            n = self._advance().value
            if n == 0:
                raise self._error("Ordinal cannot be -0")
            if self._at_keyword("of"):
                self._advance()
            child = self._parse_within()
            return PrefixExpr("ordinal", -n, child)

        if self._at_keyword("first", "last"):
            kind = self._advance().value
            tok = self._peek()
            if tok.type != TokenType.INT:
                raise self._error(f"Expected integer after '{kind}', got {tok.value!r}")
            n = self._advance().value
            if self._at_keyword("of"):
                self._advance()
            child = self._parse_within()
            return PrefixExpr(kind, n, child)

        return self._parse_within()

    def _parse_within(self, child: ASTNode | None = None) -> ASTNode:
        """Parse `inside`/`outside` constructs attached to a child atom.

        Grammar:
            inside_expr ::= atom ('inside'|'outside') tail
            tail        ::= INT ['to' INT] ('days'|'events') [dir ['of'] ref]
                          | EXPR ['and' EXPR]
            dir         ::= 'before' | 'after' | 'around'

        Notes:
            - `inside N days [dir [Y]]` — time window, direction and ref
              both optional.
            - `inside N events dir Y` — event window, direction and ref
              required.
            - `inside EXPR` — positional span. `inside EXPR and EXPR` —
              positional bounds.
            - `outside …` gives the row-level complement (restricted to
              evaluable persons) for each of the above.
        """
        if child is None:
            child = self._parse_atom()
        if not self._at_keyword("inside", "outside"):
            return child

        keyword = self._advance().value
        outside = keyword == "outside"

        tok = self._peek()

        # Numeric form: `inside [-]N [to [-]M] days/events ...`
        # A leading '-' is allowed (consumed as an OP token by the lexer).
        if tok.type == TokenType.INT or (
            tok.type == TokenType.OP and tok.value == "-"
            and self.pos + 1 < len(self.tokens)
            and self.tokens[self.pos + 1].type == TokenType.INT
        ):
            first = self._parse_signed_int()
            min_val = 0
            max_val = first
            if self._at_keyword("to"):
                self._advance()
                min_val = first
                max_val = self._parse_signed_int()
            return self._finish_numeric_inside(
                child, min_val, max_val, outside
            )

        # EXPR form: positional span or positional bounds
        bound_start = self._parse_prefix()
        if self._at_keyword("and"):
            self._advance()
            bound_end = self._parse_prefix()
            return BetweenExpr(child, bound_start, bound_end, outside=outside)
        return WithinSpanExpr(child, bound_start, outside=outside)

    def _parse_signed_int(self) -> int:
        """Consume `[-]INT` and return the signed value."""
        negate = False
        tok = self._peek()
        if tok.type == TokenType.OP and tok.value == "-":
            self._advance()
            negate = True
        tok = self._peek()
        if tok.type != TokenType.INT:
            raise self._error(f"Expected integer, got {tok.value!r}")
        value = self._advance().value
        return -value if negate else value

    def _finish_numeric_inside(
        self,
        child: ASTNode,
        min_val: int,
        max_val: int,
        outside: bool,
    ) -> ASTNode:
        """Finish parsing the unit + direction tail of a numeric inside/outside."""
        unit_tok = self._peek()
        if unit_tok.type != TokenType.KEYWORD or unit_tok.value not in (
            "days", "event", "events",
        ):
            raise self._error(
                f"Expected 'days' or 'events', got {unit_tok.value!r}"
            )
        unit_raw = self._advance().value
        unit = "events" if unit_raw in ("event", "events") else unit_raw

        direction: str | None = None
        ref: ASTNode | None = None
        if self._at_keyword("before", "after", "around"):
            direction = self._advance().value
            if self._at_keyword("of"):
                self._advance()
            rhs_q = self._try_quantified()
            ref = rhs_q if rhs_q is not None else self._parse_prefix()

        # Sign validation: negatives are only allowed with `around`.
        has_negative = min_val < 0 or max_val < 0
        if has_negative and direction != "around":
            raise self._error(
                "Negative offsets are only allowed with 'around' "
                "(use 'before' with a positive number instead)"
            )
        if min_val > max_val:
            raise self._error(
                f"Range must be ascending: got {min_val} to {max_val}"
            )

        if unit == "days":
            return WithinExpr(
                child, max_val, min_val, direction, ref, outside=outside
            )
        # events
        if direction is None or ref is None:
            raise self._error(
                "Event window requires a direction (before/after/around) "
                "and a reference expression"
            )
        # Shorthand `inside N events after Y` means positions +1..+N
        # (Y itself is excluded). With an explicit range, both bounds are
        # taken literally, including sign for `around`.
        if min_val == 0 and max_val > 0 and direction != "around":
            min_events = 1
            max_events = max_val
        else:
            min_events = min_val
            max_events = max_val
        return InsideExpr(
            child, not outside, min_events, max_events, direction, ref
        )

    def _parse_atom(self) -> ASTNode:
        tok = self._peek()

        # Parenthesized sub-expression
        if tok.type == TokenType.LPAREN:
            self._advance()
            node = self._parse_or()
            if self._peek().type != TokenType.RPAREN:
                raise self._error("Expected ')'")
            self._advance()
            return node

        # Universal-row atom: `event` or `events` matches every row.
        if self._at_keyword("event", "events"):
            self._advance()
            return EventAtom()

        # Try column comparison: IDENT OP NUMBER
        if self._is_comparison_ahead():
            return self._parse_comparison()

        # Code expression (possibly comma-separated, with optional column spec)
        return self._parse_code_expr()

    def _is_comparison_ahead(self) -> bool:
        """Look ahead to see if this is IDENT OP NUMBER (not a code list)."""
        if self.pos + 2 >= len(self.tokens):
            return False
        t0 = self.tokens[self.pos]
        t1 = self.tokens[self.pos + 1]
        # Must be: CODE/IDENT (non-keyword), then a comparison OP (not `-`).
        if t0.type == TokenType.CODE and t1.type == TokenType.OP:
            if t1.value not in (">", "<", ">=", "<=", "==", "!="):
                return False
            # Make sure the CODE isn't a keyword being used as ident
            if t0.value.lower() not in KEYWORDS:
                return True
        return False

    def _parse_comparison(self) -> ComparisonAtom:
        col_tok = self._advance()  # IDENT
        op_tok = self._advance()   # OP
        val_tok = self._peek()
        if val_tok.type not in (TokenType.INT, TokenType.FLOAT):
            raise self._error(f"Expected number after '{op_tok.value}', got {val_tok.value!r}")
        self._advance()
        return ComparisonAtom(col_tok.value, op_tok.value, float(val_tok.value))

    def _parse_code_expr(self) -> CodeAtom:
        codes: list[str] = []

        # First code item
        code = self._parse_code_item()
        codes.append(code)

        # Comma-separated additional codes
        while self._peek().type == TokenType.COMMA:
            self._advance()
            code = self._parse_code_item()
            codes.append(code)

        # Optional column spec: 'in' col1, col2, ...
        # Also supports wildcards (icd*) and ranges (icd1-icd10)
        _COL_TYPES = (TokenType.CODE, TokenType.IDENT, TokenType.STAR_CODE, TokenType.RANGE_CODE, TokenType.SLICE_CODE)
        columns: tuple[str, ...] | None = None
        if self._at_keyword("in"):
            self._advance()
            cols: list[str] = []
            tok = self._peek()
            if tok.type not in _COL_TYPES:
                raise self._error(f"Expected column name after 'in', got {tok.value!r}")
            cols.append(self._advance().value)
            while self._peek().type == TokenType.COMMA:
                self._advance()
                tok = self._peek()
                if tok.type not in _COL_TYPES:
                    raise self._error(f"Expected column name, got {tok.value!r}")
                cols.append(self._advance().value)
            columns = tuple(cols)

        return CodeAtom(tuple(codes), columns)

    def _parse_code_item(self) -> str:
        tok = self._peek()

        # @variable reference
        if tok.type == TokenType.AT:
            self._advance()
            name_tok = self._peek()
            if name_tok.type not in (TokenType.CODE, TokenType.IDENT):
                raise self._error(f"Expected variable name after '@', got {name_tok.value!r}")
            return "@" + self._advance().value

        # Star code: K50*
        if tok.type == TokenType.STAR_CODE:
            return self._advance().value

        # Range code: K50-K53
        if tok.type == TokenType.RANGE_CODE:
            return self._advance().value

        # Plain code: K50, K51.2, 4AB02
        if tok.type == TokenType.CODE:
            val = tok.value
            # Don't consume keywords as codes (except after specific contexts)
            if isinstance(val, str) and val.lower() in KEYWORDS:
                raise self._error(
                    f"Expected a code, got keyword '{val}'. "
                    f"Use parentheses if you mean a code that matches a keyword name."
                )
            return self._advance().value

        raise self._error(f"Expected a code expression, got {tok.value!r}")


def _is_single_date_expr(node: ASTNode) -> bool:
    """True if `node` is known to produce a single date per person.

    Used to validate the child of a `± N days` shift suffix.
    """
    if isinstance(node, ShiftExpr):
        return True
    if isinstance(node, PrefixExpr) and node.kind == "ordinal":
        return True
    return False


def parse(expr: str) -> ASTNode:
    """Parse a tquery expression string into an AST.

    Args:
        expr: A tquery expression like 'K50 before K51' or
              'min 3 of K50* within 100 days after 1st K51'.

    Returns:
        The root ASTNode of the parsed expression tree.

    Raises:
        TQuerySyntaxError: If the expression has invalid syntax.
    """
    tokens = tokenize(expr)
    parser = Parser(tokens, expr)
    return parser.parse()
