"""Tests for the tquery tokenizer and parser."""

import pytest

from tquery._ast import (
    BinaryLogical,
    CodeAtom,
    ComparisonAtom,
    InsideExpr,
    NotExpr,
    PrefixExpr,
    TemporalExpr,
    WithinExpr,
)
from tquery._parser import Token, TokenType, parse, tokenize
from tquery._types import TQuerySyntaxError


# ---------------------------------------------------------------------------
# Tokenizer tests
# ---------------------------------------------------------------------------

class TestTokenizer:
    def test_simple_code(self):
        tokens = tokenize("K50")
        assert tokens[0].type == TokenType.CODE
        assert tokens[0].value == "K50"
        assert tokens[1].type == TokenType.EOF

    def test_star_code(self):
        tokens = tokenize("K50*")
        assert tokens[0].type == TokenType.STAR_CODE
        assert tokens[0].value == "K50*"

    def test_range_code(self):
        tokens = tokenize("K50-K53")
        assert tokens[0].type == TokenType.RANGE_CODE
        assert tokens[0].value == "K50-K53"

    def test_ordinal(self):
        for text, expected in [("1st", 1), ("2nd", 2), ("3rd", 3), ("10th", 10)]:
            tokens = tokenize(text)
            assert tokens[0].type == TokenType.ORDINAL
            assert tokens[0].value == expected

    def test_integer(self):
        tokens = tokenize("42")
        assert tokens[0].type == TokenType.INT
        assert tokens[0].value == 42

    def test_float(self):
        tokens = tokenize("8.5")
        assert tokens[0].type == TokenType.FLOAT
        assert tokens[0].value == 8.5

    def test_operators(self):
        for op in [">", "<", ">=", "<=", "==", "!="]:
            tokens = tokenize(op)
            assert tokens[0].type == TokenType.OP
            assert tokens[0].value == op

    def test_keywords_recognized(self):
        tokens = tokenize("before after and or not within")
        kw_values = [t.value for t in tokens if t.type == TokenType.KEYWORD]
        assert kw_values == ["before", "after", "and", "or", "not", "within"]

    def test_parens_and_comma(self):
        tokens = tokenize("(K50, K51)")
        types = [t.type for t in tokens[:-1]]  # exclude EOF
        assert types == [
            TokenType.LPAREN,
            TokenType.CODE,
            TokenType.COMMA,
            TokenType.CODE,
            TokenType.RPAREN,
        ]

    def test_at_sign(self):
        tokens = tokenize("@myvar")
        assert tokens[0].type == TokenType.AT
        assert tokens[1].type == TokenType.CODE
        assert tokens[1].value == "myvar"

    def test_complex_expression(self):
        tokens = tokenize("min 3 of K50* within 100 days after 1st K51")
        types = [t.type for t in tokens[:-1]]
        assert types == [
            TokenType.KEYWORD,   # min
            TokenType.INT,       # 3
            TokenType.KEYWORD,   # of
            TokenType.STAR_CODE, # K50*
            TokenType.KEYWORD,   # within
            TokenType.INT,       # 100
            TokenType.KEYWORD,   # days
            TokenType.KEYWORD,   # after
            TokenType.ORDINAL,   # 1st
            TokenType.CODE,      # K51
        ]

    def test_dotted_code(self):
        tokens = tokenize("K50.1")
        assert tokens[0].type == TokenType.CODE
        assert tokens[0].value == "K50.1"

    def test_position_tracking(self):
        tokens = tokenize("K50 before K51")
        assert tokens[0].pos == 0   # K50
        assert tokens[1].pos == 4   # before
        assert tokens[2].pos == 11  # K51

    def test_invalid_character(self):
        with pytest.raises(TQuerySyntaxError, match="Unexpected character"):
            tokenize("K50 # K51")


# ---------------------------------------------------------------------------
# Parser tests — atoms
# ---------------------------------------------------------------------------

class TestParserAtoms:
    def test_single_code(self):
        ast = parse("K50")
        assert isinstance(ast, CodeAtom)
        assert ast.codes == ("K50",)
        assert ast.columns is None

    def test_star_code(self):
        ast = parse("K50*")
        assert isinstance(ast, CodeAtom)
        assert ast.codes == ("K50*",)

    def test_range_code(self):
        ast = parse("K50-K53")
        assert isinstance(ast, CodeAtom)
        assert ast.codes == ("K50-K53",)

    def test_comma_separated_codes(self):
        ast = parse("K50, K51, K52")
        assert isinstance(ast, CodeAtom)
        assert ast.codes == ("K50", "K51", "K52")

    def test_code_with_column_spec(self):
        ast = parse("K50 in icd1, icd2")
        assert isinstance(ast, CodeAtom)
        assert ast.codes == ("K50",)
        assert ast.columns == ("icd1", "icd2")

    def test_variable_reference(self):
        ast = parse("@antibiotics")
        assert isinstance(ast, CodeAtom)
        assert ast.codes == ("@antibiotics",)

    def test_comparison(self):
        ast = parse("glucose > 8")
        assert isinstance(ast, ComparisonAtom)
        assert ast.column == "glucose"
        assert ast.op == ">"
        assert ast.value == 8.0

    def test_comparison_float(self):
        ast = parse("hba1c >= 6.5")
        assert isinstance(ast, ComparisonAtom)
        assert ast.column == "hba1c"
        assert ast.op == ">="
        assert ast.value == 6.5


# ---------------------------------------------------------------------------
# Parser tests — prefix expressions
# ---------------------------------------------------------------------------

class TestParserPrefix:
    def test_min(self):
        ast = parse("min 3 of K50")
        assert isinstance(ast, PrefixExpr)
        assert ast.kind == "min"
        assert ast.n == 3
        assert isinstance(ast.child, CodeAtom)
        assert ast.child.codes == ("K50",)

    def test_max(self):
        ast = parse("max 5 of K50")
        assert isinstance(ast, PrefixExpr)
        assert ast.kind == "max"
        assert ast.n == 5

    def test_exactly(self):
        ast = parse("exactly 2 of K50")
        assert isinstance(ast, PrefixExpr)
        assert ast.kind == "exactly"
        assert ast.n == 2

    def test_ordinal(self):
        ast = parse("1st of K50")
        assert isinstance(ast, PrefixExpr)
        assert ast.kind == "ordinal"
        assert ast.n == 1

    def test_first_n(self):
        ast = parse("first 5 of K50")
        assert isinstance(ast, PrefixExpr)
        assert ast.kind == "first"
        assert ast.n == 5

    def test_last_n(self):
        ast = parse("last 3 of K50")
        assert isinstance(ast, PrefixExpr)
        assert ast.kind == "last"
        assert ast.n == 3


# ---------------------------------------------------------------------------
# Parser tests — logical and temporal operators
# ---------------------------------------------------------------------------

class TestParserOperators:
    def test_and(self):
        ast = parse("K50 and K51")
        assert isinstance(ast, BinaryLogical)
        assert ast.op == "and"
        assert isinstance(ast.left, CodeAtom)
        assert isinstance(ast.right, CodeAtom)

    def test_or(self):
        ast = parse("K50 or K51")
        assert isinstance(ast, BinaryLogical)
        assert ast.op == "or"

    def test_not(self):
        ast = parse("not K50")
        assert isinstance(ast, NotExpr)
        assert isinstance(ast.child, CodeAtom)

    def test_before(self):
        ast = parse("K50 before K51")
        assert isinstance(ast, TemporalExpr)
        assert ast.op == "before"
        assert isinstance(ast.left, CodeAtom)
        assert isinstance(ast.right, CodeAtom)

    def test_after(self):
        ast = parse("K50 after K51")
        assert isinstance(ast, TemporalExpr)
        assert ast.op == "after"

    def test_simultaneously(self):
        ast = parse("K50 simultaneously K51")
        assert isinstance(ast, TemporalExpr)
        assert ast.op == "simultaneously"


# ---------------------------------------------------------------------------
# Parser tests — within/inside
# ---------------------------------------------------------------------------

class TestParserWithin:
    def test_within_days(self):
        ast = parse("K50 within 100 days")
        assert isinstance(ast, WithinExpr)
        assert ast.days == 100
        assert ast.direction is None
        assert ast.ref is None
        assert isinstance(ast.child, CodeAtom)

    def test_within_days_after(self):
        ast = parse("K50 within 100 days after K51")
        assert isinstance(ast, WithinExpr)
        assert ast.days == 100
        assert ast.direction == "after"
        assert isinstance(ast.ref, CodeAtom)
        assert ast.ref.codes == ("K51",)

    def test_within_days_before(self):
        ast = parse("K50 within 30 days before K51")
        assert isinstance(ast, WithinExpr)
        assert ast.direction == "before"

    def test_inside_events(self):
        ast = parse("K50 inside 5 events after K51")
        assert isinstance(ast, InsideExpr)
        assert ast.inside is True
        assert ast.n_events == 5
        assert ast.direction == "after"

    def test_outside_events(self):
        ast = parse("K50 outside 10 events before K51")
        assert isinstance(ast, InsideExpr)
        assert ast.inside is False
        assert ast.n_events == 10
        assert ast.direction == "before"


# ---------------------------------------------------------------------------
# Parser tests — compound expressions
# ---------------------------------------------------------------------------

class TestParserCompound:
    def test_precedence_and_or(self):
        # 'or' binds looser than 'and'
        ast = parse("A or B and C")
        assert isinstance(ast, BinaryLogical)
        assert ast.op == "or"
        assert isinstance(ast.left, CodeAtom)  # A
        assert isinstance(ast.right, BinaryLogical)
        assert ast.right.op == "and"

    def test_precedence_temporal_and(self):
        # 'and' binds looser than 'before'
        ast = parse("A before B and C")
        assert isinstance(ast, BinaryLogical)
        assert ast.op == "and"
        assert isinstance(ast.left, TemporalExpr)
        assert ast.left.op == "before"

    def test_parenthesized_subexpr(self):
        ast = parse("(K50 or K51) before K52")
        assert isinstance(ast, TemporalExpr)
        assert isinstance(ast.left, BinaryLogical)
        assert ast.left.op == "or"
        assert isinstance(ast.right, CodeAtom)

    def test_nested_parens(self):
        ast = parse("(min 2 of K50) and (1st of K51)")
        assert isinstance(ast, BinaryLogical)
        assert isinstance(ast.left, PrefixExpr)
        assert isinstance(ast.right, PrefixExpr)

    def test_complex_expression(self):
        ast = parse("min 3 of K50* within 100 days after 1st of K51")
        # With correct precedence: min 3 of (K50* within 100 days after (1st of K51))
        assert isinstance(ast, PrefixExpr)
        assert ast.kind == "min"
        assert ast.n == 3
        # child is WithinExpr
        assert isinstance(ast.child, WithinExpr)
        assert ast.child.days == 100
        assert ast.child.direction == "after"
        # within's child is CodeAtom(K50*)
        assert isinstance(ast.child.child, CodeAtom)
        assert ast.child.child.codes == ("K50*",)
        # within's ref is PrefixExpr(ordinal, 1, CodeAtom(K51))
        assert isinstance(ast.child.ref, PrefixExpr)
        assert ast.child.ref.kind == "ordinal"

    def test_multiple_and(self):
        ast = parse("A and B and C")
        # Left-associative: (A and B) and C
        assert isinstance(ast, BinaryLogical)
        assert ast.op == "and"
        assert isinstance(ast.left, BinaryLogical)
        assert isinstance(ast.right, CodeAtom)

    def test_not_with_temporal(self):
        ast = parse("not K50 before K51")
        # 'not' binds tighter than 'before'
        assert isinstance(ast, TemporalExpr)
        assert isinstance(ast.left, NotExpr)
        assert isinstance(ast.right, CodeAtom)


# ---------------------------------------------------------------------------
# Parser tests — error cases
# ---------------------------------------------------------------------------

class TestParserErrors:
    def test_empty_expression(self):
        with pytest.raises(TQuerySyntaxError):
            parse("")

    def test_unclosed_paren(self):
        with pytest.raises(TQuerySyntaxError, match="Expected '\\)'"):
            parse("(K50 before K51")

    def test_missing_days_keyword(self):
        with pytest.raises(TQuerySyntaxError, match="'days'"):
            parse("K50 within 100 K51")

    def test_missing_number_after_min(self):
        with pytest.raises(TQuerySyntaxError, match="Expected integer"):
            parse("min of K50")

    def test_trailing_tokens(self):
        with pytest.raises(TQuerySyntaxError, match="Unexpected token"):
            parse("K50 K51")
