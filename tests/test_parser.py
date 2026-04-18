"""Tests for the tquery tokenizer and parser."""

import pytest

from tquery._ast import (
    BetweenExpr,
    BinaryLogical,
    CodeAtom,
    ComparisonAtom,
    EventAtom,
    InsideExpr,
    NotExpr,
    PrefixExpr,
    Quantifier,
    ShiftExpr,
    TemporalExpr,
    WithinExpr,
    WithinSpanExpr,
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
        tokens = tokenize("before after and or not inside")
        kw_values = [t.value for t in tokens if t.type == TokenType.KEYWORD]
        assert kw_values == ["before", "after", "and", "or", "not", "inside"]

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
        tokens = tokenize("min 3 of K50* inside 100 days after 1st K51")
        types = [t.type for t in tokens[:-1]]
        assert types == [
            TokenType.KEYWORD,   # min
            TokenType.INT,       # 3
            TokenType.KEYWORD,   # of
            TokenType.STAR_CODE, # K50*
            TokenType.KEYWORD,   # inside
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

    def test_negative_ordinal(self):
        ast = parse("-1st K51")
        assert isinstance(ast, PrefixExpr)
        assert ast.kind == "ordinal"
        assert ast.n == -1
        assert isinstance(ast.child, CodeAtom)

    def test_negative_ordinal_event(self):
        ast = parse("-3rd event")
        assert isinstance(ast, PrefixExpr)
        assert ast.kind == "ordinal"
        assert ast.n == -3
        assert isinstance(ast.child, EventAtom)

    def test_negative_ordinal_of(self):
        ast = parse("-2nd of K50")
        assert isinstance(ast, PrefixExpr)
        assert ast.n == -2

    def test_negative_zero_ordinal_errors(self):
        with pytest.raises(TQuerySyntaxError, match="-0"):
            parse("-0th K51")

    def test_negative_ordinal_as_window_ref(self):
        # Combines the new negative ordinal with the window grammar.
        ast = parse("K50 inside 30 days after -1st K51")
        assert isinstance(ast, WithinExpr)
        assert isinstance(ast.ref, PrefixExpr)
        assert ast.ref.kind == "ordinal"
        assert ast.ref.n == -1

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
# Parser tests — inside/outside windows
# ---------------------------------------------------------------------------

class TestParserInside:
    def test_inside_days(self):
        ast = parse("K50 inside 100 days")
        assert isinstance(ast, WithinExpr)
        assert ast.days == 100
        assert ast.direction is None
        assert ast.ref is None
        assert ast.outside is False
        assert isinstance(ast.child, CodeAtom)

    def test_inside_days_after(self):
        ast = parse("K50 inside 100 days after K51")
        assert isinstance(ast, WithinExpr)
        assert ast.days == 100
        assert ast.direction == "after"
        assert isinstance(ast.ref, CodeAtom)
        assert ast.ref.codes == ("K51",)

    def test_inside_days_before(self):
        ast = parse("K50 inside 30 days before K51")
        assert isinstance(ast, WithinExpr)
        assert ast.direction == "before"

    def test_outside_days(self):
        ast = parse("K50 outside 100 days after K51")
        assert isinstance(ast, WithinExpr)
        assert ast.outside is True
        assert ast.days == 100
        assert ast.direction == "after"

    def test_inside_events(self):
        ast = parse("K50 inside 5 events after K51")
        assert isinstance(ast, InsideExpr)
        assert ast.inside is True
        # Shorthand: +1..+5 events after Y
        assert ast.min_events == 1
        assert ast.max_events == 5
        assert ast.direction == "after"

    def test_outside_events(self):
        ast = parse("K50 outside 10 events before K51")
        assert isinstance(ast, InsideExpr)
        assert ast.inside is False
        assert ast.min_events == 1
        assert ast.max_events == 10
        assert ast.direction == "before"

    def test_inside_events_range(self):
        ast = parse("K50 inside 3 to 5 events after K51")
        assert isinstance(ast, InsideExpr)
        assert ast.min_events == 3
        assert ast.max_events == 5
        assert ast.direction == "after"

    def test_inside_days_range(self):
        ast = parse("K50 inside 5 to 7 days after K51")
        assert isinstance(ast, WithinExpr)
        assert ast.min_days == 5
        assert ast.days == 7
        assert ast.direction == "after"

    def test_inside_signed_around(self):
        ast = parse("K50 inside -5 to 20 days around K51")
        assert isinstance(ast, WithinExpr)
        assert ast.min_days == -5
        assert ast.days == 20
        assert ast.direction == "around"

    def test_negative_with_after_errors(self):
        with pytest.raises(TQuerySyntaxError, match="Negative offsets"):
            parse("K50 inside -5 days after K51")

    def test_descending_range_errors(self):
        with pytest.raises(TQuerySyntaxError, match="ascending"):
            parse("K50 inside 3 to -2 days around K51")


# ---------------------------------------------------------------------------
# Parser tests — event/events atom
# ---------------------------------------------------------------------------

class TestParserEventAtom:
    def test_event_singular(self):
        assert isinstance(parse("event"), EventAtom)

    def test_events_plural(self):
        assert isinstance(parse("events"), EventAtom)

    def test_ordinal_event(self):
        ast = parse("5th event")
        assert isinstance(ast, PrefixExpr)
        assert ast.kind == "ordinal"
        assert ast.n == 5
        assert isinstance(ast.child, EventAtom)

    def test_ordinal_of_events(self):
        ast = parse("3rd of events")
        assert isinstance(ast, PrefixExpr)
        assert ast.kind == "ordinal"
        assert ast.n == 3
        assert isinstance(ast.child, EventAtom)

    def test_last_n_events(self):
        ast = parse("last 5 events")
        assert isinstance(ast, PrefixExpr)
        assert ast.kind == "last"
        assert ast.n == 5
        assert isinstance(ast.child, EventAtom)

    def test_first_n_of_events(self):
        ast = parse("first 3 of events")
        assert isinstance(ast, PrefixExpr)
        assert ast.kind == "first"
        assert ast.n == 3
        assert isinstance(ast.child, EventAtom)

    def test_min_n_event(self):
        ast = parse("min 2 of event")
        assert isinstance(ast, PrefixExpr)
        assert ast.kind == "min"
        assert ast.n == 2
        assert isinstance(ast.child, EventAtom)

    def test_before_nth_event(self):
        ast = parse("K50 before 5th event")
        assert isinstance(ast, TemporalExpr)
        assert ast.op == "before"
        assert isinstance(ast.left, CodeAtom)
        assert isinstance(ast.right, PrefixExpr)
        assert isinstance(ast.right.child, EventAtom)

    def test_after_nth_event(self):
        ast = parse("K50 after 3rd event")
        assert isinstance(ast, TemporalExpr)
        assert ast.op == "after"
        assert isinstance(ast.right, PrefixExpr)
        assert isinstance(ast.right.child, EventAtom)


# ---------------------------------------------------------------------------
# Parser tests — inside EXPR and EXPR (positional bounds)
# ---------------------------------------------------------------------------

class TestParserInsideBounds:
    def test_bounds_ordinals_same_code(self):
        ast = parse("K50 inside 1st K51 and 5th K51")
        assert isinstance(ast, BetweenExpr)
        assert isinstance(ast.child, CodeAtom)
        assert ast.child.codes == ("K50",)
        assert isinstance(ast.bound_start, PrefixExpr)
        assert ast.bound_start.kind == "ordinal"
        assert ast.bound_start.n == 1
        assert isinstance(ast.bound_end, PrefixExpr)
        assert ast.bound_end.n == 5
        assert ast.outside is False

    def test_bounds_different_codes(self):
        ast = parse("K50 inside 1st K51 and 3rd K52")
        assert isinstance(ast, BetweenExpr)
        assert ast.bound_start.child.codes == ("K51",)
        assert ast.bound_end.child.codes == ("K52",)

    def test_bounds_events(self):
        ast = parse("K50 inside 1st event and 10th event")
        assert isinstance(ast, BetweenExpr)
        assert isinstance(ast.bound_start.child, EventAtom)
        assert isinstance(ast.bound_end.child, EventAtom)

    def test_bounds_first_last_of(self):
        ast = parse("K50 inside 1st K51 and last 1 of K51")
        assert isinstance(ast, BetweenExpr)
        assert ast.bound_end.kind == "last"
        assert ast.bound_end.n == 1

    def test_range_days(self):
        # `inside N to M days` replaces the old `between N and M days` form.
        ast = parse("K50 inside 30 to 90 days after K51")
        assert isinstance(ast, WithinExpr)
        assert ast.min_days == 30
        assert ast.days == 90
        assert ast.direction == "after"

    def test_outside_bounds(self):
        ast = parse("K50 outside 1st K51 and 5th K51")
        assert isinstance(ast, BetweenExpr)
        assert ast.outside is True


# ---------------------------------------------------------------------------
# Parser tests — shifted anchors (`± N days`)
# ---------------------------------------------------------------------------

class TestParserShiftedAnchors:
    def test_shift_negative(self):
        ast = parse("1st K51 before 1st K50 - 100 days")
        assert isinstance(ast, TemporalExpr)
        assert ast.op == "before"
        assert isinstance(ast.right, ShiftExpr)
        assert ast.right.offset_days == -100
        assert isinstance(ast.right.child, PrefixExpr)
        assert ast.right.child.kind == "ordinal"
        assert ast.right.child.n == 1

    def test_shift_positive(self):
        ast = parse("1st K51 before 1st K50 + 30 days")
        assert isinstance(ast, TemporalExpr)
        assert isinstance(ast.right, ShiftExpr)
        assert ast.right.offset_days == 30

    def test_shift_parenthesized(self):
        # Parens are optional — equivalent AST
        ast1 = parse("1st K51 before 1st K50 - 100 days")
        ast2 = parse("1st K51 before (1st K50 - 100 days)")
        assert ast1 == ast2

    def test_shift_chain(self):
        ast = parse("1st K51 before 1st K50 - 30 days - 7 days")
        assert isinstance(ast, TemporalExpr)
        outer = ast.right
        assert isinstance(outer, ShiftExpr)
        assert outer.offset_days == -7
        inner = outer.child
        assert isinstance(inner, ShiftExpr)
        assert inner.offset_days == -30

    def test_shift_in_window_ref(self):
        ast = parse("K50 inside 30 days after 1st K51 + 7 days")
        assert isinstance(ast, WithinExpr)
        assert ast.days == 30
        assert ast.direction == "after"
        assert isinstance(ast.ref, ShiftExpr)
        assert ast.ref.offset_days == 7

    def test_shift_in_both_bounds(self):
        ast = parse("K50 inside 1st K51 - 7 days and 5th K51 + 7 days")
        assert isinstance(ast, BetweenExpr)
        assert isinstance(ast.bound_start, ShiftExpr)
        assert ast.bound_start.offset_days == -7
        assert isinstance(ast.bound_end, ShiftExpr)
        assert ast.bound_end.offset_days == 7

    def test_shift_on_negative_ordinal(self):
        ast = parse("K50 before -1st K51 - 30 days")
        assert isinstance(ast, TemporalExpr)
        assert isinstance(ast.right, ShiftExpr)
        assert ast.right.offset_days == -30
        assert ast.right.child.n == -1

    def test_shift_requires_single_date_plain_code(self):
        with pytest.raises(TQuerySyntaxError, match="single-date"):
            parse("K50 - 30 days")

    def test_shift_requires_single_date_first_n(self):
        with pytest.raises(TQuerySyntaxError, match="single-date"):
            parse("first 2 of K51 - 30 days")

    def test_shift_requires_single_date_min_n(self):
        with pytest.raises(TQuerySyntaxError, match="single-date"):
            parse("min 2 of K51 + 30 days")


# ---------------------------------------------------------------------------
# Parser tests — inside EXPR (positional span)
# ---------------------------------------------------------------------------

class TestParserInsideSpan:
    def test_span_last_n_events(self):
        ast = parse("K50 inside last 5 events")
        assert isinstance(ast, WithinSpanExpr)
        assert isinstance(ast.child, CodeAtom)
        assert isinstance(ast.ref, PrefixExpr)
        assert ast.ref.kind == "last"
        assert ast.ref.n == 5
        assert isinstance(ast.ref.child, EventAtom)
        assert ast.outside is False

    def test_span_first_n_of_k51(self):
        ast = parse("K50 inside first 3 of K51")
        assert isinstance(ast, WithinSpanExpr)
        assert ast.ref.kind == "first"
        assert ast.ref.n == 3

    def test_span_last_n_of_code(self):
        ast = parse("K50 inside last 2 of K50")
        assert isinstance(ast, WithinSpanExpr)
        assert ast.ref.kind == "last"

    def test_outside_span(self):
        ast = parse("K50 outside last 5 events")
        assert isinstance(ast, WithinSpanExpr)
        assert ast.outside is True

    def test_numeric_days_still_works(self):
        ast = parse("K50 inside 100 days after K51")
        assert isinstance(ast, WithinExpr)
        assert ast.days == 100
        assert ast.direction == "after"


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
        ast = parse("min 3 of K50* inside 100 days after 1st of K51")
        # With correct precedence: min 3 of (K50* inside 100 days after (1st of K51))
        assert isinstance(ast, PrefixExpr)
        assert ast.kind == "min"
        assert ast.n == 3
        # child is WithinExpr
        assert isinstance(ast.child, WithinExpr)
        assert ast.child.days == 100
        assert ast.child.direction == "after"
        # window's child is CodeAtom(K50*)
        assert isinstance(ast.child.child, CodeAtom)
        assert ast.child.child.codes == ("K50*",)
        # window's ref is PrefixExpr(ordinal, 1, CodeAtom(K51))
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
            parse("K50 inside 100 K51")

    def test_missing_number_after_min(self):
        with pytest.raises(TQuerySyntaxError, match="Expected integer"):
            parse("min of K50")

    def test_trailing_tokens(self):
        with pytest.raises(TQuerySyntaxError, match="Unexpected token"):
            parse("K50 K51")


# ---------------------------------------------------------------------------
# Parser tests — every / any quantifiers
# ---------------------------------------------------------------------------

class TestParserQuantifiers:
    def test_every_on_left(self):
        ast = parse("every K50 after K51")
        assert isinstance(ast, TemporalExpr)
        assert ast.op == "after"
        assert isinstance(ast.left, Quantifier)
        assert ast.left.kind == "every"
        assert isinstance(ast.left.child, CodeAtom)
        assert ast.left.child.codes == ("K50",)
        assert isinstance(ast.right, CodeAtom)

    def test_every_on_right(self):
        ast = parse("K50 after every K51")
        assert isinstance(ast, TemporalExpr)
        assert isinstance(ast.left, CodeAtom)
        assert isinstance(ast.right, Quantifier)
        assert ast.right.kind == "every"

    def test_every_on_both(self):
        ast = parse("every K50 before every K51")
        assert isinstance(ast, TemporalExpr)
        assert ast.op == "before"
        assert isinstance(ast.left, Quantifier)
        assert isinstance(ast.right, Quantifier)

    def test_any_is_elided(self):
        # `any K50 after any K51` is structurally identical to `K50 after K51`
        ast = parse("any K50 after any K51")
        assert isinstance(ast, TemporalExpr)
        assert isinstance(ast.left, CodeAtom)
        assert isinstance(ast.right, CodeAtom)

    def test_every_on_within_ref(self):
        ast = parse("K50 inside 100 days after every K51")
        assert isinstance(ast, WithinExpr)
        assert ast.days == 100
        assert ast.direction == "after"
        assert isinstance(ast.ref, Quantifier)
        assert ast.ref.kind == "every"

    def test_each_is_synonym_for_every(self):
        ast1 = parse("K50 inside 100 days after each K51")
        ast2 = parse("K50 inside 100 days after every K51")
        assert ast1 == ast2

    def test_always_is_synonym_for_every(self):
        ast1 = parse("always K50 inside 100 days after K51")
        ast2 = parse("every K50 inside 100 days after K51")
        assert ast1 == ast2

    def test_never_is_synonym_for_not(self):
        ast1 = parse("never K50 inside 100 days after K51")
        ast2 = parse("not K50 inside 100 days after K51")
        assert ast1 == ast2

    def test_every_simultaneously(self):
        ast = parse("every K50 simultaneously K51")
        assert isinstance(ast, TemporalExpr)
        assert ast.op == "simultaneously"
        assert isinstance(ast.left, Quantifier)

    def test_every_without_temporal_op_errors(self):
        with pytest.raises(TQuerySyntaxError, match="requires a temporal operator"):
            parse("every K50")

    def test_every_with_paren_errors(self):
        with pytest.raises(TQuerySyntaxError, match="parenthesized group"):
            parse("every (K50 or K51) after K52")

    def test_every_with_min_errors(self):
        with pytest.raises(TQuerySyntaxError, match="cannot be combined with 'min'"):
            parse("every min 2 of K50 after K51")

    def test_every_with_not_errors(self):
        with pytest.raises(TQuerySyntaxError, match="cannot be combined with 'not'"):
            parse("every not K50 after K51")

    def test_any_with_paren_errors(self):
        with pytest.raises(TQuerySyntaxError, match="parenthesized group"):
            parse("any (K50 or K51) after K52")

    def test_every_with_variable(self):
        ast = parse("K50 after every @cohort")
        assert isinstance(ast, TemporalExpr)
        assert isinstance(ast.right, Quantifier)
        assert ast.right.child.codes == ("@cohort",)
