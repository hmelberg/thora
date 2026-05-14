# tquery grammar

The DSL is parsed by a hand-written recursive-descent parser in `tquery/_parser.py`. This document is the normative grammar.

## Lexical structure

Whitespace is insignificant except as a separator between tokens. Tokens are matched longest-first.

| Token | Regex | Examples |
|---|---|---|
| `CODE` | `[A-Za-z_]\w*(?:\.\w+)?` | `K50`, `K50.1`, `glucose`, `4AB02` |
| `STAR_CODE` | `[A-Za-z_]\w*(?:\.\w+)?\*` | `K50*`, `icd*` |
| `RANGE_CODE` | `[A-Za-z]\w*(?:\.\w+)?-[A-Za-z]\w*(?:\.\w+)?` | `K50-K53` |
| `SLICE_CODE` | `[A-Za-z_]\w*:[A-Za-z_]\w*` | `icd1:icd10` (pandas-style column slice) |
| `INT_RANGE` | `\d+-\d+` | `2-5` (count range) |
| `ORDINAL` | `\d+(st\|nd\|rd\|th)\b` | `1st`, `2nd`, `21st` |
| `FLOAT` | `\d+\.\d+` | `8.5` |
| `INT` | `\d+` | `100`, `3` |
| `OP` | `>=\|<=\|!=\|==\|[><+\-]` | `>`, `>=`, `+`, `-` |
| `LPAREN` / `RPAREN` | `(` / `)` | grouping |
| `COMMA` | `,` | code lists, column lists |
| `AT` | `@` | variable reference prefix |
| `PERCENT` | `%` | relative-magnitude suffix on aggregate threshold (v0.2.3) |
| `KEYWORD` | reserved word match | see below |

### Keywords

```
before  after  simultaneously
inside  outside
and  or  not  never
min  max  exactly
of  in  to
first  last
days  event  events
around
every  any  each  always
sum  mean  avg  median  sd  var  count  n     (aggregate functions, v0.2)
range                                          (aggregate function, v0.2.1)
rise  fall                                     (signed range, v0.2.2)
```

**`min` / `max` disambiguation.** Both keywords serve double duty:
- `min INT` / `max INT` — count-prefix quantifiers (existing).
- `min(IDENT)` / `max(IDENT)` — aggregate function calls (new in v0.2).

The parser distinguishes by 1-token lookahead: a `LPAREN` immediately after `min`/`max` selects the aggregate path; anything else selects the prefix path.

A `CODE` token whose lowercase value equals a keyword is re-tagged as a `KEYWORD`. Quoting/escaping codes that collide with keywords is not supported — use parentheses to disambiguate where needed.

### Tokenizer note: `INT_RANGE` vs. `RANGE_CODE`

`2-5` is `INT_RANGE`. `K50-K53` is `RANGE_CODE`. A leading letter or underscore distinguishes them. `2-K50` is illegal.

## Grammar (EBNF)

Precedence is encoded by production nesting: `or` is loosest, the atom level tightest. `inside`/`outside` binds tighter than count/ordinal prefixes — so `min 2 of X inside 50 days after Y` parses as `min 2 of (X inside 50 days after Y)`.

```ebnf
query         = or_expr ;

or_expr       = and_expr , { "or" , and_expr } ;
and_expr      = temporal_expr , { "and" , temporal_expr } ;
temporal_expr = lhs , [ ( "before" | "after" | "simultaneously" ) , rhs ] ;
lhs           = quantified | not_expr ;
rhs           = quantified | not_expr ;
quantified    = ( "every" | "each" | "always" | "any" ) , code_expr , [ within_tail ] ;

not_expr      = [ "not" | "never" ] , prefix_expr ;
prefix_expr   = prefix_core , { shift_suffix } ;
shift_suffix  = ( "+" | "-" ) , INT , "days" ;     (* requires single-date core *)

prefix_core   = count_prefix , within_or_atom
              | range_prefix , within_or_atom
              | ordinal_prefix , within_or_atom
              | first_last_prefix , within_or_atom
              | within_or_atom ;

count_prefix      = ( "min" | "max" | "exactly" ) , INT , [ "of" ] ;
range_prefix      = INT_RANGE , [ "of" ] ;
ordinal_prefix    = [ "-" ] , ORDINAL , [ "of" ] ;     (* "-2nd" = 2nd-from-last *)
first_last_prefix = ( "first" | "last" ) , INT , [ "of" ] ;

within_or_atom    = atom , [ within_tail ] ;
within_tail       = ( "inside" | "outside" ) , inside_body ;
inside_body       = numeric_window | expr_window ;
numeric_window    = signed_int , [ "to" , signed_int ] , ( "days" | "event" | "events" )
                  , [ direction , [ "of" ] , window_ref ] ;
expr_window       = prefix_expr , [ "and" , prefix_expr ] ;   (* span vs. between *)
direction         = "before" | "after" | "around" ;
window_ref        = quantified | prefix_expr ;
signed_int        = [ "-" ] , INT ;

atom              = "(" , query , ")"
                  | event_atom
                  | comparison_atom
                  | aggregate_atom            (* v0.2 *)
                  | code_expr ;

aggregate_atom    = agg_func , "(" , IDENT , ")" , op , number , [ "%" ] ;
                  (* `%` suffix (v0.2.3): only valid with `rise`/`fall`;
                     normalises the threshold to a fraction (10% → 0.10) *)
agg_func          = "sum" | "mean" | "avg" | "min" | "max"
                  | "median" | "sd" | "var" | "count" | "n"
                  | "range"        (* v0.2.1: max - min per person/window *)
                  | "rise" | "fall" ; (* v0.2.2: max drawup / max drawdown *)
op                = ">" | "<" | ">=" | "<=" | "==" | "!=" ;
number            = INT | FLOAT ;
event_atom        = "event" | "events" ;
comparison_atom   = CODE , OP , ( INT | FLOAT ) ;     (* OP ∈ { >, <, >=, <=, ==, != } *)
code_expr         = code_item , { "," , code_item } , [ column_spec ] ;
code_item         = AT , ident                        (* @variable *)
                  | STAR_CODE | RANGE_CODE | CODE ;
column_spec       = "in" , column_item , { "," , column_item } ;
column_item       = CODE | IDENT | STAR_CODE | RANGE_CODE | SLICE_CODE ;
ident             = CODE | IDENT ;
```

## Aggregate atoms and windowing

An `aggregate_atom` is an atom — it slots into the same productions as `code_expr` or `comparison_atom`. That means the existing `within_tail` machinery (`inside`/`outside`) can wrap an aggregate exactly like it wraps a code expression. The same surface syntax (`inside N days [after Y]`) carries TWO distinct semantics depending on what's wrapped:

| Form | Semantics |
|---|---|
| `sum(col) > 300` | Sum `col` across ALL of the person's rows; threshold the result. |
| `sum(col) > 300 inside 90 days` | **Sliding** — does there exist ANY 90-day stretch in the person's timeline where the rolling sum exceeds 300? |
| `sum(col) > 300 inside 90 days after I21` | **Anchored** — sum `col` across the person's rows within 90 days after an I21 event; threshold the result. |
| `sum(col) > 300 inside 30 to 90 days after I21` | Anchored with a range window — sum over rows whose date is 30–90 days after an I21. |
| `sum(col) > 300 outside 90 days after I21` | Row-level complement (rows OUTSIDE the window, restricted to evaluable persons). |
| `range(col) > 30 inside 5 events` | **Sliding event window** (v0.2.1) — for each row `r`, the window is the `5` consecutive rows ending at `r`; person matches if `range(col)` over any such window exceeds 30. |
| `range(col) > 30 inside 5 events after I21` | **Anchored event window** (v0.2.1) — aggregate over the 5 rows immediately following an `I21` event. |

A bare `inside N days` over a NON-aggregate child still means "within N days of the person's first event" (existing semantics). The semantic shift to sliding is specifically when the child is an `AggregateExpr`. The parser produces the same `WithinExpr` AST node in both cases; the evaluator dispatches by child type.

## Parsing constraints (errors thrown by the parser)

These are not pure grammar rules — the parser enforces them after parsing.

1. **Shift suffix requires a single-date anchor.** `± N days` is only valid attached to an `ordinal_prefix` or another `ShiftExpr`. `K50 - 30 days` is a parse error.
2. **Negative offset in `inside` is only valid with `around`.** `inside -5 days after Y` is rejected; use `inside 5 days before Y` instead. `inside -5 to 10 days around Y` is allowed.
3. **Ascending range required.** `2-5` is OK; `5-2` is rejected. Same for `inside M to N` where `M ≤ N`.
4. **`-0th` is rejected.** Ordinals must be non-zero.
5. **`every` / `each` / `always`** only accept a bare code expression (no parentheses, no count/ordinal prefix, no `not`/`never`).
6. **`any` is elided.** `any K50` parses as just `K50`. The keyword exists for readability only.
7. **Event windows require direction and ref.** `inside 5 events` alone is a parse error; `inside 5 events after Y` is OK. (Days windows allow direction and ref to be omitted — then the window is measured from the first event per person.)
8. **`simultaneously`, `before`, `after` outside of a temporal context with a quantifier are rejected.** `every K50` alone is a parse error: a quantifier must be paired with a temporal or within operator.
9. **Token at end of input.** Any non-EOF token after the top-level `or_expr` is a parse error.
10. **Aggregate argument must be an IDENT.** Inside `sum(...)` etc., only a plain identifier (column name) is accepted. Code patterns, wildcards, and expressions are rejected — code-pattern filtering belongs in the surrounding `and`/`or`, not inside the aggregate.
11. **Aggregate comparator is required.** `sum(col)` standalone is a parse error — must be followed by `>` / `<` / `>=` / `<=` / `==` / `!=` and a numeric literal.
12. **`outside` over a sliding aggregate is rejected.** The combination `AGG(col) OP NUM outside N days` (no anchor, child is an `AggregateExpr`) has no sensible semantics and is a parse error in v0.2. `outside N days after Y` (anchored) is allowed and gives row-level complement of the anchored form.
13. **`inside N events` without an anchor is allowed only over an `AggregateExpr`** (v0.2.1). For non-aggregate children, `inside N events` still requires an anchor (`after`/`before`/`around` + ref) — without an anchor it has no defined semantics. The bare form binds an aggregate to a sliding event window.
14. **`%` threshold is only valid for `rise` and `fall`** (v0.2.3). `sum(col) > 300%`, `range(col) > 10%`, etc. are parse errors. For `rise(col) > X%` and `fall(col) > X%` the threshold is normalised to a fraction at parse time (`10%` becomes `0.10` in the AST), and the AggregateExpr's `relative` field is set to `true`.

## Output

The parser produces an `ASTNode` as defined in `ast.md`. Whitespace and case are normalised away: keywords are lowercased; codes are preserved verbatim.
