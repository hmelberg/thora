# tquery AST

The Abstract Syntax Tree is the contract between parser and evaluator and the unit of cross-language portability. Every port must produce equivalent AST shapes for the same input expression.

The reference is `tquery/_ast.py`. There are 14 node types. All Python reference nodes are frozen, hashable dataclasses (used as cache keys).

## JSON serialisation convention

For cross-language testing, the AST is serialised to JSON as:

```json
{ "_node": "<NodeName>", "<field1>": <value>, "<field2>": <value> }
```

Tuples are serialised as JSON arrays. `null` for absent optional fields. Recursive children are nested objects. Example:

```json
{
  "_node": "TemporalExpr",
  "op": "before",
  "left":  { "_node": "CodeAtom", "codes": ["K50"], "columns": null },
  "right": { "_node": "CodeAtom", "codes": ["K51"], "columns": null }
}
```

## Node catalogue

### `CodeAtom`

A reference to one or more medical codes, optionally restricted to specific columns.

| Field | Type | Notes |
|---|---|---|
| `codes` | `tuple[str, ...]` | One or more code patterns. May contain plain codes (`K50`), wildcards (`K50*`), ranges (`K50-K53`), or variable refs (`@antibiotics`). |
| `columns` | `tuple[str, ...] \| null` | If present, restricts matching to these columns. May contain wildcards / ranges / slices. |

**Invariant.** `codes` is non-empty.

---

### `EventAtom`

Matches every row in the timeline. Spelled `event` or `events`. No fields.

Purpose: lets positional selectors (`5th event`, `last 5 events`) and windows (`inside 5 events after Y`) target the timeline itself rather than a code pattern.

---

### `ComparisonAtom`

A column comparison like `glucose > 8`.

| Field | Type | Notes |
|---|---|---|
| `column` | `str` | Column name. |
| `op` | `str` | One of `">"`, `"<"`, `">="`, `"<="`, `"=="`, `"!="`. |
| `value` | `float` | Comparison value (integers stored as floats). |

---

### `AggregateExpr` (v0.2)

A person-level aggregate over a numeric column, compared against a threshold. Produces a person-level boolean broadcast to row level (every row of a matching person becomes True).

| Field | Type | Notes |
|---|---|---|
| `func` | `str` | One of `"sum"`, `"mean"`, `"avg"`, `"min"`, `"max"`, `"median"`, `"sd"`, `"var"`, `"count"`, `"n"`, `"range"`, `"rise"`, `"fall"`. `"avg"` is a synonym for `"mean"`; `"n"` is a synonym for `"count"`. `"range"` = `max(col) - min(col)` (v0.2.1). `"rise"` = max drawup; `"fall"` = max drawdown (v0.2.2). |
| `column` | `str` | Numeric column name. The column must exist in the input data and be numeric. |
| `op` | `str` | Comparison operator: `">"`, `"<"`, `">="`, `"<="`, `"=="`, `"!="`. |
| `value` | `float` | Threshold value. For `relative=true` (v0.2.3), the parser stores the fraction (10% → 0.10), so the evaluator compares the aggregate ratio directly to this field. |
| `relative` | `bool` | When `true` (v0.2.3), the aggregate is computed as a relative magnitude: `(v[j] - v[i]) / v[i]` for `rise`, `(v[i] - v[j]) / v[i]` for `fall`. Only `rise` and `fall` accept `relative=true`. Default `false`. |

**Invariants.**
- `func` is one of the listed values; `"min"` / `"max"` are *aggregate* operations here, not the existing count-prefix `min` / `max`. They are distinguished syntactically (function-call form).
- `column` is a plain identifier — no patterns, wildcards, or expressions.
- An `AggregateExpr` may appear as the child of a `WithinExpr`. When it does, the wrapper's semantics shift between "sliding" and "anchored" depending on whether the wrapper carries a direction + ref. See `semantics.md`.

---

### `PrefixExpr`

A quantifier prefix on a child expression.

| Field | Type | Notes |
|---|---|---|
| `kind` | `str` | `"min"`, `"max"`, `"exactly"` (count predicates); `"ordinal"` (positional 1st/2nd/…); `"first"`, `"last"` (first/last N occurrences). |
| `n` | `int` | The numeric argument. For `"ordinal"`, negative values mean from-the-end (`-1` = last). For all other kinds, `n` is positive. |
| `child` | `ASTNode` | The expression being quantified. |

**Invariants.**
- For `kind != "ordinal"`, `n ≥ 1`.
- For `kind == "ordinal"`, `n ≠ 0`.

---

### `RangePrefixExpr`

Count range: persons with between `min_n` and `max_n` matching events (inclusive).

| Field | Type | Notes |
|---|---|---|
| `min_n` | `int` | Lower bound. |
| `max_n` | `int` | Upper bound. |
| `child` | `ASTNode` |  |

**Invariant.** `min_n ≤ max_n` and both `≥ 0`.

---

### `NotExpr`

Logical negation.

| Field | Type | Notes |
|---|---|---|
| `child` | `ASTNode` |  |

The DSL keyword `never` is parsed as `not` (`never X` ⇒ `NotExpr(X)`).

---

### `BinaryLogical`

Logical AND/OR. Person-level semantics (not row-level): `A and B` means the person matches both A and B.

| Field | Type | Notes |
|---|---|---|
| `op` | `str` | `"and"` or `"or"`. |
| `left` | `ASTNode` |  |
| `right` | `ASTNode` |  |

---

### `TemporalExpr`

Temporal ordering between two expressions.

| Field | Type | Notes |
|---|---|---|
| `op` | `str` | `"before"`, `"after"`, or `"simultaneously"`. |
| `left` | `ASTNode` |  |
| `right` | `ASTNode` |  |

`left` and `right` may each be wrapped in a `Quantifier` for universal semantics.

---

### `WithinExpr`

Time-window constraint: child events within a day range of reference events.

| Field | Type | Notes |
|---|---|---|
| `child` | `ASTNode` |  |
| `days` | `int` | Upper bound (inclusive), in days. |
| `min_days` | `int` | Lower bound (inclusive), in days. Default `0`. May be negative only when `direction == "around"`. |
| `direction` | `str \| null` | `"before"`, `"after"`, `"around"`, or `null` (no direction). Interpretation of `null` depends on `child` type — see `semantics.md`: for non-aggregate children, `null` means "anchored to first event per person"; for `AggregateExpr` children (v0.2), `null` means **sliding** rolling-window aggregate. |
| `ref` | `ASTNode \| null` | Reference expression. `null` together with `direction = null` selects the first-event-anchored / sliding interpretation per the rule above. |
| `outside` | `bool` | If true, row-level complement restricted to evaluable persons. Default `false`. |

**Invariants.**
- `min_days ≤ days`.
- `min_days < 0` only when `direction == "around"`.
- `ref == null` only when `direction == null`.

---

### `InsideExpr`

Event-window constraint: child events at offset positions from each reference row.

| Field | Type | Notes |
|---|---|---|
| `child` | `ASTNode` |  |
| `inside` | `bool` | `true` = inside window, `false` = outside (row-level complement). |
| `min_events` | `int` | Lower offset bound. |
| `max_events` | `int` | Upper offset bound. |
| `direction` | `str \| null` | `"before"`, `"after"`, `"around"`, or `null`. `null` is only allowed when `child` is an `AggregateExpr` — see below. |
| `ref` | `ASTNode \| null` | Reference expression. `null` only with `direction = null` for the sliding-aggregate form. |

**v0.2.1 — sliding aggregate path.** When `child` is an `AggregateExpr` and both `direction` and `ref` are `null`, the node represents a SLIDING right-anchored event window: for each row `r`, the window is the `max_events` consecutive rows ending at `r` (`min_events` is ignored in this mode). The person matches if the aggregate satisfies its predicate for at least one window. See `semantics.md`. For non-aggregate children this configuration is a parse error.

The window is `[min_events, max_events]` in row-position offsets from each ref row (positive = after, negative = before in `around`; for `"before"` direction, offsets are mirrored).

Shorthand `inside N events after Y` ⇒ `min_events=1, max_events=N`.

---

### `BetweenExpr`

Positional-bounds window: rows whose date falls in `[min(bound_start dates), max(bound_end dates)]` per person.

Spelled as `CHILD inside EXPR_A and EXPR_B`, e.g., `K50 inside 1st K51 and 5th K51`.

| Field | Type | Notes |
|---|---|---|
| `child` | `ASTNode` |  |
| `bound_start` | `ASTNode` |  |
| `bound_end` | `ASTNode` |  |
| `outside` | `bool` | Row-level complement, restricted to evaluable persons. Default `false`. |

---

### `WithinSpanExpr`

Positional-span window: child rows whose date falls in `[min(ref dates), max(ref dates)]` per person — the date range covered by a single multi-row selector.

Spelled as `CHILD inside EXPR`, e.g., `K50 inside last 5 events`.

| Field | Type | Notes |
|---|---|---|
| `child` | `ASTNode` |  |
| `ref` | `ASTNode` |  |
| `outside` | `bool` | Row-level complement, restricted to evaluable persons. Default `false`. |

---

### `ShiftExpr`

Per-person synthetic date: child's date shifted by an integer number of days.

| Field | Type | Notes |
|---|---|---|
| `child` | `ASTNode` | Must be a single-date expression: an `"ordinal"` `PrefixExpr` or another `ShiftExpr`. |
| `offset_days` | `int` | Signed; negative = earlier, positive = later. |

**Invariant.** Only meaningful as a reference in temporal / within / between positions. Standalone evaluation should raise.

---

### `Quantifier`

Universal or existential quantifier over a code atom in a temporal context.

| Field | Type | Notes |
|---|---|---|
| `kind` | `str` | `"any"` (existential, no-op) or `"every"` (universal). |
| `child` | `ASTNode` | Always a `CodeAtom`. Enforced at parse time. |

**Parser normalisation.** `kind="any"` is elided — `any K50` produces `CodeAtom(("K50",))` directly, not a `Quantifier`. Therefore a `Quantifier` node in the AST always has `kind="every"`. The keyword variants `each` and `always` also map to `kind="every"`.

---

## Union type

```python
ASTNode = (
    CodeAtom | EventAtom | ComparisonAtom | AggregateExpr
    | PrefixExpr | RangePrefixExpr
    | NotExpr | BinaryLogical
    | TemporalExpr | WithinExpr | WithinSpanExpr | InsideExpr | BetweenExpr
    | ShiftExpr | Quantifier
)
```

## Cross-language invariants

These must hold in every port:

1. **Equivalent parser.** For any query string accepted by Python `tquery.parse`, the port's parser produces the same AST shape (up to JSON-serialisation equality).
2. **Equivalent rejection.** For any query rejected by Python's parser, the port also rejects it.
3. **No new node types.** Ports may not invent additional AST node kinds. Backend-specific optimisations are below the AST.
4. **No field aliasing.** Field names and value sets are exactly as listed above.
