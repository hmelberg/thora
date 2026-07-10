# tquery semantics

This document is normative. It specifies what each AST node *computes*, including edge cases and tie-breaking rules. The Python reference in `tquery/_evaluator.py`, `tquery/_temporal.py`, and `tquery/_prefix.py` is the source of truth — this document is a faithful description of that behaviour.

## Universal preconditions

The evaluator operates on a tabular dataset with at least three logical columns:

- **`pid`** — person identifier. Any type with a notion of equality (integer, string, factor).
- **`date`** — event date. Day-precision (sub-day precision is silently truncated).
- One or more **code columns** — string columns holding medical codes. Multiple cells per row are supported via a separator (`","` by default).

The dataset MUST be sorted by `(pid, date)` for correct evaluation. The Python reference does not re-sort; ports may assume the same or enforce sorting at the boundary.

## Mask semantics

The evaluator produces two kinds of masks per AST node:

- **Row-level mask.** A boolean array of length `nrow(table)`. `True` rows contribute to the result.
- **Person-level mask.** Broadcast back to row-level by setting every row of a matching person to `True` (the convention used throughout `tquery/_temporal.py`).

Most nodes naturally produce a row-level mask. Person-level results (counts, pids) are derived from the final row-level mask by collecting unique `pid` values where the mask is `True`.

The public API exposes both via `TQueryResult` (`tquery/_types.py`).

## Per-node semantics

### `CodeAtom`

Matches each row whose value (in each searched column) equals one of `codes` after pattern expansion.

**Pattern expansion** (`tquery/_codes.py`):
- Plain code `K50` matches cells exactly equal to `K50`.
- Wildcard `K50*` matches any cell starting with `K50`.
- Range `K50-K53` matches `K50`, `K51`, `K52`, `K53` lexicographically (codes in the codebook with prefix in the range).
- Variable `@name` is replaced by the values from the active `variables` dict before matching.

**Columns**:
- If `columns is None`, the active config's `cols` (or all object columns) are searched.
- If `columns` is given, exact column names, wildcards (`icd*`), ranges (`icd1-icd10`), and slices (`icd1:icd10`) are all expanded against the dataset's columns.

**Multi-code cells.** If the config specifies a `sep`, cell values are split by `sep` and any part matching a code counts as a hit.

Returns a row-level mask.

---

### `EventAtom`

Matches every row. Equivalent to a constant-`True` row mask.

---

### `ComparisonAtom`

`column op value`, evaluated row-wise. Rows where `column` is `NaN`/missing produce `False` (standard pandas comparison semantics — ports must preserve this).

---

### `AggregateExpr` (v0.2)

A person-level aggregate over a numeric column, compared to a threshold.

**Algorithm (standalone, no window).**

1. Compute `agg = f(column for the person's rows, na.rm=TRUE)` per person, where `f` is the chosen function (`sum`, `mean`, etc.).
2. Test `agg op value` per person.
3. Return the row-level mask `pid ∈ {p : test(p) is TRUE}` (every row of a matching person is True).

**Aggregate functions.**

| `func` | Definition over the per-person value vector `v` | Empty `v` (no non-NA values) |
|---|---|---|
| `sum` | `Σ v` | `0` |
| `mean` / `avg` | `Σ v / |v|` | `NA` |
| `min` | `min(v)` | `NA` |
| `max` | `max(v)` | `NA` |
| `median` | sample median | `NA` |
| `sd` | sample stdev (n−1 denominator) | `NA` (also if `|v| == 1`) |
| `var` | sample variance | `NA` (also if `|v| == 1`) |
| `count` / `n` | `|v|` — count of non-NA values | `0` |
| `range` (v0.2.1) | `max(v) − min(v)` | `NA` for empty; `0` for single value |
| `rise` (v0.2.2) | Max drawup: `max over i≤j of (v[j] − v[i])` | `NA` for empty; `0` for single value |
| `fall` (v0.2.2) | Max drawdown: `max over i≤j of (v[i] − v[j])` (returned as a non-negative magnitude) | `NA` for empty; `0` for single value |
| `rise`, `relative=true` (v0.2.3) | Max relative drawup: `max over i≤j (and v[i]>0) of (v[j] − v[i]) / v[i]` | `NA` for empty; `0` for single value or if no v[i]>0 pair exists |
| `fall`, `relative=true` (v0.2.3) | Max relative drawdown magnitude: `max over i≤j (and v[i]>0) of (v[i] − v[j]) / v[i]` | `NA` for empty; `0` for single value or if no v[i]>0 pair exists |
| `range`, `relative=true` (v0.2.4) | Relative spread: `(max(v) − min(v)) / min(v)`, only when `min(v) > 0` | `NA` for empty; `0` for single value with `v > 0`; `NA` if `min ≤ 0` |

**Relative-threshold convention** (v0.2.3). The `%` suffix in the DSL is normalised to a fraction at parse time, so `rise(col) > 10%` stores `value=0.10, relative=true` in the AST. The aggregate produces a unitless ratio in `[0, ∞)`; the comparison is against `value` directly. Pairs where the denominator `v[i]` is ≤ 0 are skipped — they have no well-defined relative change. This matches the standard finance/pharmacoepi convention for "max gain / drawdown percentage".

**Comparison vs NA.** Per existing `ComparisonAtom` semantics, `NA op value` is `FALSE`. So an empty-set `mean`, `sd`, etc. on a person never matches. `sum` and `count` always have a definite answer (`0`), and threshold comparisons against `0` behave normally.

**NA in the target column.** Always skipped (`na.rm = TRUE` / `skipna = True`). The library does not provide an "include NA" mode in v0.2.

---

### `AggregateExpr` wrapped in `WithinExpr` — anchored

Form: `sum(col) > 300 inside N days [direction] Y`.

The wrapper's `child` is an `AggregateExpr` and the wrapper has a `direction` ∈ {`"before"`, `"after"`, `"around"`} with a `ref` set. Semantics:

1. Compute the row mask `M = WithinExpr_rows(non-aggregate semantics)` — the rows whose date is within `[min_days, days]` of a qualifying ref event (existing `eval_within_days` logic), with **self-exclusion disabled**: an anchor row lies inside its own window, so `sum(dose) inside 90 days after B01` includes the index B01 row's own dose.
2. For each **evaluable** person (≥ 1 ref event), aggregate the target column over rows ∈ `M`: `agg = f(col[M ∧ pid==p])`. Persons whose window is EMPTY still get an aggregate, using the empty-set defaults from the aggregate table — `sum`/`count` → `0` (which compares normally, so `count(dose) == 0 inside ... after Y` selects persons with *no* qualifying rows), everything else → `NA` (fails the comparison). Non-evaluable persons never match.
3. Apply the threshold test per person; broadcast back to rows.
4. `outside=True` flips to row-level complement, restricted to evaluable persons; empty outside-sets get the same defaults.

Range windows (`inside 30 to 90 days after Y`) work the same way — the row mask just bounds `min_days` and `days` differently.

#### Universal ref (v0.2.5): `sum(col) > 300 inside N days after every Y`

When the ref is quantified with `every`, the aggregate is evaluated **per reference event**, not over the union of windows:

1. For EACH ref row `r`, aggregate the column over the rows of the same person falling in `r`'s own day-window (same band arithmetic as above; the anchor row is inside its own window when the band contains offset 0).
2. Test the threshold per ref. Empty windows use the standard aggregate defaults — `sum`/`count` evaluate to `0` and compare normally; every other aggregate is `NA` and fails.
3. The person matches iff they have **at least one ref** (no vacuous truth) and **every** ref's window passes. All rows of matching persons are returned.

Reading: "after every MI, the cumulative dose in the following 90 days exceeded 300."

`outside ... every REF` is rejected (no defined semantics). Universal refs for **event-window** aggregates (`inside N events after every Y`) are rejected in v0.2.5 — use a day-window instead.

For multi-band windows (`around` with a positive lower bound), the two band slices are concatenated in chronological order before aggregating — this matters for the order-sensitive aggregates (`rise`, `fall`).

---

### `AggregateExpr` wrapped in `WithinExpr` — sliding

Form: `sum(col) > 300 inside N days` (no `direction`, no `ref`).

This is the analytic interpretation: "is there ANY N-day stretch in this person's timeline where the rolling aggregate satisfies the predicate?"

**Algorithm.**

1. For each person, sort rows by date (the input must already be sorted).
2. Compute, for each row `r` at date `d_r`, the rolling aggregate over rows in `[d_r - N, d_r]` (inclusive both ends, in days):
   - `sum`: rolling sum of `col` over the time window.
   - `mean`: rolling mean.
   - `count`: rolling count of non-NA.
   - `min` / `max` / `median` / `sd` / `var`: analogous rolling versions.
3. The PERSON matches if `∃ r : test(rolling_agg(r))` — i.e., if the predicate holds for at least one row's window.
4. Return the row-level mask for all rows of matching persons.

**Inclusivity** of the rolling window is `|d_window - d_r| ≤ N`. Both endpoints are inclusive — matches the existing `WithinExpr` semantics on `merge_asof` tolerance. Each row `r` is included in its own window (the window for row `r` always contains `r`).

**Anchoring choice.** The rolling window is RIGHT-ANCHORED — for each row `r`, the window is `[d_r - N, d_r]`. This is the standard pharmacoepidemiology convention ("DDD accumulated in the trailing N days"). A future v0.3 could add `inside N days centered` for centered windows; not in v0.2.

**Ties on the date column.** Rows sharing the same `(pid, date)` are all inside the same windows. No row-order dependency.

**Empty windows.** If a person has only one row, the window for that row contains just that row. Aggregates apply normally (sum of one value, mean of one value, sd is NA for one value, etc.).

**`outside=True` over a sliding aggregate.** Reserved — parse error in v0.2. (No sensible semantics for "sliding outside".)

**Performance note (non-normative).** Implementations should use a date-range non-equi self-join (pandas `rolling("Nd", on=date).agg(...)`, Polars `rolling_*_by`, data.table date-range self-join, DuckDB `RANGE BETWEEN INTERVAL 'N days' PRECEDING AND CURRENT ROW`). A naive O(rows²) scan is allowed but should be benchmarked at registry scale (≥ 50k persons) before shipping.

---

### `AggregateExpr` wrapped in `InsideExpr` — sliding event window (v0.2.1)

Form: `range(col) > 30 inside N events` (no `direction`, no `ref`).

For each row `r` at event-position `p_r` (0-indexed within the person), the window is the `N` consecutive rows ending at `r`, i.e. positions `[max(0, p_r − (N − 1)), p_r]`. The aggregate is computed over the values of `column` in these rows. The person matches if there exists at least one row whose window's aggregate satisfies the predicate.

`min_events` is ignored in this mode; `max_events` IS the window size `N`.

The `outside` flag is rejected (parse error) — same rule as `outside` over a sliding day-window aggregate.

### `AggregateExpr` wrapped in `InsideExpr` — anchored event window (v0.2.1)

Form: `range(col) > 30 inside N events after Y`.

Identical to the existing `InsideExpr` row-mask semantics (`eval_inside_outside`), but instead of returning the mask directly, the aggregate is computed over the masked rows per person and thresholded. Anchored event windows for aggregates follow the existing shorthand: `inside N events after Y` means `min_events = 1, max_events = N`.

`outside` is allowed in the anchored case; semantics are "aggregate over rows OUTSIDE the event window for evaluable persons" — analogous to `outside` over an anchored day-window aggregate.

---

### `PrefixExpr`

Operates on the row-level mask of its child, restricted within each person.

| `kind` | Semantics |
|---|---|
| `"min"` | Persons with `≥ n` matching rows; all matching rows for those persons are kept. |
| `"max"` | Persons with `≤ n` matching rows **and ≥ 1** (`max 2 of K50` reads "has K50, at most 2"); all matching rows kept. For the zero-inclusive reading, write the 0: `0-2 of K50`. |
| `"exactly"` | Persons with exactly `n` matching rows; all matching rows for those persons are kept. |

**Explicit-0 rule.** Persons with ZERO matching rows are included exactly when the expression contains a literal `0`: `exactly 0 of X` ≡ `max 0 of X` ≡ `not X`; `min 0 of X` is the tautology (every person); `0-N of X` (RangePrefixExpr) means "at most N, including none". Zero-count persons have no matching rows to mark, so they are marked on their **full timeline** — the same broadcast convention as `NotExpr` and the aggregates. Consequences (documented, same caveats as `not`): in temporal composition their dates are their whole history, `event_counts` reflects marked rows rather than code counts, and `filter()` returns all their rows. With `n ≥ 1` nothing changes: the classic behavior requires at least one match and marks only matching rows.
| `"ordinal"`, `n > 0` | Keep the `n`th matching row per person, counting from the start (row order = (pid, date) order). |
| `"ordinal"`, `n < 0` | Keep the `|n|`th matching row per person, counting from the end. `n = -1` is the last match. |
| `"first"` | Keep the first `n` matching rows per person. |
| `"last"` | Keep the last `n` matching rows per person. |

**Tie-breaking.** When multiple rows for the same `(pid, date)` exist, ordinality follows insertion order in the input table. This is why the input must be sorted by `(pid, date)` — secondary tie-breaking on row index is not specified beyond stability.

---

### `RangePrefixExpr`

Persons with between `min_n` and `max_n` matching rows (inclusive); all matching rows for those persons are kept. With `min_n = 0` (an explicit `0-N of X`), zero-count persons are included too, marked on their full timeline (see the explicit-0 rule under `PrefixExpr`).

---

### `NotExpr`

Person-level negation. A person matches `not X` if they do NOT match `X` anywhere in their timeline. Persons who are absent from the dataset entirely also satisfy `not X` (vacuously).

**Returned rows.** ALL rows of persons matching `not X` (not just the rows where the child mask is False). This is broadcast row-level from a person-level membership decision. See `tquery/_evaluator.py:_eval_not`.

---

### `BinaryLogical`

Person-level semantics.

- `A and B` — person matches iff they match A AND match B somewhere in their timeline. Returned row-level mask: `(mask_A | mask_B)` restricted to matching persons.
- `A or B` — person matches iff they match A OR match B somewhere in their timeline. Returned row-level mask: `(mask_A | mask_B)` restricted to matching persons.

(See `tquery/_evaluator.py` for the precise implementation.)

---

### `TemporalExpr`

#### Default: "first vs first"

- `A before B` ⇒ `min(date_A) < min(date_B)` per person. "The first A precedes the first B."
- `A after B` ⇒ `min(date_A) > min(date_B)` per person. "The first A follows the first B."
- `A simultaneously B` ⇒ `∃ date d : d ∈ dates_A(p) ∧ d ∈ dates_B(p)` per person.

Note this default is NOT existential: a person with K51 (2010), K50 (2012), K51 (2015) does **not** match `K50 before K51`, even though *a* K50 precedes *a* K51 — the first K50 (2012) is after the first K51 (2010). The default reads each side as its **first** event, which is the standard incidence framing ("disease onset before treatment start"). For the existential reading, use `any` (below).

#### Existential modifier: `any`

Either side may be quantified with `any`, meaning "some event on this side":

| Form | Meaning | Predicate |
|---|---|---|
| `any A before any B` | some A before some B | `min(A) < max(B)` |
| `A before any B` | first A before some B | `min(A) < max(B)` (= any/any) |
| `any A before B` | some A before first B | `min(A) < min(B)` (= default; documented no-op) |
| `any A after any B` | some A after some B | `max(A) > min(B)` |
| `any A after B` | some A after first B | `max(A) > min(B)` (= any/any) |
| `A after any B` | first A after some B | `min(A) > min(B)` (= default; documented no-op) |
| `every A before any B` | every A before some B | `max(A) < max(B)` (= `every A before B`) |
| `any A before every B` | some A before every B | `min(A) < min(B)` (= default) |

`any` requires both sides non-empty (the person must have events on each side — automatic, since an existential over an empty set is false). For `simultaneously` `any` is a no-op (already existential). In WINDOW contexts (`inside ... after any Y`, `any X inside ...`) and standalone (`any X`), `any` is elided by the parser as a no-op — windows are already existential over their events — so `Quantifier(kind="any")` only ever appears as a direct child of `TemporalExpr`.

**Result.** All rows of `A` for matching persons (left-anchored convention).

#### Universal modifiers

Either side may be wrapped in `Quantifier(kind="every", child=CodeAtom)`:

| Form | Meaning | Predicate |
|---|---|---|
| `A before every B` | every B has some A before it | `min(A) < min(B)` (= default) |
| `every A before B` | every A precedes some B | `max(A) < max(B)` |
| `every A before every B` | every A strictly before every B | `max(A) < min(B)` |
| `A after every B` | every B has some A after it | `max(A) > max(B)` |
| `every A after B` | every A follows some B | `min(A) > min(B)` (= default) |
| `every A after every B` | every A after every B | `min(A) > max(B)` |
| `A simultaneously every B` | every B has same-date A | `dates(B) ⊆ dates(A)` |
| `every A simultaneously B` | every A has same-date B | `dates(A) ⊆ dates(B)` |

**Vacuous-truth rule.** Universal modes require the quantified side's atom to be non-empty for the person. `every K51` over a person with no K51 events does NOT satisfy the predicate. (See `tquery/_temporal.py:283`.)

#### Date shifts

When the right or left side is a `ShiftExpr`, the shifted side's dates are adjusted before the comparison. `1st K51 before 1st K50 - 100 days` evaluates as: shift each person's first K50 backward by 100 days, then test first K51 < shifted first K50.

---

### `WithinExpr`

Time-window predicate. The full algorithm in `tquery/_temporal.py:eval_within_days` (band construction in `window_bands`, existence test in `band_window_match`).

**Existential (default).** The window spec is translated into one or two closed day-offset *bands*; a child row matches iff **any** reference row of the same person falls in a band. All reference rows are considered — implementations MUST NOT test only the nearest ref and then apply the lower bound (an asof/roll-nearest join followed by a `min_days` filter produces false negatives whenever the nearest ref is closer than `min_days` but a farther qualifying ref exists).

**Direction → band(s).** A ref at day `t` qualifies for a child at day `d` iff `t ∈ [d + a, d + b]`:

| `direction` | band(s) `(a, b)` | Meaning |
|---|---|---|
| `"after"` | `(−days, −min_days)` | child happens `min_days..days` after ref |
| `"before"` | `(min_days, days)` | child happens `min_days..days` before ref |
| `"around"` (unsigned, `min_days = 0`) | `(−days, days)` | child within `days` of ref, either side |
| `"around"` (unsigned, `min_days > 0`) | `(−days, −min_days)` and `(min_days, days)` | `min_days ≤ |diff| ≤ days`, either side |
| `"around"` (signed, `min_days < 0`) | `(−days, −min_days)` | signed diff `child − ref` in `[min_days, days]` (covers wholly-negative windows) |
| `null` | (no ref) | distance from each person's first event |

**Inclusivity.** Both band ends inclusive. Implementations with exclusive bounds in their join primitives MUST adjust accordingly.

**Self-exclusion.** A row never serves as its own reference. This only matters when a row matches both the child and the ref pattern (`X inside 5 days after X`, `K50* inside 5 days after K50`): the row itself does not satisfy its own window, but a *different* row on the same date does. Consequently `X inside 0 to 5 days after X` reads "the person got another X within 0–5 days of an X" — a lone X does not match; two same-date X rows do. The first-event form (`ref = null`) is anchored on a per-person *date*, not a row, so no self-exclusion applies there. Anchored aggregates also opt out (see the `AggregateExpr` anchored section).

**`outside=True`.** Row-level complement of the positive form, restricted to **evaluable persons**. A person is evaluable if they have at least one row matching the child AND (when `ref` is given) at least one row matching the ref. Non-evaluable persons contribute no `outside` matches.

**Universal modifiers.** Same logic as `TemporalExpr`. `every_left=True` ⇒ every child row in the person must satisfy the window; `every_right=True` ⇒ every ref row in the person must have a qualifying child within window.

**Ref = `null`.** When `ref` is null, the window is measured against each person's first event date.

---

### `InsideExpr`

Event-position window. The algorithm in `tquery/_temporal.py:eval_inside_outside`.

Event positions are assigned by `(pid, date)` order, 0-indexed within each person.

For each reference row at position `p`:

| `direction` | Window of positions matched |
|---|---|
| `"after"` | `[p + min_events, p + max_events]` |
| `"before"` | `[p - max_events, p - min_events]` |
| `"around"` | `[p + min_events, p + max_events]` (signed offsets) |

Child rows whose position falls in any such window are marked. `inside=False` (i.e., `outside`) yields the complement, restricted to evaluable persons.

**Parser shorthands vs literal ranges.** The bare forms get shorthands: `inside N events after Y` ⇒ `min_events=1, max_events=N` (the anchor is excluded), and `inside N events around Y` ⇒ the symmetric window `min_events=−N, max_events=N` (mirroring `inside N days around Y`). Explicit ranges (`M to N`) are ALWAYS taken literally — `inside 0 to 5 events after Y` stores `min_events=0`.

**Self-exclusion.** As with day windows, a row never matches a window anchored at itself. Positions are unique per row, so offset 0 IS the anchor row — implementations exclude `position == ref_position` from each ref's window. Without this, any window containing offset 0 (`around` ranges spanning zero, explicit `0 to N`) would trivially match self-referential patterns (`X inside -3 to 5 events around X`). Consequently, for predicates `inside 0 to N events after Y` is observably identical to `inside 1 to N events after Y`; the literal 0 only matters for anchored aggregates, which opt out of self-exclusion (the anchor row's own value is part of its window — same convention as day-window aggregates).

**Position ambiguity at equal dates.** When multiple rows share the same `(pid, date)`, their relative position depends on input row order. Specify the input order to lock determinism.

---

### `BetweenExpr` / `WithinSpanExpr`

Surface forms for `BetweenExpr`: `X inside A to B` (and `X outside A to B` for the complement) or the sugar `X between A and B`. The historical `X inside A and B` is a guided parse error — `and` after a span is reserved for logical conjunction.

Both compute a per-person date range from the reference expression(s):

- `WithinSpanExpr`: range = `[min(date(ref)), max(date(ref))]`.
- `BetweenExpr`: range = `[min(date(bound_start)), max(date(bound_end))]`.

Child rows whose date is within the range (inclusive) are marked. `outside=True` gives the row-level complement restricted to evaluable persons.

---

### `ShiftExpr`

Not directly evaluable as a top-level expression — must appear in a reference position. When evaluated in a reference position, the child's per-person date is shifted by `offset_days` (signed) before comparison or window arithmetic.

---

### `Quantifier`

A wrapper consumed by `TemporalExpr` and the within-evaluators. Existential (`kind="any"`) wrappers are elided by the parser; the only `Quantifier` value in the AST has `kind="every"`. See `TemporalExpr` / `WithinExpr` for how `every` modifies their semantics.

---

## Result derivation

Given a final row-level mask `M` aligned to the input table:

- `count` = number of unique `pid` values in rows where `M = True`.
- `pids` = set of those unique `pid` values.
- `rows` = `M` itself, as a row-aligned boolean array.
- `event_counts` = per-person counts: for each pid, number of rows where `M = True`.
- `pct` = `count / evaluable_count` (when restricted to evaluable persons) or `count / total_persons` (marginal).
- `filter(level="rows")` = the subset of input rows where `M = True`.
- `filter(level="persons")` = all input rows for persons in `pids`.

## Edge cases for `AggregateExpr` (cross-port test priority)

1. **Empty cohort for the column**: person has the column all NA — `sum` returns 0; `mean`/`sd`/etc. return NA → comparison False. The same defaults apply to a person whose anchored window contains NO rows at all (the person must still have ≥ 1 ref event to be evaluable). SQL ports beware: `SUM` over an empty or all-NULL group is NULL in SQL — coalesce to 0.
2. **Single value**: `sd`/`var` on a single value are NA → comparison False. `mean` returns the value. `median` returns the value.
3. **Boundary inclusivity (sliding)**: rolling window for date `d_r` is `[d_r - N, d_r]` INCLUSIVE both ends. A row exactly N days before `d_r` is in the window.
4. **Boundary inclusivity (anchored)**: same as existing `WithinExpr` — `min_days ≤ |delta| ≤ days`.
5. **All-NA window (sliding)**: rolling sum is 0 across all-NA windows; rolling mean is NA; comparison resolves False.
6. **Wide vs long format**: the library is column-agnostic at the aggregate layer. Users may pre-process their data to populate per-category numeric columns (e.g., `painkillers` populated only for painkiller-dispensing rows, NA elsewhere). The aggregator skips NA values, so this works without further configuration.
7. **Composition with `and` / `or`**: `K50 and sum(cost) > 1000` — person-level intersection of "has K50" and "total cost > 1000". The aggregate's broadcast-to-row mask combines via existing `BinaryLogical` semantics.
8. **Composition with `not`**: `not (sum(cost) > 1000)` — persons whose total cost is NOT above 1000, including persons absent from the dataset (per existing `NotExpr` semantics).

## Edge cases (cross-port test priority)

1. **Empty input.** All queries return an empty result.
2. **One person, no matches.** Returns `count == 0`, empty `pids`.
3. **Equal dates within a person.** Ordinality is determined by input row order; document and test.
4. **`NaN`/missing dates.** Rows with missing dates do not participate in temporal logic. Implementations may either drop them at the boundary or treat them as non-matching.
5. **Universal quantifier on empty side.** Vacuous truth is NOT permitted — the predicate fails for that person.
6. **`outside` and evaluable persons.** The complement is bounded — non-evaluable persons contribute no rows.
7. **Single-row persons.** `1st X` and `last X` coincide when there is only one match.
8. **Negative ordinals.** `-1st X` selects the last X; `-2nd X` selects the second-to-last.
9. **Shift across DST.** Day-precision dates avoid DST entirely; ports must use integer day arithmetic, not seconds.
10. **Mixed code/column case.** Codes are matched case-sensitively against cell values; keyword identification is case-insensitive on parse.
11. **Lower-bounded windows with a decoy ref.** `K50 inside 30 to 90 days after K51` where the person's *nearest* preceding K51 is 5 days back but another K51 is 60 days back MUST match. Exercised by the `3 to 10 days around` / `-10 to -3 days around` golden queries.
12. **Self-referential windows.** `X inside 5 days after X`: a lone X does not match (no *other* X); two X rows on the same date both match; `inside 1 to 5 days` excludes the same-date pair but finds an X 1–5 days back. See the `WithinExpr` self-exclusion rule.
