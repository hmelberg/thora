# R port spike — `eval_within_days`

This is a **one-shot validation spike**, not a port. It answers a single
question: does `data.table`'s rolling join reproduce pandas
`merge_asof` semantics closely enough that the R port can structurally
mirror the Python evaluator?

## What's in here

| File | Purpose |
|---|---|
| `within_days.R` | R implementation of `WithinExpr` (existential, row-anchored, unsigned window). |
| `run_spike.R` | Driver: loads the golden dataset and replays every `WithinExpr` fixture, asserting `pids` parity. |

## How to run

R and `data.table` are not installed by default. Install both (e.g., via
Homebrew or conda-forge):

```bash
# Homebrew route
brew install r
Rscript -e 'install.packages(c("data.table", "jsonlite"), repos="https://cloud.r-project.org")'

# Or conda-forge route
mamba install -n some-env r-base r-data.table r-jsonlite
```

Then:

```bash
Rscript r-spike/run_spike.R
```

Exit code is `0` if every `WithinExpr` fixture passes pid-parity with
the Python reference, `1` if any fail.

## Reference semantics (from `spec/semantics.md`)

The Python reference uses `pd.merge_asof(tolerance=N)` which matches
`|delta| <= N` **inclusive** on the upper bound. Verified empirically:

| Delta from ref | merge_asof matches with `tolerance=30`? |
|---|---|
| 30 days | **yes** (boundary inclusive) |
| 31 days | no |

The R spike applies an explicit post-join filter `abs(delta) <= N` to
guarantee inclusivity parity regardless of `data.table` version.

## Result

**12 / 12 fixtures pass.** `data.table`'s rolling join can reproduce
`pd.merge_asof` semantics for the full `WithinExpr` evaluation
(including signed `around` windows). The full R port can structurally
mirror the Python evaluator. Proceed to Phase 1.

## Findings — the three non-obvious bits

These translated directly from the spike into the future R port's
implementation guidance:

1. **`rollends` is direction-sensitive.** With `roll = +N` (direction
   "after"), `rollends = c(TRUE, TRUE)` causes false positives by
   rolling a future ref backward to fill a gap at the start of a
   group. The correct setting is `c(FALSE, TRUE)`. Symmetric for
   `roll = -N`. Plain `rollends = FALSE` over-restricts and drops
   legitimate matches at group boundaries.

2. **Column-aliased `on` needs explicit `j`.** With
   `on = .(pid, ref_date = date)`, the result's `ref_date` column
   holds the child's date (i.date), not the matched ref date. To
   recover the matched ref date, the `j` must read `x.ref_date`
   explicitly.

3. **Boundary inclusivity matches by default in `data.table` 1.18,
   but the post-join `abs(delta) <= N` filter is still recommended.**
   It costs nothing and guarantees parity across `data.table` versions
   where boundary semantics may differ.

4. **Signed `around` (min_days < 0) needs asymmetric roll bounds.**
   Backward roll uses `+days`; forward roll uses `-abs(min_days)`.
   Combined results filtered by signed delta, not `abs(delta)`.

## What this spike intentionally does NOT cover

The full R port (Phase 1) still has to address:

- Universal quantifiers (`every_left` / `every_right`).
- `ref = NULL` case (window from each person's first event).
- `outside = TRUE` (row-level complement restricted to evaluable persons).
- Every AST node type other than `WithinExpr` + `CodeAtom` child/ref.

All tracked by `spec/golden/queries.jsonl`.
