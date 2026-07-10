# tquery DSL specification

This directory is the **portable specification** for the `tquery` temporal query language. It exists so that ports of the library to other languages (R / `data.table`, Polars, DuckDB, SQL) can implement the same DSL and produce identical results.

The Python reference implementation lives in `../tquery/`. When semantics are ambiguous, the answer is whatever the Python reference produces.

## Contents

| File | Purpose |
|---|---|
| `grammar.md` | Formal EBNF grammar for the DSL. Read first to understand what queries parse. |
| `ast.md` | The intermediate representation. Every node type with its fields, types, and invariants. The contract between parser and evaluator. |
| `semantics.md` | For each AST node: what it computes, edge cases, and tie-breaking rules. |
| `golden/` | JSON test fixtures (`query`, `input_table`, `expected_pids`, `expected_rows`) used by every port. |
| `codebooks/` | ICD-10 and ATC codebook CSVs (currently lives in `../tquery/codebooks/` — to be canonicalised here). |

## How to use this spec when implementing a port

1. Read `grammar.md` → write a parser that produces nodes matching `ast.md`.
2. Read `semantics.md` → implement an evaluator that, given an AST and a data source, returns row/person masks.
3. Run the parser against every `query` in `golden/` and assert the AST shape matches the recorded shape.
4. Run the evaluator and assert `count`, `pids`, and `rows` match the recorded expectations.
5. Pass the per-node golden tests, then the end-to-end synthetic-data golden tests.
6. Run `python tools/fuzz_parity.py --iters 200` — randomized small timelines diffed across backends. The golden corpus pins agreed answers on ONE dataset; the fuzzer hunts divergence on adversarial data nobody hand-picked (this is the class of bug that once let an asof-nearest window bug survive golden parity).

## Versioning

This spec is versioned alongside the Python reference. The current spec describes Python `tquery v0.1.0` (see `../tquery/__init__.py:__version__`), with **pending additions for v0.2** marked inline (`AggregateExpr` and related grammar). When the DSL or AST changes, both the Python implementation and this spec must bump version together. The pending v0.2 additions are NOT yet present in the reference implementation — they exist in spec form to guide the upcoming work.

## Out of scope

The following Python modules are NOT part of the portable DSL spec and are not required for a v1 port:

- `_incidence.py` — bias-corrected incidence statistics
- `_string_evaluator.py` and `_stringify.py` — treatment-pattern stringification
- `_cache.py` — AST-level result caching (an implementation optimisation, not user-visible behaviour)

Ports may add their own equivalents of these later.
