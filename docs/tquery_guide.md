# tquery: A Temporal Query Language for Health Event Data

## Introduction

Healthcare researchers routinely work with event-level data: prescription records, hospital admissions, laboratory results, procedure logs. Each row represents one event for one patient, with a date, one or more medical codes, and possibly a duration.

The questions researchers ask about this data are almost always **temporal**: *Did the patient receive drug A before drug B? How many patients had at least 3 emergency visits within a year? Who started biologics without trying conventional therapy first?*

These questions are conceptually simple but surprisingly difficult to express in standard pandas code. A query like "patients who received at least 2 courses of antibiotics within 90 days before their first surgery" requires multiple groupby operations, date arithmetic, boolean masking, and careful handling of person-level vs. event-level logic.

**tquery** solves this with a domain-specific query language that reads like natural language:

```python
import tquery

# Natural language → tquery expression → answer
tquery.tquery(df, 'min 2 of J01 inside 90 days before 1st of surgery_code').count
# → 234
```

The library also includes tools for **treatment pattern visualization** (stringify functions), **medical code management** (codebooks, labels, search), and **parameterized batch queries**.

---

## Installation

```bash
pip install git+https://github.com/hmelberg/thora.git
```

---

## Quick Start

```python
import pandas as pd
import tquery

# Your event-level DataFrame
df = pd.DataFrame({
    'pid': [1, 1, 1, 2, 2, 2],
    'start_date': pd.to_datetime([
        '2020-01-01', '2020-03-01', '2020-06-01',
        '2020-01-01', '2020-02-01', '2020-04-01',
    ]),
    'icd': ['K50', 'K51', 'K52', 'K51', 'K50', 'K52'],
})

# Query: who had K50 before K51?
result = tquery.tquery(df, 'K50 before K51')
result.count     # 1 (person 1)
result.pids      # {1}

# Or use the pandas accessor
df.tq('K50 before K51').count        # 1
df.tq.count('min 2 of K50')         # 0
df.tq('K50 and K52').pids           # {1, 2}
```

---

## The Query Language

### Overview Table

#### Code Matching

| Expression | Meaning |
|-----------|---------|
| `K50` | Events with code K50 |
| `K50, K51, K52` | Events with any of these codes |
| `K50*` | Wildcard: all codes starting with K50 |
| `K50-K53` | Range: codes from K50 to K53 (alphabetic) |
| `@antibiotics` | Variable: codes from a predefined list |

#### Column Specification

| Expression | Meaning |
|-----------|---------|
| `K50 in icd` | Search in the 'icd' column only |
| `K50 in icd1, icd2, icd3` | Search in multiple named columns |
| `K50 in icd*` | Search in all columns starting with 'icd' |
| `K50 in icd1:icd10` | Search in columns icd1 through icd10 (positional slice) |
| `K50 in icd1-icd10` | Search in columns icd1 through icd10 (alphabetic range) |

#### Temporal Ordering

| Expression | Meaning |
|-----------|---------|
| `K50 before K51` | K50 occurs before K51 (for the same person) |
| `K50 after K51` | K50 occurs after K51 |
| `K50 simultaneously K51` | K50 and K51 on the same date |

#### Time Windows

| Expression | Meaning |
|-----------|---------|
| `K50 inside 90 days` | K50 within 90 days of first event per person |
| `K50 inside 90 days after K51` | K50 within 90 days following any K51 |
| `K50 inside 30 days before K51` | K50 within 30 days preceding any K51 |
| `K50 inside 60 days around K51` | K50 within 60 days in either direction (shorthand for `-60 to 60`) |
| `K50 inside 30 to 90 days after K51` | K50 between 30 and 90 days after K51 (excludes the first 30 days) |
| `K50 inside -5 to 20 days around K51` | Asymmetric window: 5 days before to 20 days after K51 |
| `K50 outside 30 days after K51` | Row-level complement of the positive form |
| `K50 outside 5 to 30 days after K51` | Row-level complement of the range form |

Negative offsets are only valid with `around`; `inside -5 days after K51` is a syntax error. Range bounds must be ascending.

#### Event Windows

| Expression | Meaning |
|-----------|---------|
| `K50 inside 5 events after K51` | K50 within the next 5 events after K51 |
| `K50 inside 3 to 5 events after K51` | K50 falling 3–5 events after K51 (range form) |
| `K50 outside 10 events before K51` | Row-level complement: K50 not in the 10 events before K51 |
| `K50 inside last 5 events` | K50 falling within the last 5 rows of the person's timeline |
| `K50 outside last 5 events` | Row-level complement of the positional span |
| `K50 inside 1st K51 and 5th K51` | K50 between the first and fifth K51 events |
| `K50 outside 1st K51 and 5th K51` | Row-level complement of the positional span |

#### The `event` / `events` Atom

`event` (or its plural `events`) is a universal-row atom — it stands for "any row in the timeline" and can be combined with positional, counting, and temporal constructs:

| Expression | Meaning |
|-----------|---------|
| `5th event` | The 5th row per person |
| `last 5 events` | The last 5 rows per person |
| `first 3 of events` | The first 3 rows per person |
| `min 3 of event` | Persons with at least 3 rows |
| `K50 before 5th event` | K50 occurs before the 5th event per person |
| `K50 after 3rd event` | K50 occurs after the 3rd event per person |
| `K50 inside last 5 events` | K50 within the last 5 events per person |

#### Quantifiers (Counts)

| Expression | Meaning |
|-----------|---------|
| `min 2 of K50` | Persons with at least 2 K50 events |
| `max 3 of K50` | Persons with at most 3 K50 events |
| `exactly 1 of K50` | Persons with exactly 1 K50 event |
| `2-5 of K50` | Persons with 2 to 5 K50 events |
| `2-5 K50` | Same (the 'of' is optional) |

#### Positional Selection

| Expression | Meaning |
|-----------|---------|
| `1st of K50` | Only the first K50 event per person |
| `2nd of K50` | Only the second K50 event per person |
| `first 3 of K50` | The first 3 K50 events per person |
| `last 2 of K50` | The last 2 K50 events per person |
| `-1st K50` | The last K50 event per person (negative ordinal) |
| `-2nd K50` | The second-to-last K50 event per person |
| `-3rd event` | The third-to-last row per person |
| `-5th of K50` | Equivalent to `5th from the end of K50` |

Negative ordinals count from the end and compose with windows, e.g. `K50 inside 60 days before -1st K51`.

#### Universal / Existential Quantifiers

| Expression | Meaning |
|-----------|---------|
| `any K50 after K51` | Default — same as `K50 after K51` (sugar) |
| `K50 after every K51` | Every K51 has a K50 after it (and K51 non-empty) |
| `K50 after each K51` | Synonym for `K50 after every K51` (reads better on the ref side) |
| `every K50 before K51` | Every K50 has a K51 after it |
| `always K50 before K51` | Synonym for `every K50 before K51` (reads better on the subject side) |
| `every K50 after every K51` | All K50s strictly follow all K51s |
| `K50 inside 100 days after every K51` | Every K51 followed by a K50 within 100 days |
| `never K50 inside 3 events after K51` | Synonym for `not (K50 inside 3 events after K51)` |

#### Logical Operators

| Expression | Meaning |
|-----------|---------|
| `K50 and K51` | Person has both codes (at any time) |
| `K50 or K51` | Person has either code |
| `not K50` | Person does not have K50 |

#### Column Comparisons

| Expression | Meaning |
|-----------|---------|
| `glucose > 8` | Rows where glucose column exceeds 8 |
| `hba1c >= 6.5` | Rows where hba1c is at least 6.5 |
| `age == 50` | Rows where age equals 50 |

---

### Compound Expressions

Expressions can be freely combined using parentheses:

```python
# Crohn's or ulcerative colitis before surgery
'(K50 or K51) before K52'

# At least 2 glucose readings above 8, within 60 days
'min 2 of glucose>8 inside 60 days'

# First K50 before first K51
'1st of K50 before 1st of K51'

# 2+ K50 events AND K51 before K52
'(min 2 of K50) and (K51 before K52)'

# No K52 AND K50 before K51
'not K52 and K50 before K51'

# Statin within 30-365 days after MI (washout period)
'C10AA inside 30 to 365 days after I21'
```

### Quantifiers: `every`, `any`, `each`, `always`, `never`

By default, temporal queries are existential — `K50 after K51` matches a person if there is *some* K50 that occurs after *some* K51. To express universal claims like "every K51 is followed by a K50 within 100 days", prefix the relevant atom with `every`. The dual `any` is the explicit form of the default and is purely sugar.

Three additional keywords are pure lexical sugar, aimed at making the query read more naturally:

- `each Y` is a synonym for `every Y` (reads more naturally on the ref side: `K50 after each K51`).
- `always X` is a synonym for `every X` (reads more naturally on the subject side: `always K50 before K51`).
- `never X` is a synonym for `not X` (applied to a whole clause: `never K50 inside 3 events after K51`).

```python
# Every diagnosis K51 is followed by a treatment K50 within 100 days
'K50 inside 100 days after every K51'
'K50 inside 100 days after each K51'    # synonym

# Every K50 the patient has is preceded by a K51 within 30 days (e.g. every
# control visit was after a recent prescription)
'every K50 inside 30 days after K51'
'always K50 inside 30 days after K51'   # synonym

# All K50s strictly follow all K51s — no overlap, K51s came first
'every K50 after every K51'

# Explicit form of the default — semantically identical to `K50 after K51`
'any K50 after any K51'

# `never` negates a whole clause (sugar for `not (...)`)
'never K50 inside 3 events after K51'
```

**Vacuous truth is excluded.** `every K51 ...` requires the person to have at least one K51 event. A patient with zero K51s does *not* satisfy `K50 after every K51`.

**Restrictions.** `every`/`any`/`each`/`always` may only precede a bare code expression or `@variable`. They cannot be combined with parentheses, counting prefixes (`min`/`max`/`exactly`/`first`/`last`/ordinals), or `not`. They also require a surrounding temporal context — `every K50` on its own is a syntax error.

**Equivalences with the existing default.** Because the historical default for `K50 before K51` already enforces "the first K50 precedes every K51" (i.e. `min(K50) < min(K51)`), adding `every K51` on the *right* of `before` is a no-op. By symmetry, `every K50 after K51` equals the default. The non-redundant additions are:

| Query | Adds |
|---|---|
| `every K50 before K51` | `max(K50) < max(K51)` — every K50 has some K51 after it |
| `K50 after every K51`  | `max(K50) > max(K51)` — every K51 has some K50 after it |
| `every K50 before every K51` | `max(K50) < min(K51)` — strict total separation |
| `every K50 after every K51`  | `min(K50) > max(K51)` — strict total separation |

Inside a time window, the universal reading on either side adds real constraints regardless of direction:

```python
# Every K51 is followed by a K50 within 100 days (cohort coverage check)
'K50 inside 100 days after every K51'

# Every K50 has a K51 in the 100 days before it (every event is "explained")
'every K50 inside 100 days after K51'
```

### Operator Precedence

From lowest (evaluated last) to highest (evaluated first):

1. `or`
2. `and`
3. `before` / `after` / `simultaneously`
4. `not`
5. `min` / `max` / `exactly` / ordinals / `first` / `last` / count ranges
6. `inside` / `outside` (time windows, event windows, positional spans)
7. Atoms (codes, comparisons, parenthesized sub-expressions)

Use parentheses to override precedence when needed:

```python
# Without parens: K51 OR (K52 before K50)  — due to precedence
'K51 or K52 before K50'

# With parens: (K51 or K52) before K50  — what you probably meant
'(K51 or K52) before K50'
```

### Shifted Anchor Dates

A reference selector can be shifted by a number of days, which is the cleanest way to express "more than N days apart" queries:

```python
# First K51 occurred more than 100 days before the first K50
'1st K51 before 1st K50 - 100 days'

# First K51 occurred before (first K50 + 30 days)
'1st K51 before 1st K50 + 30 days'

# Parens are optional but allowed
'1st K51 before (1st K50 - 100 days)'

# Chains add up: (-30) + (-7) == -37
'1st K51 before 1st K50 - 30 days - 7 days'

# Inside a window — the right bound is shifted
'K50 inside 60 days before -1st K51 - 14 days'

# As either bound of a positional `inside ... and ...` span
'K50 inside 1st K51 + 7 days and 5th K51 - 7 days'
```

**Restrictions.**

- The shifted side must be a single-date expression — i.e. an ordinal selector (`1st X`, `-2nd X`, `5th of X`). Plain code (`K51 - 30 days`), multi-row prefixes (`first 2 of K51 - 30 days`), and counting prefixes (`min 2 of K51 + 30 days`) are syntax errors.
- Shifted dates are valid only as the **reference** in:
  - the RHS of `before` / `after` / `simultaneously`
  - the ref of an `inside N days direction REF` window
  - either bound of `inside EXPR and EXPR`
- They are **not** valid as the LHS of a temporal comparison, in event-count windows (`inside N events after …`), as the ref of a positional span without bounds, as a logical operand, or as the child of a prefix.
- Only the unit `days` is supported (no `weeks`/`months`/`years`).

### Outside Complements

For every positive `inside …` form, `outside …` gives the row-level complement, restricted to evaluable persons (i.e. those for whom the corresponding positive form is well-defined):

```python
'K50 outside 30 days after K51'
'K50 outside 5 to 30 days after K51'
'K50 outside 3 events after K51'
'K50 outside last 5 events'
'K50 outside 1st K51 and 5th K51'
```

`outside` participates in the same precedence slot as `inside` and obeys the same parenthesisation rules.

---

### Variable References

Define code lists externally and reference them with `@`:

```python
antibiotics = ['J01CE02', 'J01CR02', 'J01CA04']
surgery = ['NCSP_A', 'NCSP_B']

result = tquery.tquery(df, '@antibiotics before @surgery',
                       variables={'antibiotics': antibiotics, 'surgery': surgery})
```

Variables can also be set in the configuration so they persist across queries:

```python
tquery.use(TQueryConfig(
    variables={'antibiotics': antibiotics, 'surgery': surgery}
))
df.tq('@antibiotics before @surgery').count
```

---

## Results

Every query returns a `TQueryResult` with multiple views:

```python
result = df.tq('K50 before K51')

result.count          # int: number of matching persons
result.total          # int: total persons in the DataFrame
result.evaluable      # int: persons for whom the query is well-defined
result.pct()          # float: count / evaluable * 100 (default)
result.pct(dropna=False)  # float: count / total * 100
result.pids           # set: {1, 3, 7, ...}
result.evaluable_pids # set: persons in the denominator under dropna=True
result.persons        # Series: pid → True/False
result.rows           # Series: row-level boolean mask
result.event_counts   # Series: pid → number of matching events
result.filter()       # DataFrame: all rows for matching persons
result.filter('rows') # DataFrame: only the matching rows
```

### Counting and Proportions: `count`, `pct`, and "missing"

For temporal queries the most common follow-up question is *what fraction*. There are two natural denominators, and they often differ by an order of magnitude:

```python
# Out of 100 patients in the cohort, 2 had K50 before K51.
# Out of 17 patients with both K50 and K51, 2 had K50 first.

result = df.tq('K50 before K51')
result.count             # 2
result.evaluable         # 17  — patients with both diagnoses
result.total             # 100 — all patients in df
result.pct()             # 11.8 — conditional: of those evaluable
result.pct(dropna=False) # 2.0  — marginal: of all patients
```

The default `pct()` mirrors pandas' `dropna=True` convention: persons for whom the query is *undefined* (not merely false) are excluded from the denominator. A person is "missing" only if they lack one of the events being compared in a temporal/`inside` subexpression. For non-comparative queries (`K50`, `min 2 of K51`, `K50 and K51`), every person is evaluable, so `pct()` and `pct(dropna=False)` return the same number.

The `dropna=False` form gives the marginal percentage — useful for prevalence-style questions ("what fraction of the whole cohort experienced this pattern"). The default `dropna=True` gives the conditional percentage — usually the more clinically interesting reading ("among patients eligible for the comparison, what fraction matched").

The accessor mirrors `count`:

```python
df.tq.count('K50 before K51')              # 2
df.tq.pct('K50 before K51')                # 11.8 — default conditional
df.tq.pct('K50 before K51', dropna=False)  # 2.0  — marginal
```

### Boolean Masks: `mask()`

Use `df.tq.mask("expr")` when you want a boolean Series instead of a count or percentage. By default it returns a row-aligned mask — useful for filtering directly:

```python
# Row-level mask aligned to df.index (the default)
mask = df.tq.mask('K50 inside 90 days after K51')
df[mask]                              # only the matching rows

# Quick row-selection idioms
df[df.tq.mask('last 2 of K50')]
df[df.tq.mask('after 2nd of K51')]
df[df.tq.mask('-1st K50')]            # the last K50 per person

# Person-level mask (Series indexed by pid)
persons = df.tq.mask('K50 before K51', level='persons')
persons.sum()                         # same as df.tq.count('K50 before K51')
```

`mask()` accepts the same expression language as `count()` and `pct()` and forwards `**kwargs` to the underlying query (so `cols=`, `config=`, `variables=`, etc. all work).

**Notes on edge cases:**

- If `evaluable == 0` (no person could even be evaluated), `pct()` returns `0.0`, not `NaN` or an exception.
- Wrapping a comparative subexpression in `not` (`not (K50 before K51)`) widens the evaluable set to all persons, because the existing `not` operator collapses "undefined" to True. If you want the strict conditional reading, write the positive form (`K51 before K50`) instead.
- `every X` requires X to be non-empty for the answer to be defined, so its evaluable set excludes persons with no X events.

### Shorthand Functions

```python
tquery.count_persons(df, 'K50 before K51')        # → int
tquery.event_counts(df, 'K50 inside 90 days')      # → Series (pid → count)
```

---

## Parameterized Queries: `?[...]`

Run multiple query variants at once and collect all results:

```python
# Vary a code
tquery.multi_query(df, 'K5?[0,1,2] before K51')
# K50 before K51    234
# K51 before K51      0
# K52 before K51     89

# Numeric range
tquery.multi_query(df, 'K5?[0-9] before K51')   # 10 queries

# Vary the count
tquery.multi_query(df, 'min ?[1,2,3,5] of K50')

# Vary the time window
tquery.multi_query(df, 'K50 inside ?[30,60,90,180,365] days after K51')

# Multiple slots (cartesian product)
tquery.multi_query(df, 'K5?[0,1] before K5?[2,3]')
# K50 before K52    ...
# K50 before K53    ...
# K51 before K52    ...
# K51 before K53    ...

# Via accessor
df.tq.multi('K5?[0-3] before K51')
```

**Slot syntax inside `?[...]`:**

| Syntax | Expands to |
|--------|-----------|
| `?[0,1,2]` | `0`, `1`, `2` |
| `?[K50,K51]` | `K50`, `K51` |
| `?[0-9]` | `0`, `1`, `2`, ..., `9` |
| `?[a-d]` | `a`, `b`, `c`, `d` |
| `?[50-53]` | `50`, `51`, `52`, `53` |

A shared evaluator is used across all generated queries, so common sub-expressions (like `K51` in `K5?[0-3] before K51`) are evaluated once and cached.

Safety: raises `ValueError` if the number of combinations exceeds 1000 (configurable via `max_combinations`).

---

## Configuration Profiles

Different datasets have different column names. Use `TQueryConfig` to set defaults once:

```python
from tquery import TQueryConfig, use

# Norwegian Prescription Registry
prescriptions = TQueryConfig(
    pid='lopenr',
    date='utleveringsdato',
    cols='varenr',
    sep=',',
    event_duration='ddd',
    name='Prescriptions',
)

# Hospital admissions
hospital = TQueryConfig(
    pid='pasient_id',
    date='inn_dato',
    event_end='ut_dato',
    cols=['hoved', 'bi1', 'bi2'],
    name='Hospital',
)

# Set globally
tquery.use(hospital)
df.tq('K50 before K51').count    # uses hospital column names

# Switch
tquery.use(prescriptions)
rx_df.tq('N02A before N06A').count   # uses prescription column names

# Or pass per-call
df.tq('K50', config=hospital)

# Explicit kwargs always override the config
df.tq('K50', config=hospital, cols='special_col')
```

**Config fields:**

| Field | Default | Description |
|-------|---------|-------------|
| `pid` | `'pid'` | Person ID column |
| `date` | `'start_date'` | Event date column |
| `event_end` | `None` | Event end date column (for durations) |
| `event_duration` | `None` | Duration in days column |
| `cols` | `None` | Code columns (None = auto-detect) |
| `sep` | `None` | Multi-value cell separator |
| `variables` | `None` | Dict of `@variable` code lists |
| `codebooks_dir` | `None` | Path to custom codebooks folder |
| `name` | `'default'` | Profile name (for display) |

---

## Date Filtering

Filter by date before running queries using the `.period()` and `.year()` methods:

```python
# All of 2020
df.tq.year(2020).count('K50 before K51')

# Date range
df.tq.period('2020-01-01', '2020-12-31').count('K50')

# Open-ended
df.tq.period('2020-01-01').count('K50')            # from date onwards
df.tq.period(end='2019-12-31').count('K50')         # up to date

# Chains with everything
df.tq.year(2020).event_counts('K50')
df.tq.year(2020).multi('K5?[0,1] before K51')
```

---

## Codebooks: Labels, Search, and Code Lookup

### Looking Up Labels

```python
import tquery

# Single code
tquery.get_label('K50')       # 'Diseases of the digestive system'
tquery.get_label('L04AB02')   # 'Infliximab'
tquery.get_label('N02A')      # 'Opioids'
```

Labels resolve hierarchically: if `K50.1` isn't in the codebook, it falls back to `K50`, then to the ICD-10 chapter range `K00-K93`.

### Adding Labels to a DataFrame

```python
df.tq.labels(cols='icd')
# Adds an 'icd_label' column with human-readable labels

df.tq.labels(cols=['icd', 'atc'])
# Adds both 'icd_label' and 'atc_label' columns
```

### Searching for Codes

```python
tquery.search_codes('diabetes')
#   code                   label              system
# 0  A10   Drugs used in diabetes                atc
# 1  E10   Type 1 diabetes mellitus            icd10

tquery.search_codes('inhibitor')
#   code                                    label   system
# 0  B01AF     Direct factor Xa inhibitors     atc
# 1  C10AA     HMG CoA reductase inhibitors    atc
# 2  L04AB     TNF-alpha inhibitors            atc

# Filter by system
tquery.search_codes('diabetes', system='atc')
```

### Counting Code Frequencies

```python
# How often does each code appear?
df.tq.count_codes(cols='icd')
# K50    1234
# K51     890
# I21     234

# Unique persons per code (not events)
df.tq.count_codes(cols='icd', per_person=True)

# Filter to a pattern
df.tq.count_codes(cols='icd', pattern='K5*')
# K50    1234
# K51     890
# K52     456
```

### Custom Codebooks

Place CSV files in `tquery/codebooks/` with at least `code` and `label` columns:

```csv
code,label
K50,Crohn's disease
K50.0,Crohn's disease of small intestine
K50.1,Crohn's disease of large intestine
K51,Ulcerative colitis
```

Or point to a custom directory:

```python
tquery.use(TQueryConfig(codebooks_dir='D:/my_codebooks/'))
```

**Shipped codebooks:**
- `icd10_chapters.csv` - ICD-10 chapter-level labels (21 entries)
- `atc_groups.csv` - ATC classification levels 1-5 (51 entries, including common biologics)

---

## Treatment Pattern Strings

Convert event data into string representations for pattern analysis:

### Event Order: `stringify_order`

Creates a string showing events in the order they occurred:

```python
codes = {'i': ['L04AB02'], 'a': ['L04AB04'], 'g': ['L04AB06']}

tquery.stringify_order(df, codes, cols='atc')
# pid
# 1    iiaga    (infliximab, infliximab, adalimumab, golimumab, adalimumab)
# 2    aig      (adalimumab, infliximab, golimumab)

# Remove consecutive repeats
tquery.stringify_order(df, codes, cols='atc', keep_repeats=False)
# pid
# 1    iaga

# Keep only unique codes (first occurrence of each)
tquery.stringify_order(df, codes, cols='atc', only_unique=True)
# pid
# 1    iag
```

### Time-Positioned: `stringify_time`

Each character position represents a fixed time period:

```python
tquery.stringify_time(df, codes, cols='atc', step=90)
# pid
# 1    i| |a| |i     (event i at period 0, nothing at 1, event a at 2, ...)

# Separate tracks (one column per code label)
tquery.stringify_time(df, codes, cols='atc', step=90, merge=False)
# Returns DataFrame with columns 'i', 'a', 'g'
```

### Duration-Filled: `stringify_durations`

Like `stringify_time` but events fill their entire duration:

```python
tquery.stringify_durations(df, codes, cols='atc',
                           event_duration='days', step=30)
# pid
# 1    i|i|i|a|a| |g    (infliximab for 3 periods, adalimumab for 2, ...)
```

### Querying Stringified Output

The output of `stringify_order`, `stringify_time`, and `stringify_durations` can itself be queried with tquery expressions via `string_query`, `string_query_auto`, and `cross_validate`. This lets you ask temporal questions about *patterns* of events rather than the underlying rows.

Queries reference the original codes (e.g. `L04AB02`); the `codes` dict maps each code to a single-character label used internally in the string representation.

```python
import tquery as tq

codes = {"i": ["L04AB02"], "a": ["L04AB04"], "g": ["L04AB06"]}

# 1. Auto-stringify and query in one call
pids = tq.string_query_auto(
    df, "L04AB02 before L04AB04", codes, mode="order", cols="atc",
)
# pids: set of person IDs whose pattern shows infliximab before adalimumab

# 2. Or query a pre-computed string Series
strings = tq.stringify_order(df, codes, event_start="start_date", cols="atc")
# strings: pid → 'iiai' / 'aig' / ...
pids = tq.string_query(
    "min 2 of L04AB02 before L04AB04", strings, codes, mode="order",
)

# 3. Cross-validate the DataFrame and string evaluators agree
df_pids, str_pids, match = tq.cross_validate(
    df, "L04AB02 before L04AB04", codes, mode="order", cols="atc",
)
assert match  # both evaluators returned the same set

# Same via the accessor
df_pids, str_pids, match = df.tq.cross_validate(
    "L04AB02 before L04AB04", codes, mode="order", cols="atc",
)
```

Three modes mirror the three stringify functions:

| Mode | Backed by | Use when |
|---|---|---|
| `"order"` | `stringify_order` | You only care about the *sequence* of events (ignore exact dates) |
| `"time"` | `stringify_time` | You want fixed time-step buckets (e.g. one slot per 90 days) |
| `"durations"` | `stringify_durations` | Events have durations and fill multiple slots |

The string evaluator supports the full query language — wildcards, prefixes (`min N of`, `1st of`, …), `before`/`after`, `inside N days`, `not`/`and`/`or` — but operates on the per-person string representation rather than raw rows. `cross_validate` lets you sanity-check both evaluators against each other on the same query.

### String Manipulation Helpers

```python
from tquery._stringops import del_repeats, del_singles, shorten, left_justify

# Remove consecutive duplicates
del_repeats(series)       # 'aaabbc' → 'abc'

# Remove isolated characters
del_singles(series)       # 'aabca' → 'aa'

# Reduce time resolution
shorten(series, agg=3)    # aggregate every 3 positions into 1

# Pad to uniform length
left_justify(series)      # right-pad all strings to max length
```

---

## Incidence Calculation

Counting *new* disease cases per year from register data is harder than
it looks. Two systematic biases distort the naive count:

- **Left-censoring (washout bias)** — patients whose disease started
  before the data window opens look like new cases in the early years.
  The earlier the year, the worse the inflation.
- **Right-censoring (forward bias)** — under a "≥N events" case
  definition, recent single-event patients haven't yet had time to
  accumulate the events needed to confirm a diagnosis. The latest
  years are systematically under-counted.

tquery offers three layers of incidence calculation, from naive to
fully bias-corrected.

### Naive: `raw_incidence`

Count first events per person per year, optionally requiring at least
N matching events overall before counting:

```python
import tquery as tq

# All persons whose first K50 falls in each year
tq.raw_incidence(df, "K50")

# Only persons with >=2 K50 events count
tq.raw_incidence(df, "K50", required_events=2)

# Same via the accessor
df.tq.raw_incidence("K50")
```

The result is a Series indexed by year.

### Empirical patterns: `washout_pattern` and `singles_pattern`

Inspect the bias before correcting it. `washout_pattern` shows how the
"new case" count for a year decays as more historical data is added:

```python
# How does 2018's count fall as the lookback window grows?
tq.washout_pattern(df, "K50", year=2018, step_days=200, pct=True)
# A Series indexed by lookback days, decaying from 1.0 to the
# fraction that are *truly* new in 2018.

# All years at once -> DataFrame, columns = years, index = lookback days
patterns = tq.washout_pattern(df, "K50", step_days=365, pct=True)
patterns.plot()  # visual sanity check
```

`singles_pattern` is the symmetric forward analogue, used when the case
definition requires multiple events. It tracks how many in-year
"singletons" (persons with fewer than `required_events` events that
year) remain singletons as the observation window is expanded both
backward and forward:

```python
tq.singles_pattern(
    df, "K50",
    year=2015,
    required_events=2,
    step_days=200,
    pct=True,
)
```

### Bias-corrected: `incidence`

The master function combines `raw_incidence` with washout and forward
corrections:

```python
# Default: functional washout, automatic lookahead
tq.incidence(df, "K50", required_events=2)

# Explicit options
tq.incidence(
    df, "K50",
    required_events=2,
    washout="functional",   # 'none' | 'historical' | 'functional'
    lookahead="functional", # 'none' | 'historical' | 'functional' | 'auto'
    model="exponential",    # exponential | hyperbolic | rational
    step_days=365,
)

# Same via accessor
df.tq.incidence("K50", required_events=2)
```

The functional washout fits a parametric decay curve to the empirical
washout pattern (averaged across years) and divides each year's count
by the decay value at that year's available lookback, then rescales to
the asymptote. The functional forward correction fits a similar curve
to `singles_pattern` and subtracts the estimated true-singleton
contribution.

`lookahead='auto'` (the default) enables the forward correction iff
`required_events >= 2`.

#### Worked example

```python
import tquery as tq
from tests._synthetic import make_data

# Synthetic cohort: 200 truly-new persons per year, 2000-2020.
# Restrict to a 2010-2019 observation window — both ends are biased.
df = make_data(n_per_cohort=200, start_year=2000, end_year=2020,
               cohort_duration=10, seed=0)
sample = df[(df["date"].dt.year >= 2010) & (df["date"].dt.year <= 2019)]

tq.raw_incidence(sample, date="date")
# 2010    950   <- inflated by left-censoring
# 2011    573
# 2012    387
# ...
# 2019    200

tq.incidence(sample, date="date",
             washout="functional", lookahead="none")
# 2010    198.0  <- close to the true 200
# 2011    202.8
# ...
# 2019    199.0
```

### Choosing the adjustment method and model

`incidence` exposes three knobs: `washout`, `lookahead`, and `model`.
The defaults — **`washout='functional'`**, **`lookahead='auto'`**,
**`model='exponential'`** — are deliberate.

**Why `functional` over `historical`?** The historical approach uses the
empirical mean decay directly with no smoothing. That sounds attractive
because it makes no shape assumption, but every (year, lookback-step)
cell in `washout_pattern` is a small-N estimate, so the mean decay
wiggles non-monotonically and the wiggles propagate straight into the
corrected counts. The functional approach fits a smooth, monotone
parametric curve over the same points, which strongly stabilises the
correction in the regime where the empirical estimate is noisy. Use
`washout='historical'` only as a sanity check — overlay it against
the fitted curve and look for systematic departure.

**Why `exponential` over `hyperbolic` / `rational`?** The exponential
form `a + (1-a) * exp(-b*x/365.25)` has only two parameters, decays
monotonically from 1 to the asymptote `a`, and corresponds to a
constant per-day "wash-in" rate — the cleanest assumption for a
chronic-disease registry. With a typical 5–15-year window you only
have a handful of averaged decay points to fit, and a 2-parameter
model is much more stable than a 3-parameter one. Use `hyperbolic`
or `rational` only when you have a very long observation window
(≥20 years) and visual inspection of the empirical pattern shows a
clearly heavier tail than an exponential fit.

| Situation | Recommended |
|---|---|
| Most cases (≥5 years, ≥100 cases/year) | defaults |
| Very long window with chronic disease | try `model='hyperbolic'`, compare visually |
| Cross-checking the parametric fit | `washout='historical'` |

### Curve fitting: `fit_decay`

Useful directly if you want to inspect the fitted decay curve:

```python
pattern = tq.washout_pattern(df, "K50", pct=True)
fit = tq.fit_decay(pattern, model="exponential")
fit["coeffs"]      # numpy array of parameters
fit["asymptote"]   # long-x limit (the "true" rate after full lookback)
fit["predict"]     # callable: f(days_array) -> decay values
fit["aic"]         # Akaike information criterion (lower is better)
fit["r2"]          # coefficient of determination
```

To compare all three models on the same pattern, pass `model="all"`:

```python
all_fits = tq.fit_decay(pattern, model="all")
for name, fit in all_fits.items():
    print(f"{name:12s}  AIC={fit['aic']:8.2f}  R²={fit['r2']:.4f}  "
          f"asymptote={fit['asymptote']:.3f}")
```

A model whose AIC is more than ~2 lower than the next-best is meaningfully
preferred. Differences within 2 are not distinguishable on the available
data.

`fit_decay` requires `scipy`. Install with the optional extra:

```bash
pip install tquery[incidence]
```

### Caveats

- The functional adjustment assumes the per-year washout shape is stable
  across the observation window. If true incidence is changing rapidly
  *and* the bias structure changes too, the correction can be biased.
- For `required_events >= 3`, the forward correction is approximate —
  the asymptote captures the rate of "never-confirmed" persons, not
  partial accumulators.
- The historical adjustment is provided as a sanity check; it doesn't
  smooth the empirical pattern, so noisy years can produce noisy
  corrections.

---

## Data Format Requirements

tquery expects a pandas DataFrame with:

| Column | Type | Description |
|--------|------|-------------|
| Person ID | int or str | Identifies the individual (default name: `pid`) |
| Event date | datetime | When the event occurred (default name: `start_date`) |
| Code column(s) | str (object dtype) | Medical codes (ICD, ATC, etc.) |

The DataFrame should ideally be **sorted by (pid, date)** for best performance.

**Multi-value cells** are supported: if a cell contains `"K50,K51,K52"`, set `sep=','` to search within each sub-value.

**Multiple code columns** are supported: if your data has `icd1`, `icd2`, `icd3`, set `cols=['icd1','icd2','icd3']` or use wildcards `cols='icd*'`.

---

## Medical Query Examples

### Cardiology

```python
# Heart failure 30-day readmission
df.tq('I50 inside 30 days after I50').count

# Statin within 90 days after MI (guideline adherence)
df.tq('C10AA inside 90 days after I21').count

# Stroke in AF patients without anticoagulation
df.tq('I63 inside 365 days after 1st of I48 and not B01A').count

# New statin prescription 30-365 days after MI (washout period)
df.tq('C10AA inside 30 to 365 days after I21').count
```

### Diabetes

```python
# Metformin before insulin (correct step-up)
df.tq('A10BA02 before A10AE').count

# 3+ high glucose readings within a year
df.tq('min 3 of glucose>9 inside 365 days').count

# Treatment delay: metformin within 90 days of diagnosis
df.tq('A10BA02 inside 90 days after 1st of E11').count
```

### Inflammatory Bowel Disease

```python
# Biologic before 5-ASA (top-down treatment)
df.tq('L04AB02 before A07EC02').count

# Steroid-dependent patients (3+ courses in a year)
df.tq('min 3 of H02AB06 inside 365 days').count

# Relapse within 1 year of starting biologic
df.tq('K50 inside 365 days after 1st of L04AB02').count

# Treatment pattern visualization
codes = {'i': ['L04AB02'], 'a': ['L04AB04'], 's': ['H02AB*']}
df.tq.stringify_order(codes, cols='atc')
```

### Antibiotic Stewardship

```python
# C. difficile within 60 days of antibiotics
df.tq('A047 inside 60 days after J01').count

# Repeated courses (treatment failure signal)
df.tq('min 3 of J01 inside 90 days').count

# Explore which antibiotic codes are most common
df.tq.count_codes(cols='atc', pattern='J01*')
```

### Using Variables for Readability

```python
biologics = ['L04AB02', 'L04AB04', 'L04AB06', 'L04AC05']
conventional = ['A07EC*', 'L04AX03', 'L01BA01']

df.tq('@bio before @conv',
      variables={'bio': biologics, 'conv': conventional}).count
```

---

## Performance

tquery is designed for large datasets (millions of rows):

- **Vectorized operations**: all code matching and temporal logic uses pandas/numpy, no Python loops over rows
- **Pre-computed group boundaries**: person-group structure computed once at initialization
- **AST-level caching**: identical sub-expressions evaluated once and reused
- **Efficient code matching**: `np.char.startswith` for wildcards (3-5x faster than pandas str accessor), `isin(set)` for exact matches
- **merge_asof** for time-window queries (C-optimized in pandas)

Benchmarks (on a standard laptop):
- Simple code query on 1M rows: < 1 second
- Temporal query (before/after) on 1M rows: < 2 seconds
- Compound query on 100K rows: < 1 second
