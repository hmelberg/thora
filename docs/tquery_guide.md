# tquery: A Temporal Query Language for Health Event Data

## Introduction

Healthcare researchers routinely work with event-level data: prescription records, hospital admissions, laboratory results, procedure logs. Each row represents one event for one patient, with a date, one or more medical codes, and possibly a duration.

The questions researchers ask about this data are almost always **temporal**: *Did the patient receive drug A before drug B? How many patients had at least 3 emergency visits within a year? Who started biologics without trying conventional therapy first?*

These questions are conceptually simple but surprisingly difficult to express in standard pandas code. A query like "patients who received at least 2 courses of antibiotics within 90 days before their first surgery" requires multiple groupby operations, date arithmetic, boolean masking, and careful handling of person-level vs. event-level logic.

**tquery** solves this with a domain-specific query language that reads like natural language:

```python
import tquery

# Natural language → tquery expression → answer
tquery.tquery(df, 'min 2 of J01 within 90 days before 1st of surgery_code').count
# → 234
```

The library also includes tools for **treatment pattern visualization** (stringify functions), **medical code management** (codebooks, labels, search), and **parameterized batch queries**.

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
| `K50 within 90 days` | K50 within 90 days of first event per person |
| `K50 within 90 days after K51` | K50 within 90 days following any K51 |
| `K50 within 30 days before K51` | K50 within 30 days preceding any K51 |
| `K50 within 60 days around K51` | K50 within 60 days in either direction |
| `K50 between 30 and 90 days after K51` | K50 between 30 and 90 days after K51 (excludes first 30 days) |

#### Event Windows

| Expression | Meaning |
|-----------|---------|
| `K50 inside 5 events after K51` | K50 within the next 5 events after K51 |
| `K50 outside 10 events before K51` | K50 not within 10 events before K51 |

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
'min 2 of glucose>8 within 60 days'

# First K50 before first K51
'1st of K50 before 1st of K51'

# 2+ K50 events AND K51 before K52
'(min 2 of K50) and (K51 before K52)'

# No K52 AND K50 before K51
'not K52 and K50 before K51'

# Statin within 30-365 days after MI (washout period)
'C10AA between 30 and 365 days after I21'
```

### Operator Precedence

From lowest (evaluated last) to highest (evaluated first):

1. `or`
2. `and`
3. `before` / `after` / `simultaneously`
4. `not`
5. `min` / `max` / `exactly` / ordinals / `first` / `last` / count ranges
6. `within` / `between` / `inside` / `outside`
7. Atoms (codes, comparisons, parenthesized sub-expressions)

Use parentheses to override precedence when needed:

```python
# Without parens: K51 OR (K52 before K50)  — due to precedence
'K51 or K52 before K50'

# With parens: (K51 or K52) before K50  — what you probably meant
'(K51 or K52) before K50'
```

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
result.pids           # set: {1, 3, 7, ...}
result.persons        # Series: pid → True/False
result.rows           # Series: row-level boolean mask
result.event_counts   # Series: pid → number of matching events
result.filter()       # DataFrame: all rows for matching persons
result.filter('rows') # DataFrame: only the matching rows
```

### Shorthand Functions

```python
tquery.count_persons(df, 'K50 before K51')        # → int
tquery.event_counts(df, 'K50 within 90 days')      # → Series (pid → count)
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
tquery.multi_query(df, 'K50 within ?[30,60,90,180,365] days after K51')

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
df.tq('I50 within 30 days after I50').count

# Statin within 90 days after MI (guideline adherence)
df.tq('C10AA within 90 days after I21').count

# Stroke in AF patients without anticoagulation
df.tq('I63 within 365 days after 1st of I48 and not B01A').count

# New statin prescription 30-365 days after MI (washout period)
df.tq('C10AA between 30 and 365 days after I21').count
```

### Diabetes

```python
# Metformin before insulin (correct step-up)
df.tq('A10BA02 before A10AE').count

# 3+ high glucose readings within a year
df.tq('min 3 of glucose>9 within 365 days').count

# Treatment delay: metformin within 90 days of diagnosis
df.tq('A10BA02 within 90 days after 1st of E11').count
```

### Inflammatory Bowel Disease

```python
# Biologic before 5-ASA (top-down treatment)
df.tq('L04AB02 before A07EC02').count

# Steroid-dependent patients (3+ courses in a year)
df.tq('min 3 of H02AB06 within 365 days').count

# Relapse within 1 year of starting biologic
df.tq('K50 within 365 days after 1st of L04AB02').count

# Treatment pattern visualization
codes = {'i': ['L04AB02'], 'a': ['L04AB04'], 's': ['H02AB*']}
df.tq.stringify_order(codes, cols='atc')
```

### Antibiotic Stewardship

```python
# C. difficile within 60 days of antibiotics
df.tq('A047 within 60 days after J01').count

# Repeated courses (treatment failure signal)
df.tq('min 3 of J01 within 90 days').count

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
