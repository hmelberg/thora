# thora

**T**ools for **H**ealth register analysis — a temporal query language for event-level health data in pandas.

## What is this?

Health register data is stored as events: one row per person per event, with dates, diagnosis codes (ICD-10), and drug codes (ATC). `tquery` lets you answer temporal questions about this data using a natural query language:

```python
import tquery as tq

# How many patients had Crohn's disease before starting a biologic?
tq.count_persons(df, "K50 before L04AB*")

# Crohn's patients who got surgery within 1 year of diagnosis
df.tq("K50 before JF* within 365 days after K50").count

# At least 3 antibiotic prescriptions within 100 days
df.tq("min 3 of J01* within 100 days").count

# Parameterized batch queries
df.tq.multi("K5?[0,1,2] before L04AB?[02,04,06]")
```

## Installation

```bash
pip install -e .
```

Requires Python 3.10+ and pandas.

## Quick start

```python
import pandas as pd
import tquery as tq

# Configure for your dataset
tq.use(tq.TQueryConfig(
    pid="patient_id",
    date="event_date",
    cols=["icd_main", "icd_bi1", "icd_bi2"],
))

# Query
result = df.tq("K50 before K51")
result.count       # number of persons
result.pids        # set of matching person IDs
result.filter()    # DataFrame of matching persons
```

## Query language

| Expression | Meaning |
|---|---|
| `K50` | Has diagnosis code K50 |
| `K50*` | Any code starting with K50 |
| `K50 before K51` | K50 occurs before K51 (same person) |
| `K50 after K51` | K50 occurs after K51 |
| `K50 and K51` | Person has both K50 and K51 |
| `K50 or K51` | Person has K50 or K51 |
| `not K50` | Person does not have K50 |
| `K50 within 30 days after K51` | K50 within 30 days after K51 |
| `K50 between 30 and 90 days after K51` | K50 between 30-90 days after K51 |
| `min 3 of K50` | At least 3 events with K50 |
| `2-5 of K50` | Between 2 and 5 events with K50 |
| `1st K50 before K51` | First K50 before any K51 |
| `K50 in icd*` | K50 in columns matching `icd*` |
| `glucose>8` | Comparison on numeric column |
| `@antibiotics before @surgery` | Using predefined code lists |

See [`docs/tquery_guide.md`](docs/tquery_guide.md) for the full syntax reference.

## Features

- **Temporal queries**: before, after, within N days, between M and N days, simultaneously
- **Code patterns**: wildcards (`K50*`), ranges (`K50-K53`), column slices (`icd1:icd10`)
- **Counting**: min/max/exactly N of, count ranges (2-5 of), ordinals (1st, 2nd)
- **Numeric comparisons**: `glucose>8`, `bmi>=30`
- **Variables**: define code lists and use `@variable` in queries
- **Batch queries**: `?[...]` parameterized queries with automatic combinatorial expansion
- **Codebooks**: built-in ICD-10 and ATC labels, search by keyword, count code frequencies
- **Treatment patterns**: stringify functions for visualizing treatment sequences
- **Date filtering**: `.period()` and `.year()` for time-windowed analysis
- **Config profiles**: named configurations for different datasets
- **Performance**: vectorized numpy/pandas operations, merge_asof for time windows, AST-level caching

## Documentation

Full documentation: [`docs/tquery_guide.md`](docs/tquery_guide.md)

## License

MIT
