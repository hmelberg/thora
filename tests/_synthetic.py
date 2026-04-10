"""Deterministic synthetic event generators for incidence testing.

These generators create event-level cohorts with known ground truth,
useful for validating that the incidence bias-correction recovers a
constant true incidence rate from a left- and right-censored sample.

Each call is fully reproducible via the `seed` argument.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def make_cohort(
    n: int = 1000,
    start_year: int = 1990,
    end_year: int = 2000,
    shape: float = 3.2,
    scale: float = 2.0,
    true_singles: float = 0.15,
    seed: int = 0,
) -> pd.DataFrame:
    """Generate one cohort of `n` persons whose first event is in `start_year`.

    Each person gets a Gamma-distributed number of subsequent events
    spread uniformly over [start_year, end_year]. A fraction
    `true_singles` of persons are forced to have exactly one event.

    Args:
        n: Number of persons in the cohort.
        start_year: Year in which every person's first event falls.
        end_year: Last year in which subsequent events may fall.
        shape, scale: Parameters of the Gamma distribution for visit
            counts (the count is `int(gamma) + 2`, so always >= 2 unless
            forced to 1 by `true_singles`).
        true_singles: Fraction of persons forced to have exactly one
            event.
        seed: PRNG seed.

    Returns:
        DataFrame with columns ``pid`` and ``date``, sorted by both.
    """
    rng = np.random.default_rng(seed)

    n_visits = (rng.gamma(shape, scale, size=n) + 2).astype(int)

    n_singles = int(round(true_singles * n))
    if n_singles > 0:
        single_idx = rng.choice(n, size=n_singles, replace=False)
        n_visits[single_idx] = 1

    start_date = pd.Timestamp(year=start_year, month=1, day=1)
    end_date = pd.Timestamp(year=end_year, month=12, day=31)
    span_days = (end_date - start_date).days

    pids = np.repeat(np.arange(n), n_visits)
    n_total = pids.size

    # Random offsets across the cohort window
    offsets = rng.integers(0, span_days + 1, size=n_total)

    # Force the FIRST event of each person to fall inside start_year
    # (cumulative offset positions == start of each person's run)
    starts = np.concatenate(([0], np.cumsum(n_visits)[:-1]))
    offsets[starts] = rng.integers(0, 365, size=n)

    dates = start_date + pd.to_timedelta(offsets, unit="D")

    df = pd.DataFrame({"pid": pids, "date": dates})
    return df.sort_values(["pid", "date"]).reset_index(drop=True)


def make_data(
    n_per_cohort: int = 1000,
    start_year: int = 1990,
    end_year: int = 2020,
    cohort_duration: int = 10,
    shape: float = 3.2,
    scale: float = 2.0,
    true_singles: float = 0.15,
    seed: int = 0,
) -> pd.DataFrame:
    """Stack one fresh cohort per year between start_year and end_year.

    Each cohort has `n_per_cohort` persons whose first event lands in
    that cohort's year. A new cohort is generated for every calendar
    year. The result mimics a population with constant true incidence
    over the full window.

    Args:
        n_per_cohort: Persons per yearly cohort.
        start_year: First cohort year.
        end_year: Last cohort year.
        cohort_duration: How many years a cohort's events span past its
            start year (controls the average follow-up length).
        shape, scale, true_singles: Forwarded to `make_cohort`.
        seed: PRNG seed.

    Returns:
        DataFrame with columns ``pid`` (globally unique) and ``date``,
        sorted by both.
    """
    rng = np.random.default_rng(seed)
    parts: list[pd.DataFrame] = []
    pid_offset = 0
    for year in range(start_year, end_year + 1):
        sub_seed = int(rng.integers(0, 2**31 - 1))
        cohort = make_cohort(
            n=n_per_cohort,
            start_year=year,
            end_year=year + cohort_duration,
            shape=shape,
            scale=scale,
            true_singles=true_singles,
            seed=sub_seed,
        )
        cohort["pid"] = cohort["pid"] + pid_offset
        pid_offset += n_per_cohort
        parts.append(cohort)
    df = pd.concat(parts, ignore_index=True)
    return df.sort_values(["pid", "date"]).reset_index(drop=True)
