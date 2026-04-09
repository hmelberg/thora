"""Synthetic data generation for tquery testing and benchmarks."""

from __future__ import annotations

import numpy as np
import pandas as pd


def make_test_data(
    n_persons: int = 1000,
    events_per_person: int = 10,
    codes: list[str] | None = None,
    start_year: int = 2015,
    end_year: int = 2023,
    seed: int = 42,
) -> pd.DataFrame:
    """Generate a synthetic event-level DataFrame for testing.

    Args:
        n_persons: Number of unique persons.
        events_per_person: Average events per person (actual count varies).
        codes: List of codes to sample from. Defaults to ICD-like codes.
        start_year: Earliest event year.
        end_year: Latest event year.
        seed: Random seed for reproducibility.

    Returns:
        DataFrame with columns: pid, start_date, icd — sorted by (pid, start_date).
    """
    rng = np.random.default_rng(seed)

    if codes is None:
        codes = [
            "K50", "K50.1", "K50.2", "K51", "K51.1", "K52",
            "I50", "I51", "I10", "I11",
            "S72", "S72.0", "S72.1",
            "J01", "J02", "J03", "J10", "J11",
            "E11", "E11.6", "E11.9",
            "N18", "N18.3", "N18.5",
        ]

    n_total = n_persons * events_per_person
    # Vary events per person (Poisson)
    counts = rng.poisson(events_per_person, n_persons)
    counts = np.maximum(counts, 1)  # at least 1 event per person
    n_total = int(counts.sum())

    pids = np.repeat(np.arange(1, n_persons + 1), counts)

    start = pd.Timestamp(f"{start_year}-01-01")
    end = pd.Timestamp(f"{end_year}-12-31")
    days_range = (end - start).days

    dates = start + pd.to_timedelta(rng.integers(0, days_range, n_total), unit="D")
    icd = rng.choice(codes, n_total)

    df = pd.DataFrame({
        "pid": pids,
        "start_date": dates,
        "icd": icd,
    })

    return df.sort_values(["pid", "start_date"]).reset_index(drop=True)
