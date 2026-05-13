"""Generate a larger synthetic dataset for performance benchmarking.

50k persons × ~10 events ≈ 500k rows. Written to bench/dataset_large.csv.
Run once; the file is large-ish (~12 MB) — gitignored by convention.
"""
from __future__ import annotations
from pathlib import Path

from tquery._testdata import make_test_data

REPO_ROOT = Path(__file__).resolve().parent.parent
OUT = REPO_ROOT / "bench" / "dataset_large.csv"
OUT.parent.mkdir(parents=True, exist_ok=True)

df = make_test_data(n_persons=50_000, events_per_person=10, seed=42)
df.to_csv(OUT, index=False)
print(f"wrote {OUT}  ({len(df)} rows, {df['pid'].nunique()} persons)")
