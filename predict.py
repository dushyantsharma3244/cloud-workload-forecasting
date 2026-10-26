"""Create predictions.csv with the persistence (naive) forecast.

Usage:  python -m forecast.predict --data challenge_data --output predictions.csv

The last H timestamps are the holdout. Only their timestamps are read;
their target values are never used.
"""
import argparse
import numpy as np
import pandas as pd
from .data import load_cpu, build_targets

H = 24
COLUMNS = ["timestamp", "active_vm_count", "aggregate_cpu_utilization"]


def split(targets, h=H):
    if len(targets) <= h:
        raise ValueError(f"need more than {h} timestamps, got {len(targets)}")
    return targets.iloc[:-h], targets.iloc[-h:]


def last_known(series):
    """Last non-missing training value (persistence)."""
    s = series.dropna()
    if s.empty:
        raise ValueError("no valid training observations")
    return s.iloc[-1]


def make_predictions(targets, h=H):
    train, holdout = split(targets, h)
    out = pd.DataFrame({
        "timestamp": holdout.index,                       # index only, no target values read
        "active_vm_count": np.repeat(last_known(train["active_vm_count"]), h),
        "aggregate_cpu_utilization": np.repeat(last_known(train["aggregate_cpu_utilization"]), h),
    })
    out["active_vm_count"] = out["active_vm_count"].clip(lower=0)
    out["aggregate_cpu_utilization"] = out["aggregate_cpu_utilization"].clip(0, 100)
    return out


def validate(df, h=H):
    if list(df.columns) != COLUMNS:
        raise ValueError(f"wrong columns: {df.columns.tolist()}")
    if len(df) != h:
        raise ValueError(f"expected {h} rows, got {len(df)}")
    if not (df["timestamp"].is_unique and df["timestamp"].is_monotonic_increasing):
        raise ValueError("timestamps must be unique and in chronological order")
    if not np.isfinite(df[COLUMNS[1:]].to_numpy(dtype=float)).all():
        raise ValueError("predictions must be finite")
    if (df["active_vm_count"] < 0).any():
        raise ValueError("active_vm_count must be >= 0")
    if not df["aggregate_cpu_utilization"].between(0, 100).all():
        raise ValueError("aggregate_cpu_utilization must be in [0, 100]")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="challenge_data")
    ap.add_argument("--output", default="predictions.csv")
    ap.add_argument("--horizon", type=int, default=H)
    a = ap.parse_args()
    targets = build_targets(load_cpu(a.data))
    preds = make_predictions(targets, a.horizon)
    validate(preds, a.horizon)
    preds.to_csv(a.output, index=False)
    print(f"wrote {a.output} {preds.shape}")


if __name__ == "__main__":
    main()
