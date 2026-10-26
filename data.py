"""Load the CPU readings and build the two forecasting targets."""
from pathlib import Path
import numpy as np
import pandas as pd


def load_cpu(data_dir):
    d = Path(data_dir)
    for name in ("vm_cpu_readings.csv.gz", "vm_cpu_readings.csv"):
        if (d / name).exists():
            return pd.read_csv(d / name)
    raise FileNotFoundError(f"vm_cpu_readings.csv(.gz) not found in {d}")


def build_targets(cpu):
    """One row per timestamp present in the raw readings.

    active_vm_count           = distinct VMs with a valid reading at that timestamp
    aggregate_cpu_utilization = mean of the VMs' avg_cpu at that timestamp

    Treatment: duplicate (vm_id, timestamp) rows are dropped; readings with a missing
    or out-of-range avg_cpu (outside [0, 100]) are ignored. The index keeps EVERY raw
    timestamp, so the holdout timestamps are always the final timestamps of the supplied
    data. A timestamp with no valid reading gets active_vm_count 0 and a missing CPU
    value (no value is invented).
    """
    all_ts = np.sort(cpu["timestamp"].unique())
    cpu = cpu.drop_duplicates(["vm_id", "timestamp"])
    valid = cpu[cpu["avg_cpu"].between(0, 100)]      # NaN is never "between", so it is dropped
    g = valid.groupby("timestamp")
    targets = pd.DataFrame({
        "active_vm_count": g["vm_id"].nunique(),
        "aggregate_cpu_utilization": g["avg_cpu"].mean(),
    }).reindex(all_ts)
    targets.index.name = "timestamp"
    targets["active_vm_count"] = targets["active_vm_count"].fillna(0).astype(int)
    return targets
