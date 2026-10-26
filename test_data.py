import pandas as pd
from forecast.data import build_targets


def test_vm_counted_once_and_mean():
    cpu = pd.DataFrame({
        "timestamp": [0, 0, 0, 300],
        "vm_id": ["a", "a", "b", "a"],
        "avg_cpu": [10, 10, 30, 50],
    })
    t = build_targets(cpu)
    assert t.loc[0, "active_vm_count"] == 2
    assert t.loc[0, "aggregate_cpu_utilization"] == 20
    assert t.loc[300, "active_vm_count"] == 1


def test_invalid_readings_ignored():
    cpu = pd.DataFrame({"timestamp": [0, 0, 0], "vm_id": ["a", "b", "c"],
                        "avg_cpu": [10.0, None, 150.0]})
    t = build_targets(cpu)
    assert t.loc[0, "active_vm_count"] == 1
    assert t.loc[0, "aggregate_cpu_utilization"] == 10
