import numpy as np
import pandas as pd
from forecast.data import build_targets
from forecast.predict import make_predictions, validate, H


def fake_cpu(n_vm=5, n_ts=40, seed=0):
    rng = np.random.default_rng(seed)
    rows = [(t * 300, f"vm{v}", rng.uniform(0, 30)) for t in range(n_ts) for v in range(n_vm)]
    return pd.DataFrame(rows, columns=["timestamp", "vm_id", "avg_cpu"])


def test_output_follows_contract_on_small_data():
    preds = make_predictions(build_targets(fake_cpu()))
    validate(preds)
    assert len(preds) == H


def test_holdout_values_are_not_used():
    cpu = fake_cpu()
    changed = cpu.copy()
    changed.loc[changed["timestamp"] >= (40 - H) * 300, "avg_cpu"] = 99.0   # alter the holdout
    a = make_predictions(build_targets(cpu))
    b = make_predictions(build_targets(changed))
    pd.testing.assert_frame_equal(a, b)


def test_holdout_timestamps_come_from_raw_data_even_if_readings_invalid():
    cpu = fake_cpu()
    last_ts = cpu["timestamp"].max()
    cpu.loc[cpu["timestamp"] == last_ts, "avg_cpu"] = float("nan")   # final timestamp has no valid reading
    preds = make_predictions(build_targets(cpu))
    validate(preds)
    assert preds["timestamp"].iloc[-1] == last_ts
    assert preds["timestamp"].tolist() == sorted(cpu["timestamp"].unique())[-H:]


def test_cli_end_to_end_on_tiny_dataset():
    import os, sys, tempfile
    from forecast.predict import main
    with tempfile.TemporaryDirectory() as d:
        fake_cpu(n_vm=3, n_ts=30).to_csv(os.path.join(d, "vm_cpu_readings.csv.gz"), index=False)
        out = os.path.join(d, "predictions.csv")
        old = sys.argv
        sys.argv = ["predict", "--data", d, "--output", out]
        try:
            main()
        finally:
            sys.argv = old
        p = pd.read_csv(out)
        assert list(p.columns) == ["timestamp", "active_vm_count", "aggregate_cpu_utilization"]
        assert len(p) == 24
