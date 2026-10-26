# Cloud workload forecasting

Forecasts `active_vm_count` and `aggregate_cpu_utilization` for the final two hours
(24 five-minute timestamps) of the Azure VM trace subset.

## Install
    python -m venv .venv && source .venv/bin/activate
    pip install -r requirements.txt
    pip install -e .

## Data
Extract the `data-v1` release so that `challenge_data/` contains
`vm_cpu_readings.csv.gz`, `vm_metadata.csv.gz`, `DATA_CARD.md` and `SHA256SUMS`.
The data is not committed.

## Generate predictions
    python -m forecast.predict --data challenge_data --output predictions.csv

## Tests
    pytest

## Main modeling decisions
- Targets are built per timestamp (each VM counted once; invalid avg_cpu readings ignored).
- The last 24 timestamps are the holdout. Their target values are never read.
- Final model: persistence (repeat the last training value). A tuned SVR and tree/boosting
  models (Optuna, time-ordered CV) did not beat it by a meaningful margin, see the report.
- `active_vm_count` is constant (10,000) in this subset, so persistence is exact.
- Notebooks in `notebooks/` hold the analysis; the report is in `reports/`.
