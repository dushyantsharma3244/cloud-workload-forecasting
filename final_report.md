# Cloud Workload Forecasting: Final Report

## Executive summary

**Question.** How predictable is this cloud workload, how accurately can near-term demand be forecast, and what should an infrastructure team do with the forecast?

**Answer in five points.**

1. **The fleet is stable and lightly loaded.** The 10,000 VMs keep an average CPU of 9.39% (standard deviation 0.19 points, range 8.98–9.91%). The number of active VMs is constant at 10,000 in all 134 timestamps.
2. **Individual VMs differ much more than the fleet does.** About 49% of readings are idle (average CPU of 5% or less), and about 90% are below 20%. Yet every VM category contains VMs near saturation (up to 97–99%). Averaging over 10,000 VMs hides this tail.
3. **The selected forecast is persistence** (repeat the last observed value). Over rolling 24-step forecasts inside the training window it has an MAE of 0.23 and an RMSE of 0.29 CPU points. On the final 2-hour holdout it scores MAE **[FILL: holdout MAE]** and RMSE **[FILL: holdout RMSE]**. `active_vm_count` is forecast exactly, because it is constant.
4. **Tuned ML models did not clearly beat persistence.** When tuned without seeing the test block, an SVR improved MAE by only 3.3% over persistence (0.128 vs 0.132), which is within noise. Tuning on the test block made the same model look 17% better. This is a leakage trap, and it is why the final model is the simple one.
5. **Recommendation.** Provision for the forecast plus a safety margin of 0.41 CPU points (actual demand stayed at or below forecast + margin in 90% of past forecast points) or 0.50 points (95%). Monitor the hottest VMs and the Interactive category separately, because the fleet average cannot see them.

---

## 1. Data understanding and preprocessing

**Data.** A curated subset of the Azure Public Dataset V2 (`data-v1` release): VM-level CPU readings and VM metadata.

| Property | Finding |
|---|---|
| VMs | 10,000 unique `vm_id` values, identical in the readings and the metadata |
| Time coverage | 134 unique timestamps, five minutes (300 s) apart, about 11.1 hours |
| Readings | 1,340,000 (10,000 VMs × 134) **[CONFIRM: `cpu.shape[0]`]** |
| Duplicates | No fully duplicated rows; duplicates on (`vm_id`, `timestamp`): **[CONFIRM: expected 0]** |
| Missing values | Nulls in `avg_cpu`: **[CONFIRM: expected 0]** |
| Gaps in the time grid | Differences between consecutive timestamps: **[CONFIRM: all 300 s]** |
| Value range | `avg_cpu` runs from 0.007% to 99.49%, so nothing lies outside [0, 100] |

**VM lifetimes and churn.** Every VM has `timestamp_vm_created = 0`, so all VMs existed when the window began. The earliest deletion time is 40,500 s, after the window ends at 39,900 s **[CONFIRM: `cpu['timestamp'].max()`]**, so no VM was deleted inside the window. At least 75% of VMs carry a deletion time of 2,591,400 s (about 30 days, the length of the full upstream trace), which most likely means they were still running at the end of that trace. The fleet is therefore the same 10,000 VMs for the entire window. This follows from how the subset was built: only VMs with complete observations were kept, so **this data cannot show real VM churn.**

**Target construction** (`src/forecast/data.py`).

- `active_vm_count(t)`: number of distinct `vm_id` values at timestamp `t`. Each VM is counted at most once.
- `aggregate_cpu_utilization(t)`: mean of the VMs' `avg_cpu` at timestamp `t`, in percent.
- Treatment of problems: duplicate (`vm_id`, `timestamp`) rows are dropped, readings with missing or out-of-range `avg_cpu` are ignored, and no value is invented for a timestamp without valid readings. These rules are implemented and unit-tested. In this release they change nothing **[CONFIRM: rows affected = 0]**.
- The full target series is built first. Then it is split chronologically: the first 110 points are the training window and the last 24 points (33,000–39,900 s) are the holdout. No feature, threshold or model setting uses holdout values.

**`active_vm_count` is constant.** Its value is 10,000 at all 134 timestamps. It is exactly forecastable, so no model is fitted for it, and relative error metrics are undefined for it (the baseline error is zero).

---

## 2. Workload analysis

![Figure 1. Fleet-average CPU (%, vertical axis) against seconds since the start (horizontal axis), training window.](figures/fleet_cpu_over_time.png)

### 2.1 Fleet-level behaviour

The fleet-average CPU is 9.39% on average, with a standard deviation of 0.19 points and a range of 8.98–9.91% (computed over all 134 timestamps, for description only; no modelling choice uses it). It moves noisily around 9.4%, rises to about 9.9% for roughly 90 minutes in the middle of the window (about 16,000–21,500 s), and then declines to about 9.1–9.2% by the end of the training window. There is no steady long-term trend.

*Why it matters.* The whole fleet varies by less than one percentage point, so the signal is small compared with the noise. Forecast errors that look tiny in absolute terms are comparable to the series' own variation. With only 11 hours of data, the mid-window rise is a single event. It cannot be called a daily pattern.

### 2.2 How busy individual VMs are

| `avg_cpu` band | Share of readings |
|---|---|
| 0–5% (idle) | 49.05% |
| 5–20% | 41.19% |
| 20–70% | 8.20% |
| 70–95% | 0.99% |
| above 95% | about 0.57% (derived as the remainder) |

About 90% of readings are below 20%, and 0.88% exceed 90%. The maximum CPU (`max_cpu`) of a reading is much higher than its average (`avg_cpu`) in the histogram, so many VMs have short bursts inside a five-minute window that the average hides.

*Why it matters.* The fleet is heavily under-used: roughly 90% of CPU capacity sits idle on average. Capacity pressure comes from a small group of busy VMs, not from the fleet as a whole.

![Figure 2. Share of all 5-minute readings in each average-CPU band.](figures/cpu_bands.png)

### 2.3 Differences between VM categories

| Category | VMs (share) | Mean CPU | Median | 90th pct | 95th pct | Max |
|---|---|---|---|---|---|---|
| Delay-insensitive | 6,180 (61.8%) | 8.17 | 5.88 | 14.67 | 23.62 | 99.20 |
| Interactive | 3,091 (30.9%) | 11.44 | 6.08 | 28.71 | 46.77 | 98.73 |
| Unknown | 729 (7.3%) | 10.39 | 4.03 | 30.80 | 47.46 | 96.97 |

(Values are the VM-level average CPU in percent, from the metadata statistic.)

The middle of the distribution is similar in all categories (medians of 4–6%). The categories differ in the upper tail: at the 95th percentile, Interactive and Unknown are about twice Delay-insensitive. The higher means come from a small group of busy VMs.

![Figure 3. VM-level average CPU by category and percentile (metadata statistic).](figures/category_percentiles.png)

*Why it matters.* Interactive VMs are labelled as user-facing workloads in the source data, so running short of capacity is felt there first, and their tail is heavier than Delay-insensitive. A single fleet-wide margin treats them like the quiet Delay-insensitive VMs. Unknown has only 729 VMs, so its upper percentiles rest on few VMs.

### 2.4 Peak and low-demand periods

Peak moments are defined from the training window only: fleet CPU at or above the 90th percentile (9.67%). Low moments are at or below the 10th percentile (9.18%). The gap between them is only about 0.5 points. Peak timestamps: **[FILL: do the peak rows form one block around 16,000–21,500 s, or are they scattered?]**

---

## 3. Forecasting approach

**Targets.** `active_vm_count` (constant, forecast by persistence) and `aggregate_cpu_utilization` (the modelling effort goes here). The model works at the native five-minute cadence, so no resampling or mapping back is needed.

**Baselines.** Persistence (repeat the last known value 24 times) and the historical mean of the training window.

**ML candidates.** Random Forest, AdaBoost, XGBoost and SVR, all tuned with Optuna (TPE sampler, seed 42) to minimise cross-validated MAE.

**Features.** The six previous values of the target (30 minutes of history). Every feature is computed from values available at the forecast origin. Multi-step forecasts are recursive: each prediction is fed back as the newest lag. Calendar features are not used, because 11 hours of data cannot support hour-of-day or day-of-week effects.

**Final choice.** Persistence. It is the simplest model, has nothing to tune, cannot leak, and was not clearly beaten (Section 5).

**Reproducibility.** `python -m forecast.predict --data challenge_data --output predictions.csv` rebuilds the prediction file from the release data. The holdout is always the final 24 timestamps of the supplied data, so the pipeline does not depend on the size of the dataset. Tests check the output contract and that changing the holdout values does not change the predictions.

---

## 4. Validation methodology

- **No shuffling anywhere.** Validation uses `TimeSeriesSplit(n_splits=3, test_size=24)` on the training window. Each fold learns from the past and forecasts the next 24 points, like the production task. On the 104-row lag table, the folds are fitted on 32, 56 and 80 rows.
- **Two scoring modes.** Optuna scored one-step-ahead predictions (each row uses real past values). The real task is a 24-step recursive forecast, so the final comparison uses a practice block inside the training window: learn on rows 1–86, forecast rows 87–110.
- **Leakage controls.**
  1. The holdout (final 24 points) is not used for fitting, feature construction, tuning or model selection.
  2. The first practice test overlaps Optuna's last fold (rows 87–110), so the settings were partly chosen on the test block. A second, fair test re-tunes on rows 1–86 only and then scores rows 87–110 once.
  3. Peak thresholds and safety margins use training rows only.
- **Metrics.** MAE, RMSE and the relative improvement over the baseline. No percentage-error metric is used, so near-zero denominators do not arise.

---

## 5. Results versus baseline

**One-step-ahead cross-validation (Optuna's objective).** Persistence scores MAE 0.1176 (RMSE 0.1528). Tuned models scored between 0.111 and 0.141 depending on the search space. A linear SVR reached 0.111 in one search, and an RBF SVR reached about 0.139–0.141 in later ones. The result is not stable, because the best value moves with the search setup, and none of the runs gives a reliable gain.

**24-step recursive forecast on the practice block (rows 87–110).**

| Method | MAE | vs persistence |
|---|---|---|
| Persistence | 0.1324 | reference |
| Historical mean | 0.2105 | 59% worse |
| SVR, tuned on the same block (optimistic) | 0.1093 | 17% better |
| SVR, tuned on rows 1–86 only (fair) | 0.1280 | 3.3% better |

The mean forecast is 59% worse than persistence (equivalently, persistence has a 37% lower MAE). The SVR looks clearly better only when it is tuned on the block it is tested on. Once that leak is removed, the gain shrinks from 17% to 3.3%, within the noise of a single 24-point block. I therefore kept persistence.

**Final holdout (looked at once, after `predictions.csv` was saved).**

| Target | MAE | RMSE |
|---|---|---|
| `aggregate_cpu_utilization` (persistence) | **[FILL]** | **[FILL]** |
| `active_vm_count` (persistence) | 0 | 0 |

---

## 6. Error and failure analysis

Persistence was forecast from 57 origins (training rows 30 onward) for 24 steps each, which gives 1,368 forecast points in the training window.

| Measure | Value |
|---|---|
| MAE | 0.23 |
| RMSE | 0.29 |
| Mean error (actual − forecast) | −0.02 |
| Safety margin, 90% / 95% | 0.41 / 0.50 |

The mean error is close to zero, so persistence is not biased, and its mistakes go in both directions. RMSE is 1.26 times MAE, so no handful of extreme misses dominates. The MAE of 0.23 is larger than the single-block value of 0.13 in Section 5, because it averages over many origins and all 24 steps, including the later, harder ones.

**By moment type.**

| Moment type | MAE |
|---|---|
| Peak (actual at or above 9.67%) | 0.35 |
| Low (actual at or below 9.18%) | 0.36 |
| Normal | 0.19 |

Errors are about 1.8 times larger at peaks and lows. Part of this is by construction: peaks and lows are defined by the actual value, while persistence stays near the recent level, so the extremes naturally miss more. Direction of the error at peaks and lows: **[FILL: average error at peak / low / normal]**.

**By forecast horizon.** **[FILL: MAE at 5, 30, 60, 90 and 120 minutes ahead, and whether it grows]**

**Against the historical mean over the same origins.** **[FILL: % lower MAE]**

**Where it fails.** **[FILL: the worst forecast origins, their timestamps, and whether they come just before the mid-window rise or fall]** Persistence assumes the current level continues, so it cannot anticipate a level shift such as the mid-window rise, and it lags behind whenever demand moves.

---

## 7. Capacity-planning recommendation

**Policy.** Provision CPU capacity for `forecast + margin`, where the margin is the 90th percentile of the past errors (actual minus forecast) of the persistence forecast. Actual demand then stayed at or below `forecast + margin` in 90% of past forecast points.

| Margin | Value | Covers | Cost |
|---|---|---|---|
| 90% | +0.41 CPU points (about 4.4% of the 9.39% average) | 90% of past forecast points | Smaller, runs short about once in ten forecasts |
| 95% | +0.50 CPU points | 95% of past forecast points | Larger, runs short about once in twenty |

(Units are percentage points of fleet-average CPU utilisation.)

**Under- versus over-provisioning.** A smaller margin wastes less capacity but runs short more often, and users see slow service. A larger margin makes shortages rare but pays for more unused capacity. This fleet runs at about 9% of its CPU, so even the 95% margin is cheap in absolute terms. If a shortage costs more than extra headroom (assumed here for user-facing work), the 95% margin is the sensible default.

**Segment rule.** Interactive VMs are labelled as user-facing workloads in the source data and have a much heavier tail than Delay-insensitive, so use the 95% margin for them. Delay-insensitive work can wait, so the 90% margin is enough. This is a hypothesis based on Section 2.3. Margins per category were not estimated here (see future work).

**Watch the hottest VMs separately.** The fleet average cannot see individual VMs near saturation (up to 97–99% in every category). Track the top 5% of VMs by load on top of the fleet-wide margin.

**Check on the holdout (reported once).** Share of the 24 holdout steps with actual demand at or below `forecast + 0.41`: **[FILL]**.

---

## 8. Limitations and future work

**Limitations.**

- **Short history.** 11 hours is a single window. Daily or weekly seasonality cannot be assessed, and the mid-window rise is one event.
- **No real churn.** The subset keeps only VMs with complete observations, so `active_vm_count` is constant. Conclusions about fleet size do not transfer to a real fleet.
- **Small test.** The 24-step comparison between models rests on one block of 24 points. Rolling origins overlap, so their errors are correlated.
- **Tuning noise.** With about 100 training rows, Optuna's best score is optimistic, and results moved between runs and search spaces.
- **Pooled margins.** The safety margin is pooled over all 24 steps, so it ignores any difference in accuracy between early and late steps (see the horizon table in Section 6).
- **Metadata statistic.** The category percentiles come from the metadata's VM-level average, which may cover more than the 11-hour window.
- **Segments not tested.** Whether the mid-window rise comes from one category, and the core and memory segments, were not analysed. Unknown has only 729 VMs.

**Future work.**

- Test simple tuning-free forecasts such as an exponential decay to the mean, a moving average, or exponential smoothing.
- Forecast per category, and use lagged VM-level summaries (for example the share of busy VMs) as features.
- Estimate safety margins per category, and add prediction intervals and a margin that grows with the horizon.
- Use the full 30-day upstream trace to test daily and weekly seasonality and real churn.
