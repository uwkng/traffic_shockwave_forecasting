# When Aggregate Error Misleads: Evaluating Event- and Weather-Aware Traffic Forecasting on Externally Defined Disruption Windows

Course project for *Deep Learning and Decision Making*, Technical University of Munich.

## Motivation

Standard traffic forecasting benchmarks report a single aggregate error across all timesteps. This hides performance in the conditions that matter most: event egress hours, adverse weather spells, and congestion shockwaves. The contribution of this project is the evaluation protocol, not a new model. We take an off-the-shelf STGCN (Yu et al., IJCAI 2018), condition it on weather and scheduled-event signals, and evaluate it specifically inside exogenously-defined anomalous windows.

## Data

| Source | Description |
|---|---|
| [PEMS-BAY](https://github.com/liyaguang/DCRNN) | 325 freeway sensors, San Francisco Bay Area, Jan to Jun 2017, 5-min intervals |
| ASOS (SJC + NUQ) | Precipitation and temperature, IDW-merged to sensor locations |
| ESPN / venue calendars | 95 scheduled events across 4 venues (NHL, MLB, MLS, concerts) |

The data pipeline assembles an 11-channel tensor `[52116, 325, 11]` combining speed, occupancy, time-of-day, weather, holiday, and event features. Channel selection per ablation rung is defined in `src/contract.py`.

## Method

**Model.** STGCN with parameterised `c_in`, allowing the same architecture for every ablation rung (1 to 11 input channels).

**Splits.** Rolling cross-validation with 28-day test blocks. Folds 0 to 2 cover the wet season and event-heavy months. A single 70/10/20 split is available for reproducing published baselines.

**Training.** 3 seeds per configuration, masked MAE loss on de-normalised speed (mph).

**Evaluation.** Metrics stratified by event phase (ingress, in-play, egress), weather condition, distance to venue, and shockwave regime.

## Project Structure

```
configs/
    default.yaml                Configuration: paths, thresholds, split parameters

src/
    contract.py                 Data contract: shapes, channel order, prediction format
    data/                       Stages 1 to 7: acquire, align, features, samples, split, normalise

    models/
        stgcn.py                STGCN architecture (symmetrised Laplacian)
        train.py                Training loop (rolling folds, multi-seed)
        baselines.py            Trivial baselines (historical mean, last-observed)
        loader.py               Builds DataLoaders from processed tensors
        single_split/           Single-split variant with target-window covariates

    eval/
        windows.py              Exogenous window definitions
        metrics.py              MAE, RMSE, MAPE (masked, on de-normalised mph)
        report.py               Stratified evaluation reports
        decision.py             Decision layer
        single_split/           Evaluator for the single-split variant

scripts/                        Experiment runners and result collection
results/                        Per-fold reports, main results table, figures
notebooks/                      Vanilla STGCN reproduction (stage 8)
```

## Reproducing Results

```bash
# Install dependencies
pip install numpy pandas pyyaml requests scipy torch

# Build the dataset (first run downloads PEMS-BAY, ~233 MB)
python -m src.data.build_dataset --config configs/default.yaml --acquire

# Train all configurations (requires GPU)
bash scripts/run_experiments.sh

# Generate evaluation reports
bash scripts/collect_results.sh
```

Results are written to `results/results_main.csv`. Column definitions are in `results/results_main_COLUMNS.csv`, and `results/README.md` explains the experimental setup and how to read the numbers.

## Key Design Decisions

1. **Chronological splits only.** No shuffling, to prevent future leakage.
2. **Scaler fitted on training data only**, per channel and per fold.
3. **All metrics computed after de-normalisation** in real mph.
4. **Road distance** (not great-circle) for venue proximity. This resolves opposite-carriageway ambiguity and raises the event effect from non-significant to t = −3.37.
5. **Rolling folds restricted to 0 through 2.** Folds 3 to 5 contain zero rain and zero events, so they cannot inform the research questions.

## Authors

Yilang Mei, Raouf Tadros, Uwe König
