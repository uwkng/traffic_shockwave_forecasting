# Project Status

Last updated: 2026-09-01

## Pipeline Progress

| # | Stage | File | Status | Owner |
|---|-------|------|--------|-------|
| 1 | Acquire raw sources (PEMS-BAY, weather, events) | `src/data/acquire.py` | DONE | A |
| 2-3 | Align to 5-min index + graph + distance-to-venue | `src/data/align.py` | TODO | A |
| 4 | Assemble 9-channel tensor `[T,N,9]` | `src/data/features.py` | TODO | A |
| 5 | Slice samples X`[12,N,9]` Y`[12,N,1]` | `src/data/samples.py` | TODO | A |
| 6 | Chronological 70/10/20 split | `src/data/split.py` | TODO | A |
| 7 | Z-score (train-fit only), save scaler | `src/data/normalize.py` | TODO | A |
| - | Orchestrator for stages 1-7 | `src/data/build_dataset.py` | TODO | A |
| 8 | Reproduce vanilla STGCN on plain PEMS-BAY | `src/models/` | DONE | B |
| 9 | STGCN with 9-channel input | `src/models/` | TODO | B |
| 10 | Uniform prediction I/O | `src/models/predict.py` | TODO | B |
| 11 | Metrics, windows, distance bins, onset, SEPA | `src/eval/` | TODO | C |
| 12 | Decision layer | `src/eval/decision.py` | TODO | C |

## Ownership

- **A** — Data & features (stages 1-8)
- **B** — Models & training (stages 9-10)
- **C** — Evaluation & figures (stages 11-12) ← the contribution

## What's done

- **Stage 1 — Acquire** (2026-08-27)
  - PEMS-BAY: 325 sensors × 52,116 timesteps (Jan–Jun 2017), cloned from Augmented-PEMS-BAY repo
  - Weather: 4,344 hourly rows from Open-Meteo API (temp °C, precip mm, wind km/h), single centroid
  - Events: 149 games across 5 teams (A's, Giants, Sharks, Warriors incl. playoffs, Earthquakes), exact start times sourced from MLB/NHL/NBA APIs
  - Config: `configs/default.yaml` created
  - Data contract: `src/contract.py` locked

## Decisions log

Record choices here so they don't get lost in chat.

| Date | Decision | Rationale |
|------|----------|-----------|
| 2026-08-27 | Single weather centroid (37.34, -121.94) for all sensors | Network spans ~20 km; one station is sufficient at this scale |
| 2026-08-27 | Events limited to 5 Bay Area pro sports teams | These are the major scheduled events with known attendance near the sensor network |
| 2026-08-27 | Event start times from APIs, not approximated | Approximate times can be off by hours (e.g. Sunday matinees), which misaligns the event signal with traffic |
| 2026-09-01 | Build adjacency from distances CSV, not pkl | adj_mx_bay.pkl has Python 2 pickle incompatibility; distances CSV + Gaussian kernel is cleaner |
| 2026-09-01 | NaN in speed.csv (521 values, 4 timesteps): linear interpolation + ffill/bfill | Standard approach; tiny fraction (<0.003%) of data |
| 2026-09-01 | STGCN architecture: 2 blocks [1,64,64],[64,64,64], Kt=3, Ks=3 | Faithful to Yu et al. 2018; sufficient capacity for 325-node graph |

- **Stage 8 — Vanilla STGCN reproduction** (2026-09-01)
  - `src/models/stgcn.py`: Full STGCN (ChebConv K=3, GLU temporal conv Kt=3, 2 ST-Conv blocks [1→64→64, 64→64→64], ~117k params)
  - `src/models/vanilla_loader.py`: Standalone loader for plain PEMS-BAY (1 channel, speed only); builds adjacency from distances CSV (Gaussian kernel, threshold 0.1); z-score on train only; 36,465 / 5,209 / 10,419 samples (train/val/test)
  - `src/models/train.py`: Training loop with MAE loss, Adam, StepLR, early stopping, MAE/RMSE/MAPE at 15/30/60 min, multi-seed support
  - `notebooks/train_stgcn.ipynb`: Self-contained notebook for GPU training (used A100)
  - Trained 3 seeds, results consistent with Graph WaveNet's STGCN baseline (Wu et al., 2019):
    - 15 min: MAE=1.44±0.00, RMSE=3.06±0.01, MAPE=3.03±0.01%
    - 30 min: MAE=1.93±0.00, RMSE=4.36±0.00, MAPE=4.35±0.02%
    - 60 min: MAE=2.58±0.00, RMSE=5.83±0.01, MAPE=6.25±0.05%

## Open questions

- [ ] Who fills owner A / B / C roles? Update CLAUDE.md and this file once assigned.
- [ ] DCRNN: attempt or skip? (CLAUDE.md says optional and time-boxed)

## Next steps

1. **Stage 2-3 (align.py)**: Resample weather from hourly → 5-min, map events onto the 5-min index, build adjacency matrix, compute dist_to_venue per sensor
2. **Stage 9 (9-channel STGCN)**: Change first block config `[1,64,64]` → `[9,64,64]`; blocked on stages 2-7 producing the full tensor
3. **Stage 10 (predict.py)**: Uniform prediction I/O so eval treats all models interchangeably

## How to update this file

When you finish a stage or make a decision that affects others:
1. Move the stage status from TODO → IN PROGRESS → DONE
2. Add a bullet under "What's done" with what changed
3. Log any cross-cutting decisions in the decisions table
