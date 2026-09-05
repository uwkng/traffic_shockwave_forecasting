# Project Status

Last updated: 2026-09-04

## Pipeline Progress

| # | Stage | File | Status | Owner |
|---|-------|------|--------|-------|
| 1 | Acquire raw sources (PEMS-BAY, weather, events) | `src/data/acquire.py` | DONE | A |
| 2-3 | Align to 5-min index + graph + distance-to-venue | `src/data/align.py` | DONE | A |
| 4 | Assemble 11-channel tensor `[T,N,11]` | `src/data/features.py` | DONE | A |
| 5 | Slice samples X`[12,N,C]` Y`[12,N,1]` | `src/data/samples.py` | DONE | A |
| 6 | Chronological split (single + 6 rolling folds) | `src/data/split.py` | DONE | A |
| 7 | Z-score per channel (train-fit only), save scaler | `src/data/normalize.py` | DONE | A |
| - | Orchestrator for stages 1-7 | `src/data/build_dataset.py` | DONE | A |
| 8 | Reproduce vanilla STGCN on plain PEMS-BAY | `src/models/` | DONE | B |
| - | `data/processed` -> model input | `src/models/loader.py` | DONE | A |
| 9 | STGCN with multi-channel input | `src/models/stgcn.py` | **BLOCKED** | B |
| 10 | Uniform prediction I/O | `src/models/predict.py` | TODO | B |
| 11 | Exogenous window definitions | `src/eval/windows.py` | DONE | A |
| 11 | Metrics, distance bins, onset, SEPA | `src/eval/` | TODO | C |
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

- **Stages 2-7 — full pipeline** (2026-09-04)
  - `python -m src.data.build_dataset --config configs/default.yaml` runs stages 2-7
    end to end in **14.5 s** and writes 1,053 MB to `data/processed/`.
    Add `--acquire` on a fresh clone to pull PEMS-BAY and the weather first.
  - **Master tensor `[52116, 325, 11]`**, channel order fixed by `src/contract.py`:

    | # | channel | group | # | channel | group |
    |---|---------|-------|---|---------|-------|
    | 0 | speed (TARGET) | traffic | 6 | precipitation | weather |
    | 1 | occupancy | traffic | 7 | wind_speed | weather |
    | 2 | time_of_day | calendar | 8 | dist_to_venue | event_geo |
    | 3 | day_of_week | calendar | 9 | event_active_decay | event_geo |
    | 4 | is_holiday | calendar | 10 | event_load | event_att |
    | 5 | temperature | weather | | | |

  - `contract.ABLATION_RUNGS` + `contract.rung_channels(rung)` give the column
    indices for each rung of the agreed ladder. **Build once, slice at the
    dataloader** - never rebuild the tensor per rung, or the splits and scalers
    stop being identical across rungs and the MAE numbers stop being comparable.

        0_speed         c_in= 1   [0]            <- the published configuration
        1_traffic       c_in= 2   [0,1]          <- 0 -> 1 IS occupancy's value
        2_calendar      c_in= 5   [0,1,2,3,4]
        3_weather       c_in= 5   [0,1,5,6,7]
        4_event_geo     c_in= 4   [0,1,8,9]
        5_event_geo_att c_in= 5   [0,1,8,9,10]
        6_all           c_in=11   [0..10]

    Rung 0 is speed alone, not speed+occupancy. It is the configuration every
    published PEMS-BAY number uses, so it reads against the literature directly;
    and it makes occupancy visible, which matters because occupancy is worth
    +0.153 incremental R2 - the second strongest channel measured, about a
    hundred times the whole weather group. Bundled into the baseline that
    finding cannot be seen.

  - **Splits**: `single` (70/10/20, for reproducing published numbers ONLY) and
    six expanding-window rolling folds. Counted in EPISODES, which is the
    paper's n - not timesteps, which are heavily autocorrelated:

        fold    test block                 rain  events  holidays
        fold00  2017-01-29 .. 02-26         27      6       1
        fold01  2017-02-26 .. 03-26         12     11       0
        fold02  2017-03-26 .. 04-23         12     11       0
        fold03  2017-04-23 .. 05-21          0      5       0
        fold04  2017-05-21 .. 06-18          0     12       1
        fold05  2017-06-18 .. 06-30          0      5       0
        TOTAL                               51     50       2
        single  2017-05-25 .. 06-30          0     17       1

    **The single test block contains ZERO adverse-weather episodes.** A
    weather-conditioned model cannot differ from a traffic-only one on it - the
    channel is constant across the whole block. Any weather result must come
    from rolling.

  - **Scalers are per channel and per split**, fit on that split's train span
    only. `time_of_day` / `day_of_week` / `is_holiday` pass through unscaled.
    Speed mean ranges 62.42 - 62.74 mph across folds; one global scaler would
    leak later-period statistics into every early fold. The `single` scaler
    (62.737 / 9.435 mph) matches the stage-8 notebook's (62.74 / 9.44),
    which is an independent check that the two loaders agree.
  - Acceptance checks, all passing: master shape and finiteness; `time_index`
    equals `speed.csv`'s columns element-wise; the DST gap is absent; channel 0
    equals raw `speed.csv` wherever speed was observed; all seven splits are
    chronological, disjoint and 24-sample embargoed; passthrough channels are
    identity; every split has its own scaler.

- **Stage 1 — weather repaired** (2026-09-04)
  - `configs/default.yaml` pointed at `weather_hourly.csv` while
    `acquire.build_weather()` writes `weather_5min.csv`, and a stale Open-Meteo
    export sat at the configured path. The file existed, so nothing raised: the
    pipeline silently read hourly reanalysis instead of the 5-min ASOS merge.
    Renamed to `weather_openmeteo_OLD_do_not_use.csv`, config repointed, and the
    real ASOS data (SJC 4,831 obs, NUQ 5,037 obs) committed so the pipeline no
    longer depends on an un-run fetch.
  - `venues.csv` now holds only the four venues that contribute an event.
    Coordinates cross-checked against a second, independent source (Wikidata as
    well as OpenStreetMap); worst disagreement 54 m against a 500 m tolerance.

## Handover: picking up stages 9-12

### What is blocking

**`src/models/` has no model in it.** The STGCN definition, the training loop
and the metrics live inline in `notebooks/train_stgcn.ipynb`, cells 5 / 7 / 11 /
13, so nothing downstream can import them. Stages 10 and 11 wait on:

```
src/models/stgcn.py     cell 7 as-is, with c_in a parameter instead of 1
src/models/train.py     cells 11 + 13, taking --rung and --split
src/models/predict.py   write [n_test, 12, 325] de-normalized mph
```

`src/models/loader.py` already exists and hands you a DataLoader:

```python
loaders, adj, scaler = build_loaders(rung="6_all", split="fold00")
model = STGCN(c_in=loaders["c_in"], ...)
```

### The ablation ladder

`contract.ABLATION_RUNGS`. Build the master tensor ONCE and select columns per
rung — rebuilding per rung gives each rung its own splits and scalers, and the
MAEs stop being comparable.

| rung | `c_in` | adds | note |
|---|---:|---|---|
| `0_speed` | 1 | — | the published PEMS-BAY configuration |
| `1_traffic` | 2 | occupancy | 0 -> 1 IS occupancy's value: +0.153 incremental R2 |
| `2_calendar` | 5 | time_of_day, day_of_week, is_holiday | |
| `3_weather` | 5 | temperature, precipitation, wind_speed | **rolling folds only** |
| `4_event_geo` | 4 | dist_to_venue, event_active_decay | expected to be FLAT — see below |
| `5_event_geo_att` | 5 | + event_load | 4 -> 5 isolates attendance |
| `6_all` | 11 | everything | |

Rung 4 measuring flat is the result, not a bug. Inside the egress hour, over
8,649 (event, node) pairs from 75 events: activity alone scores t = -1.02 (not
significant), the thresholded attendance weight scores t = -9.76. The
information is in the attendance, not in the fact that an event is on.

`build_loaders(..., future_covariates=True)` additionally feeds the exogenous
state of the window being predicted — the forecast and the fixture list, which
in reality are known days ahead. **Off by default**, because it changes the task
and no published baseline has that information. Worth running both ways and
reporting the difference: of samples whose target window contains adverse
weather, 22.4% have an entirely dry input window at a 60-min horizon, so the
model is blind there, not merely wrong.

### Splits

Both schemes are in `splits.npz`. Counts are EPISODES — one rain spell, one
egress hour — which is the paper's n; 5-min steps and neighbouring sensors are
heavily correlated, so a standard error over timesteps understates it by roughly
`sqrt(n_timesteps / n_episodes)`.

| split | train | val | test | test period | rain | events | holidays |
|---|---:|---:|---:|---|---:|---:|---:|
| `single` | 36,449 | 5,207 | 10,414 | 05-25 → 06-30 | **0** | 17 | 1 |
| `fold00` | 6,048 | 2,016 | 8,064 | 01-29 → 02-26 | 27 | 6 | 1 |
| `fold01` | 14,112 | 2,016 | 8,029 | 02-26 → 03-26 | 12 | 11 | 0 |
| `fold02` | 22,141 | 2,016 | 8,064 | 03-26 → 04-23 | 12 | 11 | 0 |
| `fold03` | 30,205 | 2,016 | 8,064 | 04-23 → 05-21 | 0 | 5 | 0 |
| `fold04` | 38,269 | 2,016 | 8,064 | 05-21 → 06-18 | 0 | 12 | 1 |
| `fold05` | 46,333 | 2,016 | 3,721 | 06-18 → 06-30 | 0 | 5 | 0 |
| | | | | **rolling total** | **51** | **50** | **2** |

**All 90 rain spells in the period fall inside `single`'s train span.** That is
climate, not a split ratio: California's wet season is January to April, so any
chronological split puts val and test in the dry half. No weather result can
come from `single`.

Holidays are n = 2 across the rolling test folds. Report them as cases
("Memorial Day, +11.27 mph in the PM peak"), never with an error bar.

`embargo_steps: 0`, matching the notebook and every published PEMS-BAY baseline.
That leaves up to 46 timesteps of overlap at a boundary, which `split.py` prints
rather than asserts away. Train and test are separated by the whole of val, so
no training gradient sees a test target; what it touches is early stopping. Set
`embargo_steps: 24` to remove it.

### Evaluation windows

`src/eval/windows.py`. Effects below are the speed anomaly against the same
sensor at the same time-of-week on days with no event and no rain.

| window | rule | effect |
|---|---|---|
| adverse weather | `precipitation >= 0.51 mm` | **-2.22 mph** |
| event egress | `[end_time, end_time+60min)`, attendance >= 15,000 | **-1.22 mph** (NHL, <=2 km) |
| holiday | 5 US federal holidays | **+9.50 / +11.27 mph** (AM / PM peak) |

- **Two weather tiers, not three.** `p01i` is a backward 1-hour accumulation and
  `_parse_asos` forward-fills it across the next twelve 5-min steps. That
  inflates "light rain" from 4.6% of timesteps (METAR present-weather) to 16.3%
  and dilutes its effect from -1.72 to -0.78 mph. For the same reason the
  precipitation column **cannot be summed** — each hourly reading appears twelve
  times.
- **The egress hour, not the fixture.** A doors-to-end window spans ~5.5 h,
  dilutes the dip fivefold and reads *+0.41* mph. Aligned on scheduled start over
  24 SAP Center evening NHL games, sensors within 2 km, the dip is
  **-1.69 ± 0.51 mph at +180 min** and recovers by +270.
- **Holidays are positive** — a holiday removes the commute rather than adding a
  crowd. Report as a stratum, never merge into "adverse".

Every `events.csv` column stays available for stratified reporting even though
only four enter the tensor. Stratification happens at evaluation time, indexed
by timestep: `event_type`, `confidence`, `attendance_is_estimated` and the rest
are report dimensions, not model inputs. Measured example — egress effect by
whether the attendance figure was sourced or assumed: sourced (n=41)
**-0.51 ± 0.24**, assumed (n=54) **+0.29 ± 0.22**. That contrast is confounded
with attendance size and must be reported as such.

### Ideas measured but not implemented

- **Longer horizon.** `HORIZON` is 12. Setting it to 36 costs ~1,500 parameters
  and no measurable time — horizon only sizes the final `Linear(64, HORIZON)`,
  while the ST-Conv blocks work on `INPUT_WINDOW`. One model would then emit
  15/30/60/120/180 as slices. Worth doing because AR(12) scores R2 0.567 at
  60 min but 0.147 at 3 h: at 60 min speed history already explains most of the
  variance, which is why the exogenous channels measure as worth so little
  there. See the note in `contract.py`.
- **Per-node weather.** Rejected: IDW to each sensor is worth +0.00008
  incremental R2 over one network-wide series, despite the two stations
  disagreeing in 42.7% of wet hours.
- **`rain_intensity`** (METAR present-weather ladder). Rejected: needs
  `wxcodes`, which `acquire.fetch_asos` does not request.
- **A time-of-week climatology as a channel.** Rejected: it is the proposal's
  own historical-average baseline (MAE 2.63 at 60 min against the trained
  STGCN's 2.58), and folding the baseline into the model destroys that
  comparison. It is the strongest single predictor measured (+0.216 R2), which
  is itself the finding.


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
| 2026-09-04 | Weather stays as the ASOS/IDW implementation on `main`; only the config path and the machine's CA bundle were fixed | Per-node IDW is worth +0.00008 incremental R2 and all three precipitation parameterisations put the adverse-weather slowdown at the same -3.1 mph. Not worth a rewrite before the deadline |
| 2026-09-04 | Two weather tiers (adverse / normal at 0.51 mm), not three | `p01i` is a backward 1-hour accumulation and `_parse_asos` forward-fills it, inflating "light rain" from 4.6% of timesteps to 16.3% and diluting its effect from -1.72 to -0.78 mph. A light tier read off this series understates the effect for a data-processing reason |
| 2026-09-04 | `event_load` split into `event_active_decay` (no attendance) + `event_load` (thresholded, attendance-weighted) | The ablation ladder needs rungs 4 and 5 to differ. Measured in the egress hour over 8,649 (event,node) pairs: activity alone t=-1.02 (not significant), thresholded attendance t=-9.76. The information is in the attendance |
| 2026-09-04 | 15,000 attendance floor, applied to `event_load` only | Below it there is no egress effect: >=40k -0.74+-0.56, 15k-40k -0.45+-0.22, <15k +0.35+-0.41 (wrong sign). Putting the floor in both channels would leak attendance into the activity-only rung |
| 2026-09-04 | `is_holiday` added as a channel | Largest exogenous effect in the project: Memorial Day +9.50 mph in the AM peak, +11.27 in the PM, against -4.01 for heavy rain. Sign is POSITIVE, so it is reported as a stratum, never merged into "adverse" |
| 2026-09-04 | `rain_intensity` (METAR ladder) dropped | `acquire.fetch_asos` does not request `wxcodes`, so it cannot be built from what this repo downloads. Re-downloading buys nothing on the headline number |
| 2026-09-04 | Node-by-time-of-week climatology NOT a channel | It is the proposal's own historical-average baseline (MAE 2.63 at 60 min against the trained STGCN's 2.58). Folding the baseline into the model destroys that comparison |
| 2026-09-04 | Rolling: 28-day warm-up, 28-day blocks, 6 folds (was 42/14/10) | Fold count does not change how much signal reaches test, only how it is partitioned. 28-day blocks give a median of 10 rain spells per fold against 4 for 14-day. min_train 28 rather than 42 puts 51 of the 90 rain spells into test instead of 37, because January is the wettest month |
| 2026-09-04 | Weekends and holidays KEPT, against Yu et al. | The paper drops them "to eliminate atypical traffic". 46 of 95 events (48%) fall on a weekend; that exclusion removes precisely the windows this project evaluates |

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
2. **Stage 9 (multi-channel STGCN)**: set the first block's `c_in` from `contract.rung_channels(rung)` - it is 1, 2, 4, 5 or 11 depending on the ablation rung, NOT a fixed number. Stages 2-7 now produce the tensor, so this is unblocked. NOTE `src/models/` does not exist on any branch: the model and training loop live inline in `notebooks/train_stgcn.ipynb` cells 5/7/11/13 and need extracting before stages 10-11 can import them.
3. **Stage 10 (predict.py)**: Uniform prediction I/O so eval treats all models interchangeably

## How to update this file

When you finish a stage or make a decision that affects others:
1. Move the stage status from TODO → IN PROGRESS → DONE
2. Add a bullet under "What's done" with what changed
3. Log any cross-cutting decisions in the decisions table
