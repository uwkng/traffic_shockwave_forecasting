# Project Status

Last updated: 2026-09-06

## READ THIS FIRST: the run is done, here is where everything is

The full experiment finished on an H200 on 2026-09-05. **81 trained
configurations** = 9 variants x 3 rolling folds x 3 seeds.

| you want | look at |
|---|---|
| **what we did and what the numbers mean** | **`results/README.md`** - start here |
| **the one results table** | **`results/results_main.csv`** - opens in Excel |
| what each column of that table means | `results/results_main_COLUMNS.csv` |
| per-fold numbers with standard deviations | `results/report__fold0*.{json,md}` |
| alarm counts and the cost sweep | `results/decision__fold0*.json` |
| the tables that needed no GPU | `notes/paper_tables.md` |
| the 87 prediction files (8.1 GB) | `results_bundle_v2_diffusion.tgz`, NOT in git - ask A |
| **the write-up** | **`paper/main.pdf`**, source in `paper/main.tex` |

Start with `results/README.md`. It explains the problem, the configurations and
every column of `results_main.csv` in plain language, and it names the json field
behind each number.

Training and testing use three consecutive 28-day blocks (29 Jan - 23 Apr). The
period after April is dry - zero rain spells - so it cannot inform the weather
question and is not used. That selection uses only the calendar and the weather
record, both known before any model was trained; `data/processed/splits_meta.json`
records the counts.

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
| 9 | STGCN, `c_in` parameterised, Chebyshev or diffusion operator | `src/models/stgcn.py` | DONE | A |
| 10 | Training loop, weighted loss, trivial baselines | `src/models/{train,baselines}.py` | DONE | A |
| 11 | Exogenous window definitions | `src/eval/windows.py` | DONE | A |
| 11 | Metrics, windows, propagation, stratified report | `src/eval/{metrics,report}.py` | DONE | A |
| 12 | Decision layer (confusion + cost sweep) | `src/eval/decision.py` | DONE | A |

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

### Start here

```bash
git fetch && git checkout mei_modeling
pip install numpy pandas pyyaml requests        # pipeline
pip install torch scipy                         # training only

# first run also clones PEMS-BAY (233 MB) and downloads the weather (~5 min)
python -m src.data.build_dataset --config configs/default.yaml --acquire

# rebuilds take ~15 s afterwards
python -m src.data.build_dataset --config configs/default.yaml

# does it look right? prints every rung, every split, and one real sample
python -m src.models.loader --split single
```

On macOS, `SSL: CERTIFICATE_VERIFY_FAILED` means your Python has no CA bundle -
the usual python.org-installer situation. Run
`open "/Applications/Python 3.x/Install Certificates.command"` once. It is a
machine problem; do not patch `acquire.py`.

**Nothing large is committed.** The repo is ~4 MB and you rebuild the ~1 GB of
tensors locally. Do not commit `data/processed/`.

### Where the results are

`data/processed/`, after the build:

| File | Shape | What |
|---|---|---|
| `master.npy` | `[52116, 325, 11]` | the whole period, every channel. 745 MB; open with `mmap_mode="r"` |
| `splits.npz` | 21 arrays | sample indices: `single__train`, `fold00__test`, … |
| `scalers.json` | — | per-channel mean/std, one set per split |
| `eval_mask.npy` | `[52116, 325]` | True where speed was observed, not imputed |
| `adj_mx.npy` | `[325, 325]` | Gaussian-kernel adjacency, threshold 0.1 |
| `align_meta.json`, `features_meta.json`, `samples_meta.json`, `splits_meta.json` | — | per-stage provenance, human-readable — open these to see what each stage did |

`master.npy` is **not** split. It is the full six months; `splits.npz` indexes
into it. Same tensor, different bookmarks. A sample id `t` means input
`master[t:t+12]`, target `master[t+12:t+24, :, 0]` - so one id addresses 24
timesteps, not one.

### How to use it

```python
from src.models.loader import build_loaders

loaders, adj, scaler = build_loaders(rung="6_all", split="fold00")
model = STGCN(c_in=loaders["c_in"], ...)          # c_in is 1, 2, 4, 5 or 11

for X, Y in loaders["train"]:                     # X normalised, Y raw mph
    ...

pred_mph = scaler.to_mph(pred)                    # ALWAYS before a metric
mask = loaders["eval_mask"]                       # exclude imputed values
```

`rung` picks channels, `split` picks bookmarks. Nothing else changes between
configurations.

The loader owns seven things that are easy to get wrong and that never fail
loudly - they only move the MAE: which columns, which split's scaler, per
channel rather than global, passthrough channels, the raw target,
de-normalisation, the eval mask. Its docstring explains each one. Please do not
reimplement them.

Without torch installed, `build_loaders` returns the raw `WindowDataset` instead
of a `DataLoader`, so the historical-average and persistence baselines can share
the same splits and de-normalisation without a framework.

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
| 2026-09-05 | Predict speed ONLY; the shockwave label drops the occupancy term | A second occupancy head was proposed so the evaluation could score an occupancy-based label without leaking future ground truth. Measured, the speed-only label reaches every conclusion: rain no-effect either way; events 2.50x z=3.24 against 5.25x z=3.66; upstream propagation 1.71x against 1.88x. The occupancy label is a strict SUBSET of the speed-only one (all 90,220 cells inside the 135,187). So the label lost occupancy instead of the model gaining an output - which keeps rung 0 comparable to every published PEMS-BAY number, all single-channel speed. Occupancy stays the most valuable INPUT channel (+0.220 incremental R2 in commute hours). |
| 2026-09-05 | Train folds 00-02 only, not all six | Selected on test-block content, known before any model trains: folds 3-5 contain zero rain episodes and zero NHL fixtures while carrying 73% of the compute (114,807 of 157,108 train samples). Stated as a method, not a result. |
| 2026-09-05 | Do NOT headline "rain x event" compound performance | It occurs 3 times in six months, 18 timesteps, one of them a single step. Reported as a negative result; `weather_commute` (31 episodes, 19 days) is the compound window that exists. |
| 2026-09-05 | Keep precipitation forward-filled at full magnitude; fix the LABEL, not the data | `p01i` is a backward 1-hour accumulation. Forward-filling is causal (the value at t describes [t-60, t], all past) and shifting the flag later weakens the measured effect, so the current alignment is right. Dividing by 12 changes nothing - the threshold is a percentile of the same series and the model z-scores it. Leaving off-hour steps empty DOES break things: 104 contiguous rain episodes become 213 isolated points and there is no window left to define. Call the column "1-hour accumulation (mm), forward-filled" and never sum it. |
| 2026-09-07 | REVERSED the 2026-09-05 row below: `--future-covariates` implemented and run, 45 models on `mei_modeling_on` | The blocker was the train/serve mismatch, and it dissolved once a 2017 forecast archive was found: `acquire.fetch_weather_forecast()` pulls the ECMWF IFS short-range archive (Open-Meteo Historical Forecast API), which is what was PREDICTED at the time, not a reanalysis. Validated against the observed ASOS series: 94.9% hourly agreement, **61.4% recall on wet hours**. `loader.build_loaders` raises rather than falling back to observed weather if that file is absent, so the leakage path is closed by construction. Result: pooled over 15 rung x fold pairs lookahead is HARMFUL at every horizon (+0.005/+0.009/+0.015/+0.017 mph, paired t(14)=+2.42/+2.64/+2.56/+2.14). Only `4_event_geo` improves, in 12 of 12 fold x horizon cells, growing with lead time. Rule: a known-future channel pays only if it is not derivable from the present AND known exactly - calendar fails the first, a real forecast fails the second, and of the eight knowable channels only the event schedule satisfies both. On the blind-sample cost quantified in the row below: the 23.7% weather figure REPRODUCES exactly (849/3,580, input window t..t+L-1, target t+L..t+L+H-1). The 15.2% (420/2,763) event figure DOES NOT - no setting of min_attendance, min_confidence or egress_min reproduces the 2,763 denominator, and the correct measurement under the shipped window definitions is 52.2% (432/828). The corrected number is the stronger one and it points the same way: for more than half of egress targets a contemporaneous model cannot see the fixture at all, which is exactly the rung the known-future block helps. |
| 2026-09-07 | `src/eval/decision.py` seed grouping FIXED; `onset_recall` gains a provenance column | Two bugs, one visible only together. `re.sub(r"__seed\d+$", ...)` is end-anchored, so variant runs (`..__seed42__future`) never grouped while plain rungs all collapsed to one key - and the loop then did `results[model][scope] = entry`, an assignment, so each seed OVERWROTE the last and only seed 44 survived. Seven rows of `results_main.csv` were therefore single-seed while presenting as 3-seed. Measured impact: recall varies by up to **0.067 between seeds of the same model on the same fold**, against a between-model spread of ~0.09, so the column could not rank models. Fixed to `__seed\d+(?=__|$)` with per-seed accumulation and an sd. All rows then re-measured from all three seeds LOCALLY - the predictions for the six affected rungs were still inside the `v2`, `new_rungs` and `future_preds` bundles, so no GPU and no retraining were needed. Two conclusions changed: `3_weather` is no longer the worst-recall trained model (0.688 -> 0.696, `0_speed` at 0.693 is), and the MAE-vs-onset-recall rank correlation fell from -0.65 (p=0.032) to -0.57 (p=0.066), i.e. from significant to not. `results_main.csv` carries `onset_recall_seeds` so the provenance is in the data file. |
| 2026-09-05 | `future_covariates` stays FALSE; no `--future` run (SUPERSEDED above; its 15.2%/2,763 event figure is also WRONG, see the 2026-09-07 row) | Not an optimism problem, a TRAIN/SERVE MISMATCH: the model would learn `observed future precipitation -> speed` and be served a nowcast, so the mapping itself is wrong, and no 2017 forecast archive is available to train on instead. The schedule channels have no forecast error and the objection does not apply to them, but their benefit is confined to 15.2% of event-window samples and events are already the weakest signal (+0.0018 incremental R2). Quantified cost of the choice, to be reported rather than apologised for: within the reported windows, 23.7% of adverse-weather samples (849/3,580) and 15.2% of event samples (420/2,763) carry no trace of the exogenous signal anywhere in the 60-min input window. Rejected reason: "shockwaves are too short-lived for a forecast to help" - the WAVE is short (median 15 min, p90 20) but the congestion it triggers is not (median 60 min, p75 150), and 59.6% of onsets produce congestion outlasting a 45-min horizon. |

## Session 2026-09-05: eight changes, all measured before and after

Run `python -m scripts.verify` after pulling. It exits non-zero on any failure
and covers the contract, artifact shapes, channel semantics, staleness, split
chronology, scaler spans, window contents, the graph operator's spectrum, the
shockwave label and a loader round-trip on all seven rungs.

| # | Change | Why, measured |
|---|--------|---------------|
| 0 | `dist_to_venue` is now DIRECTED ROAD distance (`align.road_distance_km`) | 128 sensor pairs sit within 150 m on opposite carriageways; great-circle put them a median 22 m apart while their speeds correlate 0.115. Egress effect at 5 km went from -0.41 +- 0.23 (t=-1.75, not significant) to -0.89 +- 0.26 (t=-3.37). `tau_km` re-swept: 4.0 still optimal. |
| 1 | Event channels AND eval windows filter to `confidence == high` | 58 of 95 rows carry a per-type default attendance. By tier the egress shockwave odds ratio is high 5.25x (z=3.66), medium 0.00x, low 1.00x. Dropping 30 of 67 fixtures raises effect size AND significance. |
| 2 | `scaled_laplacian` symmetrises before `eigsh` | `eigsh` assumes symmetry and does not check; on the directed adjacency it returned lambda_max 1.2670 against a true 1.0013, so the Chebyshev rescaling was 27% off and the spectrum sat in [-1, 0.58]. Now exactly [-1, 1]. Numbers move ~2%; stage 8's reproduction predates the fix. |
| 3 | Reported horizons 15/30/**45**, not 60 | At 60 min a per-node AR(12) is out by 17.56 mph below 35 mph while scoring 3.20 in aggregate. A 60-min column reports mostly noise where the project claims to be useful. `HORIZON` is unchanged at 12. |
| 4 | `metrics.shockwave_label` + `metrics.upstream_propagation` | The title needed a metric. Upstream/downstream co-occurrence is 1.53x at 10 min and 1.71x at 15 min - shockwaves travel against the flow and the directed graph sees it. Label is speed-only on purpose (see the decisions log). |
| 5 | `train.py --loss weighted`, OFF by default | 90.4% of the target is free-flow carrying 63.7% of the absolute error at MAE 2.18; below 45 mph is 6.0% carrying 26.1% at MAE 12-14. Plain L1 optimises the regime we do not care about. Off by default so rung 0 stays comparable to the literature. |
| 6 | `weather_commute` window; `distance_bins_km` extended to 50 | Rain during an egress hour happens 3 times in six months. Rain during the weekday peak: 31 episodes over 19 days, and it is the most discriminating window in the report (fold00 `0_speed` 4.42 in against 2.39 out at 45 min). The old top bin (10-20 km) left 127 nodes unbinned once distance became road-based. |
| 7 | `scripts/run_experiments.sh`, `src/eval/figures.py`, `scripts/verify.py` | The queue is ordered so a partial run is still usable and its header records why folds 3-5 are excluded. Figures: space-time diagram (a band leaning backwards IS the shockwave) and the propagation-ratio curve. |

**Shipping this to a GPU box.** Either clone it, or send the zip built with:

```bash
cd ..   # the directory ABOVE the repo
zip -qr tsf_gpu.zip traffic_shockwave_forecasting \
  -x '*/data/processed/*' '*/data/raw/events/_cache/*' '*/__pycache__/*' \
     '*/data/raw/augmented-pems-bay/.git/*' '*/checkpoints/*.pt' \
     '*/notes_local/*' '*.DS_Store'
```

52 MB. Note the exclusions: `data/processed/` is 1.0 GB and rebuilds in ~15 s,
`events/_cache/` is 61 MB of scraped HTML, and `augmented-pems-bay/.git` is a
47 MB nested clone. OUR `.git` is kept deliberately - 4.3 MB, and without it
results come back with no commit to attribute them to. Verified by extracting
elsewhere, rebuilding and running `scripts.verify`: 14.9 s, all checks pass,
no network needed. On the box: `bash scripts/setup_gpu_box.sh` detects that the
raw data is already there and skips the clone.

**Rebuild after pulling**: `python -m src.data.build_dataset --config
configs/default.yaml --force`. The `--force` is not optional - each stage is
skipped when its outputs merely exist, so a config change alone leaves stale
arrays on disk. This bit once during this session.

## Open questions

- [ ] Who fills owner A / B / C roles? Update CLAUDE.md and this file once assigned.
- [ ] DCRNN: attempt or skip? (CLAUDE.md says optional and time-boxed)
- [ ] Re-run stage 8 under the fixed Laplacian so the literature comparison and
      the ablation share one operator? Numbers move ~2%; low priority.
- [ ] Road distance leaves 54 of 325 nodes unreachable from any venue. Their
      event decay is 0, which is the right semantics, but `dist_to_venue` caps
      them at 48.29 km - an arbitrary finite stand-in for "never".

## Next steps

Stages 2-12 are written and pass `python -m scripts.verify` (90 checks). What
remains is compute and writing.

1. **Run the queue on a GPU.** `DEV=cuda bash scripts/run_experiments.sh`.
   Batch 0 is a 3-epoch timing probe on the widest rung and the largest fold -
   read its seconds/epoch before deciding how far down the list to go, because
   every runtime figure in this repo is extrapolated from an A100 number quoted
   in a notebook, not measured on the card you are holding.
2. **Build the tables.** `python -m src.eval.report --split fold0{0,1,2}`.
3. **Draw the figures.** `python -m src.eval.figures --date <a rainy commute
   day in the fold's test block> --freeway 101-N --predictions <npz files>`.
4. **`notes/paper_tables.md` is the writing plan.** Tables 1-7 are filled in
   and need no GPU: they come from data statistics and from two parameter-free
   baselines. Tables 8-11 and Figure 1 are placeholders the run fills in. Start
   writing from Table 1 today rather than waiting for the queue.
5. **Write the negative results up.** Three of them are findings, not gaps:
   compound rain-and-event exposure does not exist here (3 episodes / 18
   timesteps); rain does not cause breakdowns, it uniformly reduces capacity;
   and 58 of 95 fixtures carry a per-type default attendance, so an attendance
   threshold filters our own defaults rather than the world.

## How to update this file

When you finish a stage or make a decision that affects others:
1. Move the stage status from TODO → IN PROGRESS → DONE
2. Add a bullet under "What's done" with what changed
3. Log any cross-cutting decisions in the decisions table
