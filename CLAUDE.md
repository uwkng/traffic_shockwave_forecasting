# CLAUDE.md - Traffic Shockwave Forecasting

## What this project is
Course project for *Deep Learning and Decision Making* (TUM), M.Sc. MDT.
The contribution is an **evaluation protocol, not a new model**. Off-the-shelf
spatio-temporal graph models (STGCN = main, DCRNN = optional) are conditioned on
weather + scheduled-event signals and evaluated specifically inside
exogenously-defined anomalous windows (events, adverse weather) and by distance
from the venue, rather than on average error.

Constraints: Deadline at 05.09.2026, 3 contributors. Prefer small changes against
fixed interfaces over large rewrites.

## The one rule that stops three people colliding
Everything is built against the **data contract** in `src/contract.py`
(shapes, channel order, split ratios, prediction format). Do not change it
without updating the file and telling the team. All models consume the same
normalized tensors and emit the same prediction format, so eval treats them
interchangeably.

## Pipeline (data flows strictly left -> right)
| # | Stage | File |
|---|-------|------|
| 1 | Acquire raw sources (PEMS-BAY, weather, events) | `src/data/acquire.py` |
| 2-3 | Align to 5-min index + graph + distance-to-venue | `src/data/align.py` |
| 4 | Assemble 11-channel tensor `[T,N,11]` | `src/data/features.py` |
| 5 | Slice samples  X`[12,N,C]` Y`[12,N,1]` | `src/data/samples.py` |
| 6 | Split: single 70/10/20 + 6 rolling folds | `src/data/split.py` |
| 7 | Z-score PER CHANNEL (train-fit only) | `src/data/normalize.py` |
| - | Orchestrator for stages 1-7 | `src/data/build_dataset.py` |
| - | `data/processed` -> model input | `src/models/loader.py` |
| 9 | STGCN, `c_in` parameterised | `src/models/stgcn.py` |
| 10 | Training loop; trivial baselines | `src/models/{train,baselines}.py` |
| 11 | Windows, metrics, stratified report | `src/eval/{windows,metrics,report}.py` |
| 12 | Decision layer | `src/eval/decision.py` |

## Ownership (fill in names)
- Data & features (stages 1-8): `<owner A>`
- Models & training (stages 9-10): `<owner B>`
- Evaluation & figures (stages 11-13): `<owner C>`  <- the contribution; strongest engineer here

## Non-negotiables (known failure modes for this kind of project)
- Chronological split, **NO shuffling** - shuffling leaks the future.
- Fit the scaler on **train only** - fitting on all data silently inflates results.
- All metrics computed **after de-normalization** (real mph) - carry the scaler to eval.
- `dist_to_venue` is load-bearing: the only static spatial anchor for the event
  channels. Do not drop it.
- Build the master tensor ONCE with all 11 channels and select COLUMNS per
  ablation rung via `contract.rung_channels(rung)`. Rebuilding per rung gives
  each rung its own splits and scalers, and the MAEs stop being comparable.
- The scaler is PER CHANNEL and per split. One global mean/std would average mph
  with a 0/1 holiday flag, kilometres and millimetres.
- Never build the time axis with `pd.date_range`: 2017-03-12 02:00-02:55 does not
  exist in local wall-clock time, so 181x288=52,128 but the file has 52,116
  columns. Take the axis from `speed.csv`'s columns and join everything BY LABEL.
- Reproduce vanilla STGCN on plain PEMS-BAY BEFORE adding channels. (Done:
  stage 8, `notebooks/train_stgcn.ipynb`.)
- DCRNN is optional and time-boxed (legacy TF1.x); if it fights the environment, drop it.
  STGCN + baselines + full eval is already the complete contribution.

## Measured facts that decide how results are read

Speed anomalies are a sensor against ITSELF at the same time-of-week on days
with no event and no rain, unless stated otherwise. "Incremental R2" is against
the residual of a per-node AR(12) baseline. Derivations are in `STATUS.md`; the
thresholds live in `src/eval/windows.py` and `configs/default.yaml` with the
sweep that chose them.

### Corrections made 2026-09-05 - do not re-derive the superseded claims

- ~~"At 60 min history already answers most of it (AR R2 0.567), so exogenous
  channels have little room"~~ - that R2 is an AGGREGATE and 90.2% of
  observations are free-flow. By regime, AR(12) MAE in mph:

      horizon   <35 mph   35-45   >55 free-flow   ALL
      15 min      5.35     6.60       1.21        1.63
      30 min      9.97     9.01       1.58        2.25
      60 min     17.56    12.28       2.17        3.20

  Naive extrapolation fails badly INSIDE shockwaves even at 15 min. There is
  plenty of room; a longer horizon is not needed to make one. `HORIZON` stays
  12 and reporting moved to 15/30/45.
- ~~"Shockwaves and the exogenous windows barely intersect"~~ - that comparison
  was confounded by time of day (events end at 22:00, shockwaves peak at 08:00
  and 16:00). Controlled, see below: rain genuinely does nothing to shockwave
  frequency, events roughly double it.

### The phenomenon

- **Shockwaves are a commute phenomenon.** Share of timesteps with at least one
  sensor in shockwave, by hour: 06 53%, 08 69%, 10 36%, 12 26%, 14 71%,
  16 77%, 18 46%, 20 6%, 22 3%. Weekday peaks carry 88.4% against 18.2%
  off-peak.
- **They travel upstream, and the directed graph can see it.** Upstream over
  downstream co-occurrence: 1.53x at 10 min, 1.71x at 15 min
  (`metrics.upstream_propagation`). A ratio near 1 would mean the label is
  picking up network-wide congestion rather than a wave.
- **Rain does NOT cause breakdowns.** Controlling for commute: shockwave rate
  91.8% in rain against 93.1% dry; given a shockwave, 6.25 sensors involved in
  rain against 6.46 dry. Rain is a uniform capacity reduction - -2.22 mph
  overall, -3.04 mph in the commute peak - not a trigger.
- **Events DO cause breakdowns, modestly.** High-confidence fixtures, egress
  hour, matched controls (same weekday and clock, +-7/14 days, no event),
  sensors within 5 km road distance: 14/36 against 4/54, i.e. 5.25x, z=3.66.
  Speed anomaly -0.89 +- 0.26 mph (t=-3.37).
- **Holidays are the largest exogenous effect and POSITIVE.** Memorial Day
  +9.50 (AM) / +11.27 (PM). A holiday removes the commute. Report as a stratum,
  never merge into "adverse".
- **The target is 90.4% free-flow, carrying 63.7% of the absolute error** at
  MAE 2.18, while <45 mph is 6.0% of observations carrying 26.1% at MAE 12-14.
  A plain L1 optimises the regime the project does not care about; hence
  `train.py --loss weighted`, off by default.

### Provenance decides the answer

- **Only 37 of 95 events have an observed attendance.** The other 58 carry a
  per-type default, so filtering them by `min_attendance` filters our own
  defaults. By tier, egress hour, matched controls: high 5.25x z=3.66; medium
  0.00x z=-1.63 (0/19); low 1.00x. Dropping 30 of 67 qualifying fixtures RAISES
  effect size and significance. `min_confidence: high` everywhere.
- **Great-circle distance to a venue is wrong in a measurable way.** 128 sensor
  pairs lie within 150 m on OPPOSITE carriageways; great-circle separates them
  by a median 22 m while their speeds correlate at 0.115 (same-direction pairs
  up to 1 km apart: 0.850). Switching to directed road distance from
  `distances_bay_2017.csv`: within 5 km goes from 177 sensors to 54, and the
  egress effect from -0.41 +- 0.23 (t=-1.75, NOT significant) to -0.89 +- 0.26
  (t=-3.37). `tau_km = 4.0` was re-swept and is still optimal.
- **Compound exposure does not exist here.** Rain during an egress hour occurs
  3 times in six months, 18 timesteps total. Any "rain x event" headline on
  PEMS-BAY is noise. The compound window that DOES exist is rain x commute
  peak: 34 episodes over 19 days, 500 timesteps, -3.04 mph.
- **The precipitation column cannot be summed.** `p01i` is a backward 1-hour
  accumulation forward-filled across the next twelve 5-min steps. Thresholds
  and episode counts are unaffected (per-step comparisons); totals are 12x
  high. Label it "1-hour accumulation (mm), forward-filled", never "5-min
  rainfall". Dividing by 12 would change nothing - the threshold is a
  percentile of the same series and the model z-scores it - but leaving the
  off-hour steps EMPTY would: 104 contiguous rain episodes become 213 isolated
  points and the window concept collapses.

### Where the data is

- **The `single` 70/10/20 test block has ZERO adverse weather.** California's
  wet season is Jan-Apr, so any chronological split puts test in the dry half.
  Weather results must come from the rolling folds.
- **Rain and NHL live in folds 0-2 only.** Test-block contents: fold00 28 rain
  / 4 NHL / 1 holiday, fold01 16 / 8 / 0, fold02 17 / 7 / 0, folds 3-5 zero and
  zero. Folds 3-5 carry 73% of the compute (114,807 of 157,108 train samples).
  `scripts/run_experiments.sh` runs folds 00-02 and says why in its header.
- **Count EPISODES, not timesteps.** A standard error over node-timesteps
  understates by ~sqrt(n_timesteps / n_episodes). Holidays are n=1-2 per test
  fold: report as cases, never with an error bar.
- **Channel value, measured on folds 00-02, commute hours, 30-min horizon**
  (incremental R2 over AR(12); univariate and linear, so it UNDERSTATES
  channels that only act interactively, such as precipitation):
  occupancy 0.220; event_active_decay 0.0023; time_of_day 0.0023; event_load
  0.0018; temperature 0.0006; precipitation 0.0005; wind_speed 0.0004;
  is_holiday 0.0003; dist_to_venue 0.0002; day_of_week 0.0002. Occupancy is
  worth more than everything else combined by two orders of magnitude.
- **Weekends and holidays are KEPT**, against Yu et al., who drop them. 46 of
  95 events (48%) fall on a weekend.
- **The PEMS-BAY STGCN baseline is Wu et al. (2019) Table 2**, not Yu et al.
  (2018), who used BJER4 and PeMSD7 with a month-based split. Their row is
  1.36 / 1.81 / 2.49 at 15/30/60 min; our stage-8 reproduction is
  1.438 / 1.928 / 2.578 (obtained BEFORE the `eigsh` fix below).

### Settled, do not reopen

- **Predict speed only.** Predicting occupancy as a second head was proposed to
  let the evaluation score an occupancy-based shockwave label without leaking.
  Measured, the speed-only label reaches every conclusion the occupancy one
  does (rain no-effect; events 2.50x z=3.24 against 5.25x z=3.66; upstream
  1.71x against 1.88x), and the occupancy label is a strict subset of it. So
  the label dropped occupancy instead, the model stays single-output, and rung
  0 stays comparable to the published single-channel numbers. Occupancy remains
  an INPUT channel, where it is the single most valuable one.

## Our STGCN vs the official code (VeritasYin/STGCN_IJCAI-18)

Line-by-line diff done 2026-09-04 against `models/layers.py`, `base_model.py`,
`trainer.py`, `math_graph.py`, `main.py`, `tester.py`. Ours is the STGCN
*architecture*, NOT a port of that repo. Say so in the paper; the appendix
needs this list.

Same: ST-Conv block TC->GC->TC, GLU temporal conv Kt=3 VALID, Chebyshev Ks=3,
scaled Laplacian `2L/lmax - I`, LayerNorm+dropout at block end, n_his=12,
lr 1e-3 decayed by 0.7.

FIXED 2026-09-05: we called `eigsh` (symmetric-only, and it does not check) on a
Laplacian built from the DIRECTED adj_mx_bay - 1,818 of 2,694 edges have no
reverse. It returned lambda_max 1.2670 against a true 1.0013, so the Chebyshev
rescaling was off by 27% and the spectrum landed in [-1, 0.58]. `scaled_laplacian`
now symmetrises with `max(A, A.T)` first (Chebyshev is defined on undirected
graphs), giving lambda_max 1.2408 and a spectrum of exactly [-1, 1]. Numbers
move ~2%. The RAW directed matrix is still what `metrics.upstream_propagation`
reads - do not symmetrise `data/processed/adj_mx.npy` itself.

Deviations that change results:
- **Multi-step.** They predict ONE step and recurse at test time, feeding the
  prediction back 9 times (`tester.py:38-42`); loss is on that one step only
  (`base_model.py:40`). We emit all 12 steps in one pass. Ours is the
  DCRNN/Graph-WaveNet convention our baseline numbers come from - keep it, but
  do not claim to reproduce their implementation.
- **Loss.** Theirs is `tf.nn.l2_loss` (MSE); ours is L1. MSE punishes large
  errors harder, i.e. it leans TOWARD shockwaves. Switching to L1 moved us away
  from what this project cares about - this is an argument for the weighted loss.
- **Blocks.** Theirs `[[1,32,64],[64,32,128]]` (the paper's bottleneck design);
  ours `[[1,64,64],[64,64,64]]`. Ours is what stage 8 matched Wu et al. with,
  so changing it means re-running stage 8. Keep, but document.

Structural, effect unknown:
- Residual placement. They put residuals INSIDE each sublayer - inside the GLU
  (`(x_conv + x_input) * sigmoid(gate)`, `layers.py:85`) and after the graph
  conv (`relu(x_gc + x_input)`) - and have NO block-level residual. We have a
  plain `p*sigmoid(q)`, a linear ChebConv, and a block-level residual + relu.
- LayerNorm axes. They normalise over (node, channel) with per-node per-channel
  affine `[1,1,N,C]`; we use `nn.LayerNorm(c_out)`, affine `[C]`. Ours is less
  prone to overfitting single sensors, but their per-node affine was partly
  learning per-segment baselines - which is what our climatology/dist channels
  now cover.

Their bugs - do NOT copy:
- Dropout never fires: `trainer.py:80` feeds `keep_prob: 1.0` during training.
- Weight decay is collected all over `layers.py` and never added to the loss.
Ours has dropout 0.3 and weight_decay 1e-4 actually active.

Minor / ours is better: they zero-pad to widen channels (we 1x1 conv); one
global mean/std (we must be per-channel with 11 channels); they build the
Gaussian-kernel adjacency themselves (we use the published `adj_mx_bay.pkl`,
standard for PEMS-BAY); PeMSD7 228 nodes 34/5/5 days; batch 50; RMSProp.

## Compute (measured, so plans are not guesses)

Training is compute-bound - data loading is 1-2% of an epoch. Per-sample cost:
this Mac's MPS ~5,800 us, an A100 ~406 us (the stage-8 notebook's 14.8 s/epoch
over 36,465 samples). An H200 is NOT 3x an A100 here: the model is 117k
parameters, the dominant op is `[325,325] @ [325,64]`, and batch is 64, so the
card is starved. Expect 1.3-2x unless the batch size is raised.

Six rolling folds total 157,108 train samples against 36,449 for `single`, so a
rolling run costs 4.3x a single-split run of the same configuration.

## Conventions
- Python 3.11+, PyTorch. Config lives in `configs/*.yaml`; no hardcoded paths.
- Seed everything; every trained config runs 3 seeds.
- Functions stay close to pure: consume arrays/paths, produce arrays/files.
- Work one problem per Claude Code session; start a fresh session per stage.
- **Update `STATUS.md`** after every meaningful change: move stage status
  (TODO → IN PROGRESS → DONE), log decisions, note what's next. This is how
  the team stays in sync — treat it as part of finishing the task.

## Commands (wire up as files get implemented)
```bash
# stages 1-7. --force is REQUIRED after editing a stage: each stage is skipped
# when its outputs merely exist, so a config change alone leaves stale arrays.
python -m src.data.build_dataset --config configs/default.yaml --force

# stage 9-12, the whole queue, ordered so a partial run is still usable
DEV=cuda PY=python3 bash scripts/run_experiments.sh

# or one configuration at a time
python -m src.models.train --rung 0_speed --split fold00 --seeds 3 --device cuda
python -m src.models.baselines --split fold00
python -m src.eval.report --split fold00
python -m src.eval.figures --date 2017-02-08 --freeway 101-N
```