# Results — what we did and what the numbers mean

Everything here comes from one experiment run on 2026-09-05.
**99 trained models = 11 configurations × 3 time periods × 3 random seeds.**

| file | what it is |
|---|---|
| **`results_main.csv`** | **the one table.** One row per configuration, one column per metric. Open in Excel. |
| `results_main_COLUMNS.csv` | what each column of that table means, in one sentence each |
| `report__fold0*.json` | the same numbers broken down by time period, with standard deviations |
| `decision__fold0*.json` | how often each model raised an alarm, and how often it was wrong |
| `figures/spacetime_101-N_*.png` | the picture of a shockwave — see section 11 |
| `figures/propagation_ratio.png` | the measurement behind `propagation_fidelity` |

---

## 1. What is the problem

A road sensor reports a speed every 5 minutes. We give the model **the last 60 minutes**
of readings for all 325 sensors, and it predicts **the next 60 minutes** for all of them.

We measure how wrong it is, in **miles per hour**. That is `MAE`. Lower is better.

A model that is off by 2.5 mph on average is a good model. Published STGCN on this
dataset is around 2.5 mph at the 60-minute horizon, and so is ours.

## 2. What is a shockwave, and why does this project exist

When traffic breaks down, the jam does not sit still. The first driver brakes, the
driver behind brakes, and the slow patch **travels backwards** along the road, against
the direction of travel. That travelling patch is the shockwave.

We measured it: a breakdown appears at an upstream sensor **1.68× more often** than at
a downstream one, 15 minutes later. That is the `propagation_fidelity` number, and
**1.683 is reality**. A model that scores 1.0 has learned no sense of direction at all.

The problem this project is about is simple:

> Shockwaves are rare. **90% of all readings are free-flowing traffic above 55 mph.**
> So the average error is almost entirely made of "predicted 65, it was 63".
> A model can improve its average error while getting *worse* at the thing we care about.

So we do not report one number. We report five, and we check whether they agree.

## 3. The 11 channels

The published STGCN reads **one** channel: speed. We built a tensor of eleven, once,
and every configuration selects columns from that same tensor — so all of them share
the same samples, the same fold boundaries and the same scaler, and their errors are
directly comparable. The definition lives in `src/contract.py`.

| # | channel | group | varies over | where it comes from |
|---|---|---|---|---|
| 0 | **speed** | speed | node × time | PEMS-BAY. This is also the prediction target |
| 1 | **occupancy** | occupancy | node × time | PEMS-BAY. Fraction of the 5 minutes a vehicle sat over the loop — this is *density*, and density moves before speed does |
| 2 | time of day | calendar | time only | the clock |
| 3 | day of week | calendar | time only | the clock |
| 4 | is holiday | calendar | time only | 5 US federal holidays in the period |
| 5 | temperature | weather | time only | ASOS stations SJC + NUQ |
| 6 | precipitation | weather | time only | ASOS `p01i` |
| 7 | wind speed | weather | time only | ASOS |
| 8 | **distance to venue** | event\_geo | node only | great-circle km from each sensor to the nearest active venue. Static |
| 9 | event active decay | event\_geo | node × time | is a fixture on, weighted by `exp(-distance / 4 km)` |
| 10 | event load | event\_att | node × time | the same, but multiplied by `attendance / 20,000`, and only for fixtures with ≥15,000 attendees |

**The column that matters most is "varies over".** Channels 2–7 are a single number
copied to all 325 sensors. Channels 8–10 are different for every sensor. Section 5(b)
shows that this distinction, not the number of channels, decides the result.

Why ≥15,000 on channel 10: below that threshold the measured speed drop in the egress
hour is **+0.35 ± 0.41 mph** — the wrong sign. Above it, the effect is real
(t = −9.76). The threshold was measured on the data before any model was trained.

## 4. The configurations

`src/contract.py` defines a seven-rung ablation ladder. All seven were trained,
plus two variants of the headline pair:

| configuration | what it isolates |
|---|---|
| `0_speed` | the published baseline. Everything is measured against this |
| `1_traffic` | + occupancy |
| `2_calendar` | + time of day, day of week, is holiday — **all broadcast to every node** |
| `3_weather` | + temperature, precipitation, wind — **also all broadcast** |
| `4_event_geo` | + venue distance and event activity, **without attendance** |
| `5_event_geo_att` | the same, **with attendance** |
| `6_all` | all 11 |
| `0_speed` / `6_all` + **weighted loss** | changes only the training objective |
| `0_speed` / `6_all` + **directed operator** | changes only the graph convolution |

Three of these are controlled pairs, which is what makes the table readable:

- `2_calendar` vs `3_weather` vs `5_event_geo_att` — **identical channel count
  (C=5) and near-identical parameter count (118,732)**. They differ only in what
  the three added channels contain.
- `4_event_geo` vs `5_event_geo_att` — differ by exactly one thing, whether
  attendance enters.
- `6_all` vs its weighted-loss and directed-operator variants — same channels,
  one component changed.

### The two changes that are not channels

- **congestion-weighted loss** — during training, errors at low speed count more:
  weight `1 + max(0, (60 − speed)/20)`, so 1.0 at free flow, 1.5 at 50 mph, 2.5 at 30.
  Nothing else changes, and every reported number is still ordinary *unweighted* MAE.
- **directed operator** — the published model must treat "the sensor ahead of me"
  and "the sensor behind me" as the same kind of neighbour, because Chebyshev graph
  convolution needs a symmetric matrix. We swapped in the DCRNN/Graph WaveNet dual
  random walk, which keeps the directions apart. 5 supports instead of 3,
  +16,384 parameters, no other change.

## 5. The five things we measure

| column in the CSV | the question it answers |
|---|---|
| `MAE_60min` | how wrong on average — **the number everyone reports** |
| `MAE_rain_x_commute_60min` | how wrong when it is raining during rush hour — **where the errors actually are** |
| `propagation_fidelity` | did it learn that jams travel backwards (reality = 1.683); measured on the 60-min forecasts |
| `onset_recall` | of the traffic breakdowns that really happened, what share did it flag |
| `false_alarms` | how often it cried wolf |

## 6. The result

### The headline: the two signals in the title do not work the way we assumed

| rung | MAE@60 | rain × commute | event egress | propagation | onset recall |
|---|---:|---:|---:|---:|---:|
| `0_speed` | 2.502 | 5.349 | 1.307 | 1.375 | 0.702 |
| `1_traffic` | 2.369 | 5.147 | 1.250 | 1.452 | 0.726 |
| **`2_calendar`** | **2.279** | **4.861** | **1.237** | **1.489** | 0.750 |
| `3_weather` | 2.485 | 5.559 | 1.302 | 1.454 | **0.688** |
| `4_event_geo` | 2.318 | 5.070 | 1.255 | 1.348 | 0.763 |
| `5_event_geo_att` | 2.303 | 4.991 | 1.239 | 1.324 | 0.752 |
| `6_all` | 2.361 | 5.283 | 1.406 | 1.389 | 0.738 |
| *observed truth* | — | — | — | **1.683** | — |

**(a) Weather channels make the model worse, and the more weather-specific the
window, the worse they get.** `3_weather` against `1_traffic`, the same model
with three channels added:

| measured in | change |
|---|---|
| overall | **+0.116 mph (4.9% worse)** |
| inside adverse weather | **+0.170 mph (6.6% worse)** |
| inside rain × commute | **+0.412 mph (8.0% worse)** |

It also has the worst onset recall of any trained model, 0.688.

We knew why before training. Controlling for time of day, rain does not change
how *often* breakdowns happen (91.8% of commute-peak timesteps carry a shockwave
in rain against 93.1% dry) or how *large* they are (6.25 sensors involved against
6.46). Rain is a uniform capacity reduction of about −3.04 mph, and a uniform
slowdown is the easiest thing to read out of a speed history. The model does not
need to be told it is raining; it can see that everything is 3 mph slower.

**(b) Event channels help, but not because they are about events.** They do lower
error — `5_event_geo_att` beats `1_traffic` by 0.066 mph overall. But:

| comparison | result |
|---|---|
| inside the egress hour: `2_calendar` (no event channels) vs `5_event_geo_att` (all of them) | **1.237 vs 1.239 — a difference of 0.002 mph** |
| `4_event_geo` (no attendance) vs `5_event_geo_att` (with attendance) | 2.318 vs 2.303 — 0.015 mph, against a seed-and-fold spread of ±0.15–0.25 |
| where `5_event_geo_att` actually wins | the **rain × commute** window, which contains no fixtures at all |

In the hour after a fixture ends, a model that has never heard of the fixture
performs identically to one that knows its time, its venue and its attendance.
Removing attendance changes nothing. And the gains appear in windows with no
events in them.

The only mechanism that fits: `dist_to_venue` is static and **per-node**, and
STGCN has no node embeddings. It is the one channel that tells the model *which*
sensor it is looking at. It is functioning as a positional encoding, not as event
information.

**(c) What actually helps is the calendar — and it is broadcast too.** This is
the result we did not expect. `2_calendar` wins on overall error, on both weather
windows, on the egress window, and on propagation fidelity, using three channels
that are each a single number copied to all 325 sensors — structurally identical
to the weather channels, at the same parameter count.

So "a channel with no spatial variation is harmful" is **wrong**. The distinction
that survives is redundancy:

> A broadcast channel helps when it carries something the target's own history
> cannot supply, and hurts when it does not. Rain is already visible in the speed
> history as a uniform slowdown, so the weather channels are redundant and cost
> capacity. Time-of-day and day-of-week place the model in the daily and weekly
> cycle, which 60 minutes of speed cannot pin down — a quiet hour looks the same
> at 03:00 on a Tuesday and 11:00 on a Sunday.

### What the metrics agree and disagree about

| | Spearman vs overall MAE | p |
|---|---:|---:|
| rain × commute MAE | +0.79 | 0.004 |
| onset recall | −0.65 | 0.032 |
| **propagation fidelity** | **−0.35** | **0.298** |

Aggregate MAE does predict the other *error* metrics — unsurprising, they are all
error. It does **not** predict whether the model reproduces the physics.
`5_event_geo_att` ranks 2nd on error and 8th on propagation; `3_weather` ranks 8th
on error and 2nd on propagation. And the cleanest case is the directed operator:
it leaves aggregate MAE unchanged (2.502 → 2.503) while propagation falls from
1.375 to 1.246, worse in 9 of 9 paired runs.

**Stated narrowly, and it holds:** aggregate error ranks models the same way other
error metrics do, and tells you nothing about whether the model learned that jams
travel backwards.

### The honest summary

| what the title promises | what the results show |
|---|---|
| Weather-aware | ❌ weather channels are 8% **worse** inside rain × commute |
| Event-aware | ⚠️ the channels help, but strip out the event content and nothing changes; the gain is `dist_to_venue` acting as node identity |
| Shockwave | ✅ real and measured — observed 1.68×, models reproduce 1.25–1.49, and the directed operator makes it worse |

This is a negative result with a mechanism attached, which is more useful than a
positive one without. It also yields a rule that transfers: **the value of an
exogenous channel is not what it describes, but whether it carries information the
target's own history does not already contain.**

### Still open

The forecast-conditioned version was never tested where it matters. We measured
that 22–24% of adverse-weather target windows have a completely dry input window —
the rain starts inside the forecast horizon, so a contemporaneous channel cannot
see it at all. The `modeling` branch built exactly that mechanism using an ECMWF
IFS forecast archive, but its test block contains zero rain. Nobody has yet
answered whether a *forecast* of weather helps.

## 7. How to read the table without being misled

**Compare down a column, never across.** The columns are in different units and
different regimes.

**Do not compare a window against the overall column.** Two traps we hit ourselves:

- `MAE_event_egress_60min` is *lower* than the overall column for every model. That is
  not because events are easy to predict. Event egress happens at 21:30–23:00 when the
  road is empty. It is the clock, not the fixture.
- The rain window looks about 2× harder than everything else. Most of that is also the
  clock: on dry data alone, rush hour is 2.55× harder than off-peak in all three time
  periods. Rain adds between −0.65 and +1.16 mph on top of that, and its sign is not
  even consistent.

**Count events, not rows.** The rain window covers 51 rain spells; the event window
covers 22 fixtures. Neighbouring 5-minute readings are nearly identical, so a standard
deviation computed over rows would be far too small. Where we quote a spread, it is
across the 9 runs, not across rows.

---

## 8. Which time periods, and why

We train and test on three consecutive 28-day blocks, each testing on the month after
it trained. The reason is in the data:

| test block | rain spells | fixtures | holidays |
|---|---|---|---|
| 29 Jan – 26 Feb | 27 | 4 | 1 |
| 26 Feb – 26 Mar | 12 | 10 | 0 |
| 26 Mar – 23 Apr | 12 | 8 | 0 |
| 23 Apr – 30 Jun | **0** | 5 | 1 |

California's wet season is January to April. Everything after April is dry, so the last
period holds 73% of the training cost and cannot say anything about weather at all. We
excluded it. The exclusion uses only the calendar and the weather record, both known
before any model was trained.

## 9. What we could not measure

- **Warning time.** The natural operational number — how many minutes earlier than a
  simple reactive rule the model raises the alarm — is **0 minutes for every model**,
  including the trivial baselines. The forecast horizon is 60 minutes and the reactive
  rule needs 15 minutes to confirm, so there is not enough room to separate models. We
  report recall and false alarms instead, which measure the same question and do separate.
- **Rain during an event.** Adverse rain overlapping an event egress hour happens
  **3 times in six months, 18 five-minute steps total**, one of them a single step.
  It is the natural headline for a paper like this. On this data it would be noise.

## 10. How we compare against STGCN, DCRNN and Graph WaveNet

`results_main.csv` now carries the published PEMS-BAY numbers as extra rows, taken from
Wu et al. (2019), *Graph WaveNet for Deep Spatial-Temporal Graph Modeling*, Table 2.

**Read the `evaluation` column first. Rows are only comparable within the same value.**

| evaluation | which rows | MAE at 15 / 30 / 60 min |
|---|---|---|
| conventional 70/10/20 (published) | Graph WaveNet | 1.30 / 1.63 / **1.95** |
| conventional 70/10/20 (published) | DCRNN | 1.38 / 1.74 / 2.07 |
| conventional 70/10/20 (published) | STGCN | 1.36 / 1.81 / 2.49 |
| conventional 70/10/20 (published) | WaveNet | 1.39 / 1.83 / 2.35 |
| conventional 70/10/20 (published) | FC-LSTM | 2.05 / 2.20 / 2.37 |
| conventional 70/10/20 (published) | ARIMA | 1.62 / 2.33 / 3.38 |
| conventional 70/10/20 (**this work**) | **STGCN, our reproduction** | **1.44 / 1.93 / 2.58** |
| rolling folds 00-02 (this work) | everything else in the file | 1.37–1.57 / 1.79–2.16 / 2.28–3.04 |

### The reproduction is the comparable row, and it lands

Our speed-only STGCN under the **same** split the literature uses gives
**1.44 / 1.93 / 2.58** against the published **1.36 / 1.81 / 2.49** — within 0.1 mph at
every horizon.

> ⚠️ **Provenance, and a caveat that has not been closed.** That row comes from
> `notebooks/train_stgcn.ipynb` cell 13, three seeds (42/43/44), mean
> 1.4381 ± 0.0023 / 1.928 / 2.578, run on **2026-09-04**. Commit `5d2621e`, on
> **2026-09-05**, then fixed `scaled_laplacian`: `eigsh` assumes symmetry and does not
> check it, so on the directed adjacency it returned λ_max = 1.2670 against a true
> 1.0013 — the Chebyshev rescaling was **27% off** and the spectrum sat in [−1, 0.58]
> instead of [−1, 1]. **The reproduction has not been re-run since that fix.** Only
> `checkpoints/stgcn_seed42.pt` survives; the other two seeds' weights were not kept.
>
> The number is a real three-seed measurement and it does land within 0.1 mph. But it
> cannot currently be read as "the graph code is sound", because the graph code changed
> after it was taken. Re-running it is one command against the current code —
> `python -m src.models.train --rung 0_speed --split single --seeds 3 --epochs 100` —
> and costs about 45 minutes on an H200.

### Our main rows are NOT comparable to the published ones

They are trained and tested on three 28-day rolling blocks, not on the standard
70/10/20 chronological split. Different training data, different test data, different
amount of it. A number from that experiment placed beside Graph WaveNet's 1.95 would
mean nothing, in either direction.

We use the rolling folds anyway, for the reason in section 8: **the conventional split
has zero rain in its test block.** It is the better split for reproducing a published
number and a useless one for asking whether weather information helps.

### And Graph WaveNet is not a competitor here

It is a stronger forecaster than STGCN on aggregate error — 1.95 against 2.49 mph at
60 minutes, a 22% improvement, and we do not dispute it. But every number in that
column is aggregate error, and section 6 shows that aggregate error says nothing
about whether a model reproduces the upstream propagation the title is about.
Swapping in a stronger backbone would change the aggregate column and leave the
questions this project asks untouched. We kept STGCN unchanged on purpose: the
contribution is the evaluation protocol, not the model.

---

## 11. The picture

`figures/spacetime_101-N_2017-03-24.png` is the one figure worth showing.

Read it like this: **x is position along US-101 north** in road order, so left is
upstream and right is downstream. **y is time of day**, 05:00 at the bottom.
**Colour is speed** — green is free flow, red is a jam.

Three panels: what actually happened, then what the speed-only model forecast
60 minutes ahead, then the all-channel model with the weighted loss.

A jam that sat still would appear as a **vertical** red bar. A jam that travels
backwards up the road appears as a band **leaning to the upper left** — later in
time, further upstream. That lean is the shockwave, and it is what
`propagation_fidelity` measures numerically.

The three days were chosen by rainfall during the weekday peak inside each test
block, before looking at any model output. Holidays are excluded from that
choice on purpose: the first version of the selector returned Presidents' Day
for the first block, which is the wettest commute-hour day there and has no
commute at all — US holidays run +9.50 mph in the morning peak.
