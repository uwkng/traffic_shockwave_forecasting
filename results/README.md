# Results — what we did and what the numbers mean

Everything here comes from one experiment run on 2026-09-05.
**81 trained models = 9 configurations × 3 time periods × 3 random seeds.**

| file | what it is |
|---|---|
| **`results_main.csv`** | **the one table.** One row per configuration, one column per metric. Open in Excel. |
| `results_main_COLUMNS.csv` | what each column of that table means, in one sentence each |
| `report__fold0*.json` | the same numbers broken down by time period, with standard deviations |
| `decision__fold0*.json` | how often each model raised an alarm, and how often it was wrong |
| `figures/spacetime_101-N_*.png` | the picture of a shockwave — see section 10 |
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

## 3. What we added to the model

The published STGCN sees **only speed**. We added channels one at a time:

| configuration | what the model can see |
|---|---|
| `0_speed` | speed only — this is the published baseline |
| `1_traffic` | + **occupancy** (what fraction of time a car sits over the sensor — this is density, and density leads speed) |
| `3_weather` | + temperature, rainfall, wind |
| `5_event_geo_att` | + distance to each stadium, whether an event is on, how many people are attending |
| `6_all` | everything, 11 channels |

Two further changes, each touching exactly one thing:

- **congestion-weighted loss** — during training, count errors at low speed more heavily.
  Nothing else changes. The reported numbers are still ordinary unweighted MAE.
- **directed operator** — the published model is forced to treat "the sensor ahead of me"
  and "the sensor behind me" as the same kind of neighbour. We replaced that part so it
  can tell them apart.

## 4. The five things we measure

| column in the CSV | the question it answers |
|---|---|
| `MAE_60min` | how wrong on average — **the number everyone reports** |
| `MAE_rain_x_commute_60min` | how wrong when it is raining during rush hour — **where the errors actually are** |
| `propagation_fidelity` | did it learn that jams travel backwards (reality = 1.683) |
| `onset_recall` | of the traffic breakdowns that really happened, what share did it flag |
| `false_alarms` | how often it cried wolf |

## 5. The result

**No two of those five metrics pick the same winner.**

| metric | winner |
|---|---|
| lowest average error | `5_event_geo_att` |
| lowest error when it rains in rush hour | `1_traffic` |
| best at reproducing the physics | `3_weather` |
| catches the most breakdowns | `6_all` + weighted loss |
| fewest false alarms | `0_speed` + directed operator |

The rank correlation between average error and propagation fidelity is **−0.23**, which
for nine models is indistinguishable from no relationship at all.

**That is the paper.** Choosing a model by average error does not merely understate the
differences that matter — it picks a *different model* from the one any operationally
meaningful metric picks.

### Three things that make the point concrete

**(a) Weather channels made the model worse, not just "not better".**
`3_weather` has the worst error of any trained model when it rains during rush hour
(5.559 vs 5.349 for the speed-only baseline) and the worst onset recall (0.688).
We know why, and we knew before training: controlling for time of day, rain does not
change how *often* breakdowns happen (91.8% in rain vs 93.1% dry) or how *big* they are.
Rain just slows everyone down uniformly by about 3 mph — and a uniform slowdown is
already visible in the speed history the model has.

**(b) What helps is not more information — it is information that varies by location.**
`3_weather` and `5_event_geo_att` have **the same number of parameters (118,732) and the
same number of channels (5)**. The only difference: weather is one number copied to all
325 sensors, while event distance is different for every sensor. Their ordering reverses
in every window and every time period, 5.559 vs 4.991.

> On a graph model, a channel with no spatial variation is not a weak feature.
> It is a harmful one — it fills three slots of capacity saying nothing about *which*
> sensor this is.

**(c) The obvious architectural fix did not work, and that is the cleanest evidence.**
We gave the model the ability to distinguish upstream from downstream, precisely because
shockwaves travel upstream 1.68× more often. Average error barely moved
(2.361 → 2.338, well inside the ±0.25 spread across runs). Propagation fidelity got
**worse** (1.389 → 1.315, worse in 7 of 9 paired runs; for the speed-only model,
1.375 → 1.246, worse in **9 of 9**).

An average-error-only report would have logged that as a small improvement.

---

## 6. How to read the table without being misled

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

## 7. Which time periods, and why

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

## 8. What we could not measure

- **Warning time.** The natural operational number — how many minutes earlier than a
  simple reactive rule the model raises the alarm — is **0 minutes for every model**,
  including the trivial baselines. The forecast horizon is 60 minutes and the reactive
  rule needs 15 minutes to confirm, so there is not enough room to separate models. We
  report recall and false alarms instead, which measure the same question and do separate.
- **Rain during an event.** Adverse rain overlapping an event egress hour happens
  **3 times in six months, 18 five-minute steps total**, one of them a single step.
  It is the natural headline for a paper like this. On this data it would be noise.

## 9. Sanity check

Before adding anything, we reproduced the plain speed-only STGCN under the conventional
setup and got **1.44 / 1.93 / 2.58 mph** at 15/30/60 minutes, against the published
**1.36 / 1.81 / 2.49**. Within 0.1 mph at every horizon — the pipeline is sound.

---

## 10. The picture

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
