# Paper tables — what is already measured, and what the GPU run fills in

Every number marked **[measured]** is on disk now and needs no training. Numbers
marked `[GPU]` are placeholders that `scripts/run_experiments.sh` fills in.

Regenerate everything measured here with:

```bash
python -m src.data.build_dataset --config configs/default.yaml --force
python -m scripts.verify
for f in fold00 fold01 fold02; do
  python -m src.models.baselines --split $f
  python -m src.eval.report --split $f
  python -m src.eval.decision --split $f
done
python -m src.eval.figures --date 2017-02-08 --freeway 101-N
```

---

## Table 1 — The window definition decides whether the effect is visible

**[measured]** 45-min MAE (mph), inside the window against its complement.
Two baselines with no fitted parameters: persistence copies the last
observation, historical average is the per-node time-of-week mean of the
training span.

| fold | window | n_ep | persist. IN | OUT | ratio | HA IN | OUT | ratio |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| 00 | `adverse_weather` | 27 | 2.834 | 2.521 | 1.12 | 3.754 | 2.801 | 1.34 |
| 01 | `adverse_weather` | 12 | 3.307 | 2.752 | 1.20 | 3.282 | 2.860 | 1.15 |
| 02 | `adverse_weather` | 12 | 2.382 | 2.544 | 0.94 | 2.229 | 2.522 | 0.88 |
| 00 | **`weather_commute`** | 10 | 4.361 | 2.487 | **1.75** | 6.388 | 2.785 | **2.29** |
| 01 | **`weather_commute`** | 4 | 6.169 | 2.734 | **2.26** | 6.649 | 2.830 | **2.35** |
| 02 | **`weather_commute`** | 2 | 7.360 | 2.514 | **2.93** | 5.434 | 2.493 | **2.18** |
| 00 | `event_egress` | 4 | 1.270 | 2.581 | 0.49 | 1.955 | 2.948 | 0.66 |
| 01 | `event_egress` | 10 | 1.477 | 2.822 | 0.52 | 2.143 | 2.906 | 0.74 |
| 02 | `event_egress` | 8 | 1.240 | 2.565 | 0.48 | 1.535 | 2.528 | 0.61 |

**The claim.** Same rain, same data. An "adverse weather" window sits at ~1x —
it looks like nothing is happening. Intersecting it with the weekday commute
peak gives 1.75-2.93x. Six independent measurements (3 folds x 2 baselines)
agree on the ordering, so this is a property of the data, not of one baseline.

**The trap, and it belongs in the text.** `event_egress` reads 0.48-0.74, i.e.
*easier* than its complement, because egress happens at 21:30-23:00 when the
network is empty. In-window and out-of-window difficulty are not comparable.
Only model-against-model within one window is.

## Table 2 — Where the shockwaves are

**[measured]** Share of timesteps with at least one sensor in shockwave
(`metrics.shockwave_label`: >10 mph fall over 15 min, ending below 45 mph).

| stratum | share |
|---|---:|
| weekday commute peak | 88.4% |
| off-peak | 18.2% |
| by hour: 06 / 08 / 10 / 12 / 14 / 16 / 18 / 20 / 22 | 53 / 69 / 36 / 26 / 71 / 77 / 46 / 6 / 3 % |

Shockwaves are a commute phenomenon. This is why Table 1's compound window is
rain-and-peak and not rain-and-event.

## Table 3 — Rain reduces capacity; it does not trigger breakdowns

**[measured]** Controlling for time of day.

| | timesteps | any shockwave | sensors involved, given one |
|---|---:|---:|---:|
| commute, dry | 10,420 | 93.1% | 6.46 |
| commute, rain | 500 | 91.8% | 6.25 |
| off-peak, dry | 39,192 | 18.2% | — |
| off-peak, rain | 2,004 | 19.0% | — |

Mean speed: commute dry 54.26, commute rain 51.22 mph (**-3.04**); over all
hours **-2.22**. A uniform capacity reduction, not a trigger.

## Table 4 — Events do trigger breakdowns, and provenance decides whether you see it

**[measured]** Egress hour, matched controls (same weekday and clock, +-7/14
days, no event), sensors within 5 km road distance.

| tier | n | speed anomaly | shockwave: event vs control | odds ratio | z |
|---|---:|---|---|---:|---:|
| **high** (ESPN, announced attendance) | 37 | -0.63 +- 0.26 | 14/36 vs 4/54 | **5.25x** | **3.66** |
| medium (per-type default attendance) | 19 | +0.18 +- 0.11 | 0/19 vs 5/39 | 0.00x | -1.63 |
| low | 11 | -0.06 +- 0.12 | 1/11 vs 1/11 | 1.00x | 0.00 |
| all | 67 | -0.31 +- 0.15 | 15/66 vs 10/104 | 2.36x | 2.35 |

**The claim.** 58 of 95 fixtures carry a per-type DEFAULT attendance, so
filtering by an attendance threshold filters our own defaults. Dropping 30 of
67 qualifying fixtures raises both the effect size and the significance.

## Table 5 — Great-circle proximity hides the effect

**[measured]** High-confidence fixtures, egress hour.

| near-sensor definition | n sensors | speed anomaly | t | shockwave odds |
|---|---:|---|---:|---:|
| great-circle <= 5 km | 177 | -0.41 +- 0.23 | -1.75 (n.s.) | 2.50x |
| **road-network <= 5 km** | **54** | **-0.89 +- 0.26** | **-3.37** | **5.25x** |

128 sensor pairs lie within 150 m on opposite carriageways. Great-circle
separates them by a median 22 m; their speeds correlate at 0.115, against 0.850
for same-direction pairs up to 1 km apart. Only one carriageway carries egress.

## Table 6 — Negative results that are findings

**[measured]**

| claim | evidence |
|---|---|
| Compound rain-and-event exposure does not exist on PEMS-BAY 2017 H1 | 3 episodes, 18 timesteps in six months; one is a single 5-min step |
| A 60-min reporting horizon hides failure | AR(12) MAE by regime: <35 mph 17.56, 35-45 12.28, >55 2.17, ALL 3.20 |
| Aggregate MAE is dominated by free-flow | 90.4% of observations, 63.7% of absolute error, MAE 2.18 |
| Chronological splits put the wet season in train | `single` test block: 0 rain episodes. Folds 3-5: 0 rain, 0 NHL |

## Table 7 — Shockwaves travel upstream

**[measured]** `metrics.upstream_propagation` over the directed adjacency.
adj[i,j] > 0 means j is reachable driving from i.

| lag | upstream | downstream | ratio |
|---:|---:|---:|---:|
| 5 min | 136,588 | 108,582 | 1.26 |
| 10 min | 119,491 | 78,181 | **1.53** |
| 15 min | 96,042 | 56,044 | **1.71** |
| 30 min | 51,746 | 37,219 | 1.39 |

A ratio near 1 would mean the label was picking up network-wide congestion
rather than a travelling wave. Figure: `results/figures/propagation_ratio.png`.

---

# What the GPU run fills in

## Table 8 — Ablation, aggregate  `[GPU]`

| rung | c_in | 15min | 30min | 45min |
|---|---:|---|---|---|
| persistence **[measured]** | - | 1.520 / 1.635 / 1.548 | 2.099 / 2.272 / 2.099 | 2.566 / 2.783 / 2.535 |
| historical average **[measured]** | - | 2.937 / 2.885 / 2.506 | 2.937 / 2.884 / 2.506 | 2.937 / 2.884 / 2.506 |
| `0_speed` | 1 | `[GPU]` | | |
| `1_traffic` | 2 | `[GPU]` | | |
| `3_weather` | 5 | `[GPU]` | | |
| `5_event_geo_att` | 5 | `[GPU]` | | |
| `6_all` | 11 | `[GPU]` | | |

(three folds, comma-separated; mean +- sd over 3 seeds)

**Expected**: near-identical. Say so in the text BEFORE the table, or a reader
reads it as a failure. The point of Table 8 is that it cannot answer the
question - Table 9 can.

## Table 9 — Ablation inside the windows  `[GPU]`  <- the headline

45-min MAE inside each window. Compare DOWN a column only.

| rung | `weather_commute` | `adverse_weather` | `event_egress` | `holiday` |
|---|---|---|---|---|
| `0_speed` | `[GPU]` | | | |
| `1_traffic` | `[GPU]` | | | |
| `3_weather` | `[GPU]` | | | |
| `5_event_geo_att` | `[GPU]` | | | |
| `6_all` | `[GPU]` | | | |

n per fold: `weather_commute` 10/4/2 episodes, `adverse_weather` 27/12/12,
`event_egress` 4/10/8, `holiday` 1/0/0. Quote episodes, never samples. The
holiday column is n=1: a case, not an estimate.

**If this table is flat**, that is a result and it has an explanation ready:
the exogenous channels enter contemporaneously, and by the time it is raining
the slowdown is already in the speed history. Within these windows, 23.7% of
adverse-weather samples and 15.2% of event samples carry no trace of the
exogenous signal anywhere in the 60-min input window - a contemporaneous-only
model cannot see them at all. Using forecast values instead would be a
train/serve mismatch without a 2017 forecast archive to train on.

## Table 10 — Weighted loss  `[GPU]`

`--loss weighted` against plain L1, same rung, same folds, same seeds.

| | aggregate MAE | MAE where true < 45 mph | shockwave recall | lead (median, min) |
|---|---|---|---|---|
| `6_all` L1 | `[GPU]` | | | |
| `6_all` weighted | `[GPU]` | | | |

**Expected**: aggregate gets WORSE, low-speed gets BETTER. That trade is the
finding - it is the loss-function version of the paper's whole argument.

## Table 11 — Decision layer  `[GPU]`

| model | onsets | recall | false alarms | lead (median, min) |
|---|---:|---:|---:|---:|
| persistence **[measured]** | 16,743 | 0.67 | 1,822 | **0.0** |
| historical average **[measured]** | 16,743 | 0.75 | 4,507 | **0.0** |
| `0_speed` | | | | `[GPU]` |
| `6_all` | | | | `[GPU]` |

Median lead 0 for both trivial baselines is the control, not a bug: persistence
copies the last observation and cannot see an onset before it happens. Any
positive lead a trained model shows is measured against that floor. Pair with
the cost sweep in `results/decision__fold*.json` (miss:false-alarm ratios
1, 2, 5, 10, 20, 50).

## Figure 1 — Space-time diagram  `[GPU predictions]`

`python -m src.eval.figures --date <day> --freeway 101-N --predictions ...`

Observed | `0_speed` | `6_all`, side by side. A band leaning backwards is a
shockwave travelling upstream. Pick a rainy commute day from fold00's test
block (2017-01-29..02-26) - NOT a rain-and-event day, there are only three in
the corpus and they are case studies at n=1.
