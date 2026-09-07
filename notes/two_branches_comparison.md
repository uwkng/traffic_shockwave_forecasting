# `mei_modeling` vs `modeling` — what differs, and what to do about it

Two branches trained the same STGCN on the same 11 channels and reached different
numbers. This note records where the difference comes from and what we propose to
do with both sets of results. Every claim here was checked against the branches,
not assumed.

---

## Part 1 — The differences

### The three that matter

| | `mei_modeling` | `modeling` | Why it changes the answer |
|---|---|---|---|
| **Data split** | Rolling folds 00–02: three consecutive 28-day test blocks, 29 Jan – 23 Apr | Single chronological 70/10/20; test ≈ 25 May – 30 Jun | The single split's test block contains **zero adverse-weather episodes** — the Bay Area's wet season is Jan–Apr, so any chronological split puts every rain spell in train. The rolling folds carry **51 rain spells** and **19 NHL fixtures**. Both branches found this independently; `modeling/RESULTS.md` prints `n=0 samples` for its adverse-weather stratum and names the same cause. |
| **Future covariates** | **Off.** The model sees only the past 60 min of all 11 channels | **On.** The known-future value at `t+H` is appended for calendar, weather and event channels, so `c_in` reaches **19** (11 historical + 8 future) | The two branches answer different questions. One: *does knowing the current weather and fixture state help?* The other: *does knowing the forecast and the published schedule help?* Exogenous channels can only add value where they carry information the speed history does not, and only the future-facing version does. |
| **Seeds** | **3** (42, 43, 44); every number is a mean over 3 seeds × 3 folds = 9 runs | **1** (42) | Single-seed gaps cannot be separated from initialisation noise. Measured on the rolling folds, the seed-and-fold spread of aggregate MAE at 60 min is **±0.15 to ±0.25 mph** — larger than several rung-to-rung gaps in the single-seed table (e.g. `4_event_geo` 2.214 vs `5_event_geo_att` 2.212, a difference of 0.002). |

### Secondary differences

| | `mei_modeling` | `modeling` |
|---|---|---|
| Weather in the **input** window | ASOS SJC + NUQ, observed | ASOS SJC + NUQ, observed — **identical** |
| Weather in the **future** window | n/a | ECMWF IFS short-range forecast (Open-Meteo Historical Forecast API) |
| Ablation rungs trained | 5 of 7 (`2_calendar`, `4_event_geo` pending) | all 7 |
| Extra configurations | congestion-weighted loss, directed diffusion operator | none |
| Metrics | MAE/RMSE/MAPE, window strata, propagation fidelity, onset recall, false alarms | MAE/RMSE/MAPE, window strata, distance bins |
| Adjacency | published `adj_mx_bay.pkl` | rebuilt from the distances CSV with the same Gaussian kernel (the pickle is a Python-2 pickle and fails on 3.11+) |

### The `modeling` numbers are genuinely lower, and the reason is not an easier test set

`modeling` reports MAE@60 of 2.428 for `0_speed` and 2.107 for `6_all`, against
2.502 and 2.361 on the rolling folds. We checked whether its test block is simply
easier, using persistence — a model-free reference that needs no training:

| test block | persistence @15 | @30 | **@60** | mean speed | share < 45 mph |
|---|---:|---:|---:|---:|---:|
| single (`modeling`), 25 May – 30 Jun | 1.592 | 2.173 | **3.041** | 62.47 | 6.15% |
| folds 00–02 (`mei_modeling`), 29 Jan – 23 Apr | 1.568 | 2.157 | **3.043** | 62.87 | 5.66% |

**3.041 against 3.043.** The two test blocks are equally hard. The lower numbers
are real, and they have two legitimate causes:

1. **2.6× more training data.** The single split trains on 36,449 samples; the
   rolling folds train on 6,048 / 14,112 / 22,141, averaging 14,100. `fold00`
   trains on three weeks. Same model, same code, more data.
2. **Future covariates carry information the other branch does not receive.**

### Two things the forecast weather does right

Worth stating because they are easy to get wrong, and `modeling` did not:

- **Only the future portion is swapped.** The input window keeps the observed
  ASOS weather; the forecast replaces it only at `t+H`. Using observed values in
  the future portion would be leakage, and the code does not do that.
- **It is a forecast, not a reanalysis.** The Historical Forecast API returns the
  shortest-lead-time prediction from each successive model run, so every value
  uses only information available before its own timestamp.

This closes a gap the `mei_modeling` write-up currently calls impossible. Its
Limitations section says a forecast-conditioned model "would be a train/serve
mismatch without a 2017 forecast archive to train on". The archive exists and
`modeling` used it. **That sentence has to be corrected.**

---

## Part 2 — What to do

### Use both. They are not competing, and neither answers the other's question.

| use | to answer | goes where in the paper |
|---|---|---|
| `modeling`, single split, future covariates | Does exogenous information help at all, and how does the model compare to published PEMS-BAY numbers? | one aggregate table, plus the comparison to STGCN / DCRNN / Graph WaveNet |
| `mei_modeling`, rolling folds | *Where* does it help, and does aggregate MAE rank models correctly? | main results, window strata, propagation fidelity, decision layer |

### The sentence that only works with both

> Future covariates (lookahead) buy a 13.2% aggregate improvement. Contemporaneous
> covariates buy almost nothing in aggregate and are actively worse inside the
> rain × commute window. **The value of exogenous information is not what it
> describes, but whether it carries lookahead the speed history does not already
> contain.**

Neither branch can make that claim alone. It is a better finding than either
branch's headline, and it costs nothing extra to state — both experiments are
already run.

### Why we should not simply adopt the `modeling` numbers as the result

The stated contribution of this project is an **evaluation protocol**, not a model.
The `modeling` table is exactly the aggregate-MAE table the protocol argues is
insufficient. Replacing the rolling-fold results with it would remove the
contribution and leave a conventional ablation paper. Concretely, three claims
would become unsupportable:

- **Anything about weather.** Its test block has zero rain. `RESULTS.md` says so.
- **Any rung-to-rung ordering.** Single seed, and several gaps are smaller than
  the ±0.15–0.25 mph spread measured across seeds and folds.
- **The central finding.** Aggregate MAE and propagation fidelity rank the nine
  configurations with a Spearman correlation of −0.23. That cannot be shown from
  a table that reports only aggregate MAE.

### One question to settle before either table is written up

`2_calendar` is the single largest gain in the `modeling` ablation: −10.5% at
60 min. The channels are time-of-day, day-of-week and is-holiday, and their
*future* values are what the model receives. But a model that already sees 60
minutes of speed history can infer the time of day from it — the measured
incremental R² of `time_of_day` against an AR(12) residual is **+0.019**.

So: why would knowing that the target timestep is 17:00 be worth 10.5%? A
plausible reading is that the future calendar channel is acting as a **positional
encoding** that lets the model align to the daily cycle, rather than as exogenous
information. If that is what it is, it should be described that way — it is still
a real gain, but it is an architectural finding, not an argument that external
data helps.

### Concrete next steps

1. Correct the Limitations paragraph in `paper/main.tex`: the forecast archive
   exists, `modeling` used it correctly, and the open question is that its test
   block has no rain to forecast.
2. Add one aggregate table from `modeling` to the paper, clearly labelled as the
   single split, next to the published PEMS-BAY rows already in
   `results/results_main.csv`.
3. Keep the rolling-fold results as the main experiment.
4. Ask about `2_calendar` before writing it up.
5. Re-run `modeling`'s configurations with 3 seeds if GPU time allows. This is
   the cheapest thing on the list that changes what can be claimed.

---

## Settled 2026-09-07: the future covariates were run on the rolling folds

`mei_modeling_on` ran all five applicable rungs with known-future covariates,
3 folds x 3 seeds = 45 models, on the same folds, seeds and scaler as the
contemporaneous ablation. That makes each `__future` run a controlled test
against its own base rung, which is what neither branch had before.

**Result: pooled over the 15 rung x fold pairs, lookahead is harmful at every
horizon** (+0.005 / +0.009 / +0.015 / +0.017 mph at 15/30/45/60, paired
t(14) = +2.42 / +2.64 / +2.56 / +2.14, p = 0.030 / 0.020 / 0.023 / 0.051).

### This answers "one question to settle" above, and the answer is no

The question was whether `modeling`'s -10.5% at 60 min on `2_calendar` came from
the *future* calendar values acting as a positional encoding. It did not come
from the future values at all:

| | 60-min MAE |
|---|---:|
| `2_calendar`, contemporaneous | **2.279** |
| `2_calendar__future` | 2.290 |

0 of 3 folds improve. Tuesday 18:00 is a closed-form function of Tuesday 17:00,
so `time_of_day@future` restates a channel the model already holds; it buys
nothing and costs 1,344 parameters. The gain `modeling` measured is real, but it
belongs to **having the calendar channels at all** - which the rolling-fold
ablation also finds, `2_calendar` being the best of all seven rungs - not to
seeing them ahead of time. Item 4 of "concrete next steps" is closed.

### And it closes item 1 as well

The Limitations paragraph no longer needs correcting toward "the open question is
that `modeling`'s test block has no rain". The question is answered on folds that
*do* have rain: 51 spells across the three test blocks. Future weather makes
every rung that carries it worse (`3_weather` +0.040, `6_all` +0.051 at 60 min,
0/3 folds each), and the mechanism is measured - the ECMWF IFS archive recalls
only **61.4%** of wet hours, so it is fed in as fact while being wrong about 4 in
10 of exactly the timesteps that matter.

### What did help, and the rule it gives

`4_event_geo__future` is the only configuration that improves, in **all 12
fold x horizon cells**, monotonically with lead time (-0.002 / -0.006 / -0.013 /
-0.021). Its future block is one channel: the event schedule.

> A known-future channel pays only if it is **not derivable from the present**
> *and* **known exactly**. Calendar fails the first. A real weather forecast
> fails the second. Of the eight knowable channels only the event schedule
> satisfies both.

Adding attendance to that same future block cancels the whole gain
(-0.021 -> +0.006), which is now the third independent measurement in this
project saying crowd size carries nothing beyond fixture presence.

### Bearing on "should we adopt the teammate's results"

Unchanged, and now better supported. Keep both branches: they answer different
questions, and the `__future` runs show the difference between them is not what
either of us assumed. The `modeling` numbers are lower because of 2.6x more
training data, not because lookahead helps - lookahead, measured properly, hurts.

