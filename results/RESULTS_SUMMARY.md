# Results — H200 run v2 (with directed diffusion operator), 2026-09-05

**81 trained configurations**: 9 variants × 3 rolling folds × 3 seeds.
STGCN, 117–138k parameters. NVIDIA H200 NVL, ~3 s/epoch, 65–100 epochs.
Folds 00–02 only — folds 3–5 contain zero rain episodes and zero NHL fixtures.

The 9 variants: 5 ablation rungs (`0_speed`, `1_traffic`, `3_weather`,
`5_event_geo_att`, `6_all`); `0_speed` and `6_all` again with a
congestion-weighted L1 loss; and again with a directed diffusion graph operator
(`--graph-conv diffusion`, DCRNN/Graph WaveNet dual random walk) in place of
Chebyshev convolution.

---

# THE HEADLINE: different metrics select different models

All numbers averaged over 3 seeds and 3 folds, 45-min horizon. Rank in parentheses.

| model | Overall MAE ↓ | Rain×commute MAE ↓ | Propagation ↑ | Onset recall ↑ | False alarms ↓ |
|---|---|---|---|---|---|
| `5_event_geo_att` | **2.070 (1)** | **4.551 (1)** | 1.424 (7) | 0.752 (3) | 1,823 (7) |
| `6_all` + diffusion | 2.108 (2) | 4.946 (8) | 1.362 (8) | 0.740 (4) | 1,784 (6) |
| `1_traffic` | 2.116 (3) | 4.622 (2) | **1.522 (1)** | 0.726 (6) | 1,770 (4) |
| `6_all` + weighted | 2.118 (4) | 4.765 (4) | 1.435 (6) | **0.780 (1)** | 2,445 (9) |
| `6_all` | 2.124 (5) | 4.797 (6) | 1.452 (5) | 0.738 (5) | 1,775 (5) |
| `3_weather` | 2.206 (6) | 4.954 (8) | 1.516 (2) | 0.688 (9) | **1,381 (1)** |
| `0_speed` + diffusion | 2.213 (7) | 4.782 (5) | 1.348 (9) | 0.701 (8) | 1,380 (1) |
| `0_speed` (= published STGCN) | 2.214 (8) | 4.739 (3) | 1.466 (4) | 0.702 (7) | 1,433 (3) |
| `0_speed` + weighted | 2.280 (9) | 4.997 (9) | 1.401 (7) | 0.747 (4) | 2,354 (8) |
| persistence | 2.628 | 5.963 | — | 0.654 | 1,748 |
| historical average | 2.776 | 6.157 | — | 0.770 | 4,773 |

Reference: observed propagation ratio **1.683** (per fold 1.574 / 1.639 / 1.835).

**Spearman rank correlation between overall MAE and propagation fidelity:
−0.03 (n = 9, p = 0.93).** None. The model that wins on MAE
(`5_event_geo_att`) ranks 7th of 9 on reproducing the physics. The model that
best reproduces the physics (`1_traffic`) ranks 3rd on MAE and 6th on recall.
The model that catches the most onsets (`6_all` + weighted) is 4th on MAE and
carries the most false alarms.

**This is the paper.** Aggregate MAE does not merely under-state the differences
that matter — it selects a *different model* from the one any decision-relevant
metric selects.

---

# Full error table

45-min unless stated; mean over 3 seeds × 3 folds.

| model | 15min | 30min | 45min | adverse weather | rain×commute | event egress | holiday |
|---|---|---|---|---|---|---|---|
| `5_event_geo_att` | 1.370 | 1.789 | 2.070 | 2.285 | 4.551 | 1.181 | 1.781 |
| `6_all` + diffusion | 1.390 | 1.820 | 2.108 | 2.436 | 4.946 | 1.254 | 2.350 |
| `1_traffic` | 1.371 | 1.810 | 2.116 | 2.306 | 4.622 | 1.183 | 1.783 |
| `6_all` + weighted | 1.402 | 1.832 | 2.118 | 2.399 | 4.765 | 1.230 | 2.134 |
| `6_all` | 1.395 | 1.830 | 2.124 | 2.437 | 4.797 | 1.304 | 2.327 |
| `3_weather` | 1.402 | 1.868 | 2.206 | 2.442 | 4.954 | 1.222 | 2.175 |
| `0_speed` + diffusion | 1.390 | 1.864 | 2.213 | 2.417 | 4.782 | 1.230 | 1.819 |
| `0_speed` | 1.392 | 1.865 | 2.214 | 2.415 | 4.739 | 1.234 | 1.841 |
| `0_speed` + weighted | 1.412 | 1.906 | 2.280 | 2.513 | 4.997 | 1.272 | 1.900 |
| persistence | 1.568 | 2.157 | 2.628 | 2.841 | 5.963 | 1.329 | 1.930 |
| historical average | 2.776 | 2.776 | 2.776 | 3.088 | 6.157 | 1.878 | 3.786 |

Episodes per fold: `adverse_weather` 27/12/12, `weather_commute` 10/4/2,
`event_egress` 4/10/8, `holiday` 1/0/0. **Quote episodes, never the samples
they contain.** The holiday column is n=1: a case, not an estimate — and note
that `event_egress` reads *lower* than the overall column because egress falls
at 21:30–23:00 when the network is empty. Compare down a column only.

---

# What each metric measures

| metric | question | definition |
|---|---|---|
| **Overall MAE** | how wrong on average | mean abs. error in mph, 45-min horizon, all test samples |
| **Rain×commute MAE** | how wrong when it matters | same, restricted to adverse precipitation (≥0.51 mm) during weekday 07–10 / 15–19 |
| **Propagation** | did it learn the physics | upstream/downstream co-occurrence of shockwave onsets at a 15-min lag over the **directed** adjacency, computed on the model's own predictions. Observed truth = 1.683; 1.0 would mean no directional preference at all |
| **Onset recall** | fraction caught | share of observed congestion onsets (< 45 mph, 15 min) the model flagged |
| **False alarms** | cost of catching them | flagged onsets that did not occur |

---

# Finding 1 — Weather channels make the model worse

`3_weather` loses to `1_traffic` on every error metric, and loses to the plain
speed-only baseline on rain×commute MAE (4.954 vs 4.739, **+4.5%**), adverse
weather MAE (2.442 vs 2.415, +1.1%) and onset recall (0.688 vs 0.702 — the
worst recall of any trained model).

Consistent with what the data said before any training: controlling for time of
day, rain does **not** change how often breakdowns occur (91.8% of commute-peak
timesteps carry a shockwave in rain against 93.1% dry) nor how large they are
(6.25 sensors involved against 6.46). Rain is a uniform capacity reduction —
−3.04 mph in the peak — and a uniform slowdown is already legible in the speed
history the model has.

# Finding 2 — Spatial structure decides everything

Two rungs, **identical parameter count (118,732) and identical channel count
(c_in = 5)**, differing only in what the three added channels contain:

| | added channels | spatial structure | rain×commute MAE |
|---|---|---|---|
| `3_weather` | temperature, precipitation, wind | **none** — one value broadcast to all 325 nodes | **4.954** |
| `5_event_geo_att` | venue distance, event activity, attendance load | **per-node**, exponentially decayed over directed road distance | **4.551** |

Same capacity, opposite outcome, in every window and on every fold.

**On a graph model a channel with no spatial variation is not a weak feature —
it is a harmful one.** Passed through a graph convolution, a broadcast scalar is
close to a global bias, and it spends three dimensions of capacity saying
nothing about which node is which.

This also reframes `5_event_geo_att`: it wins in windows with no events in them
(rain×commute, adverse weather), so the gain is unlikely to be event timing. The
likely mechanism is `dist_to_venue` — STGCN has no node embeddings, and it is
the only channel that distinguishes one sensor from another.

Finding 5 sharpens this further.

# Finding 3 — The weighted loss trades false alarms for recall, at no cost in MAE

| | overall MAE | onset recall | false alarms |
|---|---|---|---|
| `0_speed` | 2.214 | 0.702 | 1,433 |
| `0_speed` + weighted | 2.280 (−3.0%) | **0.747** (+4.5 pt) | 2,354 (+64%) |
| `6_all` | 2.124 | 0.738 | 1,775 |
| `6_all` + weighted | **2.118 (unchanged)** | **0.780** (+4.2 pt) | 2,445 (+38%) |

Weighting the L1 loss towards low speeds costs 3% of aggregate MAE on `0_speed`
and **nothing at all** on `6_all` (2.124 → 2.118, seed sd ±0.14), and in
exchange raises onset recall by roughly 4 points in both cases, at 38–64% more
false alarms.

Whether that trade is worth taking depends on the cost ratio, and the two users
of this forecast sit at opposite ends of it: a network operator pre-deploying a
signal plan pays staff hours for a false alarm and hours of egress delay for a
miss (ratio 20–50), while a navigation service pays one driver a 2–5 minute
detour against a measured median 1.3 and p90 5.3 minute delay for driving into
the jam (ratio 1–3). `results/decision__fold*.json` sweeps ratios 1, 2, 5, 10,
20, 50 — do not pick one, name which end belongs to whom.

# Finding 4 — The models do learn propagation, but imperfectly

Observed upstream/downstream ratio at a 15-min lag is **1.683**. Every
Chebyshev model reproduces 83–90% of it — none collapses to 1.0, so they are
learning a travelling wave rather than a smooth conditional mean. But none
reaches the truth either, and the ranking is uncorrelated with MAE.

# Finding 5 — A directed graph operator makes propagation *worse*

**The hypothesis, written before the run.** Chebyshev convolution requires a
symmetric Laplacian, so the model cannot tell an upstream neighbour from a
downstream one — while the phenomenon travels upstream 1.68× more often than
downstream. Given Finding 2, spatial structure is what this model responds to,
so a directed operator looked like the most promising remaining change.

**The result reverses it.** 3 folds × 3 seeds, 45-min horizon, 15-min lag:

| | fold00 | fold01 | fold02 | **mean (n=9)** | sd |
|---|---|---|---|---|---|
| observed truth | 1.574 | 1.639 | 1.835 | **1.683** | — |
| `0_speed` | 1.510 | 1.453 | 1.435 | **1.466** | 0.042 |
| `0_speed` + diffusion | 1.379 | 1.310 | 1.355 | **1.348** | 0.053 |
| `6_all` | 1.493 | 1.432 | 1.430 | **1.452** | 0.065 |
| `6_all` + diffusion | 1.365 | 1.363 | 1.357 | **1.362** | 0.070 |

Paired by (fold, seed):

```
0_speed   diffusion − Chebyshev:  −0.118 ± 0.053   t = −6.64   worse in 9/9 pairs
6_all     diffusion − Chebyshev:  −0.090 ± 0.090   t = −3.00   worse in 6/9 pairs
```

Meanwhile aggregate MAE barely moves: `6_all` 2.124 → 2.108 and `0_speed`
2.214 → 2.213, both inside the seed sd of ±0.17. Parameters rise from 121k to
138k and from 117k to 133k.

This is the cleanest single-variable experiment in the set — only the spatial
operator changes; channels, loss, data and splits are held fixed — and it is a
third independent instance of the headline: **aggregate MAE would have reported
this change as a marginal improvement while the model moved further from the
physics.**

It also refines Finding 2. What helps is a channel that tells one node from
another (`dist_to_venue`), not a mechanism for directional propagation on the
graph. Giving the operator the ability to distinguish upstream from downstream
did not make it learn the upstream wave; it flattened it.

---

# Caveats to state out loud

- **Lead time cannot be reported from this dataset.** `metrics.onset_lead_time`
  (model prediction vs a reactive rule of < 45 mph for 15 min, 28 sensors within
  2 km of a venue) returns a **median of 0.0 min for all 19 model entries**, with
  means between 0.00 and 0.32 min. The cause is structural: the forecast horizon
  is 60 min and the reactive rule carries its own 15-min confirmation delay, so
  the resolvable range is very narrow. Onset recall and false alarms measure the
  same thing and do discriminate. *An earlier draft reported leads of 15.6 /
  25.6 / 40.0 min; those do not reproduce from the prediction files and have
  been withdrawn.*
- Folds 00–02 only, selected on test-block content **before** any training.
  A conventional chronological 70/10/20 split of PEMS-BAY has **zero rain
  episodes in its test block** — California's wet season is January–April — so
  it cannot answer the weather question at all.
- Event windows are small: 4, 10 and 8 episodes per fold. Quote episodes, never
  the samples they contain.
- **Compound rain-and-event exposure does not exist here**: 3 episodes,
  18 timesteps in six months, one of them a single 5-min step. Any benchmark
  headlining compound-event performance on this corpus is reporting noise.
- All event numbers use the 37 fixtures with ESPN-verified attendance; the other
  58 carry a per-type default we assigned ourselves.
- `historical_average` shows competitive recall (0.770) but carries 4,773 false
  alarms against `6_all`'s 1,775. Do not report it as competitive.
- The `event_egress` column reads *lower* than the overall column for every
  model. Egress falls at 21:30–23:00 on an empty network; this is the clock, not
  the fixture. Compare models down a column, never a window against its
  complement.

---

# Where the numbers come from

Bundle: `results_bundle_v2_diffusion.tgz` (8.1 GB, 87 predictions, 81
checkpoints). It is a strict superset of the earlier `results_bundle.tgz`.

| number | source |
|---|---|
| overall / per-horizon MAE | `checkpoints/*.json` → `test.45min.mae` |
| window MAE | `results/report__fold0*.json` → `<window>.inside.45min.mean` |
| onset recall, false alarms | `results/decision__fold0*.json` → `models.<v>.all.{recall,fp}` |
| propagation fidelity | `metrics.upstream_propagation(pred[:, 8, :], adj, lags=(3,))` on predictions sorted by `timesteps`; `adj` must be the **directed** `data/processed/adj_mx.npy` (1,789 one-way edges), not the symmetrised Laplacian the model uses |

Figures already rendered in the bundle: `results/figures/spacetime_101-N_2017-02-17.png`,
`…_2017-03-24.png`, `…_2017-04-17.png`, `results/figures/propagation_ratio.png`.
