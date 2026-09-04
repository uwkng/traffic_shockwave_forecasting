# Weather pipeline — handover notes

**From:** the events/stage-1b side · **To:** whoever owns stage-1 weather
**Date:** 2026-09-04 · **Status:** input for the owner, not a change request

Before the ownership split (weather → you, events → me) I had independently
rewritten stage-1 weather from Open-Meteo to ASOS. That work is on
`data_preparation_1`; yours is on `main` and therefore in `data_pipeline`.
**Yours is the version we are keeping.** This note exists only so the
measurements I made along the way are not lost — several of them contradict
choices in the current implementation, and I would rather hand you the numbers
than let them disappear with a stale branch.

Everything below was measured on this dataset. Nothing here is a style opinion.

---

## 0. Where we already agree

Worth stating first, because it is most of the file:

- **ASOS over reanalysis.** Same conclusion, independently. ERA5 over-reports
  precipitation by **+53% (NUQ) / +66% (SJC)** against the gauges.
- **SJC + NUQ as the two stations.**
- **Reindexing onto `speed.csv`'s own columns.** Your `_parse_asos` does
  `idx = pd.to_datetime(speed_cols); out.reindex(idx)`. That is exactly right and
  it is the single easiest way to destroy val+test — the naive 5-min grid has
  52,128 slots, `speed.csv` has 52,116, and the missing hour is the DST
  spring-forward on 2017-03-12. Building the axis with `pd.date_range()` would
  shift everything after that date by an hour.
- **Linear interpolation for temperature and wind.** You do
  `.interpolate(method="time")`; I reached the same place. ASOS reports
  temperature to 1 °F, so the median within-hour range is 1.11 °C — exactly 2 °F,
  i.e. quantisation noise — while the median hour-to-hour change is 0.61 °C.
  Nothing real happens inside the hour, so interpolating is honest and ffill
  would just add stair-steps.
- **Keeping Open-Meteo as a cross-check** rather than deleting it.

---

## 1. The one that matters most: weather has no spatial dimension

**Current behaviour.** `build_weather()` computes one IDW weight per station
against the *network centroid*:

```python
centroid = _sensor_centroid()
weights[station] = 1.0 / math.sqrt(dlat**2 + dlon**2)
result[col] = (vals * w).sum(axis=1) / w.sum(axis=1)      # -> one column
```

The output is `[T]`. All 325 sensors read the same value at each timestep.

**Why this matters here specifically.** The proposal's contribution is evaluating
*inside exogenously-defined anomalous windows*, and weather windows are half of
that. Measured on this network:

| Measurement | Value |
|---|---|
| Wet hours where NUQ and SJC **disagree** (12 km apart) | **42.7%** |
| Traffic signature difference depending on **which side** is wet | **1.46 mph, p = 1.7e-4** |
| Network extent | 28.9 km |

So in 4 out of 10 wet hours the two gauges are telling different stories, and the
traffic response *flips sign* with which one is raining. A `[T]` series cannot
represent that — not as a tuning problem, but structurally. It is also the exact
reason I dropped ERA5 as the primary source: its ~31 km grid is larger than the
28.9 km network, so every node read the same cell.

**Suggested change.** Compute the IDW weights per *node* rather than per
centroid, so the output is `[T, N]`:

```python
w = 1.0 / np.maximum(dist[station, node], 0.1) ** power    # [S, N]
precip = np.einsum("st,sn->tn", vals, w) / np.einsum("st,sn->tn", ok, w)
```

This is ~10 lines and does not change anything else about your pipeline. My
version is `align_asos()` in `src/data/align.py` on `data_preparation_1` if you
want a reference, but the change is small enough to write directly.

---

## 2. `p01i` is a backward 1-hour accumulation, and ffill changes what it means

**Current behaviour.**

```python
out.index = out.index.round("5min")
out = out.groupby(out.index).mean()
...
out["precipitation"] = out["precipitation"].ffill().fillna(0.0)
```

**The issue.** METAR `p01i` is *precipitation accumulated over the past hour*, not
a rate and not a 5-minute amount. Two consequences:

1. A storm hour carries SPECI reports in addition to the routine `:53`
   observation, and **every one of them repeats the past-hour total**. Their
   windows overlap. Averaging co-binned reports, or forward-filling from
   whichever came last, mixes overlapping windows — the resulting number is
   neither an hourly total nor a rate.
2. Forward-filling holds one hour's total constant across up to twelve 5-minute
   steps, so a single hour of rain reads as a plateau rather than an amount.

**What I measured.** Traffic integrates rainfall — wet pavement outlasts the
shower — so the useful predictor is an accumulation, but the *window* has to be
consistent. On true 1-minute gauge data, R² against the speed anomaly:

| Representation | R² |
|---|---|
| Instantaneous 5-min rate | 0.032 |
| Hourly step (what ffill approximates) | 0.064 |
| **1-hour trailing accumulation** | **0.070** |

(With ERA5 hourly input the same change gives 0.0471 → 0.0543.)

**Suggested change.** Take the single observation per hour whose minute is
closest to `:53` (the routine report, non-overlapping), then build the 5-min
series as a trailing accumulation:

```python
cum  = hourly.cumsum()
cum5 = cum.reindex(cum.index.union(time_index)).interpolate(method="time").reindex(time_index)
precip = (cum5 - cum5.shift(span)).fillna(cum5).clip(lower=0.0)   # span = 12
```

Interpolating the cumulative sum and differencing is algebraically the same as
interpolating the rate, and it keeps the total mass right.

**One caveat that cuts the other way:** an interpolated series must *not* be used
to define the evaluation windows themselves, or rain gets smeared into dry hours
and the window labels become circular. I kept the raw hourly series separately
for that purpose. Worth doing whatever the final representation is.

---

## 3. `wxcodes` is not being downloaded, and it is the strongest weather channel

**Current behaviour.** The IEM request asks for three fields:

```python
"&data=tmpf&data=p01i&data=sknt"
```

**What is missing.** METAR present-weather codes (`-RA`, `RA`, `+RA`, `TS`, …).
Because they are not in the request, they are not in `asos_*.csv`, so this is not
a processing choice — the raw files cannot support it without a re-download.

**What I measured.** A categorical intensity ladder built from those codes,
against the network speed anomaly:

| Level | Effect |
|---|---|
| none | +0.19 mph |
| light (`-RA`, `-DZ`, `-SN`) | −1.86 mph |
| moderate (`RA`, `DZ`, `SN`) | −3.03 mph |
| heavy (`+RA`, `TS`, `TSRA`, …) | −3.38 mph |

Per individual code it separates further: none +0.18 · `BR` (mist, no rain)
+0.66 · `-RA` −1.65 · `RA` −2.67 · `RA BR` −4.17 mph. **It discriminates better
than mm/h does.** I gave it its own channel.

**Suggested change.** Add `&data=wxcodes` to the request (and `&data=vsby` while
you are there, see below). One note on handling: take the code from each node's
*nearest* station rather than interpolating — averaging an ordinal category is
meaningless — while amounts use IDW. Also take the *worst* code seen in the hour,
not the last: a 20-minute downpour matters even if the hour closes dry.

**Visibility was tested and rejected.** `vsby` is worth downloading for the
audit trail, but after controlling for rain, low-visibility *dry* hours are
**faster** (+0.55 mph) and it adds only +0.018 R². I did not give it a channel.

---

## 4. Smaller items

**Station qualification.** RHV and PAO look like reasonable extra stations and
are not. Both have **zero night observations** and gaps up to **38 hours**. Event
egress happens at night, which is precisely when we need the data. My
`acquire.py` asserts on it:

```python
if night == 0:
    raise ValueError(f"{name} has no night observations - unusable")
```

Worth adding before anyone extends the station list.

**Station coordinates differ between our versions:**

| Station | Yours | Mine |
|---|---|---|
| SJC | 37.3626, −121.9291 | 37.3594, −121.9244 |
| NUQ | 37.4161, −122.0494 | 37.4059, −122.0490 |

0.5–1.2 km apart. Barely matters for centroid IDW; it matters more under
per-node IDW. Worth pinning against an authoritative source — I verified venue
coordinates against OpenStreetMap Nominatim for the events work and it was cheap
(note: Nominatim *requires* a User-Agent header, unlike the ESPN API which
rejects one).

**Distance uses a planar approximation** (`dlat * 111`, `dlon * 111 * cos(lat)`).
Fine at this scale; haversine costs nothing if you are touching the code anyway.

---

## 5. Two things that will bite someone regardless of the above

**(a) The docstring and the code disagree about the output filename.**

```python
#           build_weather() -> IDW merge -> weather_hourly.csv     <- comment
out_path = WEATHER_DIR / "weather_5min.csv"                        <- code
```

`data/raw/weather/weather_hourly.csv` still exists on disk and still holds the
**old Open-Meteo hourly data** (4,344 rows, generated 2026-08-28). Anyone
following the comment reads stale reanalysis instead of the new gauge data —
**and it will not error**. Either rename the output or delete the old file.

**(b) No weather data file is committed.** `main` carries the code but neither
`asos_SJC.csv`, `asos_NUQ.csv` nor `weather_5min.csv`. A fresh clone must have
network access and run `fetch_asos()` twice plus `build_weather()` before
stage 2-3 can do anything. For comparison, `events.csv` is committed as a
finished artefact and reproduces byte-identically offline from a cached
request store. Committing the two raw ASOS CSVs (~10 MB total) would make the
whole pipeline runnable offline.

Related: `build_weather()` short-circuits on `if out_path.exists(): return`, so
changing a parameter silently reuses the old file. Worth a `--force` flag or a
parameter hash in the filename.

---

## Priority, if you only do some of this

| # | Change | Why |
|---|---|---|
| 1 | **Per-node IDW → `[T, N]`** | The 1.46 mph / p=1.7e-4 spatial effect is unobservable without it, and it is half the paper's contribution |
| 2 | **Add `wxcodes` to the request** | Needs a re-download, so the sooner the better; strongest measured weather signal |
| 3 | **Trailing accumulation for precip** | R² 0.064 → 0.070, and fixes the overlapping-window problem |
| 4 | **Fix the filename mismatch / delete the stale `weather_hourly.csv`** | Silent wrong-data trap in stage 2-3 |
| 5 | Commit the raw ASOS CSVs | Makes the pipeline reproducible offline |
| 6 | Night-observation assertion, coordinates, haversine | Cheap guards |

Reference implementation for 1–3: `src/data/align.py` (`align_asos`,
`_station_hourly`, `_trailing_accum`) on `data_preparation_1`. Happy to walk
through any of it, or to open a PR against whichever branch you prefer — this is
your area, so the call on all of it is yours.
