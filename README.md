# Traffic Shockwave Forecasting

Event- and weather-aware spatio-temporal graph networks for traffic shockwave
forecasting. Course project, *Deep Learning and Decision Making* (TUM).

**The contribution is the evaluation protocol, not the model.** Off-the-shelf
STGCN, conditioned on weather and scheduled-event signals, evaluated *inside
exogenously-defined anomalous windows* — an event egress hour, an adverse-weather
spell — rather than on average.

- **`STATUS.md`** — what is done, what is not, and the decision log with the
  measurement behind each choice. Read it before changing anything.
- **`src/contract.py`** — shapes, channel order, splits, prediction format.
  The interface all three workstreams build against.

## Quick start

```bash
git clone <repo-url> && cd traffic_shockwave_forecasting
pip install numpy pandas pyyaml requests        # pipeline
pip install torch scipy                         # training only

# first run also clones PEMS-BAY (233 MB) and downloads the weather
python -m src.data.build_dataset --config configs/default.yaml --acquire

# rebuilds take ~15 s afterwards
python -m src.data.build_dataset --config configs/default.yaml

# sanity check: prints every rung, every split, and one real sample
python -m src.models.loader --split single
```

If the download fails with `SSL: CERTIFICATE_VERIFY_FAILED`, your Python has no
CA bundle — the usual python.org-installer situation on macOS. Fix the machine,
not `acquire.py`: `open "/Applications/Python 3.x/Install Certificates.command"`.

## What you get

`data/processed/`, ~1 GB, gitignored — rebuild it, do not ship it.

| File | Shape | What |
|---|---|---|
| `master.npy` | `[52116, 325, 11]` | the whole period, every channel. 745 MB; open with `mmap_mode="r"` |
| `splits.npz` | 21 arrays | sample indices: `single__train`, `fold00__test`, … |
| `scalers.json` | — | per-channel mean/std, one set per split |
| `eval_mask.npy` | `[52116, 325]` | True where speed was observed, not imputed |
| `adj_mx.npy` | `[325, 325]` | Gaussian-kernel adjacency, threshold 0.1 |
| `*_meta.json` | — | per-stage provenance, human-readable |

`master.npy` is not split. It is the full six months; `splits.npz` indexes into
it. Same tensor, different bookmarks.

## Using it

```python
from src.models.loader import build_loaders

loaders, adj, scaler = build_loaders(rung="6_all", split="fold00")
model = STGCN(c_in=loaders["c_in"], ...)

for X, Y in loaders["train"]:        # X normalised, Y raw mph
    ...
pred_mph = scaler.to_mph(pred)       # before any metric
```

`rung` selects channels (`"0_speed"` … `"6_all"`), `split` selects bookmarks
(`"single"`, `"fold00"`…`"fold05"`). Nothing else changes.

The loader owns the seven things that are easy to get wrong and never fail
loudly — which columns, which split's scaler, per channel not global,
passthrough channels, raw target, de-normalisation, the eval mask. Its docstring
explains each. Don't reimplement them.

### Which split

**Use one split for the whole paper.** Two configurations evaluated on different
test sets cannot be subtracted.

| | `single` (70/10/20) | `fold00`…`fold05` (rolling) |
|---|---|---|
| for | reproducing published numbers | everything else |
| test rain episodes | **0** | 51 |
| test event episodes | 17 | 50 |

> ⚠️ The `single` test block (2017-05-25 → 06-30) contains **zero**
> adverse-weather episodes — California's wet season is January to April. A
> weather-conditioned model cannot differ from a traffic-only one there, because
> the channel is constant across the whole block. `split.py` prints this when it
> runs. Weather results must come from the rolling folds.

## Raw data

| Source | Location | Committed |
|---|---|---|
| PEMS-BAY (speed, occupancy, graph, labelled congestion blocks) | `data/raw/augmented-pems-bay/` | no — cloned by `acquire.py` |
| Weather — ASOS SJC + NUQ, IDW-merged | `data/raw/weather/weather_5min.csv` | yes, 3 MB |
| Events — 95 fixtures, 4 venues | `data/raw/events/events.csv` | yes |

`data/raw/events/README.md` is the events data dictionary: every column, its
source, and whether each value was observed or assumed.

## Traps

1. **Never build the time axis with `pd.date_range`.** PEMS-BAY timestamps are
   local wall-clock US Pacific and 2017-03-12 02:00–02:55 does not exist (DST).
   181 × 288 = 52,128, but the file has 52,116 columns. Generating the axis adds
   12 phantom steps and silently shifts every later timestamp by an hour. The
   axis comes from `speed.csv`'s own column names; join weather and events **by
   label**.
2. **The scaler is per channel, and per split.** One global mean/std would
   average mph with a 0/1 holiday flag, kilometres and millimetres.
3. **We keep weekends and holidays**, unlike Yu et al. 48% of our events fall on
   a weekend; excluding them removes the windows this project evaluates.
4. **`weather_hourly.csv` is gone.** It was a stale Open-Meteo export sitting at
   the path the config pointed to while `build_weather()` wrote
   `weather_5min.csv`. The file existed, so nothing raised. If you see that name
   anywhere, it is stale.

## Layout

```
configs/default.yaml    paths, thresholds, split parameters
src/contract.py         shapes, channel order, splits  <- the interface
src/data/               stages 1-7
src/models/loader.py    data/processed -> model input
src/eval/windows.py     the exogenous window definitions
data/raw/               downloads; only weather + events are committed
data/processed/         built tensors (gitignored, rebuild in 15 s)
notebooks/              stage 8 reproduction
STATUS.md               status, ownership, decisions
CLAUDE.md               rules for Claude Code sessions
```

## Reproducibility

Chronological splits, no shuffling. Scalers fit on train only, per channel, per
fold. All metrics after de-normalization, masked with `eval_mask.npy`. Seed
everything; 3 seeds per trained config.
