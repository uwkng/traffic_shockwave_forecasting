# Traffic Shockwave Forecasting

Event- and weather-aware spatio-temporal graph networks for traffic shockwave
forecasting. Course project, *Deep Learning and Decision Making* (TUM).

**The contribution is the evaluation protocol, not the model.** We take
off-the-shelf STGCN (main) and DCRNN (optional), condition them on weather and
scheduled-event signals, and evaluate accuracy inside exogenously-defined
anomalous windows and by distance from the venue.

## Quick start

```bash
# 1. Clone and set up environment
git clone <repo-url> && cd traffic_shockwave_forecasting
uv sync                          # installs all dependencies from pyproject.toml

# 2. Acquire raw data (~200 MB PEMS-BAY clone + weather API call)
python -m src.data.acquire
```

## Project status

**Check `STATUS.md`** for what's done, what's in progress, who owns what,
and what to work on next. Update it after every meaningful change.

## Data contract (the interface everyone builds against)

Defined in `src/contract.py`. Summary:

| Artifact | Shape | Notes |
|---|---|---|
| master tensor | `[52116, 325, 9]` | `[T, N, channels]` |
| sample X | `[12, 325, 9]` | 60 min history, all channels |
| sample Y | `[12, 325, 1]` | 60 min ahead, speed only |
| split | 70 / 10 / 20 | chronological, no shuffle |
| predictions | `[n_test, 12, 325]` | **de-normalized** speed (mph), same format for every model |

9 channels, fixed order: `speed, time_of_day, day_of_week, precipitation,
temperature, wind_speed, event_active, event_magnitude, dist_to_venue`.

## Raw data sources

| Source | Location | How acquired |
|--------|----------|--------------|
| PEMS-BAY (speed, graph, sensor metadata) | `data/raw/augmented-pems-bay/` | `git clone` via `acquire.py` |
| Weather (temp, precip, wind) | `data/raw/weather/weather_hourly.csv` | Open-Meteo Historical API |
| Events (149 games, 5 teams) | `data/raw/events/events.csv` | MLB/NHL/NBA APIs, exact start times |

## Layout

```
configs/         yaml configs (paths, hyperparams, thresholds)
data/raw/        untouched downloads (gitignored)
data/processed/  built tensors + scaler (gitignored)
src/contract.py  <- single source of truth for shapes/format
src/data/        stages 1-7: acquire -> align -> features -> samples -> split -> normalize
src/models/      baselines, stgcn, prediction I/O
src/eval/        metrics, windows, spatial (distance+onset), sepa, decision
results/         tables + figures (gitignored)
report/          Overleaf paper source
STATUS.md        <- live project tracker, update after every change
CLAUDE.md        <- project rules for Claude Code sessions
```

## Working with Claude Code

`CLAUDE.md` (repo root) is read automatically at the start of every Claude Code
session and holds the project rules, pipeline, and ownership. Keep it current.
Install / usage details: https://docs.claude.com/en/docs/claude-code/overview
Personal, non-shared instructions go in `CLAUDE.local.md` (gitignored).

## Reproducibility

Seed everything; 3 seeds per trained config; chronological split; scaler fit on
train only; all metrics after de-normalization.
