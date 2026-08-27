# Project Status

Last updated: 2026-08-27

## Pipeline Progress

| # | Stage | File | Status | Owner |
|---|-------|------|--------|-------|
| 1 | Acquire raw sources (PEMS-BAY, weather, events) | `src/data/acquire.py` | DONE | A |
| 2-3 | Align to 5-min index + graph + distance-to-venue | `src/data/align.py` | TODO | A |
| 4 | Assemble 9-channel tensor `[T,N,9]` | `src/data/features.py` | TODO | A |
| 5 | Slice samples X`[12,N,9]` Y`[12,N,1]` | `src/data/samples.py` | TODO | A |
| 6 | Chronological 70/10/20 split | `src/data/split.py` | TODO | A |
| 7 | Z-score (train-fit only), save scaler | `src/data/normalize.py` | TODO | A |
| - | Orchestrator for stages 1-7 | `src/data/build_dataset.py` | TODO | A |
| 8 | Reproduce vanilla STGCN on plain PEMS-BAY | `src/models/` | TODO | B |
| 9 | STGCN with 9-channel input | `src/models/` | TODO | B |
| 10 | Uniform prediction I/O | `src/models/predict.py` | TODO | B |
| 11 | Metrics, windows, distance bins, onset, SEPA | `src/eval/` | TODO | C |
| 12 | Decision layer | `src/eval/decision.py` | TODO | C |

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

## Decisions log

Record choices here so they don't get lost in chat.

| Date | Decision | Rationale |
|------|----------|-----------|
| 2026-08-27 | Single weather centroid (37.34, -121.94) for all sensors | Network spans ~20 km; one station is sufficient at this scale |
| 2026-08-27 | Events limited to 5 Bay Area pro sports teams | These are the major scheduled events with known attendance near the sensor network |
| 2026-08-27 | Event start times from APIs, not approximated | Approximate times can be off by hours (e.g. Sunday matinees), which misaligns the event signal with traffic |

## Open questions

- [ ] Who fills owner A / B / C roles? Update CLAUDE.md and this file once assigned.
- [ ] DCRNN: attempt or skip? (CLAUDE.md says optional and time-boxed)

## Next steps

1. **Stage 2-3 (align.py)**: Resample weather from hourly → 5-min, map events onto the 5-min index, build adjacency matrix, compute dist_to_venue per sensor
2. **Stage 8 (vanilla STGCN)**: Can start in parallel once someone picks up owner B — only needs the raw PEMS-BAY speed data (already acquired)

## How to update this file

When you finish a stage or make a decision that affects others:
1. Move the stage status from TODO → IN PROGRESS → DONE
2. Add a bullet under "What's done" with what changed
3. Log any cross-cutting decisions in the decisions table
