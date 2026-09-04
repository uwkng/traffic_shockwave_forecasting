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
| 4 | Assemble 9-channel tensor `[T,N,9]` | `src/data/features.py` |
| 5 | Slice samples  X`[12,N,9]` Y`[12,N,1]` | `src/data/samples.py` |
| 6 | Chronological 70/10/20 split | `src/data/split.py` |
| 7 | Z-score (train-fit only), save scaler | `src/data/normalize.py` |
| - | Orchestrator for stages 1-7 | `src/data/build_dataset.py` |
| 9 | Models -> de-normalized predictions | `src/models/` |
| 10 | Uniform prediction I/O | `src/models/predict.py` |
| 11 | Metrics, windows, distance bins, onset, SEPA | `src/eval/` |
| 12 | Decision layer | `src/eval/decision.py` |

## Ownership (fill in names)
- Data & features (stages 1-8): `<owner A>`
- Models & training (stages 9-10): `<owner B>`
- Evaluation & figures (stages 11-13): `<owner C>`  <- the contribution; strongest engineer here

## Non-negotiables (known failure modes for this kind of project)
- Chronological split, **NO shuffling** - shuffling leaks the future.
- Fit the scaler on **train only** - fitting on all data silently inflates results.
- All metrics computed **after de-normalization** (real mph) - carry the scaler to eval.
- Channel 8 `dist_to_venue` is load-bearing: event features are broadcast to every
  node identically, so distance is the ONLY thing giving them a spatial anchor. Do not drop it.
- Reproduce vanilla STGCN on plain PEMS-BAY BEFORE adding the 1->9 channel change.
- DCRNN is optional and time-boxed (legacy TF1.x); if it fights the environment, drop it.
  STGCN + baselines + full eval is already the complete contribution.

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
python -m src.data.build_dataset --config configs/default.yaml
python -m src.models.train --model stgcn --config configs/default.yaml
python -m src.eval.run --config configs/default.yaml
```