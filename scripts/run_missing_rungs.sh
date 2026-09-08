#!/usr/bin/env bash
# The two ablation rungs the first GPU run did not reach.
#
#   DEV=cuda bash scripts/run_missing_rungs.sh 2>&1 | tee run_rungs24.log
#
# ~84 min on an H200, measured, not extrapolated: batch 0 of the first run
# timed the widest config on the largest fold at 5.5 s/epoch over 22,141
# samples = 0.248 ms/sample. One seed across folds 00-02 is 42,301 samples, so
# ~10.5 s/epoch; at ~80 epochs that is ~14 min per seed, ~42 min per rung.
#
# WHY THESE TWO, and not "the run was already good enough":
#
#   4_event_geo  is the missing half of a controlled pair. It differs from
#                5_event_geo_att by exactly one thing - whether ATTENDANCE
#                enters the channel. Without it, "crowd size carries signal
#                beyond a fixture merely happening" rests on a statistic
#                measured on the raw data (t = -1.02 against t = -9.76), not
#                on a trained model.
#
#   2_calendar   is a second, INDEPENDENT group of broadcast-only channels
#                (time of day, day of week, is holiday - each one number copied
#                to all 325 nodes). The strongest claim in the paper is that a
#                channel with no spatial variation is harmful on a graph model,
#                and right now that claim rests on a single pair, 3_weather
#                against 5_event_geo_att. This replicates it or qualifies it.
#                Expected outcome: close to 1_traffic or slightly worse -
#                time-of-day is already implicit in 60 minutes of speed history
#                (measured incremental R2 +0.019). Recording an expected result
#                is the point; skipping it is what makes a ladder look picked.
#
# Run this on the box that already holds the first run's predictions, so
# src.eval.report sees all 11 models at once. On a fresh box it still works,
# but the regenerated tables will contain only these two rungs - in that case
# send back the predictions and merge them where the others live.
set -u
PY=${PY:-python3}
DEV=${DEV:-cuda}
FOLDS=${FOLDS:-"fold00 fold01 fold02"}
EPOCHS=${EPOCHS:-100}
SEEDS=${SEEDS:-3}

for rung in 2_calendar 4_event_geo; do
  for f in $FOLDS; do
    echo "--- $rung $f seeds=$SEEDS ---"
    $PY -m src.models.train --rung "$rung" --split "$f" --seeds "$SEEDS" \
        --epochs "$EPOCHS" --device "$DEV" || echo "!! FAILED: $rung $f"
  done
done

echo "###### regenerating tables over every model on disk ######"
for f in $FOLDS; do
  $PY -m src.eval.report   --split "$f" || true
  $PY -m src.eval.decision --split "$f" || true
done

FULL=${FULL:-1} PY=$PY FOLDS="$FOLDS" bash scripts/collect_results.sh
