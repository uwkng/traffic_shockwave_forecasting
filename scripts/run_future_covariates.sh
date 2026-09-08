#!/usr/bin/env bash
# The known-future half of the experiment.
#
#   DEV=cuda bash scripts/run_future_covariates.sh 2>&1 | tee run_future.log
#
# ~3.5 h on an H200. Every run here PAIRS with one already on disk: same rung,
# same folds, same seeds, differing only in whether the exogenous channels are
# also supplied for the window being predicted. That pairing is the result -
# what lookahead buys over observation - and it is why nothing already trained
# needs re-running.
#
# WHY THESE FIVE. Only five rungs have any known-future channel at all;
# 0_speed and 1_traffic hold only speed and occupancy, which must be observed,
# so the flag is a no-op there and they are deliberately absent.
#
#   2_calendar        +3 future  time_of_day, day_of_week, is_holiday
#   3_weather         +3 future  temperature, precipitation, wind_speed
#   4_event_geo       +1 future  event_active_decay
#   5_event_geo_att   +2 future  event_active_decay, event_load
#   6_all             +8 future  all of the above        (c_in 11 -> 19)
#
# 4_event_geo against 5_event_geo_att is the one that matters most for this
# project: it isolates whether knowing the SIZE of a crowd that will leave
# during the predicted window adds anything over knowing that a fixture ends in
# it. Contemporaneously attendance was worth 0.015 mph against a +-0.15-0.25
# spread, i.e. nothing - but contemporaneous attendance should not help, the
# cars are still inside the arena. Anticipated attendance is the claim the
# project actually makes, and it has never been tested.
#
# THE WEATHER IS A FORECAST, NOT THE OBSERVATION. loader.build_loaders
# substitutes ECMWF IFS short-range values into the future block for the three
# weather channels and refuses to run if that archive is missing, rather than
# falling back to the observed series - which would be leakage reported as a
# result. The input window keeps the observed ASOS series, because at serving
# time the past really is observed. Measured against the observations, the
# forecast recovers 61.4% of adverse steps and over-calls somewhat (3,240
# forecast against 2,504 observed), so it is visibly not a copy of the truth.
#
# Output files gain a `__future` suffix, so the OFF twins are not overwritten.
set -u
PY=${PY:-python3}
DEV=${DEV:-cuda}
FOLDS=${FOLDS:-"fold00 fold01 fold02"}
EPOCHS=${EPOCHS:-100}
SEEDS=${SEEDS:-3}

test -f data/processed/weather_forecast.npy || {
  echo "!! data/processed/weather_forecast.npy is missing."
  echo "   $PY -c 'from src.data.acquire import fetch_weather_forecast as f; f()'"
  echo "   $PY -m src.data.build_dataset --config configs/default.yaml --force"
  exit 1
}

for rung in 2_calendar 3_weather 4_event_geo 5_event_geo_att 6_all; do
  for f in $FOLDS; do
    echo "--- $rung $f future seeds=$SEEDS ---"
    $PY -m src.models.train --rung "$rung" --split "$f" --seeds "$SEEDS" \
        --epochs "$EPOCHS" --device "$DEV" --future-covariates \
      || echo "!! FAILED: $rung $f"
  done
done

echo "###### regenerating tables over every model on disk ######"
for f in $FOLDS; do
  $PY -m src.eval.report   --split "$f" || true
  $PY -m src.eval.decision --split "$f" || true
done

FULL=${FULL:-0} PY=$PY FOLDS="$FOLDS" bash scripts/collect_results.sh
