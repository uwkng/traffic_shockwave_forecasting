#!/usr/bin/env bash
# Stage 9-12 experiment queue, ordered so that whatever finishes is usable.
#
# Run it and walk away. Every batch writes checkpoints/*.json and
# data/processed/predictions/*.npz as it goes, so killing this script at any
# point leaves a coherent partial result set - it never has to finish to be
# useful.
#
# WHY FOLDS 00-02 AND NOT ALL SIX. Selected on the exogenous window content of
# each test block, which is known before any model is trained:
#     fold00  test 01-29..02-26   28 rain episodes, 4 NHL, 1 holiday
#     fold01  test 02-26..03-26   16 rain,          8 NHL
#     fold02  test 03-26..04-23   17 rain,          7 NHL
#     fold03  test 04-23..05-21    0 rain,          0 NHL
#     fold04  test 05-21..06-18    0 rain,          0 NHL
#     fold05  test 06-18..06-30    0 rain,          0 NHL
# California's wet season is Jan-Apr and the NHL regular season ends in April,
# so folds 3-5 cannot inform either question - while carrying 73% of the compute
# (114,807 of 157,108 training samples, because the training span grows). The
# `single` 70/10/20 split has zero rain in test for the same reason.
#
# BATCH 0 IS NOT OPTIONAL. Every runtime estimate in this repo is extrapolated
# from an A100 figure quoted in a notebook. Measure the real card first, then
# decide how far down this list you can get.
set -u
PY=${PY:-python3}
DEV=${DEV:-cuda}
FOLDS=${FOLDS:-"fold00 fold01 fold02"}
EPOCHS=${EPOCHS:-100}

run () {  # run <rung> <n_seeds> [extra flags...]
  local rung=$1 seeds=$2; shift 2
  for f in $FOLDS; do
    echo "--- $rung $f seeds=$seeds $* ---"
    $PY -m src.models.train --rung "$rung" --split "$f" --seeds "$seeds" \
        --epochs "$EPOCHS" --device "$DEV" "$@" || echo "!! FAILED: $rung $f"
  done
}

echo "###### batch 0: timing probe - the widest rung on the largest fold ######"
time $PY -m src.models.train --rung 6_all --split fold02 --seeds 1 \
     --epochs 3 --device "$DEV"
echo
echo "Read the seconds/epoch above before continuing. One config-seed over"
echo "folds 00-02 is 42,301 training samples per epoch."
echo

echo "###### batch 1: trivial baselines (minutes, not hours) ######"
for f in $FOLDS; do $PY -m src.models.baselines --split "$f" || true; done

echo "###### batch 2: the headline pair -> the paper's main table ######"
run 0_speed         3
run 6_all           3

echo "###### batch 3: occupancy, the largest measured single channel ######"
run 1_traffic       3

echo "###### batch 4: weather ######"
run 3_weather       3

echo "###### batch 5: events, high-confidence fixtures only ######"
run 5_event_geo_att 3

echo "###### batch 6: congestion-weighted loss as an ablation dimension ######"
run 6_all           3 --loss weighted
run 0_speed         3 --loss weighted

echo "###### done. build the tables: ######"
for f in $FOLDS; do echo "  $PY -m src.eval.report --split $f"; done
