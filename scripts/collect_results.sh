#!/usr/bin/env bash
# Bundle everything needed to reconstruct, re-analyse and defend the results.
#
#   bash scripts/collect_results.sh          # tables + weights + provenance, ~80 MB
#   FULL=1 bash scripts/collect_results.sh   # + every prediction tensor, ~700 MB
#
# MODEL WEIGHTS ARE INCLUDED BY DEFAULT and that is a deliberate reversal. The
# STGCN has 116,940 parameters, so a checkpoint is ~1.7 MB and 45 runs come to
# ~76 MB - not the gigabytes it is easy to assume. With the weights, any later
# question can be answered by re-running inference in seconds; without them the
# answer is a five-hour retrain, and on the zip transfer route there is no going
# back to the GPU box for them.
#
# Run this ON the GPU box after run_experiments.sh. It regenerates the reports
# and figures first, so nothing depends on being re-derivable later.
#
# WHY THE PROVENANCE FILES ARE IN HERE. A results table on its own cannot be
# defended six weeks later: you cannot tell which channels, which confidence
# floor, which distance field or which split produced it. align_meta.json,
# features_meta.json, samples_meta.json, splits_meta.json and scalers.json are a
# few hundred KB between them and pin all of that down. The git commit is
# recorded for the same reason.
set -u
PY=${PY:-python3}
FULL=${FULL:-0}
FOLDS=${FOLDS:-"fold00 fold01 fold02"}
OUT=results_bundle

echo "== regenerating tables and figures from whatever finished =="
for f in $FOLDS; do
  $PY -m src.eval.report   --split "$f" || echo "!! report $f failed"
  $PY -m src.eval.decision --split "$f" || echo "!! decision $f failed"
done
# Clear stale figures first. Without this the bundle ships whatever a previous
# run left behind - during development that included a Presidents' Day panel
# that auto-date selection had already been fixed to reject.
rm -f results/figures/*.png

# --auto picks one day per fold: the wettest commute-hour day in that fold's
# TEST block, holidays excluded. Choosing by hand is how a figure ends up
# showing the day the model happened to get right.
# ONE freeway, as a sanity check that the pipeline drew something sensible.
# Do not batch-generate figures here: with FULL=1 the prediction tensors travel
# with the bundle, so any figure - other freeways, other days, other horizons -
# is a few seconds to redraw wherever the bundle lands.
$PY -m src.eval.figures --auto --freeway 101-N \
    --predictions data/processed/predictions/*.npz || echo "!! figures failed"

echo "== recording provenance =="
rm -rf "$OUT"; mkdir -p "$OUT"/{meta,checkpoints}
cp data/processed/*_meta.json data/processed/scalers.json "$OUT/meta/" 2>/dev/null
cp -r results "$OUT/results"
cp checkpoints/*.json "$OUT/checkpoints/" 2>/dev/null
cp data/processed/predictions/*.json "$OUT/checkpoints/" 2>/dev/null
cp checkpoints/*.pt "$OUT/checkpoints/" 2>/dev/null       # ~1.7 MB each
[ -f run.log ] && cp run.log "$OUT/"
cp configs/default.yaml "$OUT/meta/"
$PY -m pip freeze > "$OUT/meta/pip-freeze.txt" 2>/dev/null
cp CLAUDE.md STATUS.md notes/paper_tables.md "$OUT/" 2>/dev/null
{
  echo "commit:   $(git rev-parse HEAD 2>/dev/null || echo 'NO GIT - shipped as zip')"
  echo "branch:   $(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo '-')"
  echo "dirty:    $(git status --porcelain 2>/dev/null | wc -l | tr -d ' ') modified files"
  echo "host:     $(hostname)"
  echo "when:     $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "gpu:      $($PY -c 'import torch;print(torch.cuda.get_device_name(0))' 2>/dev/null || echo '-')"
  echo "torch:    $($PY -c 'import torch;print(torch.__version__)' 2>/dev/null || echo '-')"
  echo "runs:     $(ls checkpoints/*.json 2>/dev/null | wc -l | tr -d ' ') trained configurations"
} > "$OUT/PROVENANCE.txt"
git diff HEAD > "$OUT/uncommitted.diff" 2>/dev/null

if [ "$FULL" = "1" ]; then
  echo "== including prediction tensors (needed to re-window or re-plot) =="
  mkdir -p "$OUT/predictions"
  cp data/processed/predictions/*.npz "$OUT/predictions/" 2>/dev/null
fi

tar czf results_bundle.tgz "$OUT"
echo
echo "  -> results_bundle.tgz   $(du -h results_bundle.tgz | cut -f1)"
cat "$OUT/PROVENANCE.txt"
echo
echo "Contents:"
echo "  results/       the tables and figures, ready for the paper"
echo "  checkpoints/   per-run metadata AND model weights (.pt)"
echo "  meta/          config, scalers, every *_meta.json, pip freeze"
echo "  PROVENANCE.txt commit, GPU, torch, run count"
echo "  *.md           CLAUDE.md, STATUS.md, paper_tables.md as of this run"
if [ "$FULL" = "1" ]; then
  echo "  predictions/   raw tensors - any new window or figure is a re-read away"
else
  echo
  echo "NOTE: prediction tensors were NOT included (set FULL=1 for them)."
  echo "The weights ARE here, so a new eval window costs an inference pass"
  echo "rather than a retrain - but that needs this repo and a GPU again."
fi
