#!/usr/bin/env bash
# Bundle everything needed to reconstruct, re-analyse and defend the results.
#
#   bash scripts/collect_results.sh            # tables + provenance, ~2 MB
#   FULL=1 bash scripts/collect_results.sh     # + every prediction tensor, ~600 MB
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
$PY -m src.eval.figures --date 2017-02-08 --freeway 101-N \
    --predictions data/processed/predictions/*.npz || echo "!! figures failed"

echo "== recording provenance =="
rm -rf "$OUT"; mkdir -p "$OUT"/{meta,checkpoints}
cp data/processed/*_meta.json data/processed/scalers.json "$OUT/meta/" 2>/dev/null
cp -r results "$OUT/results"
cp checkpoints/*.json "$OUT/checkpoints/" 2>/dev/null
cp data/processed/predictions/*.json "$OUT/checkpoints/" 2>/dev/null
[ -f run.log ] && cp run.log "$OUT/"
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
echo "Without FULL=1 the prediction tensors stay behind. That is fine for the"
echo "tables, but you cannot then add a NEW eval window or redraw a figure off"
echo "this machine - only re-read what was already computed."
