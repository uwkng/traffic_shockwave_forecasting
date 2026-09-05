#!/usr/bin/env bash
# One-shot setup on a fresh GPU machine, then hand off to the experiment queue.
#
#   bash scripts/setup_gpu_box.sh
#   DEV=cuda bash scripts/run_experiments.sh 2>&1 | tee run.log
#
# What this machine needs from the network: this repo (~50 KB of tracked files)
# and the Augmented-PEMS-BAY clone (~233 MB, done by --acquire). The ASOS
# weather CSVs and events.csv ARE tracked, so no weather download is needed and
# the run does not depend on the IEM service being up.
set -eu

echo "== 1. python and GPU =="
python3 -c "import sys; print('python', sys.version.split()[0])"
python3 - <<'PYEOF'
try:
    import torch
    print("torch", torch.__version__, "| cuda", torch.cuda.is_available(),
          "|", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "NO GPU")
except ImportError:
    print("torch NOT installed")
PYEOF

echo
echo "== 2. dependencies =="
python3 -m pip install --quiet --upgrade \
    numpy pandas pyyaml requests scipy matplotlib
echo "  (install torch separately, matched to the box's CUDA:"
echo "   https://pytorch.org/get-started/locally/ )"

echo
echo "== 3. data: build stages 1-7 =="
# Two ways this box can have arrived: a git clone (code only - PEMS-BAY has to
# be fetched, ~233 MB, needs network) or the transfer zip (raw data already
# present, needs nothing). Detect it rather than asking.
if [ -f data/raw/augmented-pems-bay/data/traffic_data/speed.csv ]; then
  echo "  PEMS-BAY already present - offline rebuild, ~20 s"
  python3 -m src.data.build_dataset --config configs/default.yaml --force
else
  echo "  PEMS-BAY missing - cloning (~233 MB) then building, ~5 min"
  python3 -m src.data.build_dataset --config configs/default.yaml --acquire --force
fi

echo
echo "== 4. verify: 90 checks, exits non-zero on any failure =="
python3 -m scripts.verify

echo
echo "== 5. ready =="
echo "  DEV=cuda bash scripts/run_experiments.sh 2>&1 | tee run.log"
