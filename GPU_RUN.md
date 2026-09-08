# Running this on the GPU box

Three commands. Everything else is automatic.

```bash
unzip tsf_gpu.zip && cd traffic_shockwave_forecasting

bash scripts/setup_gpu_box.sh                                  # ~2 min
FULL=1 DEV=cuda bash scripts/run_experiments.sh 2>&1 | tee run.log
# -> results_bundle.tgz   (~600 MB)   send this back
```

## Before you start

**Install torch yourself**, matched to this box's CUDA. The setup script
installs everything else but will not guess a CUDA build:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu124   # adjust
python3 -c "import torch; print(torch.cuda.get_device_name(0))"
```

torch and scipy must be in the SAME python. A python with scipy but no torch
fails silently partway through the self-check.

No network is needed after that: the raw data ships inside the zip. The setup
script detects this and skips the 233 MB clone.

## What happens

`setup_gpu_box.sh` installs deps, rebuilds the dataset (~20 s) and runs 90
self-checks. **If it exits non-zero, stop and report it** - do not train on a
dataset that failed verification.

`run_experiments.sh` then runs, in this order:

| batch | what | note |
|---|---|---|
| 0 | timing probe: widest config, largest fold, 3 epochs | **read the seconds/epoch** |
| 1 | trivial baselines | minutes |
| 2 | `0_speed` + `6_all`, 3 seeds | **the paper's main table** |
| 3 | `1_traffic` | |
| 4 | `3_weather` | |
| 5 | `5_event_geo_att` | |
| 6 | congestion-weighted loss | |

Each batch writes checkpoints and predictions as it finishes, so **killing this
at any point leaves a usable partial result**. Batch 2 alone is enough for the
main table.

Runtime is 3.6-5.5 h for all of it on an H200 at 100 epochs, but that number is
extrapolated from an A100 figure in a notebook, not measured on this card. Batch
0 exists to replace the guess: one config-seed covers 42,301 training samples
per epoch across folds 00-02.

Only folds 00-02 are trained. Folds 3-5 contain zero rain episodes and zero NHL
fixtures - selected on test-block content, known before any model runs - while
carrying 73% of the compute. The header of `run_experiments.sh` has the counts.

## Sending results back

`run_experiments.sh` calls `scripts/collect_results.sh` at the end and produces
`results_bundle.tgz`. It contains the reports, figures, decision output, model
weights, per-run training curves, every provenance file, the config, a pip
freeze, the commit hash and the GPU model. With `FULL=1` it also contains the
raw prediction tensors, which is what allows new evaluation windows and new
figures to be produced later without touching a GPU again.

~600 MB with `FULL=1`, ~80 MB without. Google Drive is fine.

If something fails partway, run `bash scripts/collect_results.sh` by hand -
it regenerates the tables from whatever did finish.
