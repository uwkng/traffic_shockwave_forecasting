"""
Orchestrator - run the whole data pipeline, stages 1-7, in order.

    python -m src.data.build_dataset --config configs/default.yaml

Each stage is skipped if its outputs already exist and are newer than its
inputs, unless `--force` is given. Stage 1 (the ~233 MB clone) and stage 1b
(network calls) are skipped by default; pass `--acquire` to include them.

FIRST RUN ON A NEW MACHINE: use `--acquire`. If the ASOS download fails with
`SSL: CERTIFICATE_VERIFY_FAILED`, your Python has no CA bundle - this is the
standard python.org-installer situation on macOS. Fix it once, at the machine
level, rather than patching acquire.py:

    open "/Applications/Python 3.x/Install Certificates.command"

Nothing else is needed; the plain urllib call in acquire.py works afterwards.
"""

from __future__ import annotations

import argparse
import pathlib
import time

STAGES = [
    # module,                      label,                              outputs
    ("src.data.fetch_events", "1b  events (network)", ["data/raw/events/events.csv"]),
    ("src.data.acquire", "1   acquire + verify (network)", ["data/raw/weather/weather_5min.csv"]),
    ("src.data.align", "2-3 align", ["data/processed/speed.npy",
                                     "data/processed/event_load.npy",
                                     "data/processed/event_active_decay.npy"]),
    ("src.data.features", "4   11-channel tensor", ["data/processed/master.npy"]),
    ("src.data.samples", "5   sample index", ["data/processed/sample_index.npy"]),
    ("src.data.split", "6   chronological split", ["data/processed/splits.npz"]),
    ("src.data.normalize", "7   scaler (train-only)", ["data/processed/scalers.json"]),
]
NETWORK_STAGES = {"src.data.fetch_events", "src.data.acquire"}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--force", action="store_true", help="re-run stages whose outputs exist")
    ap.add_argument("--acquire", action="store_true",
                    help="also run stages 1/1b (git clone + API calls)")
    args = ap.parse_args()

    import runpy
    import sys

    timings: list[tuple[str, float]] = []
    t_all = time.time()
    print("=== build_dataset: stages 1-7 ===\n")
    for module, label, outputs in STAGES:
        if module in NETWORK_STAGES and not args.acquire:
            print(f"[skip] {label}  (pass --acquire to include)")
            continue
        done = all(pathlib.Path(o).exists() for o in outputs)
        if done and not args.force:
            print(f"[skip] {label}  (outputs present; --force to redo)")
            continue
        print(f"\n{'=' * 70}\n[run ] {label}\n{'=' * 70}")
        t0 = time.time()
        sys.argv = [module, "--config", args.config]
        # every stage ends with `raise SystemExit(main())`; without this the first
        # stage's SystemExit(0) would tear down the orchestrator itself
        try:
            runpy.run_module(module, run_name="__main__")
        except SystemExit as exc:
            if exc.code not in (0, None):
                print(f"[FAIL] {label}  exit code {exc.code}")
                return int(exc.code)
        timings.append((label, time.time() - t0))
        print(f"[done] {label}  {time.time() - t0:.1f}s")

    print(f"\n{'=' * 70}\nPipeline complete in {time.time() - t_all:.1f}s")
    if timings:
        print("\n  stage timings")
        for label, dt_ in timings:
            print(f"    {dt_:7.1f}s  {label}")
    print("\n  artifacts in data/processed/")
    total = 0
    for f in sorted(pathlib.Path("data/processed").iterdir()):
        total += f.stat().st_size
        print(f"    {f.stat().st_size / 1e6:9.1f} MB  {f.name}")
    print(f"    {total / 1e6:9.1f} MB  TOTAL")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
