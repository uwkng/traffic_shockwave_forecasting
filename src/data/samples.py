"""
Stage 5 - Build the sample index. Deliberately does NOT materialise samples.

A sample `i` reads inputs from master[i : i+12] and targets from
master[i+12 : i+24, :, speed]. Materialising those windows would write
`52,093 x 12 x 325 x 9 x 4 B = 7.3 GB` of X plus 1.0 GB of Y - a 12x redundant
copy of a 0.61 GB tensor. So this stage stores only the start indices (0.4 MB)
and the Dataset slices views out of the memory-mapped master tensor.

Two kinds of sample are excluded:

* **Windows that straddle the DST discontinuity.** The time axis jumps 65 minutes
  at 2017-03-12 01:55 -> 03:00. A window spanning that jump covers 65 minutes of
  wall-clock in 12 steps, so its temporal spacing is a lie. 23 windows are
  affected out of 52,093; dropping them is free.
* **Windows with no observed target at all**, which would contribute nothing to
  any metric. (With 0.003% missing this is normally empty, but assert-don't-trust.)

Outputs (`data/processed/`):

    sample_index.npy      [n]  int32, the start index of each valid sample
    samples_meta.json          counts + exactly what was dropped and why

Usage
-----
    python -m src.data.samples --config configs/default.yaml
    python -m src.data.samples --config configs/default.yaml --self-test
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import pathlib

import numpy as np
import pandas as pd
import yaml

from src import contract


def load_config(path: str) -> dict:
    with open(path, encoding="utf-8") as fh:
        return yaml.safe_load(fh)


# ---------------------------------------------------------------------------
# the lazy accessor every downstream stage should use
# ---------------------------------------------------------------------------

class WindowedDataset:
    """Lazy [T,N,C] -> (X[12,N,C], Y[12,N,1], mask[12,N]) windows.

    Backed by a memory-mapped master tensor, so constructing this is free and
    several folds can share one mapping. Torch-agnostic on purpose: stage 9 wraps
    it in a `torch.utils.data.Dataset` with three lines.
    """

    def __init__(self, processed_dir, indices=None, mmap=True):
        p = pathlib.Path(processed_dir)
        self.master = np.load(p / "master.npy", mmap_mode="r" if mmap else None)
        self.eval_mask = np.load(p / "eval_mask.npy", mmap_mode="r" if mmap else None)
        self.indices = (np.load(p / "sample_index.npy") if indices is None
                        else np.asarray(indices, dtype=np.int64))
        self.w, self.h = contract.INPUT_WINDOW, contract.HORIZON
        self.target = contract.TARGET_CHANNEL

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, k):
        i = int(self.indices[k])
        x = np.asarray(self.master[i:i + self.w])                       # [12,N,C]
        y = np.asarray(self.master[i + self.w:i + self.w + self.h, :, self.target])
        m = np.asarray(self.eval_mask[i + self.w:i + self.w + self.h])  # [12,N]
        return x, y[..., None], m


# ---------------------------------------------------------------------------

def build_index(cfg: dict):
    P = pathlib.Path(cfg["data"]["processed_dir"])
    for f in ("master.npy", "eval_mask.npy", "time_index.npy"):
        if not (P / f).exists():
            raise FileNotFoundError(
                f"{P / f} missing - run stage 4 (`python -m src.data.features`) first")

    time_index = pd.DatetimeIndex(np.load(P / "time_index.npy"))
    eval_mask = np.load(P / "eval_mask.npy", mmap_mode="r")
    T = len(time_index)
    w, h = contract.INPUT_WINDOW, contract.HORIZON
    span = w + h

    candidates = np.arange(T - span + 1, dtype=np.int64)

    # --- drop windows that straddle a non-5-minute step -------------------
    step = pd.Timedelta(minutes=cfg["data"]["freq_min"])
    deltas = np.diff(time_index.to_numpy()).astype("timedelta64[s]").astype(np.int64)
    irregular = np.flatnonzero(deltas != int(step.total_seconds()))
    bad = np.zeros(len(candidates), dtype=bool)
    for j in irregular:                     # the jump sits between j and j+1
        lo = max(0, j - span + 2)
        bad[lo:j + 1] = True
    n_dst = int(bad.sum())

    # --- drop windows whose target horizon is entirely unobserved ---------
    kept = candidates[~bad]
    usable = np.array([bool(eval_mask[i + w:i + w + h].any()) for i in kept])
    n_empty = int((~usable).sum())
    index = kept[usable].astype(np.int32)

    return index, {
        "timesteps": int(T),
        "input_window": w, "horizon": h,
        "candidates": int(len(candidates)),
        "dropped_straddling_time_gap": n_dst,
        "irregular_steps_at": [str(time_index[j]) for j in irregular],
        "dropped_no_observed_target": n_empty,
        "samples": int(len(index)),
        "first_sample_starts": str(time_index[index[0]]),
        "last_sample_targets_end": str(time_index[index[-1] + span - 1]),
        "materialised_X_would_be_GB": round(len(index) * w * eval_mask.shape[1]
                                            * contract.N_CHANNELS * 4 / 1e9, 2),
        "index_size_MB": round(index.nbytes / 1e6, 3),
    }


def self_test(cfg: dict) -> None:
    """Prove the lazy windows really line up with the master tensor."""
    P = pathlib.Path(cfg["data"]["processed_dir"])
    ds = WindowedDataset(P)
    master = np.load(P / "master.npy", mmap_mode="r")
    rng = np.random.default_rng(contract.SEED)
    for k in rng.integers(0, len(ds), size=5):
        x, y, m = ds[int(k)]
        i = int(ds.indices[k])
        assert x.shape == (contract.INPUT_WINDOW, master.shape[1], contract.N_CHANNELS)
        assert y.shape == (contract.HORIZON, master.shape[1], 1)
        assert m.shape == (contract.HORIZON, master.shape[1])
        assert np.array_equal(x, master[i:i + contract.INPUT_WINDOW])
        assert np.array_equal(
            y[..., 0],
            master[i + contract.INPUT_WINDOW:i + contract.INPUT_WINDOW + contract.HORIZON,
                   :, contract.TARGET_CHANNEL])
        assert np.isfinite(x).all() and np.isfinite(y).all()
    print(f"  self-test OK on 5 random samples (of {len(ds):,})")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()
    cfg = load_config(args.config)
    P = pathlib.Path(cfg["data"]["processed_dir"])

    print("=== Stage 5: sample index (lazy, not materialised) ===\n")
    index, meta = build_index(cfg)
    print(f"  candidate windows            : {meta['candidates']:,}")
    print(f"  dropped, straddle the DST gap: {meta['dropped_straddling_time_gap']}"
          f"  (irregular step at {meta['irregular_steps_at']})")
    print(f"  dropped, no observed target  : {meta['dropped_no_observed_target']}")
    print(f"  usable samples               : {meta['samples']:,}")
    print(f"  first starts {meta['first_sample_starts']}, "
          f"last target ends {meta['last_sample_targets_end']}")
    print(f"\n  index on disk {meta['index_size_MB']} MB "
          f"vs {meta['materialised_X_would_be_GB']} GB if X were materialised")

    np.save(P / "sample_index.npy", index)
    meta["generated"] = dt.datetime.now().isoformat(timespec="seconds")
    (P / "samples_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"  sample_index.npy -> {P}")

    if args.self_test:
        print()
        self_test(cfg)
    print("\nDone.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
