"""
Stage 7 - Fit the Z-score scaler. On TRAIN ONLY. Per fold.

Two decisions this stage encodes, both of which silently inflate results if you
get them wrong:

* **Train-only fit.** Statistics computed over all data leak test-period
  information into every training batch. Under `rolling`, "train only" means
  *that fold's* train span - the scaler is re-fit per fold, never once globally.
* **Not every channel gets scaled.** Z-scoring `time_of_day` or `day_of_week`
  destroys a bounded cyclical encoding for no benefit. `contract` names exactly
  which channels are scaled and which pass through.

The master tensor is NOT rewritten. Ten folds x 610 MB would be 6.1 GB of nearly
identical copies; instead the scaler is stored as parameters and applied on the
fly, which is also what lets eval de-normalize predictions back to mph.

Outputs (`data/processed/`):

    scalers.json    per split/fold: mean + std for each scaled channel, plus the
                    speed mean/std broken out so eval can de-normalize

Usage
-----
    python -m src.data.normalize --config configs/default.yaml
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import pathlib

import numpy as np
import yaml

from src import contract


def load_config(path: str) -> dict:
    with open(path, encoding="utf-8") as fh:
        return yaml.safe_load(fh)


# ---------------------------------------------------------------------------

class Scaler:
    """Applies / inverts the per-channel Z-score described by `scalers.json`."""

    def __init__(self, spec: dict):
        self.mean = np.asarray(spec["mean"], dtype=np.float32)     # [C]
        self.std = np.asarray(spec["std"], dtype=np.float32)       # [C], 1.0 where unscaled
        self.target = contract.TARGET_CHANNEL

    def transform(self, x: np.ndarray) -> np.ndarray:
        """x: [..., C] -> normalized."""
        return (x - self.mean) / self.std

    def inverse_target(self, y: np.ndarray) -> np.ndarray:
        """Normalized speed -> mph. Every metric must go through this first."""
        return y * self.std[self.target] + self.mean[self.target]


def load_scaler(processed_dir, split: str) -> Scaler:
    spec = json.loads((pathlib.Path(processed_dir) / "scalers.json").read_text())
    return Scaler(spec["splits"][split])


# ---------------------------------------------------------------------------

def fit(master: np.ndarray, train_idx: np.ndarray) -> dict:
    """Fit over the timesteps the TRAIN samples actually cover, nothing else."""
    span = contract.INPUT_WINDOW + contract.HORIZON
    lo, hi = int(train_idx.min()), int(train_idx.max()) + span
    block = master[lo:hi]                                # [t, N, C], memory-mapped

    names = [n for n, _ in contract.CHANNELS]
    mean = np.zeros(len(names), dtype=np.float64)
    std = np.ones(len(names), dtype=np.float64)
    for i, name in enumerate(names):
        if name not in contract.NORMALIZE_CHANNELS:
            continue                                     # mean 0, std 1 -> identity
        ch = np.asarray(block[:, :, i], dtype=np.float64)
        mean[i] = ch.mean()
        s = ch.std()
        std[i] = s if s > 1e-8 else 1.0                  # a constant channel stays as-is
    return {
        "timesteps_used": int(hi - lo),
        "train_time_slice": [lo, hi],
        "mean": [round(float(v), 6) for v in mean],
        "std": [round(float(v), 6) for v in std],
        "channels": names,
        "scaled": list(contract.NORMALIZE_CHANNELS),
        "passthrough": list(contract.PASSTHROUGH_CHANNELS),
        "speed_mean_mph": round(float(mean[contract.TARGET_CHANNEL]), 4),
        "speed_std_mph": round(float(std[contract.TARGET_CHANNEL]), 4),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/default.yaml")
    args = ap.parse_args()
    cfg = load_config(args.config)
    P = pathlib.Path(cfg["data"]["processed_dir"])
    if not (P / "splits.npz").exists():
        raise FileNotFoundError("run stage 6 (`python -m src.data.split`) first")

    master = np.load(P / "master.npy", mmap_mode="r")
    splits = np.load(P / "splits.npz")
    names = sorted({k.split("__")[0] for k in splits.files})

    print("=== Stage 7: Z-score scaler, train-only, per fold ===\n")
    print(f"  scaled      : {contract.NORMALIZE_CHANNELS}")
    print(f"  passthrough : {contract.PASSTHROUGH_CHANNELS}\n")
    print(f"  {'split':9s}{'train steps':>12s}{'speed mean':>12s}{'speed std':>11s}")

    out = {"splits": {}}
    for name in names:
        train_idx = splits[f"{name}__train"].astype(np.int64)
        if not len(train_idx):
            continue
        spec = fit(master, train_idx)
        out["splits"][name] = spec
        print(f"  {name:9s}{spec['timesteps_used']:>12,}"
              f"{spec['speed_mean_mph']:>12.3f}{spec['speed_std_mph']:>11.3f}")

    # the whole point of per-fold fitting: the statistics really do differ
    means = [s["speed_mean_mph"] for k, s in out["splits"].items() if k.startswith("fold")]
    if means:
        print(f"\n  speed mean across rolling folds: {min(means):.3f} .. {max(means):.3f} mph"
              f"  (spread {max(means) - min(means):.3f}) - a single global scaler would")
        print("  have leaked later-period statistics into every early fold")

    # round-trip check
    s = Scaler(out["splits"][names[0]])
    probe = np.asarray(master[:32], dtype=np.float32)
    back = s.inverse_target(s.transform(probe)[..., contract.TARGET_CHANNEL])
    assert np.allclose(back, probe[..., contract.TARGET_CHANNEL], atol=1e-3), \
        "de-normalization round-trip failed"
    print("  de-normalization round-trip OK (transform -> inverse_target == mph)")

    out["generated"] = dt.datetime.now().isoformat(timespec="seconds")
    out["note"] = ("master.npy is NOT rewritten; apply Scaler.transform() in the "
                   "DataLoader and Scaler.inverse_target() before any metric")
    (P / "scalers.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\n  scalers.json ({len(out['splits'])} splits) -> {P}")
    print("\nDone.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
