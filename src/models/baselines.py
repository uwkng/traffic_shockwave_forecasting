"""
Stage 10 - the two trivial baselines, in the same prediction format as STGCN.

    python -m src.models.baselines --split single

Historical average and persistence need no training and no GPU, and they are not
a formality here. The proposal's premise is that on normal traffic a historical
average is already enough, so aggregate MAE hides what happens in the hours that
matter. That claim is only testable if HA is measured on exactly the same
samples as the model - which is what this produces.

Both write to data/processed/predictions/ with the same keys as train.py, so
src/eval reads them without knowing which produced what.

Note HA is FIT ON TRAIN ONLY, per split, like every other statistic here. A mean
taken over the whole period would let the test block set its own baseline.
"""

from __future__ import annotations

import argparse
import json
import pathlib

import numpy as np
import pandas as pd

from src import contract
from src.models.loader import build_loaders, load_config

PRED = pathlib.Path("data/processed/predictions")


def _time_of_week(time_index: pd.DatetimeIndex) -> np.ndarray:
    """0..2015: which 5-minute slot of the week. 7 days x 288 slots."""
    return (time_index.dayofweek.values * 288
            + time_index.hour.values * 12
            + time_index.minute.values // 5)


def historical_average(loaders, time_index, speed):
    """Predict the train-set mean speed for this node at this time-of-week.

    Falls back to the node's overall train mean for a slot the train span never
    covers - which happens in the early rolling folds, where train is only four
    weeks and does not see every holiday-shaped slot.
    """
    L, H = contract.INPUT_WINDOW, contract.HORIZON
    tow = _time_of_week(time_index)
    tr = loaders["datasets"]["train"].ids
    lo, hi = int(tr.min()), int(tr.max()) + L + H          # the train timesteps
    N = speed.shape[1]

    num = np.zeros((2016, N)); den = np.zeros((2016, N))
    np.add.at(num, tow[lo:hi], speed[lo:hi])
    np.add.at(den, tow[lo:hi], 1.0)
    node_mean = speed[lo:hi].mean(axis=0)
    table = np.where(den > 0, num / np.maximum(den, 1), node_mean[None, :])

    te = loaders["datasets"]["test"].timesteps()
    idx = te[:, None] + np.arange(H)[None, :]              # [n, H] target steps
    return table[tow[idx]]                                  # [n, H, N]


def persistence(loaders, speed):
    """Predict the last observed value, held flat across the horizon."""
    L, H = contract.INPUT_WINDOW, contract.HORIZON
    last = speed[loaders["datasets"]["test"].ids + L - 1]   # [n, N]
    return np.repeat(last[:, None, :], H, axis=1)


def truth(loaders, speed):
    H = contract.INPUT_WINDOW, contract.HORIZON
    L, H = H
    te = loaders["datasets"]["test"].timesteps()
    return speed[te[:, None] + np.arange(H)[None, :]]


def score(pred, true, eval_mask, timesteps):
    H = contract.HORIZON
    idx = timesteps[:, None] + np.arange(H)[None, :]
    mask = eval_mask[idx]
    out = {}
    for name, step in [("all", None)] + list(contract.EVAL_HORIZON_STEPS.items()):
        p, t, m = ((pred, true, mask) if step is None
                   else (pred[:, step - 1], true[:, step - 1], mask[:, step - 1]))
        p, t = p[m], t[m]
        d = p - t; ad = np.abs(d); nz = np.abs(t) > 1.0
        out[name] = {"mae": float(ad.mean()),
                     "rmse": float(np.sqrt((d ** 2).mean())),
                     "mape": float((ad[nz] / np.abs(t[nz])).mean() * 100),
                     "n": int(t.size)}
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", default="single")
    ap.add_argument("--all-folds", action="store_true")
    ap.add_argument("--config", default="configs/default.yaml")
    args = ap.parse_args()
    cfg = load_config(args.config)
    P = pathlib.Path(cfg["data"]["processed_dir"])

    speed = np.load(P / "speed.npy")
    speed = np.where(np.isnan(speed), np.nanmean(speed, axis=0), speed)
    time_index = pd.DatetimeIndex(np.load(P / "time_index.npy"))
    splits = ([f"fold{i:02d}" for i in range(cfg["split"]["rolling"]["n_folds"])]
              if args.all_folds else [args.split])

    PRED.mkdir(parents=True, exist_ok=True)
    print("=== stage 10: trivial baselines (no training) ===\n")
    print(f"  {'split':9s}{'model':22s}"
          + "".join(f"{h:>10s}" for h in contract.EVAL_HORIZON_STEPS))
    for split in splits:
        loaders, _, _ = build_loaders(rung="0_speed", split=split,
                                      torch_loaders=False, config=args.config)
        te = loaders["datasets"]["test"].timesteps()
        y = truth(loaders, speed)
        for name, pred in [("historical_average",
                            historical_average(loaders, time_index, speed)),
                           ("persistence", persistence(loaders, speed))]:
            m = score(pred, y, loaders["eval_mask"], te)
            tag = f"{name}__{split}"
            np.savez_compressed(
                PRED / f"{tag}.npz",
                pred=pred.astype(contract.PREDICTION_DTYPE),
                true=y.astype(contract.PREDICTION_DTYPE),
                timesteps=te, sample_ids=loaders["datasets"]["test"].ids)
            (PRED / f"{tag}.json").write_text(
                json.dumps({"model": name, "split": split, "test": m}, indent=2),
                encoding="utf-8")
            print(f"  {split:9s}{name:22s}"
                  + "".join(f"{m[h]['mae']:10.3f}" for h in contract.EVAL_HORIZON_STEPS))
    print(f"\n  predictions -> {PRED}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
