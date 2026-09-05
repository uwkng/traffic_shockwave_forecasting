"""
Stage 4 - Assemble the 10-channel master tensor `[T, N, 10]`.

Consumes the aligned arrays from `align.py`, produces exactly what the models
consume. Channel order is taken from `contract.CHANNELS` and asserted, so a
change to the contract cannot silently desynchronise this file.

Two things this stage owns:

* **Broadcasting.** `temporal` channels vary over time only and are broadcast to
  all 325 nodes; `spatial` channels vary over node only and are broadcast over
  all 52,116 timesteps. Doing this here means the models never have to know
  which channel is which.
* **Missing values.** The augmented release stores gaps as NaN (the original
  release zero-filled them). Inputs are imputed by per-sensor linear
  interpolation in time, capped at `MISSING_POLICY["max_gap_steps"]`; anything
  longer stays NaN and is carried into the eval mask. The mask records the
  ORIGINAL missing positions so an imputed value can never enter a metric.

Outputs (`data/processed/`):

    master.npy        [T, N, 10] float32, no NaN  <- what the models see
    eval_mask.npy     [T, N]     bool, True = usable target (speed was observed)
    features_meta.json           channel order, per-channel stats, imputation report

Usage
-----
    python -m src.data.features --config configs/default.yaml
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
from src.eval.windows import US_HOLIDAYS_2017_H1, holiday_mask


def load_config(path: str) -> dict:
    with open(path, encoding="utf-8") as fh:
        return yaml.safe_load(fh)


# ---------------------------------------------------------------------------

def impute_per_sensor(arr: np.ndarray, max_gap: int) -> tuple[np.ndarray, dict]:
    """Linear interpolation in time, per sensor, only across short gaps.

    Gaps longer than `max_gap` steps are left as NaN here and filled with the
    sensor's training-period-agnostic median at the very end, but they stay
    flagged in the eval mask so they never count towards a reported metric.
    """
    nan_before = np.isnan(arr)
    if not nan_before.any():
        return arr, {"missing_before": 0, "interpolated": 0, "median_filled": 0,
                     "longest_gap_steps": 0}

    df = pd.DataFrame(arr)                                   # rows = time
    filled = df.interpolate(method="linear", axis=0, limit=max_gap,
                            limit_direction="both").to_numpy(dtype=np.float32)

    still = np.isnan(filled)
    n_interp = int(nan_before.sum() - still.sum())
    if still.any():
        col_median = np.nanmedian(filled, axis=0)
        col_median = np.where(np.isnan(col_median), np.nanmedian(filled), col_median)
        filled = np.where(still, np.broadcast_to(col_median, filled.shape), filled)

    # longest run of consecutive NaN in any single sensor, for the report
    longest = 0
    for j in range(nan_before.shape[1]):
        col = nan_before[:, j]
        if not col.any():
            continue
        run = best = 0
        for v in col:
            run = run + 1 if v else 0
            best = max(best, run)
        longest = max(longest, best)

    return filled.astype(np.float32), {
        "missing_before": int(nan_before.sum()),
        "interpolated": n_interp,
        "median_filled": int(still.sum()),
        "longest_gap_steps": int(longest),
    }


def calendar_channels(time_index: pd.DatetimeIndex, encoding: str):
    """time_of_day in [0, 1), day_of_week in [0, 1]."""
    if encoding != "fraction":
        raise NotImplementedError(
            f"time_of_day encoding {encoding!r} would change the channel count; "
            "update contract.CHANNELS and tell the team first")
    seconds = (time_index.hour * 3600 + time_index.minute * 60
               + time_index.second).to_numpy(dtype=np.float32)
    tod = seconds / 86400.0
    dow = time_index.dayofweek.to_numpy(dtype=np.float32) / 6.0
    return tod, dow


# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/default.yaml")
    args = ap.parse_args()
    cfg = load_config(args.config)
    P = pathlib.Path(cfg["data"]["processed_dir"])

    need = ["time_index.npy", "speed.npy", "occupancy.npy", "missing_mask.npy",
            "weather.npy", "event_load.npy", "event_active_decay.npy",
            "dist_to_venue.npy"]
    absent = [f for f in need if not (P / f).exists()]
    if absent:
        raise FileNotFoundError(
            f"missing {absent} - run `python -m src.data.align --config {args.config}` first")

    print(f"=== Stage 4: assemble the {contract.N_CHANNELS}-channel master tensor ===\n")

    time_index = pd.DatetimeIndex(np.load(P / "time_index.npy"))
    speed = np.load(P / "speed.npy")
    occ = np.load(P / "occupancy.npy")
    missing_mask = np.load(P / "missing_mask.npy")
    weather = np.load(P / "weather.npy")            # [T, 3] precip, temp, wind
    event_load = np.load(P / "event_load.npy")
    event_active = np.load(P / "event_active_decay.npy")
    dist = np.load(P / "dist_to_venue.npy")

    T, N = speed.shape
    C = contract.N_CHANNELS
    assert T == cfg["data"]["expected_timesteps"] and N == cfg["data"]["expected_nodes"]

    # --- the contract is the authority on order; prove we implement THAT order --
    expected = ["speed", "occupancy", "time_of_day", "day_of_week", "is_holiday",
                "temperature", "precipitation", "wind_speed", "dist_to_venue",
                "event_active_decay", "event_load"]
    actual = [name for name, _ in contract.CHANNELS]
    assert actual == expected, (
        "contract.CHANNELS changed but features.py was not updated.\n"
        f"  contract: {actual}\n  this file: {expected}")

    print("[1/4] impute missing values (inputs only; eval mask keeps the truth)")
    max_gap = contract.MISSING_POLICY["max_gap_steps"]
    speed_f, sp_rep = impute_per_sensor(speed, max_gap)
    occ_f, oc_rep = impute_per_sensor(occ, max_gap)
    for name, rep in (("speed", sp_rep), ("occupancy", oc_rep)):
        print(f"  {name:10s} missing {rep['missing_before']:>6,}  "
              f"interpolated {rep['interpolated']:>6,}  "
              f"median-filled {rep['median_filled']:>5,}  "
              f"longest gap {rep['longest_gap_steps']} steps "
              f"(cap {max_gap})")

    print("\n[2/4] calendar channels")
    tod, dow = calendar_channels(time_index, cfg["features"]["time_of_day"])
    hol = holiday_mask(time_index).astype(np.float32)
    print(f"  time_of_day  {tod.min():.4f} .. {tod.max():.4f}  "
          f"({contract.TIME_OF_DAY_ENCODING})")
    print(f"  day_of_week  {dow.min():.4f} .. {dow.max():.4f}")
    print(f"  is_holiday   {int(hol.sum()):,} of {len(hol):,} timesteps "
          f"({len(US_HOLIDAYS_2017_H1)} dates)")

    print("\n[3/4] broadcast into [T, N, C]")
    master = np.empty((T, N, C), dtype=np.float32)
    master[:, :, 0] = speed_f
    master[:, :, 1] = occ_f
    master[:, :, 2] = tod[:, None]
    master[:, :, 3] = dow[:, None]
    master[:, :, 4] = hol[:, None]
    master[:, :, 5] = weather[:, 1][:, None]        # temperature
    master[:, :, 6] = weather[:, 0][:, None]        # precipitation
    master[:, :, 7] = weather[:, 2][:, None]        # wind_speed
    master[:, :, 8] = dist[None, :]
    master[:, :, 9] = event_active
    master[:, :, 10] = event_load
    precip_src = "asos_sjc_nuq_idw_single_centroid"

    assert np.isfinite(master).all(), "master tensor contains NaN/Inf after imputation"

    # broadcasting sanity: a temporal channel must be identical across nodes, a
    # spatial one identical across time. Cheap to check, catches an index slip.
    # Derive the check from the contract rather than restating it: if someone
    # reorders CHANNELS, this fails instead of silently checking the wrong index.
    for idx, (name, kind) in enumerate(contract.CHANNELS):
        same_across_nodes = np.allclose(master[:, 0, idx], master[:, -1, idx])
        same_across_time = np.allclose(master[0, :, idx], master[-1, :, idx])
        if kind == "temporal":
            assert same_across_nodes, f"{name}: declared temporal but varies by node"
        elif kind == "spatial":
            assert same_across_time, f"{name}: declared spatial but varies over time"
        else:
            assert not same_across_nodes, (
                f"{name}: declared spatiotemporal but is identical across nodes")
    assert set(np.unique(master[:, :, contract.CHANNEL_INDEX["is_holiday"]])) <= {0.0, 1.0}, \
        "is_holiday must stay a 0/1 flag - it is in contract.PASSTHROUGH_CHANNELS"

    eval_mask = ~missing_mask                        # True = speed was observed
    print(f"  master {master.shape}  {master.nbytes / 1e6:.0f} MB")
    print(f"  eval_mask: {eval_mask.sum():,} / {eval_mask.size:,} usable "
          f"({100 * eval_mask.mean():.4f}%)")

    print("\n[4/4] writing")
    np.save(P / "master.npy", master)
    np.save(P / "eval_mask.npy", eval_mask)

    stats = {}
    for i, (name, kind) in enumerate(contract.CHANNELS):
        ch = master[:, :, i]
        stats[name] = {"kind": kind,
                       "min": round(float(ch.min()), 4),
                       "max": round(float(ch.max()), 4),
                       "mean": round(float(ch.mean()), 4),
                       "std": round(float(ch.std()), 4),
                       "normalized_later": name in contract.NORMALIZE_CHANNELS}
    meta = {
        "generated": dt.datetime.now().isoformat(timespec="seconds"),
        "config": args.config,
        "shape": [T, N, C],
        "channel_order": actual,
        "time_of_day_encoding": contract.TIME_OF_DAY_ENCODING,
        "precipitation_source": precip_src,
        "holidays": list(US_HOLIDAYS_2017_H1),
        "imputation": {"policy": contract.MISSING_POLICY["impute"],
                       "max_gap_steps": max_gap,
                       "speed": sp_rep, "occupancy": oc_rep},
        "eval_mask_usable_pct": round(100 * float(eval_mask.mean()), 5),
        "channel_stats": stats,
    }
    (P / "features_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print(f"  master.npy + eval_mask.npy -> {P}")
    print(f"  provenance -> {P / 'features_meta.json'}")
    print("\n  per-channel summary")
    print(f"  {'channel':16s}{'kind':16s}{'min':>10s}{'max':>10s}{'mean':>10s}{'std':>10s}  norm?")
    for name, s in stats.items():
        print(f"  {name:16s}{s['kind']:16s}{s['min']:>10.3f}{s['max']:>10.3f}"
              f"{s['mean']:>10.3f}{s['std']:>10.3f}   {'yes' if s['normalized_later'] else 'no'}")
    print("\nDone.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
