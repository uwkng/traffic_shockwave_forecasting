"""
Stage 6 - Split the sample index chronologically. Two modes, no shuffling ever.

`single`
    The classic PEMS-BAY 70/10/20 benchmark split. Use this - and only this - when
    reproducing published vanilla-STGCN numbers, so the comparison is fair.

`rolling`
    Expanding-window (rolling-origin) backtesting. Fold k trains on everything
    before its own test block, so there is still no future leakage, but the wet
    season and the arena calendar land inside test blocks instead of being locked
    in train. Under `single`, the test period holds 0 hours of heavy rain and
    12 events; across rolling folds it is ~26 hours and ~65 events.

Both modes drop `EMBARGO = 24` samples on each side of every boundary. Chronology
alone is not enough: without the embargo the last train sample's target window
overlaps the first val sample's input window, which is leakage.

Outputs (`data/processed/`):

    splits.npz            one int32 index array per (split, part), e.g. `single__train`
    splits_meta.json      date ranges, counts, and a per-fold census of how many
                          events and heavy-rain hours each test block actually
                          contains - the number the paper has to report as n

Usage
-----
    python -m src.data.split --config configs/default.yaml
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
from src.eval import windows


def load_config(path: str) -> dict:
    with open(path, encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def _embargo(part: np.ndarray, drop: int, *, head: bool, tail: bool) -> np.ndarray:
    """Remove `drop` samples from the head and/or tail of a contiguous block."""
    lo = drop if head else 0
    hi = len(part) - drop if tail else len(part)
    return part[lo:max(lo, hi)]


def split_single(index: np.ndarray, cfg: dict) -> dict:
    r = cfg["split"]["single"]
    drop = cfg["split"]["embargo_steps"]
    n = len(index)
    i_tr = int(n * r["train"])
    i_va = int(n * (r["train"] + r["val"]))
    return {
        # train: only its tail touches a boundary; test: only its head
        "train": _embargo(index[:i_tr], drop, head=False, tail=True),
        "val": _embargo(index[i_tr:i_va], drop, head=True, tail=True),
        "test": _embargo(index[i_va:], drop, head=True, tail=False),
    }


def split_rolling(index: np.ndarray, time_index: pd.DatetimeIndex, cfg: dict):
    r = cfg["split"]["rolling"]
    drop = cfg["split"]["embargo_steps"]
    start = time_index[0]
    day = pd.Timedelta(days=1)

    starts = time_index.to_numpy()[index]          # wall-clock start of each sample
    folds = {}
    for k in range(r["n_folds"]):
        test_lo = start + (r["min_train_days"] + k * r["test_block_days"]) * day
        test_hi = test_lo + r["test_block_days"] * day
        if test_lo >= time_index[-1]:
            break
        val_lo = test_lo - r["val_days"] * day

        tr = index[starts < np.datetime64(val_lo)]
        va = index[(starts >= np.datetime64(val_lo)) & (starts < np.datetime64(test_lo))]
        te = index[(starts >= np.datetime64(test_lo)) & (starts < np.datetime64(test_hi))]
        if not len(te):
            break
        folds[f"fold{k:02d}"] = {
            "train": _embargo(tr, drop, head=False, tail=True),
            "val": _embargo(va, drop, head=True, tail=True),
            "test": _embargo(te, drop, head=True, tail=False),
            "_test_span": (test_lo, min(test_hi, time_index[-1])),
        }
    return folds


# ---------------------------------------------------------------------------
# the census: what is actually IN each test block
# ---------------------------------------------------------------------------

def census(cfg: dict, time_index: pd.DatetimeIndex, test_idx: np.ndarray) -> dict:
    """What a test block actually contains - and this, not the timestep count,
    is the paper's n.

    Counted in EPISODES, per contract.EFFECTIVE_N_UNIT. Neighbouring 5-min steps
    and neighbouring sensors are heavily correlated, so "2,504 adverse-weather
    timesteps" is not 2,504 independent observations; it is 90 rain spells. A
    standard error computed over timesteps understates it by roughly
    sqrt(n_timesteps / n_episodes).

    Read the numbers this prints before choosing a split. The single 70/10/20
    test block scores ZERO adverse-weather episodes, which is why no weather
    result may be reported on it.
    """
    empty = {"episodes_wx": 0, "episodes_events": 0, "holidays": 0,
             "steps_wx": 0, "precip_mm": 0.0}
    if not len(test_idx):
        return empty
    span = contract.INPUT_WINDOW + contract.HORIZON
    lo = time_index[test_idx.min()]
    hi = time_index[min(test_idx.max() + span - 1, len(time_index) - 1)]
    block = (time_index >= lo) & (time_index <= hi)

    def episodes(mask, gap=12):
        """Contiguous runs, merged across gaps shorter than `gap` steps."""
        i = np.flatnonzero(mask)
        return 0 if not len(i) else 1 + int((np.diff(i) > gap).sum())

    wx = windows.weather_windows(cfg["data"]["weather_csv"], time_index) & block
    ev_mask, ev_used = windows.event_windows(
        cfg["data"]["events_csv"], time_index,
        min_attendance=cfg["eval"]["min_attendance"])
    ev_mask &= block
    hol = windows.holiday_mask(time_index) & block

    w = pd.read_csv(cfg["data"]["weather_csv"], parse_dates=["timestamp"])
    w = w[(w["timestamp"] >= lo) & (w["timestamp"] <= hi)]
    return {
        "from": str(lo), "to": str(hi),
        "episodes_wx": episodes(wx),
        "episodes_events": episodes(ev_mask),
        "holidays": episodes(hol, gap=288),
        "steps_wx": int(wx.sum()),
        "precip_mm": round(float(w["precipitation"].sum()), 1),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/default.yaml")
    args = ap.parse_args()
    cfg = load_config(args.config)
    P = pathlib.Path(cfg["data"]["processed_dir"])
    if not (P / "sample_index.npy").exists():
        raise FileNotFoundError("run stage 5 (`python -m src.data.samples`) first")

    index = np.load(P / "sample_index.npy").astype(np.int64)
    time_index = pd.DatetimeIndex(np.load(P / "time_index.npy"))
    drop = cfg["split"]["embargo_steps"]

    print("=== Stage 6: chronological split ===\n")
    print(f"  {len(index):,} samples, embargo {drop} samples at every boundary\n")

    arrays, meta = {}, {"embargo_steps": drop, "mode_in_config": cfg["split"]["mode"]}

    # ---- single ----------------------------------------------------------
    parts = split_single(index, cfg)
    print("  [single] classic PEMS-BAY benchmark split")
    meta["single"] = {}
    for name, part in parts.items():
        arrays[f"single__{name}"] = part.astype(np.int32)
        c = census(cfg, time_index, part) if name == "test" else {}
        meta["single"][name] = {"n": int(len(part)),
                                "from": str(time_index[part[0]]),
                                "to": str(time_index[part[-1]]), **c}
        print(f"    {name:5s} {len(part):>7,}  {time_index[part[0]]} -> {time_index[part[-1]]}"
              + (f"   | {c['episodes_events']} event episodes, "
                 f"{c['episodes_wx']} rain episodes, {c['holidays']} holidays"
                 if c else ""))
    sc = meta["single"]["test"]
    if sc["episodes_wx"] == 0:
        print("    NOTE: the single test block contains ZERO adverse-weather "
              "episodes.\n"
              "          A weather-conditioned model cannot differ from a "
              "traffic-only one on it -\n"
              "          the channel is constant across the whole block. Use "
              "rolling for any weather result.")

    # ---- rolling ---------------------------------------------------------
    folds = split_rolling(index, time_index, cfg)
    print(f"\n  [rolling] {len(folds)} expanding-window folds")
    print("    episodes, not timesteps: contract.EFFECTIVE_N_UNIT")
    print(f"    {'fold':7s}{'train':>8s}{'val':>7s}{'test':>7s}  {'test block':38s}"
          f"{'rain':>5s}{'evts':>5s}{'hol':>4s}")
    meta["rolling"], tot_ev, tot_wx, tot_hol = {}, 0, 0, 0
    for fname, f in folds.items():
        for name in ("train", "val", "test"):
            arrays[f"{fname}__{name}"] = f[name].astype(np.int32)
        c = census(cfg, time_index, f["test"])
        tot_ev += c["episodes_events"]
        tot_wx += c["episodes_wx"]
        tot_hol += c["holidays"]
        meta["rolling"][fname] = {
            "n_train": int(len(f["train"])), "n_val": int(len(f["val"])),
            "n_test": int(len(f["test"])),
            "train_to": str(time_index[f["train"][-1]]) if len(f["train"]) else None,
            "test": c,
        }
        print(f"    {fname:7s}{len(f['train']):>8,}{len(f['val']):>7,}{len(f['test']):>7,}"
              f"  {c.get('from','-')[:16]} -> {c.get('to','-')[:16]}   "
              f"{c['episodes_wx']:>5}{c['episodes_events']:>5}{c['holidays']:>4}")
    meta["rolling_totals"] = {
        "rain_episodes_across_test_folds": tot_wx,
        "event_episodes_across_test_folds": tot_ev,
        "holidays_across_test_folds": tot_hol,
        "note": "episodes, not timesteps - see contract.EFFECTIVE_N_UNIT",
    }
    print(f"    {'TOTAL':7s}{'':>8s}{'':>7s}{'':>7s}  {'':38s}"
          f"{tot_wx:>5}{tot_ev:>5}{tot_hol:>4}")

    # ---- leakage assertions ---------------------------------------------
    for label, parts_ in [("single", parts)] + [(k, v) for k, v in folds.items()]:
        tr, va, te = parts_["train"], parts_["val"], parts_["test"]
        span = contract.INPUT_WINDOW + contract.HORIZON
        assert not (set(tr) & set(va) & set(te)), f"{label}: overlapping sample ids"
        if len(tr) and len(va):
            assert tr.max() + span <= va.min(), \
                f"{label}: train target window overlaps val input window"
        if len(va) and len(te):
            assert va.max() + span <= te.min(), \
                f"{label}: val target window overlaps test input window"
    print("\n  leakage assertions passed for every split and fold")

    np.savez_compressed(P / "splits.npz", **arrays)
    meta["generated"] = dt.datetime.now().isoformat(timespec="seconds")
    (P / "splits_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"  splits.npz ({len(arrays)} arrays) + splits_meta.json -> {P}")
    print("\nDone.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
