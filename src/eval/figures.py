"""
Stage 13 - the two figures that make "shockwave" visible.

    python -m src.eval.figures --date 2017-02-08 --freeway 101-N

Nothing else in the repo draws anything, and the project's central claim is
spatial: a breakdown forms at a bottleneck and travels UPSTREAM against the
flow. That is invisible in a table of MAEs and unmistakable in a space-time
diagram, where it appears as a coloured band leaning backwards in time.

Two figures:

  1. `spacetime` - sensors along one freeway on the x axis in road order, time
     on the y axis, speed as colour. Drawn once for the observed series and once
     per model whose predictions are on disk, side by side at a fixed horizon.
     A model that has learnt propagation reproduces the lean; one that has
     learnt a smooth conditional mean paints a flat wash.

  2. `propagation` - the lag histogram behind `metrics.upstream_propagation`:
     how long after a downstream sensor breaks down does its upstream neighbour
     follow. Observed against predicted. This is the quantitative version of
     figure 1 and it is the one to put in the paper if only one fits.

Sensor order comes from `freeway_sensor_sequence.csv`, which ships with the
augmented release - do NOT sort by postmile yourself. Postmiles restart per
freeway (US-101 runs 383-398 while CA-87 runs 0-9) and the file already encodes
direction of travel.
"""

from __future__ import annotations

import argparse
import pathlib
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

from src import contract
from src.eval import metrics as M

FIGDIR = pathlib.Path("results/figures")
# green -> yellow -> deep red, so free flow recedes and breakdown advances.
SPEED_CMAP = "RdYlGn"
SPEED_VMIN, SPEED_VMAX = 20.0, 70.0


def load_config(path="configs/default.yaml"):
    with open(path, encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def freeway_order(cfg, sensor_ids, freeway_dir: str) -> np.ndarray:
    """Node indices along one freeway, in the direction of travel."""
    p = (pathlib.Path(cfg["data"]["pems_bay_dir"]) / "data" / "sensor_graph"
         / "freeway_sensor_sequence.csv")
    seq = pd.read_csv(p, dtype=str)
    seq.columns = [c.strip() for c in seq.columns]
    fw_col, ord_col, id_col = seq.columns[0], seq.columns[1], seq.columns[2]
    sub = seq[seq[fw_col] == freeway_dir].copy()
    assert len(sub), (f"no sensors for {freeway_dir!r}; available: "
                      f"{sorted(seq[fw_col].unique())}")
    sub[ord_col] = sub[ord_col].astype(int)
    sub = sub.sort_values(ord_col)
    pos = {s: i for i, s in enumerate(sensor_ids)}
    idx = [pos[s] for s in sub[id_col] if s in pos]
    assert idx, f"none of the {freeway_dir} sensors are in the node set"
    return np.asarray(idx)


def prediction_panel(files, horizon_key: str, n_times: int, n_nodes: int):
    """[T, N] of the h-step-ahead forecast, on the SAME time grid as `speed`.

    A prediction file is indexed by SAMPLE, not by timestep, and covers a whole
    test block. `pred[i, h]` is the forecast for timestep `timesteps[i] + h`, so
    it has to be scattered back onto the global axis before it can be drawn
    beside the observed series. The first version of this module passed
    `z["pred"][:, h]` straight into `imshow` next to one day of observations;
    with `sharey=True` that squashed the observed panel into a sliver at the
    bottom of the figure and captioned the result as a comparison.

    Several files for the same model (the seeds) are averaged.
    """
    h = contract.EVAL_HORIZON_STEPS[horizon_key] - 1
    total = np.zeros((n_times, n_nodes), np.float64)
    count = np.zeros((n_times, 1), np.float64)
    for f in files:
        z = np.load(f)
        ts = z["timesteps"] + h
        keep = ts < n_times
        total[ts[keep]] += z["pred"][keep, h]
        count[ts[keep]] += 1.0
    with np.errstate(invalid="ignore"):
        out = total / np.where(count == 0, np.nan, count)
    return out.astype(np.float32)


def group_seeds(files):
    """model name -> its files, collapsing __seedNN and __foldNN."""
    import collections
    by = collections.defaultdict(list)
    for f in files:
        name = pathlib.Path(f).stem
        name = re.sub(r"__fold\d+", "", name)
        name = re.sub(r"__seed\d+(?=__|$)", "", name)
        by[name].append(f)
    return by


def spacetime(speed_panels: dict, nodes: np.ndarray, times: pd.DatetimeIndex,
              title: str, out: pathlib.Path):
    """One panel per series in `speed_panels` (name -> [T, N] on the same grid)."""
    n = len(speed_panels)
    fig, axes = plt.subplots(1, n, figsize=(3.6 * n, 5.4), sharey=True,
                             constrained_layout=True)
    axes = np.atleast_1d(axes)
    for ax, (name, arr) in zip(axes, speed_panels.items()):
        cmap = plt.get_cmap(SPEED_CMAP).copy()
        cmap.set_bad("white")
        im = ax.imshow(np.ma.masked_invalid(arr[:, nodes]), aspect="auto",
                       origin="lower", cmap=cmap, vmin=SPEED_VMIN,
                       vmax=SPEED_VMAX, interpolation="nearest")
        ax.set_title(name, fontsize=9)
    # one x label for the row: three overlapping copies is not three labels.
    axes[len(axes) // 2].set_xlabel(
        "sensor, in road order  ->  direction of travel")
    step = max(1, len(times) // 12)
    axes[0].set_yticks(np.arange(0, len(times), step))
    axes[0].set_yticklabels([t.strftime("%H:%M") for t in times[::step]])
    axes[0].set_ylabel("time of day")
    fig.colorbar(im, ax=axes, label="speed (mph)", shrink=0.85)
    fig.suptitle(title)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def propagation_figure(panels: dict, adj: np.ndarray, out: pathlib.Path):
    """Upstream/downstream co-occurrence ratio against lag, one line per series."""
    fig, ax = plt.subplots(figsize=(6.4, 4.2), constrained_layout=True)
    lags = (1, 2, 3, 4, 6, 9, 12)
    for name, arr in panels.items():
        r = M.upstream_propagation(arr, adj, lags=lags)
        xs = [lag * contract.FREQ_MIN for lag in lags
              if f"{lag * contract.FREQ_MIN}min" in r]
        ys = [r[f"{x}min"]["ratio"] for x in xs]
        ax.plot(xs, ys, marker="o", label=name)
    ax.axhline(1.0, color="0.4", lw=1, ls="--")
    ax.annotate("1.0 = no preferred direction", xy=(xs[-1], 1.0),
                xytext=(-4, 6), textcoords="offset points", ha="right",
                fontsize=8, color="0.35")
    ax.set_xlabel("lag (minutes)")
    ax.set_ylabel("upstream / downstream co-occurrence")
    ax.set_title("Shockwaves travel against the flow")
    ax.legend()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def auto_dates(cfg, time_index) -> list[str]:
    """One day per rolling fold: the wettest commute day in its TEST block.

    Choosing the day by hand is how a figure ends up showing the case the model
    happened to get right. This picks on the exogenous signal alone - rainfall
    during the weekday peak - which is known before any model runs.

    Holidays are excluded, and that is not cosmetic. The first version of this
    function returned 2017-02-20 for fold00: the wettest commute-hour day in
    that test block is Presidents' Day, which has no commute. Measured, US
    holidays run +9.50 mph in the AM peak and +11.27 in the PM - the figure
    would have shown a free-flowing network captioned as a rainy rush hour.
    """
    import json
    from src.eval import windows as W
    rain = W.weather_windows(cfg["data"]["weather_csv"], time_index)
    peak = W.commute_mask(time_index)
    meta = json.loads(
        (pathlib.Path(cfg["data"]["processed_dir"]) / "splits_meta.json")
        .read_text(encoding="utf-8"))["rolling"]
    idx = pd.DatetimeIndex(time_index)
    out = []
    for fold in sorted(meta):
        t = meta[fold]["test"]
        block = (idx >= pd.Timestamp(t["from"])) & (idx <= pd.Timestamp(t["to"]))
        wet = rain & peak & block & ~W.holiday_mask(time_index)
        if not wet.any():
            continue
        days = pd.Series(wet).groupby(idx.date).sum()
        out.append(str(days.idxmax()))
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--date", default="2017-02-08",
                    help="a day inside fold00's test block (01-29..02-26)")
    ap.add_argument("--freeway", default="101-N")
    ap.add_argument("--from-hour", type=int, default=5)
    ap.add_argument("--to-hour", type=int, default=22)
    ap.add_argument("--predictions", nargs="*", default=[],
                    help="data/processed/predictions/*.npz to overlay")
    ap.add_argument("--horizon", default="60min",
                    choices=list(contract.EVAL_HORIZON_STEPS),
                    help="which forecast horizon to draw")
    ap.add_argument("--max-panels", type=int, default=4,
                    help="refuse to draw more model panels than this")
    ap.add_argument("--auto", action="store_true",
                    help="ignore --date; draw one panel per rolling fold, each "
                         "on the wettest commute day inside that fold's TEST "
                         "block. Picking by hand risks picking a day the model "
                         "happens to do well on.")
    args = ap.parse_args()
    cfg = load_config(args.config)

    proc = pathlib.Path(cfg["data"]["processed_dir"])
    speed = np.load(proc / "speed.npy").astype(np.float32)
    speed = np.nan_to_num(speed, nan=np.nanmedian(speed))
    ids = np.load(proc / "sensor_ids.npy", allow_pickle=True).astype(str)
    adj = np.load(proc / "adj_mx.npy")
    ti = pd.DatetimeIndex(np.load(proc / "time_index.npy", allow_pickle=True))

    nodes = freeway_order(cfg, ids, args.freeway)
    dates = [args.date]
    if args.auto:
        dates = auto_dates(cfg, ti)
        print(f"  auto-selected: {', '.join(dates)}")

    written = []
    for date in dates:
        day = pd.Timestamp(date)
        sel = ((ti >= day + pd.Timedelta(hours=args.from_hour))
               & (ti < day + pd.Timedelta(hours=args.to_hour)))
        if not sel.any():
            print(f"  skipping {date}: not in the time axis")
            continue
        panels = {"observed": speed[sel]}
        grouped = group_seeds(args.predictions)
        if len(grouped) > args.max_panels:
            raise SystemExit(
                f"{len(grouped)} models would give {len(grouped) + 1} panels. "
                f"A space-time diagram is read by comparing two or three series "
                f"side by side, not {len(grouped)}. Pass fewer files, or raise "
                f"--max-panels deliberately.")
        for name, files in grouped.items():
            full = prediction_panel(files, args.horizon, len(ti), speed.shape[1])
            if np.isnan(full[sel]).all():
                print(f"  skipping {name} on {date}: no predictions cover it")
                continue
            panels[f"{name}\n({args.horizon} ahead)"] = full[sel]
        written.append(spacetime(
            panels, nodes, ti[sel],
            f"{args.freeway}, {date} - a band leaning backwards is a "
            f"shockwave travelling upstream",
            FIGDIR / f"spacetime_{args.freeway}_{date}.png"))

    written.append(propagation_figure({"observed": speed}, adj,
                                      FIGDIR / "propagation_ratio.png"))
    for w in written:
        print(f"  {w}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
