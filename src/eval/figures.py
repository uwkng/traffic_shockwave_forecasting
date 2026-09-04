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


def spacetime(speed_panels: dict, nodes: np.ndarray, times: pd.DatetimeIndex,
              title: str, out: pathlib.Path):
    """One panel per series in `speed_panels` (name -> [T, N] on the same grid)."""
    n = len(speed_panels)
    fig, axes = plt.subplots(1, n, figsize=(5.2 * n, 6.0), sharey=True,
                             constrained_layout=True)
    axes = np.atleast_1d(axes)
    for ax, (name, arr) in zip(axes, speed_panels.items()):
        im = ax.imshow(arr[:, nodes], aspect="auto", origin="lower",
                       cmap=SPEED_CMAP, vmin=SPEED_VMIN, vmax=SPEED_VMAX,
                       interpolation="nearest")
        ax.set_title(name)
        ax.set_xlabel("sensor, in road order  ->  direction of travel")
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
    args = ap.parse_args()
    cfg = load_config(args.config)

    proc = pathlib.Path(cfg["data"]["processed_dir"])
    speed = np.load(proc / "speed.npy").astype(np.float32)
    speed = np.nan_to_num(speed, nan=np.nanmedian(speed))
    ids = np.load(proc / "sensor_ids.npy", allow_pickle=True).astype(str)
    adj = np.load(proc / "adj_mx.npy")
    ti = pd.DatetimeIndex(np.load(proc / "time_index.npy", allow_pickle=True))

    day = pd.Timestamp(args.date)
    sel = ((ti >= day + pd.Timedelta(hours=args.from_hour))
           & (ti < day + pd.Timedelta(hours=args.to_hour)))
    assert sel.any(), f"{args.date} is not in the time axis"
    nodes = freeway_order(cfg, ids, args.freeway)

    panels = {"observed": speed[sel]}
    for f in args.predictions:
        z = np.load(f)
        panels[pathlib.Path(f).stem] = z["pred"][:, contract.EVAL_HORIZON_STEPS["30min"] - 1]

    p1 = spacetime(panels, nodes, ti[sel],
                   f"{args.freeway}, {args.date} - "
                   f"a band leaning backwards is a shockwave travelling upstream",
                   FIGDIR / f"spacetime_{args.freeway}_{args.date}.png")
    p2 = propagation_figure({"observed": speed}, adj,
                            FIGDIR / "propagation_ratio.png")
    print(f"  {p1}\n  {p2}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
