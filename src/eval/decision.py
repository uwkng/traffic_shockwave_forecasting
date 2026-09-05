"""
Stage 12 - turn a prediction into the decision it supports, and cost it.

    python -m src.eval.decision --split single

The proposal's framing: each forecast is converted into one operational choice -
deploy a special-event signal-timing plan on the sensors near a venue, or do not
- and compared against a REACTIVE baseline that acts only once congestion has
been measured. The question is not whose MAE is lower. It is whether a lower MAE
converts into acting sooner, and at what cost in false alarms.

Errors are asymmetric and the ratio is the whole argument. `cost_ratio` is swept
rather than fixed, because the honest claim is "the model wins for any ratio
above X", not "the model wins at the ratio we chose".

TWO USERS SIT ON ONE SWEEP. The same forecasts support two different decisions,
and they occupy opposite ends of `cost_ratios`. Name which end you are quoting.

  * Network operator - pre-deploy a special-event signal plan. A false alarm
    costs staff hours; a miss costs egress delay for tens of thousands. Ratio
    20-50, the right-hand end. Scope `egress`, and it is UNDERPOWERED: fold00
    has 21 positive cases in total. Report it as a case study, not an estimate.

  * Navigation service - warn one driver to reroute. A false alarm costs that
    driver a 2-5 minute detour. A miss costs the delay of driving through the
    jam, and that is measurable on this corpus rather than assumed:

        free-flow median speed                     65.8 mph
        contiguous <45 mph stretch along a freeway p50 1.99 km, p90 7.87 km
        delay crossing the p50 stretch at 30 mph   +1.3 min
        delay crossing the p90 stretch             +5.3 min
        one 0.74 km segment below 25 mph           +1.06 min (3.5x free-flow)

    So a miss costs roughly what a false alarm costs - ratio 1-3, the left-hand
    end. That is the demanding operating point: a driver-facing product cannot
    spend false alarms the way an operator can. Scope `all`, n = 16,743 onsets
    in fold00, which is where the statistical power is.

The asymmetry between the two scopes is itself worth stating: the decision with
the most tolerance for false alarms is the one we can least afford to measure.

Everything here is computed on the SAME predictions src/eval/report.py scores,
so the decision numbers and the MAE numbers cannot drift apart.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import re

import numpy as np
import pandas as pd

from src import contract
from src.eval.windows import event_windows
from src.models.loader import load_config

PRED = pathlib.Path("data/processed/predictions")
OUT = pathlib.Path("results")


def congested(speed: np.ndarray, threshold: float, persist: int) -> np.ndarray:
    """[n, H, N] -> [n, N] bool: does a run of `persist` low steps occur?"""
    low = speed < threshold
    if persist > 1:
        win = np.lib.stride_tricks.sliding_window_view(low, persist, axis=1)
        low = win.all(axis=-1)
    return low.any(axis=1)


def decide(pred, true, threshold, persist):
    """Confusion counts for 'deploy the plan', per (sample, node)."""
    want = congested(true, threshold, persist)     # congestion actually occurs
    act = congested(pred, threshold, persist)      # the model says deploy
    return {"tp": int((act & want).sum()), "fp": int((act & ~want).sum()),
            "fn": int((~act & want).sum()), "tn": int((~act & ~want).sum())}


def cost(c: dict, ratio: float) -> float:
    """Expected cost in false-alarm-equivalents. A miss costs `ratio` of them."""
    return c["fp"] + ratio * c["fn"]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", default="single")
    ap.add_argument("--max-km", type=float, default=2.0,
                    help="only sensors this close to a venue can be re-timed")
    ap.add_argument("--config", default="configs/default.yaml")
    args = ap.parse_args()
    cfg = load_config(args.config)
    P = pathlib.Path(cfg["data"]["processed_dir"])
    o = cfg["eval"]["onset"]
    thr, persist = o["speed_mph"], o["persist_steps"]

    dist = np.load(P / "dist_to_venue.npy")
    near = dist <= args.max_km
    time_index = pd.DatetimeIndex(np.load(P / "time_index.npy"))
    egress_mask, _ = event_windows(cfg["data"]["events_csv"], time_index,
                                   min_attendance=cfg["eval"]["min_attendance"])

    files = sorted(PRED.glob(f"*__{args.split}*.npz"))
    if not files:
        print(f"  no predictions for {args.split} in {PRED}/")
        return 1

    ratios = [1, 2, 5, 10, 20, 50]
    rows, results = [], {}
    print(f"=== stage 12: decision layer, split {args.split} ===")
    print(f"  deploy a special-event timing plan on the {near.sum()} sensors "
          f"within {args.max_km} km of a venue")
    print(f"  congestion = below {thr} mph for {persist * contract.FREQ_MIN} min "
          f"inside the {contract.HORIZON * contract.FREQ_MIN}-min horizon\n")

    for f in files:
        model = re.sub(r"__seed\d+$", "", f.stem.replace(f"__{args.split}", ""))
        z = np.load(f)
        pred, true, ts = z["pred"][:, :, near], z["true"][:, :, near], z["timesteps"]
        idx = ts[:, None] + np.arange(contract.HORIZON)[None, :]
        in_egress = egress_mask[idx].any(axis=1)

        for scope, sel in [("all", slice(None)), ("egress", in_egress)]:
            p, t = pred[sel], true[sel]
            if len(p) == 0:
                continue
            c = decide(p, t, thr, persist)
            prec = c["tp"] / max(c["tp"] + c["fp"], 1)
            rec = c["tp"] / max(c["tp"] + c["fn"], 1)
            entry = dict(c, precision=prec, recall=rec,
                         f1=2 * prec * rec / max(prec + rec, 1e-9),
                         cost={str(r): cost(c, r) for r in ratios},
                         n_samples=int(len(p)))
            results.setdefault(model, {})[scope] = entry
            rows.append((model, scope, c, prec, rec, entry["f1"]))

    print(f"  {'model':22s}{'scope':8s}{'TP':>8s}{'FP':>8s}{'FN':>8s}"
          f"{'prec':>7s}{'rec':>7s}{'F1':>7s}")
    for model, scope, c, prec, rec, f1 in rows:
        print(f"  {model:22s}{scope:8s}{c['tp']:>8,}{c['fp']:>8,}{c['fn']:>8,}"
              f"{prec:>7.3f}{rec:>7.3f}{f1:>7.3f}")

    print(f"\n  expected cost in false-alarm-equivalents (a miss costs `ratio` "
          f"of them);\n  lower is better, and the winner changing with the ratio "
          f"is the finding:")
    print(f"  {'model':22s}{'scope':8s}" + "".join(f"{'x' + str(r):>12s}" for r in ratios))
    for model, sc in results.items():
        for scope, e in sc.items():
            print(f"  {model:22s}{scope:8s}"
                  + "".join(f"{e['cost'][str(r)]:>12,.0f}" for r in ratios))

    OUT.mkdir(exist_ok=True)
    (OUT / f"decision__{args.split}.json").write_text(
        json.dumps({"split": args.split, "max_km": args.max_km,
                    "threshold_mph": thr, "persist_steps": persist,
                    "cost_ratios": ratios, "models": results}, indent=2),
        encoding="utf-8")
    print(f"\n  -> {OUT}/decision__{args.split}.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
