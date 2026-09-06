"""
Stage 11 - assemble the tables. This is the paper.

    python -m src.eval.report --split single
    python -m src.eval.report --all-folds

Reads every prediction file in data/processed/predictions/ for the requested
split, scores each one against the exogenous windows, and writes
results/report__<split>.{json,md}.

It does not care which model produced a prediction: train.py and baselines.py
write the same four keys, so STGCN, historical average and persistence are
scored by identical code on identical samples. That is the point - the
comparison against a trivial baseline is only meaningful if nothing differs but
the model.
"""

from __future__ import annotations

import argparse
import collections
import json
import pathlib
import re

import numpy as np
import pandas as pd

from src import contract
from src.eval import metrics as M
from src.eval.windows import (weather_windows, event_windows, holiday_mask,
                              weather_commute_windows)
from src.models.loader import load_config

PRED = pathlib.Path("data/processed/predictions")
OUT = pathlib.Path("results")


def load_predictions(split: str) -> dict[str, list[pathlib.Path]]:
    """Group prediction files by model name, collapsing seeds together."""
    by_model = collections.defaultdict(list)
    for f in sorted(PRED.glob(f"*__{split}*.npz")):
        # NOT anchored with $: a run tagged `__weighted` or `__diffusion`
        # carries a suffix AFTER the seed, and an anchored pattern left every
        # such run as its own single-seed "model" - 0_speed__seed42__weighted,
        # __seed43__weighted, __seed44__weighted - so the seed spread silently
        # became three separate rows with sd 0.
        name = re.sub(r"__seed\d+(?=__|$)", "", f.stem.replace(f"__{split}", ""))
        by_model[name].append(f)
    return dict(by_model)


def build_windows(cfg, time_index):
    """The exogenous windows. Holidays are reported but NEVER merged into
    'adverse': a holiday makes traffic faster, and averaging it with rain would
    cancel both.

    `weather_commute` is the compound window. It is deliberately rain-AND-peak
    rather than the rain-AND-event one a reader expects: rain overlaps an egress
    hour 3 times in six months (18 timesteps), while rain in the weekday peak
    gives 34 episodes over 19 days - and 88.4% of peak timesteps carry a
    shockwave against 18.2% off-peak. See src/eval/windows.py.
    """
    d = cfg["data"]
    return {
        "adverse_weather": weather_windows(d["weather_csv"], time_index,
                                           cfg["eval"]["weather"]["adverse_mm"]),
        "weather_commute": weather_commute_windows(
            d["weather_csv"], time_index, cfg["eval"]["weather"]["adverse_mm"]),
        "event_egress": event_windows(d["events_csv"], time_index,
                                      min_attendance=cfg["eval"]["min_attendance"])[0],
        "holiday": holiday_mask(time_index),
    }


def score_model(files, eval_mask, windows, dist_km, cfg):
    """Average metrics over seeds; report the spread so it is visible."""
    runs = []
    for f in files:
        z = np.load(f)
        pred, true, ts = z["pred"], z["true"], z["timesteps"]
        r = M.stratify(pred, true, ts, eval_mask, windows)
        r["by_distance"] = M.by_distance(pred, true, ts, eval_mask, dist_km,
                                         cfg["eval"]["distance_bins_km"])
        egress = M.samples_in_window(ts, windows["event_egress"])
        r["by_distance_egress"] = M.by_distance(pred, true, ts, eval_mask, dist_km,
                                                cfg["eval"]["distance_bins_km"], egress)
        o = cfg["eval"]["onset"]
        r["onset"] = M.onset_lead_time(pred, true, ts, o["speed_mph"],
                                       o["persist_steps"], dist_km, max_km=2.0)
        runs.append(r)

    def agg(path):
        vals = []
        for r in runs:
            v = r
            for k in path:
                v = v.get(k) if isinstance(v, dict) else None
                if v is None:
                    return None
            vals.append(v)
        return {"mean": float(np.mean(vals)), "sd": float(np.std(vals))} if vals else None

    out = {"n_seeds": len(runs), "raw": runs}
    for w in ["all"] + list(windows):
        if w == "all":
            out["all"] = {h: agg(["all", h, "mae"]) for h in contract.EVAL_HORIZON_STEPS}
        else:
            out[w] = {
                "n_episodes": runs[0][w]["n_episodes"],
                "n_samples_inside": runs[0][w]["n_samples_inside"],
                "pct_of_test": runs[0][w]["pct_of_test"],
                "inside": {h: agg([w, "inside", h, "mae"]) for h in contract.EVAL_HORIZON_STEPS},
                "outside": {h: agg([w, "outside", h, "mae"]) for h in contract.EVAL_HORIZON_STEPS},
            }
    return out


def fmt(v):
    if v is None:
        return "     -"
    return f"{v['mean']:.3f}" + (f"±{v['sd']:.3f}" if v["sd"] > 5e-4 else "")


def render(split, scored, windows, cfg) -> str:
    L = [f"# Stratified results - split `{split}`", ""]
    hs = list(contract.EVAL_HORIZON_STEPS)

    L += ["## Aggregate (the number the field reports)", "",
          "| model | " + " | ".join(hs) + " |",
          "|---" * (len(hs) + 1) + "|"]
    for m, s in scored.items():
        L.append(f"| `{m}` | " + " | ".join(fmt(s["all"][h]) for h in hs) + " |")
    L += ["", "MAE in mph, mean±sd over seeds.", ""]

    for w in windows:
        any_ = next(iter(scored.values()))[w]
        L += [f"## Window: `{w}`", "",
              f"{any_['n_episodes']} episodes, {any_['n_samples_inside']:,} test "
              f"samples ({any_['pct_of_test']}% of the split).", ""]
        if any_["n_samples_inside"] == 0:
            L += ["**This window is empty in this split.** Nothing can be "
                  "concluded from it here; use the rolling folds.", ""]
            continue
        L += ["| model | " + " | ".join(f"{h} in | {h} out" for h in hs) + " |",
              "|---" * (2 * len(hs) + 1) + "|"]
        for m, s in scored.items():
            cells = []
            for h in hs:
                cells += [fmt(s[w]["inside"][h]), fmt(s[w]["outside"][h])]
            L.append(f"| `{m}` | " + " | ".join(cells) + " |")
        L += ["", f"n = {any_['n_episodes']} episodes, not "
              f"{any_['n_samples_inside']:,} samples: neighbouring 5-min steps "
              "and sensors are correlated, so any error bar must be quoted per "
              "episode.", ""]

    L += ["## MAE by distance from the nearest venue", "",
          "If the error grows towards a venue during egress but not otherwise, "
          "the error is spatially anchored to the crowd. If it is flat, it is "
          "not.", ""]
    bins = list(next(iter(scored.values()))["raw"][0]["by_distance"])
    L += ["| model | window | " + " | ".join(bins) + " |", "|---" * (len(bins) + 2) + "|"]
    for m, s in scored.items():
        for key, lab in [("by_distance", "all"), ("by_distance_egress", "egress")]:
            d = s["raw"][0][key]
            if not d:
                continue
            L.append(f"| `{m}` | {lab} | "
                     + " | ".join(f"{d[b]['mae']:.3f}" if b in d else "-" for b in bins) + " |")

    o_cfg = cfg["eval"]["onset"]
    L += ["", "## Congestion onset: lead time over a reactive rule", "",
          f"The reactive baseline acts once congestion is OBSERVED "
          f"(< {o_cfg['speed_mph']} mph for "
          f"{o_cfg['persist_steps'] * contract.FREQ_MIN} min); the model acts "
          "when it PREDICTS that. Positive lead means the model was earlier. "
          "Sensors within 2 km of a venue.",
          "",
          "A median lead of 0 for the trivial baselines is expected and is the "
          "control: persistence copies the last observation, so it cannot see "
          "an onset before it happens, and the historical average has no idea "
          "which day it is. Any positive lead a trained model shows is measured "
          "against this floor.", "",
          "| model | onsets | detected | recall | false alarms | lead (median, min) |",
          "|---|---|---|---|---|---|"]
    for m, s in scored.items():
        o = s["raw"][0]["onset"]
        L.append(f"| `{m}` | {o['n_onsets_observed']:,} | {o['n_detected']:,} | "
                 f"{o['recall']:.2f} | {o['false_alarms']:,} | "
                 f"{o['lead_min_median']:.1f} |")
    return "\n".join(L) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", default="single")
    ap.add_argument("--all-folds", action="store_true")
    ap.add_argument("--config", default="configs/default.yaml")
    args = ap.parse_args()
    cfg = load_config(args.config)
    P = pathlib.Path(cfg["data"]["processed_dir"])

    eval_mask = np.load(P / "eval_mask.npy")
    dist_km = np.load(P / "dist_to_venue.npy")
    time_index = pd.DatetimeIndex(np.load(P / "time_index.npy"))
    windows = build_windows(cfg, time_index)

    splits = ([f"fold{i:02d}" for i in range(cfg["split"]["rolling"]["n_folds"])]
              if args.all_folds else [args.split])
    OUT.mkdir(exist_ok=True)

    for split in splits:
        found = load_predictions(split)
        if not found:
            print(f"  {split}: no predictions in {PRED}/ - run "
                  f"`python -m src.models.baselines --split {split}` and "
                  f"`python -m src.models.train --split {split}` first")
            continue
        scored = {m: score_model(f, eval_mask, windows, dist_km, cfg)
                  for m, f in found.items()}
        md = render(split, scored, windows, cfg)
        (OUT / f"report__{split}.md").write_text(md, encoding="utf-8")
        (OUT / f"report__{split}.json").write_text(
            json.dumps({m: {k: v for k, v in s.items() if k != "raw"}
                        for m, s in scored.items()}, indent=2), encoding="utf-8")
        print(md)
        print(f"  -> {OUT}/report__{split}.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
