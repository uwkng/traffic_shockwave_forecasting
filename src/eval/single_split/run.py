"""Stage 11 — evaluate predictions, stratified by window, distance, horizon.

    python -m src.eval.single_split.run --rung 0_speed --split fold00 --seed 42
    python -m src.eval.single_split.run --rung 6_all  --split fold00 --seed 42 --compare 0_speed

Loads the .npy predictions written by predict.py, slices them by evaluation
window (normal / adverse weather / event egress / holiday) and by distance
bin from the nearest venue, and reports MAE / RMSE / MAPE at each horizon.

With --compare, runs a paired Wilcoxon test over episodes between two rungs.
"""

from __future__ import annotations

import argparse
import json
import pathlib

import numpy as np
import pandas as pd

from src import contract
from src.models.loader import build_loaders, load_config
from src.eval.single_split.metrics import all_metrics
from src.eval import windows


# ---------------------------------------------------------------------------

def _load_predictions(rung: str, split: str, seed: int,
                      config: str, future_covariates: bool = False) -> np.ndarray:
    cfg = load_config(config)
    ckpt_dir = pathlib.Path(cfg["training"]["checkpoint_dir"])
    fc_tag = "_fc" if future_covariates else ""
    path = ckpt_dir / f"pred_{rung}_{split}{fc_tag}_seed{seed}.npy"
    pred = np.load(path)
    assert pred.shape[1] == contract.HORIZON
    assert pred.shape[2] == contract.N_NODES
    return pred


def _load_ground_truth(split: str, config: str):
    """Return (true [n_test, H, N], time_indices [n_test], eval_mask [T, N])."""
    loaders, _, scaler = build_loaders(
        rung="0_speed", split=split, config=config, torch_loaders=False,
    )
    ds = loaders["datasets"]["test"]
    n = len(ds)
    H, N = contract.HORIZON, contract.N_NODES
    true = np.empty((n, H, N), dtype=np.float32)
    for i in range(n):
        _, y = ds[i]
        true[i] = y  # already raw mph
    timesteps = ds.timesteps()
    eval_mask = loaders["eval_mask"]
    return true, timesteps, eval_mask


def _build_window_masks(timesteps, config):
    """Return dict of (name -> [n_test] bool) window membership masks."""
    cfg = load_config(config)
    P = pathlib.Path(cfg["data"]["processed_dir"])
    time_index = np.load(P / "time_index.npy", allow_pickle=True)

    weather_mask_full = windows.weather_windows(cfg["data"]["weather_csv"], time_index)
    event_mask_full, events_df = windows.event_windows(
        cfg["data"]["events_csv"], time_index,
    )
    holiday_mask_full = windows.holiday_mask(time_index)

    out = {}
    out["adverse_weather"] = weather_mask_full[timesteps]
    out["event_egress"] = event_mask_full[timesteps]
    out["holiday"] = holiday_mask_full[timesteps]
    out["normal"] = ~(out["adverse_weather"] | out["event_egress"] | out["holiday"])
    return out


def _build_distance_bins(config):
    """Return (bin_edges, sensor_bin_indices [N])."""
    cfg = load_config(config)
    P = pathlib.Path(cfg["data"]["processed_dir"])
    master = np.load(P / "master.npy", mmap_mode="r")
    dist_idx = contract.CHANNEL_INDEX["dist_to_venue"]
    dist = master[0, :, dist_idx]  # static channel, same at every timestep
    bins = cfg["eval"]["distance_bins_km"]
    bin_idx = np.digitize(dist, bins) - 1
    return bins, bin_idx


# ---------------------------------------------------------------------------

def evaluate_split(rung: str, split: str, seed: int, config: str,
                   future_covariates: bool = False) -> dict:
    pred = _load_predictions(rung, split, seed, config, future_covariates)
    true, timesteps, eval_mask = _load_ground_truth(split, config)
    n_test = pred.shape[0]
    assert true.shape == pred.shape, f"shape mismatch: {true.shape} vs {pred.shape}"

    cfg = load_config(config)

    # Per-sample eval_mask: [n_test, H, N] — True where speed was observed
    H = contract.HORIZON
    sample_mask = np.ones_like(pred, dtype=bool)
    for i in range(n_test):
        t = int(timesteps[i])
        sample_mask[i] = eval_mask[t:t + H, :]

    window_masks = _build_window_masks(timesteps, config)
    bins, bin_idx = _build_distance_bins(config)

    results = {"rung": rung, "split": split, "seed": seed,
               "future_covariates": future_covariates,
               "n_test": n_test, "horizons": {}, "windows": {}, "distance": {}}

    # --- Overall and per-horizon ---
    for hname, hstep in contract.EVAL_HORIZON_STEPS.items():
        h = hstep - 1  # 0-indexed
        m = sample_mask[:, h, :]
        results["horizons"][hname] = all_metrics(pred[:, h, :], true[:, h, :], m)

    results["horizons"]["all"] = all_metrics(pred, true, sample_mask)

    # --- Per window, at each horizon ---
    for wname, wmask in window_masks.items():
        n_active = int(wmask.sum())
        if n_active == 0:
            results["windows"][wname] = {"n_samples": 0, "horizons": {}}
            continue
        idx = np.where(wmask)[0]
        p_w = pred[idx]
        t_w = true[idx]
        m_w = sample_mask[idx]
        wres = {"n_samples": n_active, "horizons": {}}
        for hname, hstep in contract.EVAL_HORIZON_STEPS.items():
            h = hstep - 1
            wres["horizons"][hname] = all_metrics(p_w[:, h, :], t_w[:, h, :], m_w[:, h, :])
        wres["horizons"]["all"] = all_metrics(p_w, t_w, m_w)
        results["windows"][wname] = wres

    # --- Per distance bin, at 60 min ---
    h60 = contract.EVAL_HORIZON_STEPS["60min"] - 1
    for b in range(len(bins) - 1):
        lo, hi = bins[b], bins[b + 1]
        label = f"{lo}-{hi}km"
        nodes = np.where((bin_idx == b))[0]
        if len(nodes) == 0:
            results["distance"][label] = {"n_nodes": 0}
            continue
        p_d = pred[:, h60, nodes]
        t_d = true[:, h60, nodes]
        m_d = sample_mask[:, h60, nodes]
        results["distance"][label] = {
            "n_nodes": int(len(nodes)),
            **all_metrics(p_d, t_d, m_d),
        }

    return results


def compare_rungs(rung_a: str, rung_b: str, split: str, seed: int,
                  config: str, future_covariates: bool = False) -> dict:
    """Paired Wilcoxon over episodes between two rungs."""
    from scipy.stats import wilcoxon

    pred_a = _load_predictions(rung_a, split, seed, config, future_covariates)
    pred_b = _load_predictions(rung_b, split, seed, config, future_covariates)
    true, timesteps, eval_mask = _load_ground_truth(split, config)

    results = {}
    for hname, hstep in contract.EVAL_HORIZON_STEPS.items():
        h = hstep - 1
        # Per-sample MAE across nodes
        mae_a = np.mean(np.abs(pred_a[:, h, :] - true[:, h, :]), axis=1)
        mae_b = np.mean(np.abs(pred_b[:, h, :] - true[:, h, :]), axis=1)
        diff = mae_a - mae_b
        stat, p = wilcoxon(diff, alternative="two-sided")
        results[hname] = {
            "mae_a": float(np.mean(mae_a)),
            "mae_b": float(np.mean(mae_b)),
            "diff_mean": float(np.mean(diff)),
            "wilcoxon_stat": float(stat),
            "p_value": float(p),
        }
    return results


# ---------------------------------------------------------------------------

def _print_results(results: dict):
    print(f"\n{'=' * 65}")
    print(f"  rung={results['rung']}  split={results['split']}  "
          f"seed={results['seed']}  n_test={results['n_test']}")
    print(f"{'=' * 65}")

    print("\n  Overall:")
    print(f"    {'horizon':>8s}  {'MAE':>7s}  {'RMSE':>7s}  {'MAPE':>7s}")
    for h, m in results["horizons"].items():
        print(f"    {h:>8s}  {m['mae']:7.4f}  {m['rmse']:7.4f}  {m['mape']:6.2f}%")

    print("\n  By window:")
    for wname, wdata in results["windows"].items():
        n = wdata["n_samples"]
        print(f"    {wname} (n={n}):")
        if n == 0:
            print(f"      (no samples)")
            continue
        for h, m in wdata["horizons"].items():
            print(f"      {h:>8s}  MAE={m['mae']:.4f}  RMSE={m['rmse']:.4f}  "
                  f"MAPE={m['mape']:.2f}%")

    print("\n  By distance (60 min):")
    for label, d in results["distance"].items():
        if d["n_nodes"] == 0:
            print(f"    {label}: (no nodes)")
        else:
            print(f"    {label} ({d['n_nodes']} nodes): MAE={d['mae']:.4f}  "
                  f"RMSE={d['rmse']:.4f}  MAPE={d['mape']:.2f}%")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rung", default="0_speed",
                    choices=list(contract.ABLATION_RUNGS))
    ap.add_argument("--split", default="single")
    ap.add_argument("--seed", type=int, default=contract.SEED)
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--compare", default=None,
                    help="Second rung to compare against (paired Wilcoxon)")
    ap.add_argument("--future-covariates", action="store_true",
                    help="Evaluate predictions from future-covariates runs")
    args = ap.parse_args()

    fc = args.future_covariates
    fc_tag = "_fc" if fc else ""
    results = evaluate_split(args.rung, args.split, args.seed, args.config, fc)
    _print_results(results)

    out_dir = pathlib.Path("results")
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / f"eval_{args.rung}_{args.split}{fc_tag}_seed{args.seed}.json"
    out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\n  saved: {out_path}")

    if args.compare:
        print(f"\n  Paired Wilcoxon: {args.rung} vs {args.compare}")
        comp = compare_rungs(args.rung, args.compare, args.split, args.seed,
                             args.config, fc)
        for h, c in comp.items():
            sign = "+" if c["diff_mean"] > 0 else ""
            sig = "***" if c["p_value"] < 0.001 else "**" if c["p_value"] < 0.01 else "*" if c["p_value"] < 0.05 else "ns"
            print(f"    {h:>5s}: {args.rung} MAE={c['mae_a']:.4f}  "
                  f"{args.compare} MAE={c['mae_b']:.4f}  "
                  f"diff={sign}{c['diff_mean']:.4f}  p={c['p_value']:.4f} {sig}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
