"""
Stage 11 - metrics, and the stratified reporting the project exists to produce.

    python -m src.eval.report --split single

This module has the arithmetic; `src/eval/report.py` assembles the tables.

The one idea here: every metric is computed THREE times - over the whole test
set, inside an exogenously-defined window, and over its complement. A model can
be excellent on aggregate and useless in exactly the hours an operator would act
on, and an aggregate number cannot show you that.

Two rules that decide whether the numbers mean anything:

  * MASK THE IMPUTED VALUES. 521 speed readings were interpolated so the model
    would see no NaN. Scoring against them scores the model against our own
    interpolation.
  * COUNT EPISODES, NOT TIMESTEPS. Neighbouring 5-min steps and neighbouring
    sensors are strongly correlated, so a standard error over node-timesteps
    understates it by roughly sqrt(n_timesteps / n_episodes). `n_episodes` is
    reported next to every windowed metric and is the n a claim may quote.
"""

from __future__ import annotations

import numpy as np

from src import contract


def episodes(mask: np.ndarray, gap: int = 12) -> int:
    """Contiguous runs of True, merged across gaps shorter than `gap` steps."""
    i = np.flatnonzero(mask)
    return 0 if not len(i) else 1 + int((np.diff(i) > gap).sum())


def _mae_rmse_mape(pred, true):
    d = pred - true
    ad = np.abs(d)
    nz = np.abs(true) > 1.0                 # a percentage error below 1 mph is noise
    return {"mae": float(ad.mean()),
            "rmse": float(np.sqrt((d ** 2).mean())),
            "mape": float((ad[nz] / np.abs(true[nz])).mean() * 100) if nz.any() else float("nan"),
            "n_node_timesteps": int(true.size)}


def score(pred: np.ndarray, true: np.ndarray, obs_mask: np.ndarray,
          sample_mask: np.ndarray | None = None) -> dict:
    """Metrics at every eval horizon.

    pred / true   [n_samples, HORIZON, N], de-normalised mph
    obs_mask      [n_samples, HORIZON, N] bool, True where speed was OBSERVED
    sample_mask   [n_samples] bool, which samples to score (a window). None = all.
    """
    if sample_mask is not None:
        pred, true, obs_mask = pred[sample_mask], true[sample_mask], obs_mask[sample_mask]
    out = {"n_samples": int(len(pred))}
    if len(pred) == 0:
        return out
    for name, step in [("all", None)] + list(contract.EVAL_HORIZON_STEPS.items()):
        p, t, m = ((pred, true, obs_mask) if step is None else
                   (pred[:, step - 1], true[:, step - 1], obs_mask[:, step - 1]))
        out[name] = _mae_rmse_mape(p[m], t[m])
    return out


def target_mask(timesteps: np.ndarray, eval_mask: np.ndarray) -> np.ndarray:
    """Expand [T, N] observation mask onto [n_samples, HORIZON, N]."""
    idx = timesteps[:, None] + np.arange(contract.HORIZON)[None, :]
    return eval_mask[idx]


def samples_in_window(timesteps: np.ndarray, window: np.ndarray,
                      require: str = "any") -> np.ndarray:
    """Which samples have their TARGET window overlapping a timestep window.

    `window` is a [T] bool mask from src.eval.windows. "any" selects a sample if
    any of its HORIZON target steps falls inside; "all" requires every step.

    "any" is the right default: a sample whose horizon contains the egress hour
    is a sample the model had to get the egress right for, even if the other
    steps are quiet.
    """
    idx = timesteps[:, None] + np.arange(contract.HORIZON)[None, :]
    hit = window[idx]
    return hit.all(axis=1) if require == "all" else hit.any(axis=1)


def stratify(pred, true, timesteps, eval_mask, windows: dict) -> dict:
    """Full test set, then each named window and its complement.

    `windows` maps a name to a [T] bool mask over timesteps.
    Returns {"all": ..., "<name>": {"inside": ..., "outside": ...,
                                    "n_episodes": ...}}
    """
    obs = target_mask(timesteps, eval_mask)
    out = {"all": score(pred, true, obs)}
    for name, w in windows.items():
        inside = samples_in_window(timesteps, w)
        out[name] = {
            "inside": score(pred, true, obs, inside),
            "outside": score(pred, true, obs, ~inside),
            "n_episodes": episodes(w[timesteps.min():timesteps.max() + 1]),
            "n_samples_inside": int(inside.sum()),
            "pct_of_test": round(100 * float(inside.mean()), 2),
        }
    return out


def by_distance(pred, true, timesteps, eval_mask, dist_km: np.ndarray,
                bins: list[float], sample_mask: np.ndarray | None = None) -> dict:
    """MAE by distance from the nearest venue - the proposal's shockwave test.

    If errors grow towards a venue during an egress hour and not otherwise, the
    error is spatially anchored to the crowd. If they are flat, it is not.
    """
    obs = target_mask(timesteps, eval_mask)
    if sample_mask is not None:
        pred, true, obs = pred[sample_mask], true[sample_mask], obs[sample_mask]
    out = {}
    for lo, hi in zip(bins[:-1], bins[1:]):
        nodes = (dist_km >= lo) & (dist_km < hi)
        if not nodes.any() or len(pred) == 0:
            continue
        p, t, m = pred[:, :, nodes], true[:, :, nodes], obs[:, :, nodes]
        out[f"{lo}-{hi}km"] = dict(_mae_rmse_mape(p[m], t[m]), n_nodes=int(nodes.sum()))
    return out


# ---------------------------------------------------------------------------
# Shockwave label + propagation. This is what makes the title defensible.
# ---------------------------------------------------------------------------
# A shockwave is a sharp deceleration that travels UPSTREAM against the flow.
# The label below is deliberately SPEED-ONLY, although the textbook definition
# also wants high occupancy. Both were measured; the occupancy conjunct changes
# nothing that matters and it would force the model to predict occupancy too:
#
#     definition        cells   rain (commute-ctrl)   event (high-conf)   up/down 15min
#     drop & occ>.2 &   0.533%  83.4% vs dry 88.6%    5.25x, z=3.66       1.88x
#     drop & <45 only   0.798%  91.8% vs dry 93.1%    2.50x, z=3.24       1.71x
#
# The occupancy version is a strict SUBSET (all 90,220 of its cells are in the
# 135,187 of the speed-only one), so it only tightens the label. Every
# conclusion survives. Keeping it speed-only means the model predicts speed and
# speed alone - which also keeps rung 0 directly comparable to the published
# PEMS-BAY numbers, all of which are single-channel speed.
SHOCKWAVE_DROP_MPH = 10.0      # fall over DROP_STEPS
SHOCKWAVE_DROP_STEPS = 3       # 15 min
SHOCKWAVE_SPEED_MPH = 45.0


def shockwave_label(speed: np.ndarray) -> np.ndarray:
    """[..., T, N] speed -> bool of the same shape. Time is the second-last axis.

    True where speed fell more than SHOCKWAVE_DROP_MPH over the preceding
    SHOCKWAVE_DROP_STEPS and is now below SHOCKWAVE_SPEED_MPH. The first
    DROP_STEPS entries along time are always False: there is no history to
    difference against, and guessing one would invent onsets.
    """
    k = SHOCKWAVE_DROP_STEPS
    out = np.zeros(speed.shape, dtype=bool)
    if speed.shape[-2] <= k:
        return out
    fell = (speed[..., :-k, :] - speed[..., k:, :]) > SHOCKWAVE_DROP_MPH
    out[..., k:, :] = fell & (speed[..., k:, :] < SHOCKWAVE_SPEED_MPH)
    return out


def upstream_propagation(speed: np.ndarray, adj: np.ndarray,
                         lags: tuple[int, ...] = (1, 2, 3, 6)) -> dict:
    """Does a shockwave travel against the traffic, and does the model see it?

    `adj` must be the RAW DIRECTED adjacency (data/processed/adj_mx.npy), not the
    symmetrised one the model's Laplacian uses. adj[i, j] > 0 means j is
    reachable driving from i, i.e. j is DOWNSTREAM of i. A shockwave propagates
    the other way: j breaks down first, i follows.

    For each lag we count ordered co-occurrences in both directions and report
    the ratio. Measured on the observed series, speed-only label:
        lag 10 min  1.53x     lag 15 min  1.71x
    A ratio near 1 would mean the label is picking up network-wide congestion
    rather than a travelling wave, so this doubles as a sanity check on the
    label itself. Run it on the TRUE series and on a model's predictions: a
    model that has learnt propagation reproduces the ratio, one that has learnt
    a smooth conditional mean flattens it towards 1.
    """
    sw = shockwave_label(speed)
    src, dst = np.nonzero(adj > 0)
    keep = src != dst
    src, dst = src[keep], dst[keep]
    out = {"n_directed_edges": int(len(src)),
           "shockwave_cell_pct": round(100 * float(sw.mean()), 4)}
    for lag in lags:
        if sw.shape[0] <= lag:
            continue
        up = int(sum((sw[:-lag, j] & sw[lag:, i]).sum() for i, j in zip(src, dst)))
        dn = int(sum((sw[:-lag, i] & sw[lag:, j]).sum() for i, j in zip(src, dst)))
        out[f"{lag * contract.FREQ_MIN}min"] = {
            "upstream": up, "downstream": dn,
            "ratio": round(up / dn, 3) if dn else None,
        }
    return out


def onset_lead_time(pred, true, timesteps, speed_mph: float, persist_steps: int,
                    dist_km: np.ndarray | None = None, max_km: float | None = None):
    """How much earlier than a reactive rule would the model have triggered?

    The reactive baseline acts only once congestion is OBSERVED: speed below
    `speed_mph` for `persist_steps` consecutive steps. The model acts as soon as
    it PREDICTS that. Lead time is the difference, in minutes, and it is the
    number the decision layer is actually about - a lower MAE is only useful if
    it converts into acting sooner.

    Returns per-onset lead times in minutes (positive = the model was earlier).
    """
    nodes = slice(None) if dist_km is None or max_km is None else (dist_km <= max_km)
    p, t = pred[:, :, nodes], true[:, :, nodes]
    step_min = contract.FREQ_MIN

    def first_hit(a):
        """Earliest horizon step where `persist_steps` consecutive steps are low."""
        low = a < speed_mph
        if persist_steps > 1:
            win = np.lib.stride_tricks.sliding_window_view(low, persist_steps, axis=1)
            low = win.all(axis=-1)
        idx = np.argmax(low, axis=1).astype(float)
        idx[~low.any(axis=1)] = np.nan
        return idx

    obs, prd = first_hit(t), first_hit(p)
    both = ~np.isnan(obs) & ~np.isnan(prd)
    lead = (obs[both] - prd[both]) * step_min
    return {
        "n_onsets_observed": int((~np.isnan(obs)).sum()),
        "n_detected": int(both.sum()),
        "recall": float(both.sum() / max((~np.isnan(obs)).sum(), 1)),
        "false_alarms": int((np.isnan(obs) & ~np.isnan(prd)).sum()),
        "lead_min_mean": float(lead.mean()) if len(lead) else float("nan"),
        "lead_min_median": float(np.median(lead)) if len(lead) else float("nan"),
        "lead_min_p25_p75": ([float(np.percentile(lead, 25)),
                              float(np.percentile(lead, 75))] if len(lead) else None),
    }
