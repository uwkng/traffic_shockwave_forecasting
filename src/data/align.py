"""
Stages 2-3 - Align every source onto ONE time axis and ONE node order.

This is where the two silent-killer bugs live, so both are asserted here rather
than trusted:

1.  **The time axis is the columns of `speed.csv`.** 52,116 steps, not 52,128 -
    2017-03-12 02:00..02:55 does not exist, because PEMS-BAY timestamps are local
    wall-clock and that is the DST spring-forward hour. Building the axis with
    `pd.date_range()` yields 12 phantom steps.
2.  **Weather is joined on the timestamp LABEL, never by position.** Open-Meteo
    returns a contiguous 4,344-row hourly series that *includes* the
    non-existent 02:00. `np.repeat(weather, 12)` gives 52,128 rows; forcing that
    onto 52,116 shifts every weather value after 2017-03-12 by one hour - and
    that covers the whole validation and test period.

Node order is taken from `adj_mx_bay.pkl` (the adjacency matrix defines it) and
every other source is reindexed onto it, with an assertion that nothing silently
reordered.

Outputs (all in `data/processed/`, all on the canonical [T] x [N] grid):

    time_index.npy      [T]      datetime64[ns], the authoritative axis
    sensor_ids.npy      [N]      canonical node order (str)
    speed.npy           [T, N]   float32, mph, NaN where missing
    occupancy.npy       [T, N]   float32, fraction, NaN where missing
    missing_mask.npy    [T, N]   bool, True = speed was missing in the source
    weather.npy         [T, 3]   float32, (precipitation, temperature, wind_speed)
    event_load.npy      [T, N]   float32, distance-decayed event pressure
    dist_to_venue.npy   [N]      float32, km to the nearest included venue
    adj_mx.npy          [N, N]   float32, thresholded Gaussian kernel (as published)
    align_meta.json              provenance + every check that was run

Usage
-----
    python -m src.data.align --config configs/default.yaml
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import pathlib
import pickle
import sys

import numpy as np
import pandas as pd
import yaml

from src import contract


# ---------------------------------------------------------------------------

def load_config(path: str) -> dict:
    with open(path, encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def haversine_km(lat1, lon1, lat2, lon2):
    """Vectorised over the second point."""
    lat1, lon1 = map(np.radians, (lat1, lon1))
    lat2, lon2 = np.radians(np.asarray(lat2)), np.radians(np.asarray(lon2))
    a = (np.sin((lat2 - lat1) / 2) ** 2
         + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2)
    return 2 * 6371.0 * np.arcsin(np.sqrt(a))


# ---------------------------------------------------------------------------
# graph / node order
# ---------------------------------------------------------------------------

def load_graph(cfg: dict) -> tuple[list[str], np.ndarray]:
    """Canonical node order + adjacency, straight from the published pickle."""
    p = pathlib.Path(cfg["data"]["pems_bay_dir"]) / cfg["data"]["adjacency_pkl"]
    with open(p, "rb") as fh:
        sensor_ids, _id_to_ind, adj = pickle.load(fh, encoding="latin1")
    sensor_ids = [str(s) for s in sensor_ids]
    adj = np.asarray(adj, dtype=np.float32)

    n = cfg["data"]["expected_nodes"]
    assert len(sensor_ids) == n, f"adjacency has {len(sensor_ids)} ids, expected {n}"
    assert adj.shape == (n, n), f"adjacency shape {adj.shape}, expected ({n},{n})"
    assert len(set(sensor_ids)) == n, "duplicate sensor ids in the adjacency pickle"
    # published PEMS-BAY kernel: self-loops = 1, weights in [0, 1]
    assert np.isfinite(adj).all(), "non-finite entries in the adjacency matrix"
    assert adj.min() >= 0.0 and adj.max() <= 1.0, "adjacency weights outside [0, 1]"
    return sensor_ids, adj


# ---------------------------------------------------------------------------
# time axis + traffic matrices
# ---------------------------------------------------------------------------

def load_traffic(cfg: dict, sensor_ids: list[str]):
    """Returns (time_index, speed[T,N], occupancy[T,N]) on the canonical order.

    The source files are stored transposed: rows are sensors, columns are
    timestamps.
    """
    d = cfg["data"]
    root = pathlib.Path(d["pems_bay_dir"])

    speed_df = pd.read_csv(root / d["speed_csv"], index_col=0)
    occ_df = pd.read_csv(root / d["occupancy_csv"], index_col=0)
    speed_df.index = speed_df.index.astype(str)
    occ_df.index = occ_df.index.astype(str)

    # --- the authoritative time axis: speed.csv's own column labels ---
    time_index = pd.to_datetime(speed_df.columns)
    n_t = len(time_index)
    assert n_t == d["expected_timesteps"], \
        f"speed.csv has {n_t} timesteps, expected {d['expected_timesteps']}"
    assert time_index.is_monotonic_increasing, "speed.csv columns are not sorted in time"
    assert time_index.is_unique, "duplicate timestamps in speed.csv"

    # --- the gap must be exactly the DST hour, nothing else ---
    step = pd.Timedelta(minutes=d["freq_min"])
    gaps = time_index.to_series().diff().dropna()
    odd = gaps[gaps != step]
    naive = pd.date_range(time_index[0], time_index[-1], freq=step)
    absent = naive.difference(time_index)
    expected_absent = pd.date_range(contract.DST_GAP[0], contract.DST_GAP[1], freq=step)
    assert list(absent) == list(expected_absent), (
        f"time axis gap is not the expected DST hour.\n"
        f"  absent: {list(absent)[:15]}\n  expected: {list(expected_absent)}")

    # --- node order: reindex onto the adjacency's order, and prove it worked ---
    missing_nodes = set(sensor_ids) - set(speed_df.index)
    assert not missing_nodes, f"speed.csv is missing sensors {sorted(missing_nodes)[:5]}"
    reordered = speed_df.index.tolist() != sensor_ids
    speed_df = speed_df.reindex(sensor_ids)
    occ_df = occ_df.reindex(sensor_ids)
    assert speed_df.index.tolist() == sensor_ids
    assert occ_df.index.tolist() == sensor_ids
    assert list(occ_df.columns) == list(speed_df.columns), \
        "speed.csv and occupancy.csv do not share a time axis"

    speed = speed_df.to_numpy(dtype=np.float32).T          # -> [T, N]
    occ = occ_df.to_numpy(dtype=np.float32).T
    return time_index, speed, occ, {
        "rows_reordered_to_match_graph": reordered,
        "odd_gaps": {str(k): str(v) for k, v in odd.items()},
    }


# ---------------------------------------------------------------------------
# weather: hourly -> 5-min, BY LABEL
# ---------------------------------------------------------------------------

def align_weather(cfg: dict, time_index: pd.DatetimeIndex):
    """Weather onto the traffic axis, joined BY LABEL. Returns (features, windows, meta).

    Since 2026-09-04 `data.weather_csv` is acquire.build_weather()'s output: ASOS
    SJC + NUQ, IDW-merged, already on a 5-minute cadence built from speed.csv's
    own column labels. So there is no resampling left to do and this function is
    a LABEL JOIN plus two checks. It used to interpolate an hourly Open-Meteo
    export; that path is gone with the file it read.

    Joining on labels rather than position is the whole point. A positional join
    would work right up until 2017-03-12 02:00 - an hour that does not exist in
    local wall-clock time but does exist in a naively generated hourly series -
    and would then shift every later timestamp by one hour, silently, for the
    rest of the period. Every val and test sample would be misaligned.

    `features` is what the model sees; `windows` is what src/eval/windows.py
    reports on. Here they are the same array, because the gauge series is
    already at the target cadence and nothing is interpolated. Kept as two
    return values because that must NOT be assumed: the moment anyone smooths
    the input, the eval windows have to keep reading the raw series, or
    interpolation will spread rain into genuinely dry timesteps and label them
    adverse.
    """
    w = pd.read_csv(cfg["data"]["weather_csv"], parse_dates=["timestamp"])
    w = w.set_index("timestamp").sort_index()
    assert w.index.is_unique, "duplicate timestamps in the weather file"

    phantom = pd.Timestamp(contract.DST_GAP[0]) in w.index
    assert not phantom, (
        f"{cfg['data']['weather_csv']} contains {contract.DST_GAP[0]}, which does "
        "not exist in local wall-clock time. It was not built from speed.csv's "
        "columns - re-run acquire.build_weather().")

    cols = ["precipitation", "temperature", "wind_speed"]
    feat = w[cols].reindex(time_index)
    if feat.isna().any().any():
        n = int(feat.isna().any(axis=1).sum())
        first = feat[feat.isna().any(axis=1)].index[0]
        raise ValueError(
            f"{n} traffic timestamps are absent from the weather file, first at "
            f"{first}. The weather file must cover the traffic axis exactly; do "
            "not interpolate here, re-run acquire.build_weather().")

    arr = feat.to_numpy(dtype=np.float32)
    naive_positional = len(w)
    return arr, arr.copy(), {
        "weather_rows": int(len(w)),
        "weather_csv": cfg["data"]["weather_csv"],
        "contains_phantom_dst_hour": bool(phantom),
        "rows_actually_produced": int(len(feat)),
        "rows_a_positional_join_would_have_produced": int(naive_positional),
        "join": "by label on speed.csv's column timestamps; no resampling",
        "source": "ASOS SJC + NUQ, IDW-merged by acquire.build_weather()",
        # NO total here, deliberately. p01i is a BACKWARD 1-hour accumulation
        # and acquire._parse_asos forward-fills it across the twelve 5-min steps
        # that follow, so every hourly reading appears twelve times and summing
        # this column returns 12x the rainfall. (It reported 5980.6 mm for the
        # period; the real figure is ~500 mm, and 2017 was one of the wettest
        # first halves on record in San Jose.) For an actual total, read the
        # p01i column of the raw per-station files, where each observation
        # appears once: SJC 597.3 mm, NUQ 630.8 mm over 2017-01-01..06-30.
        "precip_column_is_not_summable": (
            "p01i is a backward 1-hour accumulation, forward-filled to 5 min; "
            "each hourly reading repeats 12 times. Sum the raw asos_*.csv p01i "
            "column instead."),
        # Counts and comparisons ARE valid on this column: repetition changes how
        # many steps carry a value, not whether a given step is above threshold.
        "wet_steps": int((feat["precipitation"] > 0).sum()),
        "max_1h_accum_mm": round(float(feat["precipitation"].max()), 3),
        "temperature_c_range": [round(float(feat["temperature"].min()), 2),
                                round(float(feat["temperature"].max()), 2)],
    }


# ---------------------------------------------------------------------------
# NOTE. There used to be an `align_asos` here that downloaded SJC/NUQ a second
# time, built a per-node IDW precipitation field and a METAR present-weather
# ladder. Both are gone, for measured reasons:
#   * per-node IDW is worth +0.00008 incremental R2 over one global series
#     (the two stations disagree in 42.7% of wet hours, but that disagreement
#     does not carry predictive information at this scale);
#   * the present-weather ladder needs `wxcodes`, which acquire.fetch_asos does
#     not request, and all precipitation parameterisations put the
#     adverse-weather slowdown at the same -3.1 mph.
# Weather now comes from exactly one place - acquire.build_weather() - so there
# is one definition of "the weather" in the repo instead of two that drift.
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# geometry: dist_to_venue + event_load
# ---------------------------------------------------------------------------

def node_coords(cfg: dict, sensor_ids: list[str]):
    p = pathlib.Path(cfg["data"]["pems_bay_dir"]) / cfg["data"]["sensor_metadata_csv"]
    meta = pd.read_csv(p, dtype={"Sensor ID": str}).set_index("Sensor ID")
    missing = set(sensor_ids) - set(meta.index)
    assert not missing, f"sensor_metadata.csv is missing {sorted(missing)[:5]}"
    meta = meta.reindex(sensor_ids)
    return (meta["Latitude"].to_numpy(dtype=np.float64),
            meta["Longitude"].to_numpy(dtype=np.float64))


def included_venues(cfg: dict) -> pd.DataFrame:
    """The venues events.csv actually points at.

    Since 2026-09-04 venues.csv holds ONLY venues that contribute an event, so
    there is nothing left to filter; the `included` branch is kept so an older
    checkout of the file still works. The full screening record - every
    candidate venue, its distance to the nearest sensor, kept or dropped - lives
    in data/raw/events/events_meta.json under "venues".
    """
    v = pd.read_csv(cfg["data"]["venues_csv"])
    if "included" in v.columns:
        v = v[v["included"].astype(str).str.lower().isin(("yes", "true", "1"))]
    assert len(v), "no venues in venues.csv - run fetch_events.py first"
    return v.reset_index(drop=True)


def compute_dist_to_venue(cfg, lat, lon, venues) -> tuple[np.ndarray, dict]:
    """km from each node to the nearest venue that ACTUALLY HOSTS an event.

    Distance is measured only to venues with >= 1 event in the study period, not
    to every geographically-close venue. Stanford Stadium is 7.59 km from the
    network but hosts nothing between January and June (college football is an
    autumn sport). Including it would hand every western node a small
    dist_to_venue while that node never sees an event - which is exactly the
    signal channel 8 is supposed to carry.
    """
    events = pd.read_csv(cfg["data"]["events_csv"], usecols=["venue_name"])
    active = set(events["venue_name"].unique())
    used = venues[venues["venue_name"].isin(active)].reset_index(drop=True)
    idle = sorted(set(venues["venue_name"]) - active)
    assert len(used), "no included venue hosts any event - check events.csv"

    d = np.stack([haversine_km(r.venue_lat, r.venue_lon, lat, lon)
                  for r in used.itertuples()])              # [n_venues, N]
    dist = d.min(axis=0).astype(np.float32)

    # what it would have been if idle venues counted, so the choice is auditable
    d_all = np.stack([haversine_km(r.venue_lat, r.venue_lon, lat, lon)
                      for r in venues.itertuples()]).min(axis=0)
    return dist, {
        "venues_measured_to": list(used["venue_name"]),
        "venues_excluded_no_events": idle,
        "median_km": round(float(np.median(dist)), 3),
        "median_km_if_idle_venues_counted": round(float(np.median(d_all)), 3),
    }


def compute_event_load(cfg, time_index, lat, lon, venues):
    """Build BOTH event channels. They differ in exactly one term.

        event_active_decay[t, n] = sum_e active_e(t) * exp(-dist(n, venue_e)/tau)
        event_load[t, n]         = same, times 1[att_e >= MIN] * att_e / ref

    Two channels, not one, because the ablation ladder has to separate "an event
    is on" from "how big it is" - and measured, all of the signal is in the
    second. Within the egress hour, over 8,649 (event, node) pairs from 75
    events: activity alone scores t = -1.02 (not significant), while the
    thresholded attendance weight scores t = -9.76.

    The attendance floor (contract.EVENT_LOAD_MIN_ATTENDANCE) is applied to
    event_load ONLY. Putting it in event_active_decay too would leak attendance
    into the activity-only rung and collapse the contrast the ladder exists to
    show.
    """
    f = cfg["features"]["event_load"]
    tau, ref = float(f["tau_km"]), float(f["attendance_ref"])
    lead = pd.Timedelta(minutes=f["lead_min"])
    lag = pd.Timedelta(minutes=f["lag_min"])

    events = pd.read_csv(cfg["data"]["events_csv"],
                         parse_dates=["start_time", "end_time"])
    known = set(venues["venue_name"])
    dropped = sorted(set(events["venue_name"]) - known)
    events = events[events["venue_name"].isin(known)].reset_index(drop=True)

    # one decay vector per venue, reused by every event there
    decay = {r.venue_name: np.exp(-haversine_km(r.venue_lat, r.venue_lon, lat, lon)
                                  / tau).astype(np.float32)
             for r in venues.itertuples()}

    floor = float(contract.EVENT_LOAD_MIN_ATTENDANCE)
    load = np.zeros((len(time_index), len(lat)), dtype=np.float32)
    active = np.zeros((len(time_index), len(lat)), dtype=np.float32)
    used = clipped = below_floor = 0
    for e in events.itertuples():
        lo = time_index.searchsorted(e.start_time + lead, side="left")
        hi = time_index.searchsorted(e.end_time + lag, side="right")
        if hi <= lo:
            continue
        if lo == 0 or hi == len(time_index):
            clipped += 1
        d = decay[e.venue_name]
        active[lo:hi] += d                       # NO attendance term, no floor
        att = float(e.expected_attendance)
        if att >= floor:
            load[lo:hi] += (att / ref) * d
        else:
            below_floor += 1
        used += 1

    nz = load > 0
    return load, active, {
        "attendance_floor": floor,
        "events_below_floor": int(below_floor),
        "timesteps_with_any_activity": int((active > 0).any(axis=1).sum()),
        "events_in_file": int(len(events) + len(dropped)),
        "events_used": int(used),
        "venues_dropped_from_events_csv": dropped,
        "events_clipped_at_period_edge": int(clipped),
        "tau_km": tau, "attendance_ref": ref,
        "active_span_min": [f["lead_min"], f["lag_min"]],
        "timesteps_with_any_load": int(nz.any(axis=1).sum()),
        "pct_timesteps_with_load": round(100 * float(nz.any(axis=1).mean()), 3),
        "max_load": round(float(load.max()), 4),
        "mean_load_where_active": round(float(load[nz].mean()), 4) if nz.any() else 0.0,
    }


# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/default.yaml")
    args = ap.parse_args()
    cfg = load_config(args.config)
    out = pathlib.Path(cfg["data"]["processed_dir"])
    out.mkdir(parents=True, exist_ok=True)

    print("=== Stages 2-3: align ===\n")

    print("[1/6] graph + canonical node order")
    sensor_ids, adj = load_graph(cfg)
    deg = (adj > 0).sum(axis=1)
    print(f"  {len(sensor_ids)} nodes, adjacency {adj.shape}, "
          f"mean degree {deg.mean():.1f} (min {deg.min()}, max {deg.max()})")

    print("\n[2/6] traffic matrices on speed.csv's own time axis")
    time_index, speed, occ, tmeta = load_traffic(cfg, sensor_ids)
    print(f"  time axis  : {len(time_index)} steps, {time_index[0]} -> {time_index[-1]}")
    print(f"  DST gap    : verified exactly {contract.DST_GAP[0]}..{contract.DST_GAP[1]}")
    print(f"  reordered speed rows to match the graph: {tmeta['rows_reordered_to_match_graph']}")
    missing_mask = np.isnan(speed)
    print(f"  speed      : {speed.shape}  missing {missing_mask.sum():,} "
          f"({100 * missing_mask.mean():.4f}%)")
    occ_nan = np.isnan(occ)
    print(f"  occupancy  : {occ.shape}  missing {occ_nan.sum():,} "
          f"({100 * occ_nan.mean():.4f}%)")

    print("\n[3/6] weather -> traffic axis, joined ON LABEL")
    weather, weather_windows, wmeta = align_weather(cfg, time_index)
    print(f"  {wmeta['weather_rows']:,} rows -> {wmeta['rows_actually_produced']:,} steps")
    print(f"  contains the phantom {contract.DST_GAP[0]}: "
          f"{wmeta['contains_phantom_dst_hour']}  "
          f"(a positional join would have made "
          f"{wmeta['rows_a_positional_join_would_have_produced']} rows -> silent 1-h shift)")
    print(f"  source        : {wmeta['source']}")
    print(f"  join          : {wmeta['join']}")
    print(f"  precipitation : {wmeta['wet_steps']:,} wet steps, "
          f"max 1-h accumulation {wmeta['max_1h_accum_mm']} mm")
    print(f"                  (no total: the column is a forward-filled 1-h "
          f"accumulation, so it is not summable)")
    print(f"  temperature   : {wmeta['temperature_c_range'][0]} .. "
          f"{wmeta['temperature_c_range'][1]} degC")

    print("\n[4/6] node coordinates + venue distances")
    lat, lon = node_coords(cfg, sensor_ids)
    venues = included_venues(cfg)
    dist, dmeta = compute_dist_to_venue(cfg, lat, lon, venues)
    print(f"  venues in range   : {list(venues['venue_name'])}")
    print(f"  measured distance to: {dmeta['venues_measured_to']}")
    if dmeta["venues_excluded_no_events"]:
        print(f"  excluded (0 events): {dmeta['venues_excluded_no_events']}"
              f"  [median would have been {dmeta['median_km_if_idle_venues_counted']} km]")
    print(f"  dist_to_venue  : min {dist.min():.2f} km, median {np.median(dist):.2f} km, "
          f"max {dist.max():.2f} km")
    for lim in (2, 5, 10):
        print(f"    nodes within {lim:2d} km of a venue: {(dist <= lim).sum():3d}")

    print("\n[5/6] event channels (event_active_decay + event_load)")
    event_load, event_active, emeta = compute_event_load(
        cfg, time_index, lat, lon, venues)
    print(f"  events used  : {emeta['events_used']} / {emeta['events_in_file']}"
          + (f"  (dropped venues: {emeta['venues_dropped_from_events_csv']})"
             if emeta["venues_dropped_from_events_csv"] else ""))
    print(f"  active span  : start{emeta['active_span_min'][0]:+d} min "
          f"-> end{emeta['active_span_min'][1]:+d} min")
    print(f"  timesteps with any load: {emeta['timesteps_with_any_load']:,} "
          f"({emeta['pct_timesteps_with_load']}%)")
    print(f"  max {emeta['max_load']:.3f}, mean where active "
          f"{emeta['mean_load_where_active']:.3f}")

    print("\n[6/6] writing")
    np.save(out / "time_index.npy", time_index.to_numpy())
    np.save(out / "sensor_ids.npy", np.array(sensor_ids))
    np.save(out / "speed.npy", speed)
    np.save(out / "occupancy.npy", occ)
    np.save(out / "missing_mask.npy", missing_mask)
    np.save(out / "weather.npy", weather)
    np.save(out / "weather_windows.npy", weather_windows)
    np.save(out / "event_load.npy", event_load)
    np.save(out / "event_active_decay.npy", event_active)
    np.save(out / "dist_to_venue.npy", dist)
    np.save(out / "adj_mx.npy", adj)

    meta = {
        "generated": dt.datetime.now().isoformat(timespec="seconds"),
        "config": args.config,
        "n_timesteps": int(len(time_index)),
        "n_nodes": len(sensor_ids),
        "time_axis_source": contract.TIME_AXIS_SOURCE,
        "time_first": str(time_index[0]), "time_last": str(time_index[-1]),
        "traffic": tmeta,
        "weather": wmeta,
        "event_load": emeta,
        "speed_missing_pct": round(100 * float(missing_mask.mean()), 5),
        "occupancy_missing_pct": round(100 * float(np.isnan(occ).mean()), 5),
        "venues_in_range": list(venues["venue_name"]),
        "dist_to_venue": dmeta | {"min": round(float(dist.min()), 3),
                                  "max": round(float(dist.max()), 3)},
        "adjacency_mean_degree": round(float(deg.mean()), 2),
    }
    (out / "align_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    total = sum(f.stat().st_size for f in out.glob("*.npy"))
    print(f"  {len(list(out.glob('*.npy')))} arrays -> {out}  ({total / 1e6:.0f} MB)")
    print(f"  provenance -> {out / 'align_meta.json'}")
    print("\nDone.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
