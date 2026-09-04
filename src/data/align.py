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


MIN_CONFIDENCE_RANK = {"low": 0, "medium": 1, "high": 2}


def qualifying_events(cfg, events: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Events whose provenance is good enough to weight a channel with.

    events.csv carries a `confidence` column and, decisively, an
    `attendance_kind` column. Only the 37 ESPN-sourced rows carry an ANNOUNCED
    attendance; the other 58 carry `type_default_estimate`, a per-event-type
    constant we assigned ourselves. Filtering those rows by an attendance
    threshold therefore filters our own defaults by construction - the AHL
    fixtures that dominate the medium tier draw a few thousand people while
    being stamped with a >=15,000 default.

    Measured on the egress hour, matched against the same weekday and clock on
    non-event days +-7/14 days, sensors within 5 km:

        confidence   n   speed anomaly        shockwave rate vs control
        high        37   -0.63 +- 0.26 mph    14/36 vs  4/54 = 5.25x, z=3.66
        medium      19   +0.18 +- 0.11 mph     0/19 vs  5/39 = 0.00x, z=-1.63
        low         11   -0.06 +- 0.12 mph     1/11 vs  1/11 = 1.00x, z=0.00
        all         67   -0.31 +- 0.15 mph    15/66 vs 10/104 = 2.36x, z=2.35

    Dropping 30 of 67 fixtures RAISES the odds ratio from 2.36x to 5.25x and z
    from 2.35 to 3.66. The excluded rows are noise, not signal.

    NOTE what "high" does and does not cover. `end_time_is_estimated` is 1 for
    all 95 rows: duration_source is `nominal_by_type` for every event, so the
    egress window is an estimate even here. High confidence means the date,
    start time and attendance are sourced, not that the end time is.
    """
    floor = MIN_CONFIDENCE_RANK[str(cfg["features"]["event_load"]
                                    .get("min_confidence", "high")).lower()]
    if "confidence" not in events.columns:
        return events, {"min_confidence": "n/a - column absent", "kept": len(events)}
    rank = events["confidence"].str.lower().map(MIN_CONFIDENCE_RANK)
    assert rank.notna().all(), (
        f"unknown confidence values: "
        f"{sorted(set(events['confidence']) - set(MIN_CONFIDENCE_RANK))}")
    keep = rank >= floor
    return events[keep].reset_index(drop=True), {
        "min_confidence": [k for k, v in MIN_CONFIDENCE_RANK.items() if v == floor][0],
        "kept": int(keep.sum()),
        "dropped": int((~keep).sum()),
        "kept_by_tier": events.loc[keep, "confidence"].value_counts().to_dict(),
        "dropped_by_tier": events.loc[~keep, "confidence"].value_counts().to_dict(),
    }


def road_distance_km(cfg, sensor_ids, lat, lon, venues,
                     anchor_radius_km: float = 1.5) -> tuple[dict, dict]:
    """Directed road-network km from each venue to each sensor.

    Great-circle distance has a specific, measurable defect on this network, not
    a vague one. 128 sensor pairs lie within 150 m of each other on OPPOSITE
    carriageways of the same freeway. Their great-circle distances to a venue
    differ by a median of 22 m - they are treated as the same place - while
    their speeds correlate at a median of 0.115 (same-direction pairs up to
    1 km apart correlate at 0.850). Egress traffic uses one carriageway; the
    other gets an identical proximity signal and contributes only noise.

    `distances_bay_2017.csv` ships with the augmented release and holds 8,358
    DIRECTED sensor-to-sensor road distances in metres - the same file the
    published adjacency was built from. Shortest paths over it, followed in the
    direction of travel, separate the two carriageways properly.

    A venue is not a sensor, so each venue seeds every sensor within
    `anchor_radius_km` great-circle (falling back to the single nearest) and the
    hop from venue to seed is added to the path. Multi-source matters: crowds
    leaving SAP Center take both 87N and 87S, and a single nearest-sensor anchor
    (87S) would bias the whole field southbound.

    Sensors the egress flow cannot reach are left as `inf`. That is the correct
    semantics - `exp(-inf/tau) = 0` switches the event channels off there - and
    the caller caps the value before writing the `dist_to_venue` channel.

    Measured effect of the switch, high-confidence fixtures, egress hour:

        near-sensor set      n     speed anomaly        shockwave vs control
        great-circle <=5km  177   -0.41 +- 0.23 (t=-1.75)  2.50x, z=3.24
        network      <=5km   54   -0.89 +- 0.26 (t=-3.37)  5.25x, z=3.66

    The 123 sensors great-circle adds at 5 km carry no egress traffic and halve
    the measured effect; at that radius the great-circle result is not
    significant and the network one is.
    """
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import dijkstra

    p = pathlib.Path(cfg["data"]["pems_bay_dir"]) / cfg["data"]["distances_csv"]
    d = pd.read_csv(p, dtype={"from": str, "to": str})
    pos = {s: i for i, s in enumerate(sensor_ids)}
    n = len(sensor_ids)
    keep = d["from"].isin(pos) & d["to"].isin(pos)
    dropped_edges = int((~keep).sum())
    d = d[keep]
    w = np.maximum(d["distance"].to_numpy(dtype=np.float64), 1e-6)   # metres
    graph = csr_matrix((w, (d["from"].map(pos).to_numpy(),
                            d["to"].map(pos).to_numpy())), shape=(n, n))

    per_venue, seeds = {}, {}
    for r in venues.itertuples():
        gc = haversine_km(r.venue_lat, r.venue_lon, lat, lon)
        src = np.flatnonzero(gc <= anchor_radius_km)
        if src.size == 0:
            src = np.array([int(gc.argmin())])
        sp = dijkstra(graph, directed=True, indices=src) / 1000.0
        per_venue[r.venue_name] = (sp + gc[src][:, None]).min(axis=0)
        seeds[r.venue_name] = {
            "n_seed_sensors": int(src.size),
            "nearest_hop_km": round(float(gc[src].min()), 3),
            "reachable": int(np.isfinite(per_venue[r.venue_name]).sum()),
        }
    return per_venue, {
        "distances_csv": str(p),
        "edges_used": int(len(d)),
        "edges_dropped_not_in_node_set": dropped_edges,
        "anchor_radius_km": anchor_radius_km,
        "per_venue": seeds,
    }


def venue_distances(cfg, sensor_ids, lat, lon, venues) -> tuple[dict, dict, dict]:
    """Both distance fields, so the choice stays auditable. Returns
    (chosen, great_circle, meta) where each field maps venue name -> [N] km."""
    mode = str(cfg["features"]["event_load"].get("distance", "haversine")).lower()
    gc = {r.venue_name: haversine_km(r.venue_lat, r.venue_lon, lat, lon)
          for r in venues.itertuples()}
    meta = {"mode": mode}
    if mode in ("road", "network"):
        net, nmeta = road_distance_km(cfg, sensor_ids, lat, lon, venues)
        meta["network"] = nmeta
        return net, gc, meta
    assert mode == "haversine", f"unknown features.event_load.distance: {mode!r}"
    return gc, gc, meta


def compute_dist_to_venue(cfg, dist_by_venue, gc_by_venue, venues,
                          active_venues) -> tuple[np.ndarray, dict]:
    """km from each node to the nearest venue that ACTUALLY HOSTS an event.

    Distance is measured only to venues with >= 1 QUALIFYING event, not to every
    geographically-close venue. Stanford Stadium is 7.59 km from the network but
    hosts nothing between January and June (college football is an autumn
    sport). Including it would hand every western node a small dist_to_venue
    while that node never sees an event - which is exactly the signal channel 8
    is supposed to carry.

    Since 2026-09-05 "qualifying" means the confidence filter too, so the same
    venue set drives this channel, the event channels and the eval windows.
    Shoreline Amphitheatre is the case that moves: it is nearest for 46 of 325
    sensors, and all 9 of its in-period shows are low-confidence Wayback rows
    whose measured effect is 1.00x - exactly the failure mode the paragraph
    above describes, with a different venue in the role.

    Unreachable nodes (road mode only) are capped at the largest finite distance
    so the CHANNEL stays finite; the event decays keep the raw inf and go to 0.
    """
    used = venues[venues["venue_name"].isin(active_venues)].reset_index(drop=True)
    idle = sorted(set(venues["venue_name"]) - set(active_venues))
    assert len(used), "no included venue hosts a qualifying event - check events.csv"

    stack = np.stack([dist_by_venue[v] for v in used["venue_name"]])
    dist = stack.min(axis=0)
    unreachable = int((~np.isfinite(dist)).sum())
    if unreachable:
        cap = float(dist[np.isfinite(dist)].max())
        dist = np.where(np.isfinite(dist), dist, cap)
    else:
        cap = None
    dist = dist.astype(np.float32)

    gc_used = np.stack([gc_by_venue[v] for v in used["venue_name"]]).min(axis=0)
    gc_all = np.stack([gc_by_venue[v] for v in venues["venue_name"]]).min(axis=0)
    return dist, {
        "venues_measured_to": list(used["venue_name"]),
        "venues_excluded_no_qualifying_events": idle,
        "median_km": round(float(np.median(dist)), 3),
        "median_km_great_circle": round(float(np.median(gc_used)), 3),
        "median_km_if_idle_venues_counted": round(float(np.median(gc_all)), 3),
        "nodes_unreachable_capped": unreachable,
        "cap_km": None if cap is None else round(cap, 3),
        "nodes_within_2km": int((dist <= 2).sum()),
        "nodes_within_5km": int((dist <= 5).sum()),
        "nodes_within_2km_great_circle": int((gc_used <= 2).sum()),
        "nodes_within_5km_great_circle": int((gc_used <= 5).sum()),
    }


def compute_event_load(cfg, time_index, dist_by_venue, venues, n_nodes):
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

    `dist_by_venue` comes from venue_distances() and may be great-circle or
    road-network km. Unreachable nodes arrive as inf and exp(-inf/tau) = 0
    switches both channels off there, which is the intended semantics: egress
    traffic never passes a sensor it cannot reach.
    """
    f = cfg["features"]["event_load"]
    tau, ref = float(f["tau_km"]), float(f["attendance_ref"])
    lead = pd.Timedelta(minutes=f["lead_min"])
    lag = pd.Timedelta(minutes=f["lag_min"])

    events = pd.read_csv(cfg["data"]["events_csv"],
                         parse_dates=["start_time", "end_time"])
    events, conf_meta = qualifying_events(cfg, events)
    known = set(venues["venue_name"])
    dropped = sorted(set(events["venue_name"]) - known)
    events = events[events["venue_name"].isin(known)].reset_index(drop=True)

    # one decay vector per venue, reused by every event there. inf -> 0.
    with np.errstate(over="ignore"):
        decay = {v: np.exp(-dist_by_venue[v] / tau).astype(np.float32)
                 for v in venues["venue_name"]}
    for v, dv in decay.items():
        assert np.isfinite(dv).all(), f"non-finite decay for {v}"

    floor = float(contract.EVENT_LOAD_MIN_ATTENDANCE)
    load = np.zeros((len(time_index), n_nodes), dtype=np.float32)
    active = np.zeros((len(time_index), n_nodes), dtype=np.float32)
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
        "confidence_filter": conf_meta,
        "attendance_floor": floor,
        "events_below_floor": int(below_floor),
        "timesteps_with_any_activity": int((active > 0).any(axis=1).sum()),
        "events_used": int(used),
        "venues_dropped_from_events_csv": dropped,
        "events_clipped_at_period_edge": int(clipped),
        "tau_km": tau, "attendance_ref": ref,
        "distance_mode": str(f.get("distance", "haversine")),
        "active_span_min": [f["lead_min"], f["lag_min"]],
        "timesteps_with_any_load": int(nz.any(axis=1).sum()),
        "pct_timesteps_with_load": round(100 * float(nz.any(axis=1).mean()), 3),
        "max_load": round(float(load.max()), 4),
        "mean_load_where_active": round(float(load[nz].mean()), 4) if nz.any() else 0.0,
        "nodes_with_zero_decay": {v: int((dv == 0).sum()) for v, dv in decay.items()},
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
    dist_by_venue, gc_by_venue, distmeta = venue_distances(
        cfg, sensor_ids, lat, lon, venues)
    print(f"  distance mode : {distmeta['mode']}")
    if "network" in distmeta:
        for v, sm in distmeta["network"]["per_venue"].items():
            print(f"    {v:<24s} {sm['n_seed_sensors']:2d} seed sensors, "
                  f"nearest hop {sm['nearest_hop_km']:.2f} km, "
                  f"reaches {sm['reachable']}/{len(sensor_ids)}")
    ev_all = pd.read_csv(cfg["data"]["events_csv"], usecols=["venue_name", "confidence"])
    ev_ok, conf_meta = qualifying_events(cfg, ev_all)
    print(f"  confidence    : keep >= {conf_meta['min_confidence']} "
          f"-> {conf_meta['kept']} of {len(ev_all)} events "
          f"({conf_meta.get('kept_by_tier', {})})")
    dist, dmeta = compute_dist_to_venue(
        cfg, dist_by_venue, gc_by_venue, venues, set(ev_ok["venue_name"].unique()))
    print(f"  venues in range   : {list(venues['venue_name'])}")
    print(f"  measured distance to: {dmeta['venues_measured_to']}")
    if dmeta["venues_excluded_no_qualifying_events"]:
        print(f"  excluded (0 qualifying events): "
              f"{dmeta['venues_excluded_no_qualifying_events']}")
    if dmeta["nodes_unreachable_capped"]:
        print(f"  unreachable by road: {dmeta['nodes_unreachable_capped']} nodes, "
              f"capped at {dmeta['cap_km']} km (their event decay is 0)")
    print(f"  dist_to_venue  : min {dist.min():.2f} km, median {np.median(dist):.2f} km, "
          f"max {dist.max():.2f} km")
    for lim in (2, 5, 10):
        print(f"    nodes within {lim:2d} km of a venue: {(dist <= lim).sum():3d}"
              f"   (great-circle would give "
              f"{dmeta['nodes_within_2km_great_circle'] if lim == 2 else dmeta['nodes_within_5km_great_circle'] if lim == 5 else '-'})")

    print("\n[5/6] event channels (event_active_decay + event_load)")
    event_load, event_active, emeta = compute_event_load(
        cfg, time_index, dist_by_venue, venues, len(sensor_ids))
    print(f"  events used  : {emeta['events_used']}"
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
        "venue_distance": distmeta,
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
