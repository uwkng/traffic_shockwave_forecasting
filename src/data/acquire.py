"""
Stage 1 — Acquire raw data sources.

Three sources:
  1. PEMS-BAY (speed, occupancy, sensor graph)  – git clone
  2. Weather (precip, temp, wind)                – ASOS ground stations (SJC + NUQ)
     via Iowa Environmental Mesonet, IDW-merged; Open-Meteo as cross-validation
  3. Events  (scheduled events with attendance)  – curated CSV (checked in)
"""

import json
import math
import pathlib
import subprocess
import urllib.request

import pandas as pd

RAW = pathlib.Path("data/raw")
PEMS_DIR = RAW / "augmented-pems-bay"
WEATHER_DIR = RAW / "weather"
EVENTS_DIR = RAW / "events"


# ---------------------------------------------------------------------------
# 1. PEMS-BAY
# ---------------------------------------------------------------------------

def fetch_pems_bay() -> pathlib.Path:
    """Clone the Augmented-PEMS-BAY repo (shallow, ~200 MB)."""
    if PEMS_DIR.exists():
        return PEMS_DIR
    subprocess.run(
        ["git", "clone", "--depth", "1",
         "https://github.com/george-j-ste/Augmented-PEMS-BAY.git",
         str(PEMS_DIR)],
        check=True,
    )
    return PEMS_DIR


# ---------------------------------------------------------------------------
# 2. Weather — ASOS ground stations via Iowa Environmental Mesonet (primary)
# ---------------------------------------------------------------------------
#
# Two ASOS stations bracket the PEMS-BAY sensor network:
#   SJC (San Jose Intl Airport)   — ~2 km from centroid, dominates IDW
#   NUQ (Moffett Federal Airfield) — ~12 km from centroid, north edge
#
# Pipeline: fetch_asos() -> raw CSV per station
#           build_weather() -> IDW merge -> weather_hourly.csv

ASOS_STATIONS = {
    "SJC": {"lat": 37.3626, "lon": -121.9291},
    "NUQ": {"lat": 37.4161, "lon": -122.0494},
}


def _sensor_centroid() -> tuple[float, float]:
    """Return (lat, lon) centroid of the PEMS-BAY sensor network."""
    meta = pd.read_csv(
        PEMS_DIR / "data" / "sensor_graph" / "sensor_metadata.csv",
    )
    return float(meta["Latitude"].mean()), float(meta["Longitude"].mean())


def fetch_asos(station: str) -> pathlib.Path:
    """Download raw ASOS/METAR observations from Iowa Environmental Mesonet.

    Saves one CSV per station with columns: station, valid, tmpf, p01i, sknt.
    Units at this stage are still imperial (°F, inches, knots).
    """
    out_path = WEATHER_DIR / f"asos_{station}.csv"
    if out_path.exists():
        return out_path

    WEATHER_DIR.mkdir(parents=True, exist_ok=True)
    url = (
        "https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py?"
        f"station={station}"
        "&data=tmpf&data=p01i&data=sknt"
        "&tz=America%2FLos_Angeles"
        "&format=onlycomma"
        "&latlon=no&elev=no"
        "&missing=M&trace=T"
        "&direct=no"
        "&report_type=3&report_type=4"
        "&year1=2017&month1=1&day1=1&hour1=0"
        "&year2=2017&month2=7&day2=1&hour2=0"
    )
    with urllib.request.urlopen(url, timeout=120) as resp:
        raw = resp.read().decode("utf-8")

    if len(raw.strip().splitlines()) < 2:
        raise RuntimeError(f"ASOS download for {station} returned no data")

    out_path.write_text(raw, encoding="utf-8")
    n_obs = len(raw.strip().splitlines()) - 1
    print(f"  ASOS {station} saved -> {out_path}  ({n_obs} observations)")
    return out_path


def _parse_asos(path: pathlib.Path) -> pd.DataFrame:
    """Parse raw ASOS CSV into 5-minute metric-unit observations.

    Timestamps are rounded to the nearest 5 minutes, then linearly
    interpolated across the full PEMS-BAY time grid (52 116 slots).
    Precipitation is forward-filled instead of interpolated.
    """
    df = pd.read_csv(path, comment="#", na_values=["M", "  M"])
    df["p01i"] = df["p01i"].replace("T", 0.005)
    df["p01i"] = pd.to_numeric(df["p01i"], errors="coerce")
    df["valid"] = pd.to_datetime(df["valid"])
    df = df.set_index("valid").sort_index()

    out = pd.DataFrame(index=df.index)
    out["temperature"] = (df["tmpf"] - 32) * 5.0 / 9.0
    out["wind_speed"] = df["sknt"] * 1.852
    out["precipitation"] = df["p01i"] * 25.4

    # Round to nearest 5 minutes; average if multiple obs land on same slot
    out.index = out.index.round("5min")
    out = out.groupby(out.index).mean()

    # Reindex to the actual PEMS-BAY time axis (skips DST gap Mar 12 02:00)
    speed_cols = pd.read_csv(
        PEMS_DIR / "data" / "traffic_data" / "speed.csv", nrows=0,
    ).columns[1:]
    idx = pd.to_datetime(speed_cols)
    out = out.reindex(idx)

    out["temperature"] = out["temperature"].interpolate(method="time").bfill()
    out["wind_speed"] = out["wind_speed"].interpolate(method="time").bfill()
    out["precipitation"] = out["precipitation"].ffill().fillna(0.0)

    return out


def build_weather() -> pathlib.Path:
    """Merge ASOS stations into one 5-minute CSV via inverse-distance weighting.

    Each station is independently parsed to the 5-minute grid (round, then
    interpolate) before merging, so observations at different times in the
    two stations both contribute temporal detail.

    Temperature & wind: IDW average (falls back to single station on gaps).
    Precipitation: max across stations (avoids washing out local events).
    """
    out_path = WEATHER_DIR / "weather_5min.csv"
    if out_path.exists():
        return out_path

    centroid = _sensor_centroid()

    parsed: dict[str, pd.DataFrame] = {}
    weights: dict[str, float] = {}
    for station, info in ASOS_STATIONS.items():
        raw_path = WEATHER_DIR / f"asos_{station}.csv"
        if not raw_path.exists():
            raise FileNotFoundError(f"Run fetch_asos('{station}') first")
        parsed[station] = _parse_asos(raw_path)
        dlat = (info["lat"] - centroid[0]) * 111.0
        dlon = ((info["lon"] - centroid[1]) * 111.0
                * math.cos(math.radians(centroid[0])))
        weights[station] = 1.0 / math.sqrt(dlat**2 + dlon**2)

    w_total = sum(weights.values())
    weights = {k: v / w_total for k, v in weights.items()}
    print(f"  IDW weights: "
          f"{', '.join(f'{k}={v:.3f}' for k, v in weights.items())}")

    stations = list(ASOS_STATIONS)
    idx = parsed[stations[0]].index
    result = pd.DataFrame(index=idx)

    for col in ("temperature", "wind_speed"):
        vals = pd.DataFrame({s: parsed[s][col] for s in stations})
        w = pd.DataFrame(
            {s: vals[s].notna().astype(float) * weights[s] for s in stations},
        )
        result[col] = ((vals.fillna(0) * w).sum(axis=1)
                       / w.sum(axis=1).replace(0, float("nan")))

    precip = pd.DataFrame({s: parsed[s]["precipitation"] for s in stations})
    result["precipitation"] = precip.max(axis=1).fillna(0.0)

    result.index.name = "timestamp"
    result = result.reset_index()
    result = result[["timestamp", "temperature", "precipitation", "wind_speed"]]
    result.to_csv(out_path, index=False)
    print(f"  Weather merged -> {out_path}  ({len(result)} rows, 5-min)")
    return out_path


# ---------------------------------------------------------------------------
# 2b. Weather — Open-Meteo reanalysis (cross-validation / fallback)
# ---------------------------------------------------------------------------

def fetch_weather_openmeteo() -> pathlib.Path:
    """Download Open-Meteo ERA5-Land reanalysis for the sensor centroid.

    Saved separately so the two sources can be compared side-by-side.
    """
    out_path = WEATHER_DIR / "weather_openmeteo.csv"
    if out_path.exists():
        return out_path

    WEATHER_DIR.mkdir(parents=True, exist_ok=True)
    lat, lon = _sensor_centroid()
    url = (
        "https://archive-api.open-meteo.com/v1/archive?"
        f"latitude={lat:.4f}&longitude={lon:.4f}"
        "&start_date=2017-01-01&end_date=2017-06-30"
        "&hourly=temperature_2m,precipitation,wind_speed_10m"
        "&timezone=America/Los_Angeles"
    )
    with urllib.request.urlopen(url, timeout=60) as resp:
        data = json.loads(resp.read())

    h = data["hourly"]
    df = pd.DataFrame({
        "timestamp": pd.to_datetime(h["time"]),
        "temperature": h["temperature_2m"],
        "precipitation": h["precipitation"],
        "wind_speed": h["wind_speed_10m"],
    })
    df.to_csv(out_path, index=False)
    print(f"  Open-Meteo saved -> {out_path}  ({len(df)} hourly rows)")
    return out_path


# ---------------------------------------------------------------------------
# 3. Events (curated CSV, checked into the repo)
# ---------------------------------------------------------------------------
# 2c. Weather - ECMWF IFS short-range FORECAST (Open-Meteo Historical Forecast)
# ---------------------------------------------------------------------------
# This is not the same thing as 2b, and the difference is the whole point.
#
# ASOS (2a) and ERA5-Land (2b) both say what the weather WAS. Fed as history
# they tell the model what already happened, which is exactly the information
# the speed history already contains: measured on this dataset, rain does not
# change how often breakdowns occur (91.8% of commute-peak timesteps in rain
# against 93.1% dry) nor how large they are - it is a uniform -3.04 mph
# capacity reduction, and a uniform slowdown is trivially readable from an hour
# of speeds. That is why the observed weather channels measure as worthless.
#
# A forecast is different information. Of the samples whose TARGET window
# contains adverse weather, the fraction whose INPUT window is completely dry -
# the rain starts inside the horizon, so a contemporaneous channel is blind to
# it - is 8.2% at 15 min, 22.4% at 60 min and 41.9% at 3 h.
#
# The Historical Forecast API stitches together the FIRST hours of each
# successive model run, so each timestamp carries the prediction with the
# shortest available lead time (~0-12 h for ECMWF IFS, which runs every 6 h).
# Every value therefore uses only information available before its own
# timestamp: this is a forecast archive, NOT a reanalysis, and feeding it into
# the future portion of the input is not leakage.
#
# Single point, like 2b. The network spans 19.7 x 21.1 km and the IFS grid is
# 9-11 km, so querying per sensor would return ~4 distinct values, not 325.

def fetch_weather_forecast() -> pathlib.Path:
    """Download the ECMWF IFS short-range forecast archive for the centroid."""
    out_path = WEATHER_DIR / "weather_forecast_hourly.csv"
    if out_path.exists():
        print(f"  forecast already present -> {out_path}")
        return out_path

    WEATHER_DIR.mkdir(parents=True, exist_ok=True)
    lat, lon = _sensor_centroid()
    url = (
        "https://historical-forecast-api.open-meteo.com/v1/forecast?"
        f"latitude={lat:.4f}&longitude={lon:.4f}"
        "&start_date=2017-01-01&end_date=2017-06-30"
        "&hourly=temperature_2m,precipitation,wind_speed_10m"
        "&models=ecmwf_ifs"
        "&timezone=America/Los_Angeles"
    )
    with urllib.request.urlopen(url, timeout=120) as resp:
        data = json.loads(resp.read())

    h = data["hourly"]
    df = pd.DataFrame({
        "timestamp": pd.to_datetime(h["time"]),
        "temperature": h["temperature_2m"],
        "precipitation": h["precipitation"],
        "wind_speed": h["wind_speed_10m"],
    })
    n_null = int(df[["temperature", "precipitation", "wind_speed"]].isna().sum().sum())
    assert len(df) > 4000, f"forecast archive returned only {len(df)} rows"
    df.to_csv(out_path, index=False)
    print(f"  ECMWF IFS forecast -> {out_path}  ({len(df)} hourly rows, "
          f"{n_null} nulls, {(df.precipitation > 0).sum()} wet hours)")
    return out_path


# ---------------------------------------------------------------------------

EVENTS_COLUMNS = [
    "start_time",           # ISO datetime, local tz (America/Los_Angeles)
    "end_time",             # ISO datetime
    "venue_name",           # human-readable label
    "venue_lat",            # WGS-84
    "venue_lon",
    "expected_attendance",  # integer — used for event_magnitude channel
    "event_type",           # e.g. NBA, NHL, MLB, MLS, concert
]


def check_events() -> pathlib.Path:
    """Validate that the curated events CSV exists and has the right schema."""
    events_path = EVENTS_DIR / "events.csv"
    if not events_path.exists():
        raise FileNotFoundError(
            f"Events file not found at {events_path}.\n"
            "Create it with columns: " + ", ".join(EVENTS_COLUMNS)
        )
    df = pd.read_csv(events_path)
    missing = set(EVENTS_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"Events CSV missing columns: {missing}")
    print(f"Events loaded -> {events_path}  ({len(df)} events)")
    return events_path


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=== Stage 1: Acquire raw data ===\n")

    print("[1/3] PEMS-BAY ...")
    pems = fetch_pems_bay()
    speed = pd.read_csv(pems / "data" / "traffic_data" / "speed.csv")
    print(f"  speed.csv: {speed.shape[0]} sensors × {speed.shape[1] - 1} timesteps")

    print("\n[2/3] Weather (ASOS ground stations) ...")
    for station in ASOS_STATIONS:
        fetch_asos(station)
    build_weather()
    print("  (run fetch_weather_openmeteo() for cross-validation)")
    print("\n[2b/3] Weather (ECMWF IFS forecast archive) ...")
    fetch_weather_forecast()

    print("\n[3/3] Events ...")
    try:
        check_events()
    except FileNotFoundError as e:
        print(f"  WARNING: {e}")
        print("  (this is expected until the events CSV is created)")

    print("\nDone.")
