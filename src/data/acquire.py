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

    print("\n[3/3] Events ...")
    try:
        check_events()
    except FileNotFoundError as e:
        print(f"  WARNING: {e}")
        print("  (this is expected until the events CSV is created)")

    print("\nDone.")
