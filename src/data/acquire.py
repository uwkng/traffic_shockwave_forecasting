"""
Stage 1 — Acquire raw data sources.

Three sources:
  1. PEMS-BAY (speed, occupancy, sensor graph)  – git clone
  2. Weather (precip, temp, wind)                – Open-Meteo Historical API
  3. Events  (scheduled events with attendance)  – curated CSV (checked in)
"""

import json
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
# 2. Weather (Open-Meteo Historical — free, no API key)
# ---------------------------------------------------------------------------

def _sensor_centroid() -> tuple[float, float]:
    """Return (lat, lon) centroid of the PEMS-BAY sensor network."""
    meta = pd.read_csv(
        PEMS_DIR / "data" / "sensor_graph" / "sensor_metadata.csv",
    )
    return float(meta["Latitude"].mean()), float(meta["Longitude"].mean())


def fetch_weather() -> pathlib.Path:
    """Download hourly weather for the sensor-network centroid, Jan–Jun 2017.

    Variables: temperature_2m (°C), precipitation (mm), wind_speed_10m (km/h).
    Saved as CSV with a ``timestamp`` column in America/Los_Angeles local time
    (matching PEMS-BAY timestamps).
    """
    out_path = WEATHER_DIR / "weather_hourly.csv"
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

    hourly = data["hourly"]
    df = pd.DataFrame({
        "timestamp": pd.to_datetime(hourly["time"]),
        "temperature": hourly["temperature_2m"],
        "precipitation": hourly["precipitation"],
        "wind_speed": hourly["wind_speed_10m"],
    })
    df.to_csv(out_path, index=False)
    print(f"Weather saved → {out_path}  ({len(df)} hourly rows)")
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
    print(f"Events loaded → {events_path}  ({len(df)} events)")
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

    print("\n[2/3] Weather ...")
    fetch_weather()

    print("\n[3/3] Events ...")
    try:
        check_events()
    except FileNotFoundError as e:
        print(f"  ⚠ {e}")
        print("  (this is expected until the events CSV is created)")

    print("\nDone.")
