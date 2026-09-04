"""
Stage 1b - Collect scheduled events, reproducibly.

Why this file exists
--------------------
The previous ``events.csv`` was a hand-made file with no fetch script. Its
CONTENTS turned out to be genuine (checked game by game against ESPN: 144/149
exact date+time matches, 136/149 attendance figures identical to the person),
but two things were wrong with how it was built:

1.  It enumerated **teams**, not **venue calendars**. SAP Center - 0.51 km from
    the nearest sensor - actually hosted 70 events in 2017 H1. Only the 27
    Sharks games were collected. The 26 AHL Barracuda games, 17 concerts
    (Ariana Grande, The Weeknd, Roger Waters, TOOL, Queen + Adam Lambert, ...),
    WWE Payback and Stars On Ice were all missed. Levi's Stadium hosted U2's
    Joshua Tree Tour on 2017-05-17 (68,500-seat stadium, 1.63 km from the
    network) - also missed.
2.  75.8% of what it DID collect was at Oracle Arena / Oakland Coliseum /
    AT&T Park, whose nearest sensor is 39-49 km away. Those events cannot move
    South Bay traffic; broadcasting them into the event channel is noise.

This script rebuilds the file from two reproducible sources:

  * **ESPN scoreboard API** - exact start times (UTC) and real per-game
    attendance for MLB / NHL / NBA / MLS. Free, no key.
  * **Wayback Machine snapshots of the venues' own event calendars** -
    everything the leagues don't cover: concerts, AHL, wrestling, ice shows.

Every output row records where its date, time and attendance came from, so the
paper can state precisely which values were observed and which were assumed.

Usage
-----
    python -m src.data.fetch_events --config configs/default.yaml
    python -m src.data.fetch_events --config configs/default.yaml --dry-run
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import gzip
import html
import json
import math
import pathlib
import re
import sys
import time
import zoneinfo

import requests
import yaml

LOCAL_TZ = zoneinfo.ZoneInfo("America/Los_Angeles")

# Do NOT set a User-Agent. ESPN's WAF answers 403 to browser-like and custom
# agents but serves requests' default one fine. Verified 2026-08-28.

# ---------------------------------------------------------------------------
# Venue registry. Coordinates are WGS-84. `sources` says how we enumerate it.
# Venues beyond events.max_venue_distance_km of every sensor are dropped later.
# ---------------------------------------------------------------------------
VENUES = [
    # name,               lat,       lon,        capacity, espn_names,                              wayback_host
    ("SAP Center",        37.332700, -121.901000, 17562,
     ("SAP Center at San Jose", "SAP Center"),                                                      "sapcenter.com"),
    ("Avaya Stadium",     37.351100, -121.925200, 18000,
     ("Avaya Stadium", "PayPal Park", "Earthquakes Stadium"),                                       None),
    ("Levi's Stadium",    37.403300, -121.969400, 68500,
     ("Levi's Stadium",),                                                                           "levisstadium.com"),
    # Outdoor amphitheatre, 1 Amphitheatre Pkwy, Mountain View. Bigger than SAP
    # Center and 1.56 km from a sensor, but its season is May-October, so it was
    # invisible to a venue list built around the winter sports calendar.
    ("Shoreline Amphitheatre", 37.426700, -122.080600, 22500,
     (),                                                                            "mountainviewamphitheater.com"),
    ("Stanford Stadium",  37.434600, -122.160900, 50424,
     ("Stanford Stadium",),                                                                         None),
    # --- kept only so the exclusion is explicit and auditable in the log ---
    ("Oracle Arena",      37.750300, -122.203100, 19596, ("Oracle Arena", "ORACLE Arena"),          None),
    ("Oakland Coliseum",  37.751600, -122.200600, 35067,
     ("Oakland Coliseum", "Oakland-Alameda County Coliseum", "RingCentral Coliseum"),               None),
    ("AT&T Park",         37.778600, -122.389300, 41915, ("AT&T Park", "Oracle Park"),              None),
]
# tuple layout: 0=name 1=lat 2=lon 3=capacity 4=espn_names 5=wayback_host
# A venue does not have ONE capacity - it has one per configuration, and they
# differ by thousands. SAP Center seats 17,562 for hockey but 19,190 for a
# concert (no ice, floor seating). Levi's seats 68,500 for football but a
# concert puts the stage in one end zone and sells far fewer: U2's own Boxscore
# shows 50,072 offered. Using the sport number for concerts is a real error, and
# it is the error the U2 row already caught - the stadium_concert default was
# 0.80 x 68,500 = 55,000 against a real house of 50,072.
#
# Every figure below is from the venue's Wikipedia infobox, checked 2026-09-02,
# except Levi's concert, which is the measured Boxscore offering.
CAPACITY_BY_USE = {
    "SAP Center":             {"hockey": 17562, "concert": 19190,
                               "wrestling": 18300, "basketball": 18543},
    "Avaya Stadium":          {"soccer": 18000},
    # Levi's is deliberately given NO concert entry. A stadium concert has no
    # stable capacity - it depends where the stage goes. U2's Boxscore shows
    # 50,072 offered in 2017; the venue's record is 80,058 (Ed Sheeran, 2023).
    # Inventing a single number here would be worse than falling back to the
    # football capacity, which at least is a real published figure.
    "Levi's Stadium":         {"football": 68500},
    "Shoreline Amphitheatre": {"concert": 22500},
    "Stanford Stadium":       {"football": 50424},
    "Oracle Arena":           {"basketball": 19596, "concert": 19596},
    "Oakland Coliseum":       {"baseball": 46847, "football": 63132},
    "AT&T Park":              {"baseball": 41915},
}

# Which configuration each event type is played in. Anything unlisted falls back
# to the venue's headline capacity.
TYPE_TO_USE = {
    "NHL": "hockey", "AHL": "hockey", "ice_show": "hockey",
    "NBA": "basketball", "MLS": "soccer", "MLB": "baseball",
    "NFL": "football", "amateur_sport": "football",
    "concert": "concert", "stadium_concert": "concert",
    "amphitheater_concert": "concert", "music_festival": "concert",
    "wrestling": "wrestling",
}


def venue_capacity(venue: str, event_type: str) -> int:
    """Seats this venue actually offers for THIS kind of event."""
    uses = CAPACITY_BY_USE.get(venue, {})
    use = TYPE_TO_USE.get(event_type)
    if use in uses:
        return uses[use]
    return VENUE_META[venue]["capacity"]


VENUE_BY_ESPN = {espn_name: v[0] for v in VENUES for espn_name in v[4]}
VENUE_META = {v[0]: {"lat": v[1], "lon": v[2], "capacity": v[3], "wayback": v[5]}
              for v in VENUES}

# Independent check on the hand-entered coordinates. The query strings use each
# venue's CURRENT name where it was renamed after 2017 (Avaya -> PayPal Park,
# Oracle Arena -> Oakland Arena, AT&T Park -> Oracle Park), because that is what
# OpenStreetMap holds today.
OSM_QUERY = {
    "SAP Center":             "SAP Center, San Jose, California",
    "Avaya Stadium":          "PayPal Park, San Jose, California",
    "Levi's Stadium":         "Levi's Stadium, Santa Clara, California",
    "Shoreline Amphitheatre": "Shoreline Amphitheatre, Mountain View, California",
    "Stanford Stadium":       "Stanford Stadium, Stanford, California",
    "Oracle Arena":           "Oakland Arena, Oakland, California",
    "Oakland Coliseum":       "Oakland Coliseum, Oakland, California",
    "AT&T Park":              "Oracle Park, San Francisco, California",
}
OSM_TOLERANCE_KM = 0.5      # a stadium footprint is ~200 m across; 500 m is generous

# Named all-day festivals whose titles carry no "festival" keyword.
MUSIC_FESTIVAL_KEYWORDS = ("bfd", "id10t", "warped tour", "audiotistic",
                           "beyond wonderland", "rock the bells", "lollapalooza",
                           "outside lands")

MONTHS = {m: i + 1 for i, m in enumerate(
    ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"])}


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def haversine_km(lat1, lon1, lat2, lon2) -> float:
    r = math.radians
    a = (math.sin(r(lat2 - lat1) / 2) ** 2
         + math.cos(r(lat1)) * math.cos(r(lat2)) * math.sin(r(lon2 - lon1) / 2) ** 2)
    return 2 * 6371.0 * math.asin(math.sqrt(a))


def load_config(path: str) -> dict:
    with open(path, encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def sensor_table(cfg: dict) -> list[dict] | None:
    """Every PEMS-BAY sensor with the fields we need, or None if not downloaded.

    Source: ``data/sensor_graph/sensor_metadata.csv`` of the Augmented-PEMS-BAY
    checkout - the SAME file `align.py` uses to place nodes, so the distances
    written into events.csv are the distances the model actually sees.
    """
    p = pathlib.Path(cfg["data"]["pems_bay_dir"]) / cfg["data"]["sensor_metadata_csv"]
    if not p.exists():
        return None
    with open(p, encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    return [{"id": r["Sensor ID"], "lat": float(r["Latitude"]),
             "lon": float(r["Longitude"]), "fwy": r["Freeway"], "dir": r["Direction"]}
            for r in rows]


def sensor_coords(cfg: dict) -> list[tuple[float, float]] | None:
    """(lat, lon) of every sensor, or None if PEMS-BAY isn't downloaded yet."""
    tbl = sensor_table(cfg)
    return None if tbl is None else [(s["lat"], s["lon"]) for s in tbl]


def venue_sensor_geometry(sensors: list[dict]) -> dict[str, dict]:
    """Per venue: which sensor is nearest, how far, and how many are close.

    `nearest_sensor_*` answers "which sensor is this event measured against?".
    The counts answer "how many nodes can this venue plausibly move?" - a venue
    whose nearest sensor is close but that has 0 other sensors within 5 km can
    only ever show up on one node.
    """
    geo = {}
    for name, meta in VENUE_META.items():
        d = [(haversine_km(meta["lat"], meta["lon"], s["lat"], s["lon"]), s)
             for s in sensors]
        d.sort(key=lambda t: t[0])
        km, near = d[0]
        geo[name] = {
            "nearest_sensor_id": near["id"],
            "nearest_sensor_km": round(km, 3),
            "nearest_sensor_freeway": f"{near['fwy']}-{near['dir']}",
            "sensors_within_2km": sum(1 for k, _ in d if k <= 2.0),
            "sensors_within_5km": sum(1 for k, _ in d if k <= 5.0),
        }
    return geo


def _osm_get(query: str, cache: pathlib.Path, cfg: dict) -> bytes | None:
    """Nominatim needs its own fetcher: unlike ESPN it REQUIRES a User-Agent and
    answers 403 without one, which is exactly what `get()` deliberately omits."""
    if cache.exists():
        raw = cache.read_bytes()
        if raw.strip() not in (b"", b"[]"):
            return raw
    ev = cfg["events"]
    for attempt in range(ev["request_retries"]):
        try:
            resp = requests.get(
                "https://nominatim.openstreetmap.org/search",
                params={"q": query, "format": "json", "limit": 1},
                headers={"User-Agent": "traffic-shockwave-forecasting/1.0 "
                                       "(course project; venue coordinate check)"},
                timeout=ev["request_timeout_s"])
            if resp.status_code == 200 and resp.content.strip() not in (b"", b"[]"):
                cache.parent.mkdir(parents=True, exist_ok=True)
                cache.write_bytes(resp.content)
                return resp.content
        except requests.RequestException:
            pass
        time.sleep(1.5 * (attempt + 1))
    print(f"  ! OSM lookup failed for {query!r}", file=sys.stderr)
    return None


def verify_venue_coords(cfg: dict) -> dict[str, dict]:
    """Cross-check the hand-entered venue coordinates against OpenStreetMap.

    `venue_lat` / `venue_lon` / `venue_capacity` are the only values in this
    pipeline that are typed rather than fetched, and they propagate into every
    `nearest_sensor_*` column and the whole exp(-d/tau) decay in `event_load`.
    A fetched value that is wrong gets caught by `source_url`; a typed one does
    not - unless something checks it. This is that check.

    Nominatim is queried once per venue and cached to disk, so reruns are
    offline and deterministic. If the network is unavailable the check reports
    `unverified` rather than failing the build.
    """
    out, cache_dir = {}, pathlib.Path(cfg["events"]["cache_dir"]) / "osm"
    for name, meta in VENUE_META.items():
        q = OSM_QUERY.get(name)
        if q is None:
            out[name] = {"status": "no_query"}
            continue
        raw = _osm_get(q, cache_dir / (re.sub(r"\W+", "_", name) + ".json"), cfg)
        if raw is None:
            out[name] = {"status": "unverified", "query": q}
            continue
        try:
            doc = json.loads(raw)
        except json.JSONDecodeError:
            out[name] = {"status": "unverified", "query": q}
            continue
        if not doc:
            out[name] = {"status": "not_found", "query": q}
            continue
        km = haversine_km(meta["lat"], meta["lon"],
                          float(doc[0]["lat"]), float(doc[0]["lon"]))
        out[name] = {"status": "ok" if km <= OSM_TOLERANCE_KM else "MISMATCH",
                     "query": q, "osm_lat": round(float(doc[0]["lat"]), 6),
                     "osm_lon": round(float(doc[0]["lon"]), 6),
                     "delta_m": round(km * 1000, 1)}
        time.sleep(1.2)                     # Nominatim asks for <=1 req/s
    return out


def get(url: str, cache: pathlib.Path, cfg: dict, *, binary=False):
    """GET with on-disk cache and retries. Returns bytes, or None if it failed.

    ESPN intermittently answers {"code":3001,"detail":"timeout error"}; those are
    NOT cached, because caching them silently drops whole days of fixtures.
    """
    if cache.exists():
        raw = cache.read_bytes()
        if raw[:2] == b"\x1f\x8b":
            raw = gzip.decompress(raw)
        if b'"code":3001' not in raw[:200] and len(raw) > 200:
            return raw
    ev = cfg["events"]
    for attempt in range(ev["request_retries"]):
        try:
            resp = requests.get(url, timeout=ev["request_timeout_s"])
            if resp.status_code == 200 and b'"code":3001' not in resp.content[:200]:
                cache.parent.mkdir(parents=True, exist_ok=True)
                cache.write_bytes(resp.content)
                return resp.content
        except requests.RequestException:
            pass
        time.sleep(1.5 * (attempt + 1))
    print(f"  ! giving up on {url}", file=sys.stderr)
    return None


def classify(title: str, league: str | None, venue: str) -> str:
    """Map a raw event title to one of the config's `defaults` keys."""
    if league:
        return league
    t = title.lower()
    if "barracuda" in t:
        return "AHL"
    if "sharks vs" in t or "sharks v." in t:
        return "NHL"
    if any(k in t for k in ("wwe", "bellator", "ufc", "boxing")):
        return "wrestling"
    if any(k in t for k in ("stars on ice", "disney on ice", "figure skating", "sesame", "paw patrol")):
        return "ice_show"
    if "49ers" in t:
        return "NFL"
    if any(k in t for k in ("earthquakes", "sounders", "galaxy", "real madrid", "gold cup")):
        return "MLS"
    # these must be tested BEFORE the venue fallback, or a food festival and a
    # high-school all-star game both get billed as a 55,000-seat rock concert
    if any(k in t for k in ("monster jam", "supercross", "motocross")):
        return "motorsport"
    # An all-day music festival is a different TRAFFIC shape from a food
    # festival and from an evening concert: arrivals spread over many hours and
    # the crowd leaves late. KITS' own history says BFD "would run all day, with
    # up-and-coming bands on the festival stage during the day, and more
    # established bands on the main stage at night" - it was being billed as a
    # 3-hour 19:00 concert because its title carries no festival keyword.
    if (any(k in t for k in MUSIC_FESTIVAL_KEYWORDS)
            or ("festival" in t and venue == "Shoreline Amphitheatre")):
        return "music_festival"
    if any(k in t for k in ("classic", "festival", "fest ", "expo", "beer", "wine", "food")):
        return "festival"
    if any(k in t for k in ("all star", "all-star", "high school", "graduation",
                            "commencement", "harlem globetrotters")):
        return "amateur_sport"
    if venue == "Levi's Stadium":
        return "stadium_concert"
    if venue == "Shoreline Amphitheatre":
        return "amphitheater_concert"
    return "concert"


# ---------------------------------------------------------------------------
# source 1: ESPN scoreboard API  (exact times + real attendance)
# ---------------------------------------------------------------------------

def collect_espn(cfg: dict) -> list[dict]:
    ev, out = cfg["events"], []
    cache_root = pathlib.Path(ev["cache_dir"]) / "espn"
    start = dt.date.fromisoformat(cfg["data"]["date_range"]["start"])
    end = dt.date.fromisoformat(cfg["data"]["date_range"]["end"])
    for league, endpoint in ev["espn_leagues"].items():
        n = 0
        day = start
        while day <= end:
            stamp = day.strftime("%Y%m%d")
            url = (f"https://site.api.espn.com/apis/site/v2/sports/{endpoint}"
                   f"/scoreboard?dates={stamp}&limit=100")
            raw = get(url, cache_root / league / f"{stamp}.json", cfg)
            day += dt.timedelta(days=1)
            if raw is None:
                continue
            try:
                doc = json.loads(raw)
            except json.JSONDecodeError:
                continue
            for item in doc.get("events", []):
                comp = item["competitions"][0]
                venue_raw = (comp.get("venue") or {}).get("fullName", "")
                venue = VENUE_BY_ESPN.get(venue_raw)
                if venue is None:
                    continue
                if not ev["include_preseason"] and item.get("season", {}).get("type") == 1:
                    continue
                local = (dt.datetime.fromisoformat(item["date"].replace("Z", "+00:00"))
                         .astimezone(LOCAL_TZ).replace(tzinfo=None))
                att = comp.get("attendance")
                out.append({
                    "start": local,
                    "venue": venue,
                    "type": league,
                    "name": item.get("shortName") or item.get("name", ""),
                    "attendance": int(att) if att else None,
                    "time_source": "espn",
                    "attendance_source": "espn" if att else "default",
                    "date_source": "espn",
                    "source": "espn",
                    "source_url": url,
                    "source_record_id": str(item.get("id", "")),
                    # ESPN publishes the kickoff in UTC ("...Z"); we convert once,
                    # here, and never again. Everything downstream is local.
                    "utc_from_source": item["date"],
                })
                n += 1
        print(f"    ESPN {league:4s}: {n} fixtures at registered venues")
    return out


# ---------------------------------------------------------------------------
# source 2: Wayback snapshots of the venues' own calendars
# ---------------------------------------------------------------------------

def wayback_snapshots(url_pattern: str, cfg: dict, limit: int) -> list[tuple[str, str]]:
    """Timestamps of archived snapshots useful for the study period.

    A venue's listing page shows only UPCOMING events, so a snapshot taken after
    the study window contributes nothing. Search from ~3 months before the study
    start (arena calendars run a few months ahead) up to the study end, and
    spread the picks evenly inside that range.
    """
    lo = (dt.date.fromisoformat(cfg["data"]["date_range"]["start"])
          - dt.timedelta(days=90)).strftime("%Y%m%d")
    hi = dt.date.fromisoformat(cfg["data"]["date_range"]["end"]).strftime("%Y%m%d")
    api = ("http://web.archive.org/cdx/search/cdx?"
           f"url={url_pattern}&from={lo}&to={hi}"
           "&output=json&filter=statuscode:200&limit=400")
    raw = get(api, pathlib.Path(cfg["events"]["cache_dir"]) / "cdx"
              / (re.sub(r"\W+", "_", url_pattern) + ".json"), cfg)
    if raw is None:
        return []
    try:
        rows = json.loads(raw)[1:]
    except (json.JSONDecodeError, IndexError):
        return []
    pairs = sorted({(r[1], r[2]) for r in rows})      # (timestamp, original url)
    if not pairs:
        return []
    if len(pairs) <= limit:
        return pairs
    step = len(pairs) / limit                        # spread evenly over the range
    return [pairs[int(i * step)] for i in range(limit)]


def collect_sapcenter(cfg: dict) -> list[dict]:
    """SAP Center's own listing: <div class="entry"> ... <span class="m/d/y"> + <h2><a>."""
    ev = cfg["events"]
    cache = pathlib.Path(ev["cache_dir"]) / "wayback" / "sapcenter"
    snaps = wayback_snapshots("sapcenter.com/events/all", cfg, ev["wayback_snapshots_per_source"])
    print(f"    SAP Center: {len(snaps)} archived snapshots")
    seen: dict[tuple[dt.date, str], dict] = {}
    for stamp, _orig in snaps:
        raw = get(f"http://web.archive.org/web/{stamp}id_/http://www.sapcenter.com/events/all",
                  cache / f"{stamp}.html", cfg)
        if raw is None:
            continue
        page = raw.decode("utf-8", "replace")
        for block in re.split(r'<div class="entry', page)[1:]:
            block = block[:2500]
            d = re.search(r'<span class="m">\s*(\w+)\s*</span>\s*'
                          r'<span class="d">\s*(\d+)\s*</span>\s*'
                          r'<span class="y">\s*,?\s*(\d{4})', block)
            t = re.search(r"<h2>\s*<a[^>]*>(.*?)</a>", block, re.S)
            if not (d and t and MONTHS.get(d.group(1)[:3])):
                continue
            date = dt.date(int(d.group(3)), MONTHS[d.group(1)[:3]], int(d.group(2)))
            title = html.unescape(re.sub(r"<[^>]+>", "", t.group(1))).strip()
            if not title:
                continue
            href = re.search(r'<h2>\s*<a[^>]*href="([^"]+)"', block)
            seen.setdefault((date, title), {
                "date": date, "venue": "SAP Center", "name": title,
                "type": classify(title, None, "SAP Center"),
                "source": f"wayback:sapcenter.com@{stamp}",
                "date_source": "venue_calendar",
                "source_url": (f"http://web.archive.org/web/{stamp}/"
                               "http://www.sapcenter.com/events/all"),
                "detail_url": href.group(1) if href else None,
                "detail_stamp": stamp,
            })
    return list(seen.values())


# ---------------------------------------------------------------------------
# SAP Center detail pages: the real doors/show times
# ---------------------------------------------------------------------------

def _to_time(text: str) -> dt.time | None:
    m = re.match(r"(\d{1,2}):(\d{2})\s*([APap])", text.strip())
    if not m:
        return None
    hour, minute, half = int(m.group(1)), int(m.group(2)), m.group(3).upper()
    if half == "P" and hour != 12:
        hour += 12
    if half == "A" and hour == 12:
        hour = 0
    return dt.time(hour, minute)


FULL_MONTHS = {m: i + 1 for i, m in enumerate(
    ["January", "February", "March", "April", "May", "June", "July", "August",
     "September", "October", "November", "December"])}


def parse_sap_detail(page: str) -> tuple[dt.time | None, dt.time | None, dt.date | None]:
    """(show_time, doors_time, page_date) from a SAP Center event page.

    `page_date` is the date the DETAIL page itself prints, and it is the guard on
    everything else: SAP's URL slugs are only unique per title, so a four-game
    home series against the same opponent produces `barracuda-vs-tucson`,
    `-tucson-1`, `-tucson-2`, ... and the listing page's link can point at the
    wrong one. Two rows were measurably affected. A time is only trusted when
    this date equals the date the listing gave.
    """
    txt = re.sub(r"\s+", " ", html.unescape(re.sub(r"<[^>]+>", " ", page)))
    page_date = None
    m = re.search(r"Event Times.{0,200}?(?:Mon|Tues|Wednes|Thurs|Fri|Satur|Sun)day,\s*"
                  r"(\w+)\s+(\d{1,2}),\s*(\d{4})", txt)
    if m and m.group(1) in FULL_MONTHS:
        page_date = dt.date(int(m.group(3)), FULL_MONTHS[m.group(1)], int(m.group(2)))
    doors = None
    m = re.search(r"Doors Open\s*(?:\|\s*)*(\d{1,2}:\d{2}\s*[APap]\.?[Mm]?)", txt)
    if m:
        doors = _to_time(m.group(1))
    show = None
    m = re.search(r"Event Times.{0,400}?"
                  r"(?:Mon|Tues|Wednes|Thurs|Fri|Satur|Sun)day,\s*\w+\s+\d{1,2},\s*\d{4}"
                  r".{0,60}?(\d{1,2}:\d{2}\s*[APap]\.?[Mm]?)", txt, re.S)
    if m:
        show = _to_time(m.group(1))
    if show is None:                      # fall back to the first time that is not doors
        for cand in re.findall(r"(\d{1,2}:\d{2}\s*[APap]\.?[Mm]?)", txt):
            parsed = _to_time(cand)
            if parsed and parsed != doors:
                show = parsed
                break
    return show, doors, page_date


def add_sap_detail_times(rows: list[dict], cfg: dict) -> dict:
    """Replace venue-convention times with the times the venue actually published.

    Measured motivation: with assumed 20:00 start times, the 6,000-16,000
    attendance tier (all concerts) showed a +0.133 mph egress effect - the WRONG
    SIGN. A one-hour error shifts the egress window by 12 five-minute steps, which
    is enough to wash the effect out or invert it.
    """
    ev = cfg["events"]
    cache = pathlib.Path(ev["cache_dir"]) / "wayback" / "sap_detail"
    got = miss = mismatch = 0
    for r in rows:
        url = r.get("detail_url")
        if r["venue"] != "SAP Center" or not url:
            continue
        slug = re.sub(r"\W+", "_", url.rstrip("/").rsplit("/", 1)[-1])[:80]
        raw = get(f"http://web.archive.org/web/{r['detail_stamp']}id_/{url}",
                  cache / f"{slug}.html", cfg)
        if raw is None:
            miss += 1
            continue
        show, doors, page_date = parse_sap_detail(raw.decode("utf-8", "replace"))
        if show is None:
            miss += 1
            continue
        if page_date is not None and page_date != r["date"]:
            # wrong page for this row (slug collision) - fall back to the
            # convention time rather than importing another game's start.
            mismatch += 1
            continue
        r["detail_show"] = show
        r["detail_doors"] = doors
        r["source_url"] = f"http://web.archive.org/web/{r['detail_stamp']}/{url}"
        got += 1
    return {"detail_pages_with_a_show_time": got, "detail_pages_failed": miss,
            "detail_pages_rejected_date_mismatch": mismatch}


def collect_levis(cfg: dict) -> list[dict]:
    """Levi's Stadium runs WordPress event-organiser; the slug carries the date."""
    ev = cfg["events"]
    cache = pathlib.Path(ev["cache_dir"]) / "wayback" / "levis"
    # Take the URLs the archive actually holds - month listings, day pages and the
    # index - rather than constructing them; guessed URLs 404 through Wayback.
    snaps = wayback_snapshots("levisstadium.com/events*", cfg,
                              ev["wayback_snapshots_per_source"] * 3)
    print(f"    Levi's Stadium: {len(snaps)} archived snapshots")
    seen: dict[tuple[dt.date, str], dict] = {}
    for i, (stamp, orig) in enumerate(snaps):
        url = f"http://web.archive.org/web/{stamp}id_/{orig}"
        raw = get(url, cache / f"{i:03d}.html", cfg)
        if raw is None:
            continue
        page = raw.decode("utf-8", "replace")
        # /event/2017-05-17-u2-the-joshua-tree-tour/  title="U2: The Joshua Tree Tour"
        for m in re.finditer(r'href="[^"]*/event/(\d{4})-(\d{2})-(\d{2})-[^"]*"\s+title="([^"]+)"', page):
            date = dt.date(int(m.group(1)), int(m.group(2)), int(m.group(3)))
            title = html.unescape(m.group(4)).strip()
            seen.setdefault((date, title), {
                "date": date, "venue": "Levi's Stadium", "name": title,
                "type": classify(title, None, "Levi's Stadium"),
                "source": f"wayback:levisstadium.com@{stamp}",
                "date_source": "venue_calendar",
                "source_url": f"http://web.archive.org/web/{stamp}/{orig}",
            })
    return list(seen.values())


def collect_shoreline(cfg: dict) -> list[dict]:
    """Shoreline Amphitheatre publishes schema.org JSON-LD on its listing page.

    Each upcoming show is emitted as a MusicEvent block carrying `name` and
    `startDate`. The DATE is authoritative; the TIME is not - every block reads
    19:00-23:00 with a nonsense +02:00 offset, i.e. a WordPress plugin default,
    not a published door time. So only the date is taken from here and the start
    falls back to the per-type convention, flagged as such.

    Season pass products ("Megaticket", "Ticket to Rock") are listed alongside
    real shows and are dropped: they are ticket bundles, not events, and would
    double-count the concerts they cover.
    """
    ev = cfg["events"]
    cache = pathlib.Path(ev["cache_dir"]) / "wayback" / "shoreline"
    snaps = wayback_snapshots("mountainviewamphitheater.com/events", cfg,
                              ev["wayback_snapshots_per_source"] * 2)
    print(f"    Shoreline Amphitheatre: {len(snaps)} archived snapshots")
    seen: dict[tuple[dt.date, str], dict] = {}
    bundles = 0
    for stamp, _orig in snaps:
        raw = get(f"https://web.archive.org/web/{stamp}id_/"
                  "http://www.mountainviewamphitheater.com/events",
                  cache / f"{stamp}.html", cfg)
        if raw is None:
            continue
        page = raw.decode("utf-8", "replace")
        for m in re.finditer(r'\{"@context":"http://schema\.org","@type":"MusicEvent".*?\}\}\}',
                             page, re.S):
            try:
                doc = json.loads(html.unescape(m.group(0)))
            except json.JSONDecodeError:
                continue
            iso, title = doc.get("startDate", "")[:10], doc.get("name", "").strip()
            if not (iso and title):
                continue
            if "Includes All Performances" in title:      # season pass, not an event
                bundles += 1
                continue
            date = dt.date.fromisoformat(iso)
            seen.setdefault((date, title), {
                "date": date, "venue": "Shoreline Amphitheatre", "name": title,
                "type": classify(title, None, "Shoreline Amphitheatre"),
                "source": f"wayback:mountainviewamphitheater.com@{stamp}",
                "date_source": "venue_calendar",
                "source_url": (f"https://web.archive.org/web/{stamp}/"
                               "http://www.mountainviewamphitheater.com/events"),
            })
    print(f"      {len(seen)} shows ({bundles} season-pass listings dropped)")
    return list(seen.values())


# ---------------------------------------------------------------------------
# Column origin taxonomy.
#
# "derived" was too coarse a word: it lumped a haversine distance (which cannot
# be wrong unless its inputs are) together with end_time (which rests entirely on
# a duration we assumed). Four categories, and only ONE of them means "assumed":
#
#   fetched      read from a source document; `source_url` retrieves it
#   computed     deterministic function of other columns; recomputable, no choice
#   assumed      rests on a value in configs/*.yaml that no source published
#   hand_entered typed into the VENUES registry in this file; NO script verifies it
#
# `assumed` is the only category that makes a value uncertain. `hand_entered` is
# not uncertain in principle - a building's coordinates are a fact - but nothing
# here checks it, so it is called out separately rather than hidden in "computed".
# ---------------------------------------------------------------------------
COLUMN_ORIGINS = {
    "event_id":                 ("computed",     "venue slug + start_time"),
    "start_time":               ("fetched|assumed", "fetched for 79 rows; assumed for the rest - read start_time_basis"),
    "end_time":                 ("assumed",      "ALWAYS start + events.defaults[type].duration_min; no source publishes a finish"),
    "timezone":                 ("computed",     "constant, copied from data.timezone"),
    "start_time_utc":           ("computed",     "start_time converted to UTC; verification only"),
    "venue_name":               ("fetched",      "venue whose calendar or ESPN record produced the row"),
    "venue_lat":                ("hand_entered", "VENUES registry; CHECKED against OpenStreetMap - see venue_coord_osm_delta_m"),
    "venue_lon":                ("hand_entered", "VENUES registry; CHECKED against OpenStreetMap - see venue_coord_osm_delta_m"),
    "venue_capacity":           ("hand_entered", "VENUES registry; NOT verified by any source - the one unchecked input left"),
    "nearest_sensor_id":        ("computed",     "argmin haversine over sensor_metadata.csv; inherits venue_lat/lon"),
    "nearest_sensor_km":        ("computed",     "haversine; inherits venue_lat/lon"),
    "nearest_sensor_freeway":   ("fetched",      "sensor_metadata.csv"),
    "sensors_within_2km":       ("computed",     "count; inherits venue_lat/lon"),
    "sensors_within_5km":       ("computed",     "count; inherits venue_lat/lon"),
    "venue_coord_osm_delta_m":  ("computed",     "metres between the hand-entered coordinate and OpenStreetMap's - the check on venue_lat/lon"),
    "expected_attendance":      ("fetched|assumed", "fetched for 37 rows; assumed for the rest - read attendance_kind"),
    "attendance_kind":          ("computed",     "label over attendance_source and venue_capacity"),
    "event_type":               ("fetched|computed", "the league name, or classify() run on the title"),
    "event_name":               ("fetched",      "title as the source published it"),
    "duration_min":             ("assumed",      "events.defaults[type].duration_min"),
    "date_source":              ("computed",     "label; the DATE itself is fetched on every row"),
    "time_source":              ("computed",     "label"),
    "attendance_source":        ("computed",     "label"),
    "duration_source":          ("computed",     "label; constant nominal_by_type"),
    "start_time_basis":         ("computed",     "exact_published vs type_convention"),
    "assumed_from":             ("computed",     "the exact config keys and values this row leans on"),
    "start_time_is_estimated":  ("computed",     "flag over time_source"),
    "end_time_is_estimated":    ("computed",     "flag; 1 on every row"),
    "attendance_is_estimated":  ("computed",     "flag over attendance_source"),
    "confidence":               ("computed",     "rollup of the two variable flags"),
    "source":                   ("computed",     "provenance tag"),
    "source_url":               ("computed",     "the request URL or archived page the row came from"),
}


# ---------------------------------------------------------------------------
# audited corrections
#
# The venue listing is not ground truth. These two lists record the cases where
# an INDEPENDENT source contradicts it. Both are applied after dedupe and both
# are reported, so the file never silently disagrees with what was scraped.
# ---------------------------------------------------------------------------

# Rows the venue calendar produced that a team source shows did not happen.
# Evidence (all on web.archive.org, 2026-09-02):
#   * sjbarracuda.com's own April 2017 gameday posts are Apr 1 vs Manitoba,
#     Apr 8 AT Stockton, Apr 9 vs Stockton, Apr 15 AT Bakersfield, Apr 21 vs
#     Stockton. Tucson/Roadrunners appears nowhere in April.
#   * The club's front page on 2017-04-22 reads "Last Game Sat, Apr 15" and
#     "Upcoming Games Home Fri, Apr 21 ... Home Sun, Apr 23".
#   * The AHL does not schedule four consecutive home dates against one opponent.
# The four rows came from stale entries left on the SAP listing (two from the
# 2016-10 snapshot, two from the 2017-02 one) after a schedule change.
EXCLUSIONS = [
    ("SAP Center", "2017-04-11", "Barracuda vs. Tucson"),
    ("SAP Center", "2017-04-12", "Barracuda vs. Tucson"),
    ("SAP Center", "2017-04-13", "Barracuda vs. Tucson"),
    ("SAP Center", "2017-04-14", "Barracuda vs. Tucson"),
]
EXCLUSION_REASON = ("sjbarracuda.com April 2017 archive shows no Tucson game and "
                    "no game at all on these dates; stale SAP listing entry")

# Real events the venue listing MISSED. Both are Barracuda Calder Cup playoff
# home games at SAP Center, absent from every archived SAP snapshot but printed
# on the club's own front page with start times.
MANUAL_ROWS = [
    {"start": dt.datetime(2017, 4, 21, 19, 0), "venue": "SAP Center", "type": "AHL",
     "name": "Barracuda vs. Stockton (Calder Cup Playoffs)",
     "time_source": "venue_site", "attendance_source": "default",
     "date_source": "team_site_verified", "source": "wayback:sjbarracuda.com@20170422",
     "source_url": "https://web.archive.org/web/20170422/http://www.sjbarracuda.com/"},
    {"start": dt.datetime(2017, 4, 23, 15, 0), "venue": "SAP Center", "type": "AHL",
     "name": "Barracuda vs. Stockton (Calder Cup Playoffs)",
     "time_source": "venue_site", "attendance_source": "default",
     "date_source": "team_site_verified", "source": "wayback:sjbarracuda.com@20170422",
     "source_url": "https://web.archive.org/web/20170422/http://www.sjbarracuda.com/"},
]


# Real attendance for rows no API covers, from Billboard Boxscore - the industry
# figure venues and promoters report, the same class of source as ESPN's
# announced attendance. Transcribed into Wikipedia tour-date tables, which cite
# billboard.com/biz/current-boxscore (archived 2017-09-27).
#
# This is the ONLY route found to real crowd sizes for concerts. It covers 4 of
# the 58 rows that would otherwise use a per-type default, because it needs both
# a Wikipedia tour page AND an editor who transcribed the Boxscore row. Searched
# and NOT found: Lady Antebellum, Live 105's BFD, Future/Migos (x2), Boston,
# Brad Paisley, Korn, ID10T (x2) - checked 2026-09-02.
#
# `sold` is what the row uses; `offered` is the capacity that show was configured
# for and is kept because it is the interesting part: a stadium concert sells a
# far smaller house than the sport does (Levi's 50,072 vs its 68,500 NFL
# capacity - the stage eats one end), and the same band on consecutive nights
# can differ by a third.
BOXSCORE = {
    ("Levi's Stadium", "2017-05-17"): {
        "sold": 50072, "offered": 50072,
        "url": "https://en.wikipedia.org/wiki/The_Joshua_Tree_Tours_2017_and_2019"},
    ("Shoreline Amphitheatre", "2017-06-02"): {
        "sold": 17305, "offered": 22000,
        "url": "https://en.wikipedia.org/wiki/Chris_Stapleton%27s_All-American_Road_Show_Tour"},
    ("Shoreline Amphitheatre", "2017-06-03"): {
        "sold": 21780, "offered": 22015,
        "url": "https://en.wikipedia.org/wiki/Dead_%26_Company_Summer_Tour_2017"},
    ("Shoreline Amphitheatre", "2017-06-04"): {
        "sold": 16422, "offered": 21949,
        "url": "https://en.wikipedia.org/wiki/Dead_%26_Company_Summer_Tour_2017"},
}


def apply_boxscore(rows: list[dict]) -> int:
    """Attach real Boxscore attendance to the raw rows that have one."""
    n = 0
    for r in rows:
        d = r.get("date") or (r["start"].date() if "start" in r else None)
        hit = BOXSCORE.get((r["venue"], d.isoformat())) if d else None
        if hit is None:
            continue
        r["attendance"] = hit["sold"]
        r["attendance_source"] = "billboard_boxscore"
        r["boxscore_offered"] = hit["offered"]
        r["source"] = r.get("source", "") + f"; attendance:{hit['url']}"
        n += 1
    return n


def apply_corrections(rows: list[dict]) -> tuple[list[dict], dict]:
    """Drop contradicted rows; the additions are injected before materialise."""
    drop = {(v, d, n) for v, d, n in EXCLUSIONS}
    kept, removed = [], []
    for r in rows:
        key = (r["venue_name"], r["start_time"].strftime("%Y-%m-%d"), r["event_name"])
        (removed if key in drop else kept).append(r)
    return kept, {"excluded": [f"{v} {d} {n}" for v, d, n in sorted(drop)],
                  "excluded_matched": len(removed),
                  "exclusion_reason": EXCLUSION_REASON}


# ---------------------------------------------------------------------------
# merge / materialise
# ---------------------------------------------------------------------------

def _to_utc(local_naive: dt.datetime) -> str:
    """Local wall-clock -> UTC string. Never guess: PEMS-BAY is local wall-clock,
    so `start_time` must stay local; this column exists only so a reader can
    verify the offset (PST -08:00 before 2017-03-12, PDT -07:00 after)."""
    return (local_naive.replace(tzinfo=LOCAL_TZ)
            .astimezone(dt.timezone.utc).strftime("%Y-%m-%d %H:%M"))


def _slug(text: str) -> str:
    return re.sub(r"_+", "_", re.sub(r"\W+", "_", text)).strip("_").lower()


def materialise(rows: list[dict], cfg: dict, geo: dict[str, dict]) -> list[dict]:
    """Give every row a concrete start_time / end_time / attendance, plus the
    provenance that says which of those three is measured and which is assumed.

    The three fields are resolved INDEPENDENTLY, each by first-match-wins down a
    priority list. Nothing here estimates a DATE: every row's date comes from a
    source, always.

      start_time   1. espn               exact published UTC kickoff -> local
                   2. venue_detail_page  the venue's own published show time
                   3. venue_site         the club's own front page
                   4. venue_convention   ASSUMED = events.defaults[type].start

      attendance   1. billboard_boxscore real reported crowd count (strongest)
                   2. espn               the league's announced figure
                   3. default            ASSUMED = events.defaults[type].attendance

      end_time     no priority list - ALWAYS assumed. No source anywhere
                   publishes a finish time, so it is
                   start + events.defaults[type].duration_min on all rows.

    `confidence` is a rollup of the two fields that VARY, by this truth table:

        start assumed?  attendance assumed?  confidence
        no              no                   high
        no              yes                  medium
        yes             no                   medium
        yes             yes                  low

    end_time is deliberately EXCLUDED from it: it is assumed on 100% of rows, so
    including it would make every row `low` and destroy the column's usefulness.
    Read `end_time_is_estimated` (always 1) for that, and `assumed_from` for the
    exact config keys any given row rests on.
    """
    defaults = cfg["events"]["defaults"]
    tz_name = cfg["data"]["timezone"]
    out = []
    for r in rows:
        kind = r["type"] if r["type"] in defaults else "other"
        dflt = defaults[kind]
        if "start" in r:                                   # ESPN: exact time
            start = r["start"]
            time_source = r["time_source"]
        elif r.get("detail_show"):                         # venue published it
            start = dt.datetime.combine(r["date"], r["detail_show"])
            time_source = "venue_detail_page"
        else:                                              # last resort
            hh, mm = (int(x) for x in dflt["start"].split(":"))
            start = dt.datetime.combine(r["date"], dt.time(hh, mm))
            time_source = "venue_convention"
        # `or` is safe only because no source reports a zero gate; a real 0 would
        # silently fall through to the default. Guard it so that stays true.
        assert r.get("attendance") != 0, f"zero attendance reported for {r.get('name')}"
        att = r.get("attendance") or dflt["attendance"]
        att_source = r.get("attendance_source", "default")
        venue = r["venue"]
        # Capacity FOR THIS CONFIGURATION, not the venue's headline number - a
        # concert at SAP Center offers 19,190 seats, a Sharks game 17,562.
        cap = venue_capacity(venue, kind)

        # NONE of these figures are turnstile counts. US pro leagues report
        # "announced attendance" = tickets sold/distributed, and the NFL row
        # proves it: 70,178 at Levi's EXCEEDS its 68,500 seating capacity. You
        # cannot seat more people than you have seats; the figure counts tickets,
        # including standing room. All 37 therefore overstate the crowd that
        # actually travelled, by the no-show rate.
        #
        # What separates this label from `announced_observed` is CENSORING, not
        # measurement. A figure landing exactly on capacity means "sold out",
        # i.e. demand >= capacity, with the true value unknowable above it. The
        # measured shape says so plainly: at SAP Center the 10 non-sellout games
        # produce 10 DISTINCT values (17,386, 17,387, 17,402, ... 17,515) - the
        # signature of a real varying quantity - while 17,562 occurs 17 times as
        # a point mass exactly at the ceiling, and nothing ever exceeds it. All
        # 9 Avaya MLS matches sit at exactly 18,000. Right-censored data, so
        # these 26 rows carry no information ABOUT EACH OTHER.
        #
        # NOTE this label inherits `cap`, which is HAND-ENTERED and unverified.
        # A wrong capacity would silently mislabel a genuine sellout as
        # `announced_observed` (or the reverse). It cannot change the NUMBER in
        # `expected_attendance`, only the label on it.
        if att_source == "billboard_boxscore":
            # A real, varying, reported crowd count - the strongest figure in the
            # file, stronger than ESPN's sellout-at-capacity rows.
            att_kind = "boxscore_observed"
        elif att_source != "espn":
            att_kind = "type_default_estimate"
        elif int(att) == cap:
            att_kind = "announced_sellout_at_capacity"
        else:
            att_kind = "announced_observed"

        time_est = time_source == "venue_convention"
        att_est = att_source not in ("espn", "billboard_boxscore")
        # Spell out, per row, exactly which config values this row rests on.
        # Every row lists end_time, because end_time is assumed on every row.
        assumed = []
        if time_est:
            assumed.append(f"start_time<-events.defaults.{kind}.start({dflt['start']})")
        assumed.append(f"end_time<-events.defaults.{kind}.duration_min({dflt['duration_min']})")
        if att_est:
            assumed.append(f"expected_attendance<-events.defaults.{kind}.attendance({dflt['attendance']})")
        confidence = ("high" if not time_est and not att_est
                      else "low" if time_est and att_est else "medium")
        g = geo[venue]
        end = start + dt.timedelta(minutes=dflt["duration_min"])
        out.append({
            "event_id": f"{_slug(venue)}_{start:%Y%m%dT%H%M}",
            "start_time": start,
            "end_time": end,
            "timezone": tz_name,
            "start_time_utc": _to_utc(start),
            "venue_name": venue,
            "venue_lat": VENUE_META[venue]["lat"],
            "venue_lon": VENUE_META[venue]["lon"],
            "venue_capacity": cap,
            "nearest_sensor_id": g["nearest_sensor_id"],
            "nearest_sensor_km": g["nearest_sensor_km"],
            "nearest_sensor_freeway": g["nearest_sensor_freeway"],
            "sensors_within_2km": g["sensors_within_2km"],
            "sensors_within_5km": g["sensors_within_5km"],
            "venue_coord_osm_delta_m": g.get("venue_coord_osm_delta_m", ""),
            "expected_attendance": int(att),
            "attendance_kind": att_kind,
            "event_type": kind,
            "event_name": r.get("name", ""),
            "duration_min": int(dflt["duration_min"]),
            "date_source": r.get("date_source", "espn"),
            "time_source": time_source,
            "attendance_source": att_source,
            "duration_source": "nominal_by_type",          # never an observed finish
            "start_time_basis": "type_convention" if time_est else "exact_published",
            "assumed_from": "; ".join(assumed),
            "start_time_is_estimated": int(time_est),
            "end_time_is_estimated": 1,                    # true for EVERY row
            "attendance_is_estimated": int(att_est),
            "confidence": confidence,
            "source": r["source"],
            "source_url": r.get("source_url", ""),
        })
    return out


def dedupe(rows: list[dict]) -> list[dict]:
    """ESPN wins over Wayback for the same venue+day (it has the exact time)."""
    best: dict[tuple[str, dt.date], dict] = {}
    for r in sorted(rows, key=lambda x: (x["time_source"] != "espn",
                                         x["attendance_source"] != "espn")):
        key = (r["venue_name"], r["start_time"].date())
        if key in best:
            # a genuine double-header / two shows in one day keeps both only if
            # the start times differ by more than 3 hours
            if abs((r["start_time"] - best[key]["start_time"]).total_seconds()) > 3 * 3600:
                best[(r["venue_name"], r["start_time"].date(), r["start_time"].hour)] = r
            continue
        best[key] = r
    return sorted(best.values(), key=lambda x: x["start_time"])


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--dry-run", action="store_true", help="collect and report, write nothing")
    args = ap.parse_args()

    cfg = load_config(args.config)
    ev = cfg["events"]
    start = dt.datetime.fromisoformat(cfg["data"]["date_range"]["start"])
    end = dt.datetime.fromisoformat(cfg["data"]["date_range"]["end"]) + dt.timedelta(days=1)

    # ---- which venues are close enough to matter? -------------------------
    sensors = sensor_table(cfg)
    print("=== Venue screening ===")
    if sensors is None:
        print("  ! PEMS-BAY not downloaded; falling back to the published sensor bbox")
        sensors = [{"id": "bbox", "lat": la, "lon": lo, "fwy": "?", "dir": "?"}
                   for la, lo in [(37.2504, -122.0795), (37.4275, -121.8403),
                                  (37.3447, -121.9401)]]
    coords = [(s["lat"], s["lon"]) for s in sensors]
    geo = venue_sensor_geometry(sensors)
    keep = {}
    for name, meta in VENUE_META.items():
        d = min(haversine_km(meta["lat"], meta["lon"], la, lo) for la, lo in coords)
        ok = d <= ev["max_venue_distance_km"]
        g = geo[name]
        print(f"  {'KEEP  ' if ok else 'DROP  '}{name:18s} nearest sensor "
              f"{g['nearest_sensor_id']} ({g['nearest_sensor_freeway']}) at {d:6.2f} km"
              f"  | sensors within 2/5 km: {g['sensors_within_2km']}/{g['sensors_within_5km']}")
        if ok:
            keep[name] = d

    # ---- are the hand-entered coordinates right? --------------------------
    print("\n=== Venue coordinate check (OpenStreetMap) ===")
    osm = verify_venue_coords(cfg)
    for name, v in osm.items():
        if v["status"] in ("ok", "MISMATCH"):
            print(f"  {v['status']:8s} {name:24s} {v['delta_m']:6.0f} m from OSM")
        else:
            print(f"  {v['status']:8s} {name:24s} (coordinates NOT verified)")
    bad = [n for n, v in osm.items() if v["status"] == "MISMATCH"]
    assert not bad, f"hand-entered coordinates disagree with OSM by >{OSM_TOLERANCE_KM} km: {bad}"

    # ---- collect ----------------------------------------------------------
    print("\n=== Collecting ===")
    raw: list[dict] = []
    if ev["sources"].get("espn"):
        raw += collect_espn(cfg)
    if ev["sources"].get("sapcenter_wayback"):
        raw += collect_sapcenter(cfg)
    if ev["sources"].get("levis_wayback"):
        raw += collect_levis(cfg)
    if ev["sources"].get("shoreline_wayback"):
        raw += collect_shoreline(cfg)
    raw += [dict(r) for r in MANUAL_ROWS]

    dmeta = None
    if ev["sources"].get("sapcenter_detail_times"):
        print("    SAP Center detail pages (real doors/show times) ...")
        dmeta = add_sap_detail_times(raw, cfg)
        print(f"      got a published show time for {dmeta['detail_pages_with_a_show_time']}, "
              f"failed on {dmeta['detail_pages_failed']}, "
              f"rejected {dmeta['detail_pages_rejected_date_mismatch']} whose page "
              f"printed a different date")

    for name, g in geo.items():
        v = osm.get(name, {})
        g["venue_coord_osm_delta_m"] = (v["delta_m"] if v.get("status") == "ok"
                                        else "")
    nbs = apply_boxscore(raw)
    print(f"    Billboard Boxscore: real attendance attached to {nbs} rows")
    rows = materialise(raw, cfg, geo)
    rows = [r for r in rows if r["venue_name"] in keep and start <= r["start_time"] < end]
    rows = dedupe(rows)
    rows, cmeta = apply_corrections(rows)
    print(f"\n=== Audited corrections ===")
    print(f"  removed {cmeta['excluded_matched']} contradicted rows "
          f"({cmeta['exclusion_reason']}):")
    for x in cmeta["excluded"]:
        print(f"    - {x}")
    print(f"  added {len(MANUAL_ROWS)} rows the venue listing missed "
          f"(Barracuda playoff home games, from the club's own site)")

    # ---- report -----------------------------------------------------------
    from collections import Counter
    print(f"\n=== {len(rows)} events kept ===")
    print("  by venue :", dict(Counter(r["venue_name"] for r in rows)))
    print("  by type  :", dict(Counter(r["event_type"] for r in rows)))
    print("  start time from ESPN       :",
          sum(1 for r in rows if r["time_source"] == "espn"))
    print("  start time from venue page :",
          sum(1 for r in rows if r["time_source"] == "venue_detail_page"))
    print("  start time still assumed   :",
          sum(1 for r in rows if r["time_source"] == "venue_convention"))
    print("  real attendance from ESPN  :",
          sum(1 for r in rows if r["attendance_source"] == "espn"))
    print("  end_time is nominal for ALL rows (no source publishes finish times)")
    print("  attendance kinds           :",
          dict(Counter(r["attendance_kind"] for r in rows)))
    print("  confidence                 :",
          dict(Counter(r["confidence"] for r in rows)))
    print(f"  all start/end times are LOCAL wall-clock in {cfg['data']['timezone']} "
          "(US Pacific), matching PEMS-BAY's own axis - NOT US Eastern, NOT UTC")

    if args.dry_run:
        print("\n--dry-run: nothing written")
        return 0

    out = pathlib.Path(cfg["data"]["events_csv"])
    out.parent.mkdir(parents=True, exist_ok=True)
    # Venue-level facts (coordinates, capacity, sensor geometry, the OSM check)
    # live in venues.csv, ONE row per venue, and join on `venue_name`. Repeating
    # them on all 95 event rows added nine columns that say the same thing 95
    # times and can drift out of step with venues.csv. `start_time_utc` is
    # likewise dropped: it is `start_time` re-expressed, carries no information,
    # and nothing downstream ever read it. `timezone` STAYS despite being
    # constant - reading these timestamps as Eastern or UTC silently destroys
    # every event window, so the file states its own convention.
    cols = ["event_id", "start_time", "end_time", "timezone",
            "venue_name",
            "expected_attendance", "attendance_kind", "event_type", "event_name",
            "duration_min",
            "date_source", "time_source", "attendance_source", "duration_source",
            "start_time_basis", "assumed_from",
            "start_time_is_estimated", "end_time_is_estimated",
            "attendance_is_estimated", "confidence", "source", "source_url"]
    with open(out, "w", newline="", encoding="utf-8") as fh:
        # `materialise` still computes the venue-level fields - venues.csv and the
        # screening log need them - so project onto `cols` rather than handing
        # DictWriter extra keys, which it rejects.
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        for r in rows:
            r = dict(r)
            r["start_time"] = r["start_time"].strftime("%Y-%m-%d %H:%M")
            r["end_time"] = r["end_time"].strftime("%Y-%m-%d %H:%M")
            w.writerow({k: r[k] for k in cols})
    print(f"\nWrote {out}  ({len(rows)} rows)")

    ven = pathlib.Path(cfg["data"]["venues_csv"])
    with open(ven, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["venue_name", "venue_lat", "venue_lon", "capacity",
                    "nearest_sensor_id", "nearest_sensor_km", "nearest_sensor_freeway",
                    "sensors_within_2km", "sensors_within_5km",
                    "osm_query", "osm_delta_m", "coord_check",
                    "n_events_in_period", "included", "inclusion_rule", "exclusion_reason"])
        n_ev = Counter(r["venue_name"] for r in rows)
        rule = f"nearest sensor <= {ev['max_venue_distance_km']} km (events.max_venue_distance_km)"
        for name, meta in VENUE_META.items():
            g = geo[name]
            inc = name in keep
            w.writerow([name, meta["lat"], meta["lon"], meta["capacity"],
                        g["nearest_sensor_id"], f"{g['nearest_sensor_km']:.2f}",
                        g["nearest_sensor_freeway"], g["sensors_within_2km"],
                        g["sensors_within_5km"],
                        osm.get(name, {}).get("query", ""),
                        osm.get(name, {}).get("delta_m", ""),
                        osm.get(name, {}).get("status", "unverified"),
                        n_ev.get(name, 0),
                        "yes" if inc else "no", rule,
                        "" if inc else f"nearest sensor {g['nearest_sensor_km']:.2f} km > "
                                       f"{ev['max_venue_distance_km']} km"])
    print(f"Wrote {ven}")

    # ---- machine-readable provenance summary ------------------------------
    # Written next to the CSV so a reader never has to re-run the script to
    # learn how the file was built or how much of it is assumed.
    meta = {
        "generated_by": "src/data/fetch_events.py",
        "generated_on_utc": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d %H:%M"),
        "config": args.config,
        "study_period": cfg["data"]["date_range"],
        "timezone": {
            "value": cfg["data"]["timezone"],
            "note": ("US PACIFIC, not US Eastern. start_time/end_time are LOCAL "
                     "wall-clock, the same convention as PEMS-BAY's speed.csv "
                     "columns, so events join to traffic by label with no shift. "
                     "start_time_utc is the same instant in UTC, for verification "
                     "only. 2017-03-12 02:00-02:59 does not exist locally (DST)."),
        },
        "sources": {
            "espn": ("site.api.espn.com scoreboard, one request per day per league; "
                     "gives the exact UTC start and the announced attendance"),
            "venue_calendars": ("Wayback Machine snapshots of sapcenter.com/events/all "
                                "and levisstadium.com/events*, plus each SAP event's "
                                "own detail page for the published show time"),
            "sensor_geometry": (str(pathlib.Path(cfg["data"]["pems_bay_dir"])
                                    / cfg["data"]["sensor_metadata_csv"])),
        },
        "venue_inclusion_rule_km": ev["max_venue_distance_km"],
        "venues": {name: dict(geo[name], included=(name in keep),
                              capacity=VENUE_META[name]["capacity"],
                              capacity_by_use=CAPACITY_BY_USE.get(name, {}),
                              n_events=int(sum(1 for r in rows if r["venue_name"] == name)))
                   for name in VENUE_META},
        "capacity_check": {
            "method": ("每个场馆的 Wikipedia infobox capacity 字段，2026-09-02 人工核对；"
                       "Levi's concert 用 U2 Boxscore 实测开放座位数"),
            "source": "en.wikipedia.org venue infoboxes",
            "type_to_configuration": TYPE_TO_USE,
            "note": ("A venue has one capacity PER CONFIGURATION. Using the sport "
                     "number for a concert is a real error: it produced a concert "
                     "default of 14,000 (0.80 x SAP's 17,562 hockey capacity) "
                     "instead of 15,352 (0.80 x its 19,190 concert capacity), and "
                     "a stadium_concert default of 55,000 against a real 50,072."),
        },
        "venue_coord_check": {
            "method": "OpenStreetMap Nominatim, one query per venue, cached to disk",
            "tolerance_km": OSM_TOLERANCE_KM,
            "results": osm,
        },
        "n_events": len(rows),
        "counts": {
            "by_venue": dict(Counter(r["venue_name"] for r in rows)),
            "by_type": dict(Counter(r["event_type"] for r in rows)),
            "time_source": dict(Counter(r["time_source"] for r in rows)),
            "attendance_source": dict(Counter(r["attendance_source"] for r in rows)),
            "attendance_kind": dict(Counter(r["attendance_kind"] for r in rows)),
            "confidence": dict(Counter(r["confidence"] for r in rows)),
        },
        "estimated": {
            "start_time": int(sum(r["start_time_is_estimated"] for r in rows)),
            "attendance": int(sum(r["attendance_is_estimated"] for r in rows)),
            "end_time": len(rows),
        },
        "detail_page_audit": dmeta if ev["sources"].get("sapcenter_detail_times") else None,
        "column_origins": {c: {"origin": COLUMN_ORIGINS[c][0], "note": COLUMN_ORIGINS[c][1]}
                           for c in cols},
        "origin_legend": {
            "fetched": "read from a source document; source_url retrieves it",
            "computed": "deterministic function of other columns; recomputable",
            "assumed": "rests on a configs/*.yaml value no source published - THE ONLY UNCERTAIN CATEGORY",
            "hand_entered": "typed into the VENUES registry in fetch_events.py; NO script verifies it",
        },
        "audited_corrections": dict(cmeta, added=[r["name"] + " @ "
                                                  + r["start"].strftime("%Y-%m-%d %H:%M")
                                                  for r in MANUAL_ROWS]),
    }
    mp = out.parent / "events_meta.json"
    mp.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote {mp}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
