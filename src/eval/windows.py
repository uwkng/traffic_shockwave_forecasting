"""Stage 11 - exogenous window definitions.

The proposal's contribution is that every metric is reported THREE times: over
the whole test set, inside exogenously-defined anomalous windows, and over the
complement. This module defines those windows and nothing else, so the choice
of window is one auditable place rather than a threshold copy-pasted into each
figure.

"Exogenous" is the load-bearing word. Every window here is derivable from a
weather observation or a published schedule, i.e. from information that exists
BEFORE the traffic does. No window may be defined from speed, occupancy, or
anything derived from them - that is the endogenous definition this project
exists to contrast against, and it is only available after congestion has
already formed.

Every threshold below was measured on 2017 H1, not chosen by convention. The
numbers in the docstrings are the measured speed anomaly relative to the same
node at the same time-of-week on days with no event and no rain.
"""

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Weather
# ---------------------------------------------------------------------------
# TWO tiers, not three. A three-tier light/moderate/heavy split is the natural
# thing to want and it does not survive contact with this precipitation series.
# acquire._parse_asos forward-fills p01i, which is a BACKWARD 1-hour
# accumulation, so one wet hour is stamped onto the twelve 5-min slots that
# follow it. That inflates the "lightly raining" category roughly threefold
# (16.3% of all timesteps here versus 4.6% by METAR present-weather codes) and
# dilutes its measured effect by more than half (-0.78 mph against -1.72 mph
# for observed light rain). Reporting a light tier from this series understates
# the effect for a data-processing reason, not a traffic one.
#
# A single threshold does not have that problem, and the result is not
# sensitive to where it is put:
#     p50 = 0.17 mm ->  9.50% of steps, -1.62 mph
#     p60 = 0.25 mm ->  6.92% of steps, -1.95 mph
#     p75 = 0.51 mm ->  4.42% of steps, -2.22 mph   <- default
#     p90 = 1.27 mm ->  1.77% of steps, -2.57 mph
# 0.51 mm is the 75th percentile of the WET steps: large enough that the window
# is real rain, small enough that it still holds 2,306 timesteps to average.
ADVERSE_PRECIP_MM = 0.51


def weather_windows(weather_csv, time_index, threshold_mm=ADVERSE_PRECIP_MM):
    """Boolean mask, True where the weather is adverse.

    `weather_csv` is cfg.data.weather_csv (the 5-min ASOS merge). It is
    reindexed onto `time_index` BY LABEL - the caller's index is the authority,
    because PEMS-BAY's axis is local wall-clock and is missing the DST
    spring-forward hour 2017-03-12 02:00..02:55. Aligning by position instead
    would slide every later timestamp by one hour.
    """
    w = pd.read_csv(weather_csv, parse_dates=["timestamp"]).set_index("timestamp")
    p = w["precipitation"].reindex(pd.DatetimeIndex(time_index))
    if p.isna().any():
        raise ValueError(
            f"{int(p.isna().sum())} of {len(p)} timestamps are absent from "
            f"{weather_csv}. The weather file must cover the traffic axis "
            "exactly; do not interpolate here, fix the acquisition.")
    return (p.to_numpy() >= threshold_mm)


# ---------------------------------------------------------------------------
# Events
# ---------------------------------------------------------------------------
# EGRESS, not "the event". Measured on 24 SAP Center evening NHL games, sensors
# within 2 km, aligned on scheduled START so the window does not depend on the
# estimated end_time:
#     +60 min  -0.49   +120 min -0.60   +150 min -0.68
#     +180 min -1.69 +- 0.51   <- 3.3 sigma, and 19:30 + 180 min = 22:30
#     +210 min -0.94 +- 0.34   +240 min -0.31   recovered by +270
# The dip is entirely in the hour the crowd leaves. A window spanning the whole
# fixture (doors to end, ~5.5 h) dilutes it about fivefold and reads +0.41 mph,
# which is how this was missed the first time.
EGRESS_MIN = (0, 60)        # minutes after end_time

# Attendance floor. Below it there is no measurable egress effect at all:
#     >= 40,000   -0.74 +- 0.56    (n=3)
#     15k - 40k   -0.45 +- 0.22    (n=64, > 2 SE)
#     <  15,000   +0.35 +- 0.41    (n=28, wrong sign, not significant)
# The 28 sub-threshold fixtures are almost all AHL: the Barracuda average 4,412
# in a 17,562-seat building. Including them in an "event window" adds 28 events
# that behave exactly like a normal evening and halves the measured effect.
MIN_ATTENDANCE = 15_000


# Provenance floor. events.csv marks 37 rows `high` (ESPN: real start time,
# ANNOUNCED attendance), 46 `medium` and 12 `low`. 42 of the 46 medium rows and
# all 12 low rows carry `attendance_kind = type_default_estimate` - a constant we
# assigned per event type - so applying MIN_ATTENDANCE to them filters our own
# defaults by construction, not the world. The medium tier is dominated by 24
# AHL fixtures stamped with a >=15,000 default; minor-league hockey draws a few
# thousand.
#
# Measured on the egress hour against matched controls (same weekday and clock,
# +-7/14 days, no event), sensors within 5 km:
#     high    37 fixtures   -0.63 +- 0.26 mph   14/36 vs  4/54 = 5.25x, z=3.66
#     medium  19            +0.18 +- 0.11        0/19 vs  5/39 = 0.00x, z=-1.63
#     low     11            -0.06 +- 0.12        1/11 vs  1/11 = 1.00x, z=0.00
#     all     67            -0.31 +- 0.15       15/66 vs 10/104 = 2.36x, z=2.35
# Dropping 30 of 67 fixtures RAISES both the effect size and the significance:
# the excluded rows are noise. Keep this in step with
# configs/default.yaml features.event_load.min_confidence, which applies the
# same floor to the input channels.
#
# What "high" does NOT cover: end_time_is_estimated is 1 for all 95 rows
# (duration_source is nominal_by_type throughout), so the egress window itself
# is an estimate even here. High means the date, start time and attendance are
# sourced.
MIN_CONFIDENCE = "high"
_CONF_RANK = {"low": 0, "medium": 1, "high": 2}


def event_windows(events_csv, time_index, *, min_attendance=MIN_ATTENDANCE,
                  egress_min=EGRESS_MIN, venues=None,
                  min_confidence=MIN_CONFIDENCE):
    """Boolean mask, True during the egress hour of a qualifying event.

    Returns the mask and the DataFrame of events that produced it, so a caller
    reporting "the event window" can state exactly which fixtures defined it.
    """
    ev = pd.read_csv(events_csv, parse_dates=["start_time", "end_time"])
    sub = ev[ev["expected_attendance"] >= min_attendance]
    if min_confidence is not None and "confidence" in sub.columns:
        floor = _CONF_RANK[str(min_confidence).lower()]
        rank = sub["confidence"].str.lower().map(_CONF_RANK)
        assert rank.notna().all(), (
            f"unknown confidence values: "
            f"{sorted(set(sub['confidence']) - set(_CONF_RANK))}")
        sub = sub[rank >= floor]
    if venues is not None:
        sub = sub[sub["venue_name"].isin(venues)]
    idx = pd.DatetimeIndex(time_index)
    mask = np.zeros(len(idx), dtype=bool)
    lo, hi = egress_min
    for end in sub["end_time"]:
        mask |= ((idx >= end + pd.Timedelta(minutes=lo))
                 & (idx < end + pd.Timedelta(minutes=hi)))
    return mask, sub


# ---------------------------------------------------------------------------
# Calendar
# ---------------------------------------------------------------------------
# US federal holidays inside the study period. This is the largest exogenous
# effect measured anywhere in this project - larger than rain, and an order of
# magnitude larger than events - because a holiday removes the commute rather
# than adding a crowd. Speed anomaly against the same weekday and time:
#     2017-01-02  New Year observed  Mon  all-day +2.29  AM +4.61  PM  +2.12
#     2017-01-16  MLK Day            Mon  all-day +2.53  AM +7.19  PM  +6.71
#     2017-02-20  Presidents Day     Mon  all-day +0.54  AM +5.31  PM  +2.68
#     2017-05-29  Memorial Day       Mon  all-day +4.43  AM +9.50  PM +11.27
# Note the SIGN: holidays make traffic FASTER, so they are not an "adverse"
# window. They belong in the metrics as a stratum to be reported or controlled
# for, never merged into the adverse windows above.
#
# Memorial Day falls inside the 70/10/20 test period, which no other exogenous
# signal here does: that test window contains zero adverse-weather timesteps.
US_HOLIDAYS_2017_H1 = {
    "2017-01-01": "New Year's Day",
    "2017-01-02": "New Year's Day (observed)",
    "2017-01-16": "Martin Luther King Jr. Day",
    "2017-02-20": "Presidents' Day",
    "2017-05-29": "Memorial Day",
}


# The compound window that actually exists. "Adverse weather during a scheduled
# event" is the natural thing to want and there is no such thing here: rain
# overlaps an egress hour 3 times in six months, 18 timesteps in total, one of
# them a single step. Any benchmark headlining compound event-and-weather
# performance on PEMS-BAY 2017 H1 is reporting noise.
#
# Rain during the weekday commute peak is a different story - 34 episodes across
# 19 distinct days, 500 timesteps - and it is where the shockwaves are: 88.4% of
# commute-peak timesteps carry at least one, against 18.2% off-peak. Rain in the
# peak costs -3.04 mph against -2.22 mph over all hours.
#
# Note what this window is NOT evidence for. Controlling for time of day, rain
# does not change how OFTEN breakdowns happen (91.8% in rain against 93.1% dry)
# nor how big they are (6.25 sensors involved against 6.46). Rain is a uniform
# capacity reduction; the peak is when the network has no slack to absorb it.
# HOW TO READ A WINDOWED METRIC. Never quote "MAE inside / MAE outside" as
# evidence that the window's phenomenon is hard to predict. A window is not a
# random sample of the test set - it sits at a particular hour, season and
# place, and the ratio absorbs all of that. Measured with persistence at 45 min:
#
#     fold   peak dry  peak RAIN  off-peak dry  off-peak RAIN
#     00        4.881      4.230         1.912          2.596
#     01        5.316      6.185         2.084          2.601
#     02        4.885      6.040         1.897          2.266
#
# Time of day ALONE, dry samples only, is 2.55x / 2.55x / 2.58x - three folds to
# two decimals. Rain then adds -0.65 to +1.16 mph on top and its sign is not
# consistent across folds. So the 1.75-2.93x that `weather_commute` shows
# against its complement is mostly the clock. `event_egress` makes the same
# point from the other side: it reads 0.48-0.74, i.e. EASIER than outside,
# because egress falls at 21:30-23:00 on an empty network.
#
# Report the 2x2. The window is still the right place to look - it is where the
# errors are - but only model-against-model inside one cell attributes them.
COMMUTE_HOURS = ((7, 10), (15, 19))     # local wall-clock, weekdays only


def commute_mask(time_index):
    """Boolean mask, True in the weekday morning or evening peak."""
    idx = pd.DatetimeIndex(time_index)
    h = idx.hour
    peak = np.zeros(len(idx), dtype=bool)
    for lo, hi in COMMUTE_HOURS:
        peak |= (h >= lo) & (h < hi)
    return peak & (idx.dayofweek < 5)


def weather_commute_windows(weather_csv, time_index,
                            threshold_mm=ADVERSE_PRECIP_MM):
    """Adverse precipitation AND the weekday commute peak."""
    return (weather_windows(weather_csv, time_index, threshold_mm)
            & commute_mask(time_index))


def holiday_mask(time_index):
    """Boolean mask, True on a US federal holiday."""
    idx = pd.DatetimeIndex(time_index)
    days = {pd.Timestamp(d).date() for d in US_HOLIDAYS_2017_H1}
    return np.isin(idx.date, list(days))
