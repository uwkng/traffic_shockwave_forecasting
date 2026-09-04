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


def event_windows(events_csv, time_index, *, min_attendance=MIN_ATTENDANCE,
                  egress_min=EGRESS_MIN, venues=None):
    """Boolean mask, True during the egress hour of a qualifying event.

    Returns the mask and the DataFrame of events that produced it, so a caller
    reporting "the event window" can state exactly which fixtures defined it.
    """
    ev = pd.read_csv(events_csv, parse_dates=["start_time", "end_time"])
    sub = ev[ev["expected_attendance"] >= min_attendance]
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


def holiday_mask(time_index):
    """Boolean mask, True on a US federal holiday."""
    idx = pd.DatetimeIndex(time_index)
    days = {pd.Timestamp(d).date() for d in US_HOLIDAYS_2017_H1}
    return np.isin(idx.date, list(days))
