"""
DATA CONTRACT - the single source of truth for shapes, channel order, splits,
and the prediction format. Every module in the repo imports from here.

Rule: do NOT change anything in this file without updating the team. All three
workstreams (data / models / eval) build against these constants, so a silent
change here breaks everyone downstream.

--------------------------------------------------------------------------
REVISION 2026-09-04 - eleven channels, grouped for ablation. Read this.
--------------------------------------------------------------------------
Three changes, each forced by a measurement rather than a preference. The
numbers quoted throughout this file are the speed anomaly of a sensor against
ITSELF at the same time-of-week on days with no event and no rain, unless
stated otherwise; "incremental R2" is measured against the residual of a
per-node AR(12) baseline that predicts 60 minutes ahead (that baseline scores
R2 0.567 on the test split, residual sd 6.29 mph).

  1. `occupancy` is not a filler channel. It is worth +0.153 incremental R2,
     second only to a node-by-time-of-week climatology and roughly a hundred
     times the whole weather group combined. It ships with the augmented
     release we already download.

  2. The single `event_load` channel is SPLIT IN TWO. It multiplied attendance
     into the decay, so "is an event on" and "how big is it" could not be
     separated - and the ablation ladder needs exactly that separation. Within
     the egress hour, over 8,649 (event, node) pairs from 75 events:
         active only, no attendance   R2 0.00012   t = -1.02   (NOT significant)
         linear attendance/20k        R2 0.00917   t = -8.95
         threshold >= 15,000          R2 0.01089   t = -9.76   <- best
     The information is in the attendance, not in the fact of an event. That
     contrast is a result, so the two channels are kept apart to show it.

  3. `is_holiday` is new and is the LARGEST exogenous effect in the project.
     Memorial Day runs +9.50 mph in the AM peak and +11.27 in the PM, against
     -4.01 for heavy rain and -1.22 for an NHL egress hour. Note the sign: a
     holiday removes the commute rather than adding a crowd.

`rain_intensity` (the METAR present-weather ladder) is GONE. It measured well
(-1.72 mph for observed light rain against -0.78 for the same hours as the
gauge series scores them), but `acquire.fetch_asos` does not request `wxcodes`,
so it cannot be built from the data this repo actually downloads. Adding it
means re-downloading both stations; measured against the headline number that
buys nothing (all precipitation parameterisations put the adverse-weather
slowdown at the same -3.1 mph), so it was dropped rather than faked.

A node-by-time-of-week climatology (`hist_mean_speed`) is deliberately NOT a
channel even though it is the strongest single predictor measured
(+0.216 incremental R2). It is the proposal's own historical-average BASELINE:
at the 60-minute horizon HA scores MAE 2.63 against the trained STGCN's 2.58,
and folding the baseline into the model destroys that comparison.
"""

# --- Graph / time dimensions (PEMS-BAY augmented release) ---
N_NODES = 325           # freeway loop detectors
N_TIMESTEPS = 52116     # 5-min steps, 1 Jan - 30 Jun 2017
FREQ_MIN = 5            # sampling interval, minutes

# The time axis is NOT a contiguous 5-min grid. 181 days x 288 = 52,128, but the
# file has 52,116 columns: 2017-03-12 02:00..02:55 is missing because PEMS-BAY
# timestamps are LOCAL WALL-CLOCK (America/Los_Angeles) and that hour does not
# exist (DST spring-forward). Never build the time axis with pd.date_range().
TIME_AXIS_SOURCE = "columns of data/raw/augmented-pems-bay/data/traffic_data/speed.csv"
DST_GAP = ("2017-03-12 02:00", "2017-03-12 02:55")   # 12 steps absent by design

# --- The 11 input channels, in this exact order ---
# variation: "spatiotemporal" = varies over time AND node
#            "temporal"        = varies over time only, broadcast to all nodes
#            "spatial"         = varies over node only, constant over time
CHANNELS = [
    ("speed",              "spatiotemporal"),  #  0  <- the prediction TARGET, mph
    ("occupancy",          "spatiotemporal"),  #  1  fraction 0..1
    ("time_of_day",        "temporal"),        #  2  fraction of day in [0, 1)
    ("day_of_week",        "temporal"),        #  3  0=Mon .. 6=Sun, scaled to [0,1]
    ("is_holiday",         "temporal"),        #  4  0/1, US federal holidays
    ("temperature",        "temporal"),        #  5  degC
    ("precipitation",      "temporal"),        #  6  mm
    ("wind_speed",         "temporal"),        #  7  km/h
    ("dist_to_venue",      "spatial"),         #  8  km to NEAREST in-network venue
    ("event_active_decay", "spatiotemporal"),  #  9  NO attendance - see below
    ("event_load",         "spatiotemporal"),  # 10  attendance-weighted
]
N_CHANNELS = len(CHANNELS)                    # 11
TARGET_CHANNEL = 0                            # speed
CHANNEL_INDEX = {name: i for i, (name, _) in enumerate(CHANNELS)}

# time_of_day is a single value, matching the standard STGCN/DCRNN PEMS-BAY
# setup. Switching to sin/cos gives 12 channels - tell the team first.
TIME_OF_DAY_ENCODING = "fraction"   # "fraction" | "sincos" (sincos => 12 channels)

# --- Channel groups: the ablation ladder ------------------------------------
# Build the master tensor ONCE with all 11 channels; each ablation rung selects
# COLUMNS at the dataloader and sets the model's c_in accordingly. Never
# rebuild the tensor per rung - the splits and scalers must be identical across
# rungs or the MAE numbers are not comparable.
CHANNEL_GROUPS = {
    "traffic":   ["speed", "occupancy"],
    "calendar":  ["time_of_day", "day_of_week", "is_holiday"],
    "weather":   ["temperature", "precipitation", "wind_speed"],
    "event_geo": ["dist_to_venue", "event_active_decay"],
    "event_att": ["event_load"],
}
# The rungs, in the order the team agreed. `traffic` is in every rung: without
# speed history there is nothing to condition ON.
ABLATION_RUNGS = {
    "1_traffic":            ["traffic"],
    "2_calendar":           ["traffic", "calendar"],
    "3_weather":            ["traffic", "weather"],
    "4_event_geo":          ["traffic", "event_geo"],
    "5_event_geo_att":      ["traffic", "event_geo", "event_att"],
    "6_all":                ["traffic", "calendar", "weather", "event_geo", "event_att"],
}


def rung_channels(rung: str) -> list[int]:
    """Column indices for an ablation rung, in CHANNELS order."""
    names = {n for g in ABLATION_RUNGS[rung] for n in CHANNEL_GROUPS[g]}
    return [i for i, (n, _) in enumerate(CHANNELS) if n in names]


# --- event_active_decay (9) and event_load (10) -----------------------------
# Both sum over events active at t and decay with distance from the venue. They
# differ in ONE thing, and that difference is the point of rungs 4 and 5:
#
#   event_active_decay[t, n] = sum_e  active_e(t) * exp(-dist(n, venue_e) / TAU)
#   event_load[t, n]         = sum_e  active_e(t) * exp(-dist(n, venue_e) / TAU)
#                                     * 1[attendance_e >= MIN] * attendance_e / REF
#
# The attendance FLOOR is in event_load only. Putting it in both would leak
# attendance information into the "activity only" rung and destroy the contrast.
# Below the floor there is no egress effect to find (sensors within 2 km,
# anomaly during the egress hour):
#       >= 40,000   -0.74 +- 0.56   (n=3 events)
#       15k - 40k   -0.45 +- 0.22   (n=64, > 2 SE)
#       <  15,000   +0.35 +- 0.41   (n=28, WRONG SIGN, not significant)
# The 28 sub-threshold fixtures are almost all AHL: the San Jose Barracuda
# average 4,412 in a 17,562-seat building. A linear attendance weight would
# score them as "a quarter of an NHL game"; measured, they are zero.
EVENT_LOAD_TAU_KM = 4.0        # distance decay constant, km (sweep 2-8 in ablation)
EVENT_LOAD_REF = 20_000        # attendance normaliser, so a full arena is ~1.0
EVENT_LOAD_MIN_ATTENDANCE = 15_000
EVENT_LOAD_DISTANCE = "haversine"   # "haversine" | "road" (distances_bay_2017.csv)

# Out-of-network venues decay to ~0 on their own (exp(-39/4) ~ 1e-5), but they
# are excluded at collection time anyway; venues.csv holds only the four venues
# that actually contribute an event.

# --- Windowing ---
INPUT_WINDOW = 12       # 12 steps = 60 min of history  -> X
HORIZON = 12            # 12 steps = 60 min ahead       -> Y
EVAL_HORIZON_STEPS = {"15min": 3, "30min": 6, "60min": 12}

# Samples whose X or Y straddles a split boundary leak the future across it.
EMBARGO = INPUT_WINDOW + HORIZON        # 24 steps, dropped on EACH side

# --- Split ------------------------------------------------------------------
# Chronological, NO shuffling, in BOTH modes.
SPLIT_MODE = "rolling"          # "single" | "rolling"

# "single": the classic PEMS-BAY benchmark split. Use this, and ONLY this, when
# reproducing published vanilla-STGCN numbers so the comparison is like for
# like. Do NOT use it for any weather result: its test period
# (2017-05-25 -> 06-30) contains ZERO adverse-weather timesteps, so a
# weather-conditioned model cannot differ from a traffic-only one there - the
# channel is constant across the whole test set.
SPLIT = {"train": 0.70, "val": 0.10, "test": 0.20}

# "rolling": expanding-window backtesting. Each fold trains strictly BEFORE it
# tests, so there is still no future leakage, but the wet season lands in test
# folds instead of being locked inside train.
#
# Fold COUNT does not change how much signal reaches test - the test span is
# always the same (181 - min_train_days) days, so every scheme sees the same
# episodes. It trades compute against per-fold density. The unit that matters is
# the EPISODE, not the timestep: the period holds 90 rain spells, 60 qualifying
# egress hours and 4 holidays, however many 5-min steps those cover.
#     block  folds  folds with rain  median rain spells per fold
#        7d    19          9/19          0     <- useless
#       14d     9          5/9           4
#       21d     6          4/6           6
#       28d     6          4/6          10     <- chosen
# min_train_days is 28, not 42: January is the wettest month of the period
# (39 of the 90 rain spells) and a 42-day warm-up locks most of it out of test
# entirely - 37 spells reach test at 42 days against 51 at 28. The cost is that
# fold 1 trains on 8,064 samples instead of 12,096. That does not confound the
# headline, because the comparison between rungs is PAIRED WITHIN A FOLD: both
# rungs see the same 8,064 samples, so a short train span moves the absolute MAE
# of both, not the difference between them. Report per-fold results and say
# which fold has the shortest train span.
ROLLING = {
    "min_train_days": 28,        # 2017-01-01 .. 01-29 warm-up before fold 1
    "test_block_days": 28,       # 4 whole weeks: every weekday appears 4 times
    "val_days": 7,               # carved off the END of each fold's train span
    "n_folds": 6,                # covers 2017-01-29 .. 06-30
}

# --- Normalization ----------------------------------------------------------
# PER CHANNEL, Z-score, fit on TRAIN ONLY, re-fit per fold under "rolling".
# One global scalar over the whole tensor - which is what the stage-8 notebook
# does, correctly, for its single speed channel - would here average mph
# together with a 0/1 holiday flag, kilometres and millimetres.
NORMALIZE_CHANNELS = [
    "speed", "occupancy", "temperature", "precipitation", "wind_speed",
    "dist_to_venue", "event_active_decay", "event_load",
]
# Z-scoring a 0/1 flag or a cyclical index is meaningless, and it makes "not a
# holiday" a non-zero value.
PASSTHROUGH_CHANNELS = ["time_of_day", "day_of_week", "is_holiday"]

# --- Missing values ---------------------------------------------------------
# The AUGMENTED release leaves gaps BLANK (NaN); the original DCRNN release
# zero-filled them. Measured on the augmented file: 521 / 16,937,700 = 0.0031%,
# so the proposal's "0.003%" holds. What differs is the REPRESENTATION: NaN, not
# 0. Anything keying off zero-valued speeds must be rewritten against NaN.
# Inputs are imputed so the model sees no NaN; evaluation masks the ORIGINAL
# missing positions so imputed values never enter a metric.
MISSING_POLICY = {
    "impute": "time_interpolate_per_sensor",   # linear in time, per sensor
    "max_gap_steps": 12,                       # longer gaps stay NaN -> masked
    "eval": "mask_original_nans",
    "mask_file": "data/processed/missing_mask.npy",   # [T, N] bool, True = missing
}

# --- Weekends and holidays are KEPT -----------------------------------------
# Yu et al. (IJCAI 2018) state: "In order to eliminate atypical traffic, only
# workday traffic data are adopted in our experiment." We do the opposite, on
# purpose. 46 of the 95 scheduled events (48%), and 33 of the 67 that clear the
# attendance floor, fall on a Saturday or Sunday; Memorial Day is both a holiday
# and the largest single exogenous effect measured. The paper's exclusion
# removes precisely the windows this project exists to evaluate.
DROP_WEEKENDS = False

# --- Canonical tensor shapes ---
#   master tensor : [N_TIMESTEPS, N_NODES, N_CHANNELS]
#   sample X      : [INPUT_WINDOW, N_NODES, N_CHANNELS]   (or a rung's subset)
#   sample Y      : [HORIZON, N_NODES, 1]                 (speed only)
#   predictions   : [n_test_samples, HORIZON, N_NODES]  DE-NORMALIZED speed (mph)
PREDICTION_DTYPE = "float32"

# --- Evaluation windows -----------------------------------------------------
# Defined in src/eval/windows.py, which owns the thresholds and the measurements
# behind them. Named here so nothing re-invents them:
#   weather   precipitation >= 0.51 mm      (p75 of wet steps; -2.22 mph)
#   events    [end_time, end_time + 60 min) for attendance >= 15,000
#   holidays  5 US federal holidays in the period; reported, never merged into
#             "adverse", because the sign is POSITIVE
EVAL_WINDOWS_MODULE = "src.eval.windows"

# Egress is where the shockwave is: the crowd leaves AFTER end_time. Aligned on
# scheduled START over 24 SAP Center evening NHL games (sensors within 2 km) the
# dip is -1.69 +- 0.51 mph at +180 min and has recovered by +270. A window
# spanning the whole fixture (doors to end, ~5.5 h) dilutes that about fivefold
# and reads +0.41 mph, which is how it was missed on the first pass.
EVENT_PHASES = {
    "ingress": (-90, 0),      # minutes relative to start_time
    "in_play": (0, None),     # start_time -> end_time
    "egress":  (0, +60),      # minutes relative to END_time  <- the shockwave
}
# end_time is a per-sport convention, not an observed finish. It is nonetheless
# CORRECT for the NHL rows: sweeping duration_min and scoring the mean anomaly
# in [end, end+60) gives 120min -0.84, 135 -1.08, 150 -1.22, 165 -1.20,
# 180 -0.95, 210 -0.41. The configured 150 is the optimum.
EVENT_END_IS_NOMINAL = True

# Congestion onset. Reuse the augmented release's labelled congestion blocks
# where they exist and fall back to this threshold rule elsewhere.
# CAUTION: the labels cover 2017-05-25 17:50 -> 06-30 23:55 ONLY - the last 36
# days. Under SPLIT_MODE "rolling" that is inside the final fold and nowhere
# else, so onset metrics are not available for folds 1-5.
ONSET = {
    "speed_mph": 45.0,        # below this ...
    "persist_steps": 3,       # ... for 15 min = congestion has begun
    "labels": "data/raw/augmented-pems-bay/data/traffic_data/congestion_blocks_mask.csv",
    "labels_cover": ("2017-05-25 17:50", "2017-06-30 23:55"),
}

# --- Statistics -------------------------------------------------------------
# The INDEPENDENT unit is the episode, not the node-timestep: neighbouring 5-min
# steps and neighbouring sensors are heavily correlated, so a standard error
# over node-timesteps understates it by ~sqrt(n_node_timesteps / n_episodes).
# Report paired per-EPISODE differences and quote n = number of episodes.
EFFECTIVE_N_UNIT = "episode"      # one event, or one contiguous rain spell
PAIRED_TEST = "wilcoxon"          # paired, non-parametric, over episodes

# --- Reproducibility ---
SEED = 42
N_SEEDS = 3
