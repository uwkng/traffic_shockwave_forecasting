"""
DATA CONTRACT - the single source of truth for shapes, channel order, splits,
and the prediction format. Every module in the repo imports from here.

Rule: do NOT change anything in this file without updating the team. All three
workstreams (data / models / eval) build against these constants, so a silent
change here breaks everyone downstream.
"""

# --- Graph / time dimensions (PEMS-BAY augmented release) ---
N_NODES = 325           # freeway loop detectors
N_TIMESTEPS = 52116     # 5-min steps, 1 Jan - 30 Jun 2017
FREQ_MIN = 5            # sampling interval, minutes

# --- The 9 input channels, in this exact order ---
# variation: "spatiotemporal" = varies over time AND node
#            "temporal"        = varies over time only, broadcast to all nodes
#            "spatial"         = varies over node only, constant over time
CHANNELS = [
    ("speed",            "spatiotemporal"),  # 0  <- also the prediction TARGET
    ("time_of_day",      "temporal"),        # 1  encode sin/cos or 0..1
    ("day_of_week",      "temporal"),        # 2
    ("precipitation",    "temporal"),        # 3  single centroid, broadcast
    ("temperature",      "temporal"),        # 4
    ("wind_speed",       "temporal"),        # 5
    ("event_active",     "temporal"),        # 6  0/1 flag
    ("event_magnitude",  "temporal"),        # 7  attendance-scaled
    ("dist_to_venue",    "spatial"),         # 8  haversine, LOAD-BEARING
]
N_CHANNELS = len(CHANNELS)                    # 9
TARGET_CHANNEL = 0                            # speed

# --- Windowing ---
INPUT_WINDOW = 12       # 12 steps = 60 min of history  -> X
HORIZON = 12            # 12 steps = 60 min ahead       -> Y
# horizons we report at, in step index (1-based conceptually):
EVAL_HORIZON_STEPS = {"15min": 3, "30min": 6, "60min": 12}

# --- Split (chronological, NO shuffling) ---
SPLIT = {"train": 0.70, "val": 0.10, "test": 0.20}

# --- Canonical tensor shapes ---
#   master tensor : [N_TIMESTEPS, N_NODES, N_CHANNELS]
#   sample X      : [INPUT_WINDOW, N_NODES, N_CHANNELS]
#   sample Y      : [HORIZON, N_NODES, 1]           (speed only)
#   predictions   : [n_test_samples, HORIZON, N_NODES]  DE-NORMALIZED speed (mph)
#
# Every model writes predictions in this identical format so the eval suite can
# treat baselines / STGCN / DCRNN interchangeably.
PREDICTION_DTYPE = "float32"