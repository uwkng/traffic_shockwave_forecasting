# Stratified results - split `fold02`

## Aggregate (the number the field reports)

| model | 15min | 30min | 45min |
|---|---|---|---|
| `historical_average` | 2.506 | 2.506 | 2.506 |
| `persistence` | 1.548 | 2.099 | 2.535 |

MAE in mph, mean±sd over seeds.

## Window: `adverse_weather`

12 episodes, 460 test samples (5.7% of the split).

| model | 15min in | 15min out | 30min in | 30min out | 45min in | 45min out |
|---|---|---|---|---|---|---|
| `historical_average` | 2.337 | 2.516 | 2.287 | 2.519 | 2.229 | 2.522 |
| `persistence` | 1.473 | 1.553 | 1.988 | 2.106 | 2.382 | 2.544 |

n = 12 episodes, not 460 samples: neighbouring 5-min steps and sensors are correlated, so any error bar must be quoted per episode.

## Window: `weather_commute`

2 episodes, 35 test samples (0.43% of the split).

| model | 15min in | 15min out | 30min in | 30min out | 45min in | 45min out |
|---|---|---|---|---|---|---|
| `historical_average` | 5.944 | 2.491 | 5.745 | 2.491 | 5.434 | 2.493 |
| `persistence` | 3.045 | 1.541 | 5.272 | 2.085 | 7.360 | 2.514 |

n = 2 episodes, not 35 samples: neighbouring 5-min steps and sensors are correlated, so any error bar must be quoted per episode.

## Window: `event_egress`

8 episodes, 184 test samples (2.28% of the split).

| model | 15min in | 15min out | 30min in | 30min out | 45min in | 45min out |
|---|---|---|---|---|---|---|
| `historical_average` | 1.613 | 2.526 | 1.575 | 2.527 | 1.535 | 2.528 |
| `persistence` | 0.995 | 1.561 | 1.174 | 2.121 | 1.240 | 2.565 |

n = 8 episodes, not 184 samples: neighbouring 5-min steps and sensors are correlated, so any error bar must be quoted per episode.

## Window: `holiday`

0 episodes, 0 test samples (0.0% of the split).

**This window is empty in this split.** Nothing can be concluded from it here; use the rolling folds.

## MAE by distance from the nearest venue

If the error grows towards a venue during egress but not otherwise, the error is spatially anchored to the crowd. If it is flat, it is not.

| model | window | 0-1km | 1-2km | 2-5km | 5-10km | 10-20km | 20-50km |
|---|---|---|---|---|---|---|---|
| `historical_average` | all | 2.379 | 2.599 | 2.626 | 2.183 | 2.593 | 2.524 |
| `historical_average` | egress | 1.186 | 1.801 | 1.762 | 1.409 | 1.584 | 1.568 |
| `persistence` | all | 2.114 | 2.022 | 2.237 | 1.884 | 2.204 | 2.070 |
| `persistence` | egress | 1.080 | 1.166 | 1.401 | 0.988 | 1.170 | 1.055 |

## Congestion onset: lead time over a reactive rule

The reactive baseline acts once congestion is OBSERVED (< 45.0 mph for 15 min); the model acts when it PREDICTS that. Positive lead means the model was earlier. Sensors within 2 km of a venue.

A median lead of 0 for the trivial baselines is expected and is the control: persistence copies the last observation, so it cannot see an onset before it happens, and the historical average has no idea which day it is. Any positive lead a trained model shows is measured against this floor.

| model | onsets | detected | recall | false alarms | lead (median, min) |
|---|---|---|---|---|---|
| `historical_average` | 12,501 | 10,436 | 0.83 | 5,420 | 0.0 |
| `persistence` | 12,501 | 7,996 | 0.64 | 1,448 | 0.0 |
