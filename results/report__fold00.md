# Stratified results - split `fold00`

## Aggregate (the number the field reports)

| model | 15min | 30min | 45min |
|---|---|---|---|
| `historical_average` | 2.937 | 2.937 | 2.937 |
| `persistence` | 1.520 | 2.099 | 2.566 |

MAE in mph, mean±sd over seeds.

## Window: `adverse_weather`

27 episodes, 1,155 test samples (14.32% of the split).

| model | 15min in | 15min out | 30min in | 30min out | 45min in | 45min out |
|---|---|---|---|---|---|---|
| `historical_average` | 3.767 | 2.798 | 3.762 | 2.799 | 3.754 | 2.801 |
| `persistence` | 1.611 | 1.505 | 2.284 | 2.069 | 2.834 | 2.521 |

n = 27 episodes, not 1,155 samples: neighbouring 5-min steps and sensors are correlated, so any error bar must be quoted per episode.

## Window: `weather_commute`

10 episodes, 341 test samples (4.23% of the split).

| model | 15min in | 15min out | 30min in | 30min out | 45min in | 45min out |
|---|---|---|---|---|---|---|
| `historical_average` | 6.305 | 2.788 | 6.373 | 2.785 | 6.388 | 2.785 |
| `persistence` | 2.280 | 1.487 | 3.437 | 2.040 | 4.361 | 2.487 |

n = 10 episodes, not 341 samples: neighbouring 5-min steps and sensors are correlated, so any error bar must be quoted per episode.

## Window: `event_egress`

4 episodes, 92 test samples (1.14% of the split).

| model | 15min in | 15min out | 30min in | 30min out | 45min in | 45min out |
|---|---|---|---|---|---|---|
| `historical_average` | 2.139 | 2.946 | 2.047 | 2.947 | 1.955 | 2.948 |
| `persistence` | 0.992 | 1.527 | 1.180 | 2.110 | 1.270 | 2.581 |

n = 4 episodes, not 92 samples: neighbouring 5-min steps and sensors are correlated, so any error bar must be quoted per episode.

## Window: `holiday`

1 episodes, 299 test samples (3.71% of the split).

| model | 15min in | 15min out | 30min in | 30min out | 45min in | 45min out |
|---|---|---|---|---|---|---|
| `historical_average` | 3.788 | 2.904 | 3.787 | 2.904 | 3.786 | 2.904 |
| `persistence` | 1.264 | 1.530 | 1.652 | 2.117 | 1.930 | 2.590 |

n = 1 episodes, not 299 samples: neighbouring 5-min steps and sensors are correlated, so any error bar must be quoted per episode.

## MAE by distance from the nearest venue

If the error grows towards a venue during egress but not otherwise, the error is spatially anchored to the crowd. If it is flat, it is not.

| model | window | 0-1km | 1-2km | 2-5km | 5-10km | 10-20km | 20-50km |
|---|---|---|---|---|---|---|---|
| `historical_average` | all | 2.767 | 3.001 | 3.311 | 2.620 | 3.023 | 2.910 |
| `historical_average` | egress | 1.744 | 2.625 | 2.510 | 1.979 | 2.074 | 1.844 |
| `persistence` | all | 2.141 | 2.174 | 2.186 | 1.851 | 2.203 | 2.088 |
| `persistence` | egress | 1.059 | 1.450 | 1.264 | 0.977 | 1.205 | 1.079 |

## Congestion onset: lead time over a reactive rule

The reactive baseline acts once congestion is OBSERVED (< 45.0 mph for 15 min); the model acts when it PREDICTS that. Positive lead means the model was earlier. Sensors within 2 km of a venue.

A median lead of 0 for the trivial baselines is expected and is the control: persistence copies the last observation, so it cannot see an onset before it happens, and the historical average has no idea which day it is. Any positive lead a trained model shows is measured against this floor.

| model | onsets | detected | recall | false alarms | lead (median, min) |
|---|---|---|---|---|---|
| `historical_average` | 16,743 | 12,513 | 0.75 | 4,507 | 0.0 |
| `persistence` | 16,743 | 11,204 | 0.67 | 1,822 | 0.0 |
