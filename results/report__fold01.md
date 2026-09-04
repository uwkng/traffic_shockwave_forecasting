# Stratified results - split `fold01`

## Aggregate (the number the field reports)

| model | 15min | 30min | 45min |
|---|---|---|---|
| `historical_average` | 2.885 | 2.884 | 2.884 |
| `persistence` | 1.635 | 2.272 | 2.783 |

MAE in mph, mean±sd over seeds.

## Window: `adverse_weather`

12 episodes, 460 test samples (5.73% of the split).

| model | 15min in | 15min out | 30min in | 30min out | 45min in | 45min out |
|---|---|---|---|---|---|---|
| `historical_average` | 3.251 | 2.862 | 3.278 | 2.860 | 3.282 | 2.860 |
| `persistence` | 1.844 | 1.623 | 2.652 | 2.249 | 3.307 | 2.752 |

n = 12 episodes, not 460 samples: neighbouring 5-min steps and sensors are correlated, so any error bar must be quoted per episode.

## Window: `weather_commute`

4 episodes, 115 test samples (1.43% of the split).

| model | 15min in | 15min out | 30min in | 30min out | 45min in | 45min out |
|---|---|---|---|---|---|---|
| `historical_average` | 6.491 | 2.832 | 6.638 | 2.830 | 6.649 | 2.830 |
| `persistence` | 3.114 | 1.614 | 4.851 | 2.234 | 6.169 | 2.734 |

n = 4 episodes, not 115 samples: neighbouring 5-min steps and sensors are correlated, so any error bar must be quoted per episode.

## Window: `event_egress`

10 episodes, 230 test samples (2.86% of the split).

| model | 15min in | 15min out | 30min in | 30min out | 45min in | 45min out |
|---|---|---|---|---|---|---|
| `historical_average` | 2.120 | 2.907 | 2.130 | 2.907 | 2.143 | 2.906 |
| `persistence` | 1.073 | 1.652 | 1.332 | 2.299 | 1.477 | 2.822 |

n = 10 episodes, not 230 samples: neighbouring 5-min steps and sensors are correlated, so any error bar must be quoted per episode.

## Window: `holiday`

0 episodes, 0 test samples (0.0% of the split).

**This window is empty in this split.** Nothing can be concluded from it here; use the rolling folds.

## MAE by distance from the nearest venue

If the error grows towards a venue during egress but not otherwise, the error is spatially anchored to the crowd. If it is flat, it is not.

| model | window | 0-1km | 1-2km | 2-5km | 5-10km | 10-20km | 20-50km |
|---|---|---|---|---|---|---|---|
| `historical_average` | all | 2.742 | 2.953 | 3.281 | 2.475 | 2.932 | 2.913 |
| `historical_average` | egress | 1.432 | 2.395 | 2.914 | 1.873 | 2.062 | 2.143 |
| `persistence` | all | 2.338 | 2.344 | 2.342 | 2.084 | 2.333 | 2.271 |
| `persistence` | egress | 1.103 | 1.663 | 1.734 | 1.263 | 1.299 | 1.156 |

## Congestion onset: lead time over a reactive rule

The reactive baseline acts once congestion is OBSERVED (< 45.0 mph for 15 min); the model acts when it PREDICTS that. Positive lead means the model was earlier. Sensors within 2 km of a venue.

A median lead of 0 for the trivial baselines is expected and is the control: persistence copies the last observation, so it cannot see an onset before it happens, and the historical average has no idea which day it is. Any positive lead a trained model shows is measured against this floor.

| model | onsets | detected | recall | false alarms | lead (median, min) |
|---|---|---|---|---|---|
| `historical_average` | 17,281 | 12,553 | 0.73 | 4,391 | 0.0 |
| `persistence` | 17,281 | 11,308 | 0.65 | 1,975 | 0.0 |
