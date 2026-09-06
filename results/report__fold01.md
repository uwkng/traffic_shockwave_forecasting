# Stratified results - split `fold01`

## Aggregate (the number the field reports)

| model | 15min | 30min | 45min |
|---|---|---|---|
| `0_speed` | 1.464±0.004 | 1.985±0.005 | 2.372±0.010 |
| `0_speed__seed42__diffusion` | 1.457 | 1.975 | 2.359 |
| `0_speed__seed42__weighted` | 1.481 | 2.025 | 2.428 |
| `0_speed__seed43__diffusion` | 1.459 | 1.984 | 2.372 |
| `0_speed__seed43__weighted` | 1.499 | 2.058 | 2.497 |
| `0_speed__seed44__diffusion` | 1.470 | 1.988 | 2.378 |
| `0_speed__seed44__weighted` | 1.488 | 2.033 | 2.450 |
| `1_traffic` | 1.441±0.004 | 1.930±0.007 | 2.274±0.012 |
| `3_weather` | 1.464±0.002 | 1.972±0.009 | 2.344±0.016 |
| `5_event_geo_att` | 1.446±0.003 | 1.910±0.005 | 2.226±0.012 |
| `6_all` | 1.453±0.005 | 1.908±0.006 | 2.210±0.014 |
| `6_all__seed42__diffusion` | 1.443 | 1.892 | 2.188 |
| `6_all__seed42__weighted` | 1.462 | 1.920 | 2.219 |
| `6_all__seed43__diffusion` | 1.444 | 1.901 | 2.197 |
| `6_all__seed43__weighted` | 1.449 | 1.889 | 2.166 |
| `6_all__seed44__diffusion` | 1.442 | 1.903 | 2.211 |
| `6_all__seed44__weighted` | 1.466 | 1.910 | 2.188 |
| `historical_average` | 2.885 | 2.884 | 2.884 |
| `persistence` | 1.635 | 2.272 | 2.783 |

MAE in mph, mean±sd over seeds.

## Window: `adverse_weather`

12 episodes, 460 test samples (5.73% of the split).

| model | 15min in | 15min out | 30min in | 30min out | 45min in | 45min out |
|---|---|---|---|---|---|---|
| `0_speed` | 1.647±0.006 | 1.453±0.004 | 2.344±0.008 | 1.963±0.005 | 2.884±0.015 | 2.341±0.010 |
| `0_speed__seed42__diffusion` | 1.635 | 1.447 | 2.326 | 1.953 | 2.859 | 2.329 |
| `0_speed__seed42__weighted` | 1.663 | 1.470 | 2.385 | 2.003 | 2.931 | 2.397 |
| `0_speed__seed43__diffusion` | 1.641 | 1.448 | 2.342 | 1.962 | 2.880 | 2.341 |
| `0_speed__seed43__weighted` | 1.687 | 1.487 | 2.425 | 2.036 | 3.013 | 2.466 |
| `0_speed__seed44__diffusion` | 1.643 | 1.459 | 2.345 | 1.966 | 2.898 | 2.346 |
| `0_speed__seed44__weighted` | 1.670 | 1.477 | 2.400 | 2.011 | 2.968 | 2.418 |
| `1_traffic` | 1.614±0.006 | 1.431±0.003 | 2.256±0.010 | 1.910±0.007 | 2.720±0.019 | 2.247±0.013 |
| `3_weather` | 1.666±0.005 | 1.452±0.002 | 2.311±0.011 | 1.952±0.009 | 2.783±0.025 | 2.317±0.017 |
| `5_event_geo_att` | 1.624±0.004 | 1.435±0.003 | 2.244±0.007 | 1.889±0.005 | 2.672±0.014 | 2.199±0.013 |
| `6_all` | 1.664±0.021 | 1.440±0.006 | 2.308±0.045 | 1.884±0.009 | 2.764±0.059 | 2.176±0.017 |
| `6_all__seed42__diffusion` | 1.641 | 1.431 | 2.296 | 1.868 | 2.753 | 2.153 |
| `6_all__seed42__weighted` | 1.691 | 1.448 | 2.357 | 1.893 | 2.850 | 2.181 |
| `6_all__seed43__diffusion` | 1.640 | 1.433 | 2.257 | 1.880 | 2.678 | 2.168 |
| `6_all__seed43__weighted` | 1.668 | 1.436 | 2.274 | 1.865 | 2.671 | 2.136 |
| `6_all__seed44__diffusion` | 1.630 | 1.431 | 2.249 | 1.882 | 2.684 | 2.182 |
| `6_all__seed44__weighted` | 1.687 | 1.453 | 2.285 | 1.887 | 2.677 | 2.158 |
| `historical_average` | 3.251 | 2.862 | 3.278 | 2.860 | 3.282 | 2.860 |
| `persistence` | 1.844 | 1.623 | 2.652 | 2.249 | 3.307 | 2.752 |

n = 12 episodes, not 460 samples: neighbouring 5-min steps and sensors are correlated, so any error bar must be quoted per episode.

## Window: `weather_commute`

4 episodes, 115 test samples (1.43% of the split).

| model | 15min in | 15min out | 30min in | 30min out | 45min in | 45min out |
|---|---|---|---|---|---|---|
| `0_speed` | 2.850±0.026 | 1.444±0.003 | 4.455±0.034 | 1.949±0.005 | 5.644±0.050 | 2.324±0.010 |
| `0_speed__seed42__diffusion` | 2.817 | 1.438 | 4.402 | 1.939 | 5.562 | 2.312 |
| `0_speed__seed42__weighted` | 2.859 | 1.461 | 4.487 | 1.989 | 5.671 | 2.380 |
| `0_speed__seed43__diffusion` | 2.834 | 1.439 | 4.445 | 1.948 | 5.620 | 2.325 |
| `0_speed__seed43__weighted` | 2.903 | 1.479 | 4.572 | 2.021 | 5.821 | 2.449 |
| `0_speed__seed44__diffusion` | 2.839 | 1.450 | 4.460 | 1.952 | 5.683 | 2.330 |
| `0_speed__seed44__weighted` | 2.895 | 1.468 | 4.571 | 1.996 | 5.815 | 2.401 |
| `1_traffic` | 2.792±0.013 | 1.422±0.003 | 4.324±0.025 | 1.895±0.007 | 5.401±0.049 | 2.229±0.012 |
| `3_weather` | 2.843±0.017 | 1.444±0.002 | 4.408±0.036 | 1.937±0.008 | 5.534±0.046 | 2.297±0.016 |
| `5_event_geo_att` | 2.818±0.011 | 1.426±0.003 | 4.290±0.028 | 1.875±0.005 | 5.263±0.056 | 2.182±0.013 |
| `6_all` | 2.823±0.004 | 1.433±0.005 | 4.307±0.009 | 1.873±0.006 | 5.286±0.036 | 2.165±0.013 |
| `6_all__seed42__diffusion` | 2.823 | 1.423 | 4.374 | 1.856 | 5.402 | 2.141 |
| `6_all__seed42__weighted` | 2.892 | 1.441 | 4.450 | 1.883 | 5.465 | 2.172 |
| `6_all__seed43__diffusion` | 2.788 | 1.425 | 4.237 | 1.867 | 5.170 | 2.154 |
| `6_all__seed43__weighted` | 2.846 | 1.429 | 4.318 | 1.854 | 5.253 | 2.121 |
| `6_all__seed44__diffusion` | 2.773 | 1.423 | 4.240 | 1.869 | 5.224 | 2.167 |
| `6_all__seed44__weighted` | 2.799 | 1.447 | 4.267 | 1.876 | 5.182 | 2.144 |
| `historical_average` | 6.491 | 2.832 | 6.638 | 2.830 | 6.649 | 2.830 |
| `persistence` | 3.114 | 1.614 | 4.851 | 2.234 | 6.169 | 2.734 |

n = 4 episodes, not 115 samples: neighbouring 5-min steps and sensors are correlated, so any error bar must be quoted per episode.

## Window: `event_egress`

10 episodes, 230 test samples (2.86% of the split).

| model | 15min in | 15min out | 30min in | 30min out | 45min in | 45min out |
|---|---|---|---|---|---|---|
| `0_speed` | 1.023±0.006 | 1.477±0.004 | 1.268±0.005 | 2.006±0.005 | 1.422±0.011 | 2.400±0.010 |
| `0_speed__seed42__diffusion` | 1.019 | 1.470 | 1.266 | 1.996 | 1.432 | 2.386 |
| `0_speed__seed42__weighted` | 1.034 | 1.494 | 1.301 | 2.046 | 1.480 | 2.455 |
| `0_speed__seed43__diffusion` | 1.019 | 1.472 | 1.261 | 2.005 | 1.411 | 2.400 |
| `0_speed__seed43__weighted` | 1.042 | 1.512 | 1.287 | 2.081 | 1.464 | 2.528 |
| `0_speed__seed44__diffusion` | 1.023 | 1.483 | 1.270 | 2.009 | 1.426 | 2.406 |
| `0_speed__seed44__weighted` | 1.046 | 1.501 | 1.308 | 2.054 | 1.486 | 2.478 |
| `1_traffic` | 0.995±0.001 | 1.454±0.004 | 1.218±0.004 | 1.951±0.007 | 1.356±0.007 | 2.302±0.013 |
| `3_weather` | 1.003±0.001 | 1.478±0.002 | 1.231±0.010 | 1.994±0.009 | 1.379±0.021 | 2.372±0.016 |
| `5_event_geo_att` | 1.019±0.003 | 1.458±0.003 | 1.233±0.006 | 1.930±0.005 | 1.361±0.003 | 2.252±0.012 |
| `6_all` | 1.024±0.008 | 1.466±0.005 | 1.244±0.009 | 1.928±0.006 | 1.384±0.012 | 2.234±0.014 |
| `6_all__seed42__diffusion` | 1.024 | 1.455 | 1.238 | 1.912 | 1.369 | 2.212 |
| `6_all__seed42__weighted` | 1.028 | 1.475 | 1.250 | 1.940 | 1.389 | 2.244 |
| `6_all__seed43__diffusion` | 1.019 | 1.457 | 1.250 | 1.920 | 1.393 | 2.221 |
| `6_all__seed43__weighted` | 1.024 | 1.462 | 1.243 | 1.908 | 1.374 | 2.190 |
| `6_all__seed44__diffusion` | 1.020 | 1.455 | 1.243 | 1.923 | 1.389 | 2.235 |
| `6_all__seed44__weighted` | 1.034 | 1.479 | 1.253 | 1.929 | 1.387 | 2.211 |
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
| `0_speed` | all | 2.005 | 1.998 | 2.086 | 1.807 | 1.980 | 1.978 |
| `0_speed` | egress | 1.031 | 1.639 | 1.676 | 1.227 | 1.196 | 1.135 |
| `0_speed__seed42__diffusion` | all | 1.993 | 1.964 | 2.084 | 1.816 | 1.979 | 1.962 |
| `0_speed__seed42__diffusion` | egress | 1.022 | 1.611 | 1.700 | 1.228 | 1.222 | 1.133 |
| `0_speed__seed42__weighted` | all | 2.037 | 2.044 | 2.129 | 1.857 | 2.018 | 2.022 |
| `0_speed__seed42__weighted` | egress | 1.035 | 1.659 | 1.696 | 1.278 | 1.216 | 1.196 |
| `0_speed__seed43__diffusion` | all | 2.023 | 1.980 | 2.103 | 1.826 | 1.985 | 1.967 |
| `0_speed__seed43__diffusion` | egress | 1.024 | 1.626 | 1.690 | 1.230 | 1.208 | 1.116 |
| `0_speed__seed43__weighted` | all | 2.108 | 2.123 | 2.182 | 1.925 | 2.070 | 2.050 |
| `0_speed__seed43__weighted` | egress | 1.063 | 1.686 | 1.754 | 1.302 | 1.225 | 1.156 |
| `0_speed__seed44__diffusion` | all | 2.011 | 1.969 | 2.104 | 1.825 | 1.990 | 1.986 |
| `0_speed__seed44__diffusion` | egress | 1.021 | 1.580 | 1.694 | 1.231 | 1.225 | 1.134 |
| `0_speed__seed44__weighted` | all | 2.047 | 2.055 | 2.149 | 1.885 | 2.043 | 2.027 |
| `0_speed__seed44__weighted` | egress | 1.054 | 1.728 | 1.738 | 1.295 | 1.240 | 1.180 |
| `1_traffic` | all | 1.967 | 1.921 | 2.020 | 1.734 | 1.900 | 1.897 |
| `1_traffic` | egress | 1.046 | 1.520 | 1.620 | 1.150 | 1.189 | 1.076 |
| `3_weather` | all | 2.048 | 1.975 | 2.085 | 1.795 | 1.957 | 1.943 |
| `3_weather` | egress | 1.042 | 1.574 | 1.647 | 1.189 | 1.192 | 1.097 |
| `5_event_geo_att` | all | 1.929 | 1.915 | 1.994 | 1.704 | 1.902 | 1.908 |
| `5_event_geo_att` | egress | 1.025 | 1.596 | 1.719 | 1.138 | 1.189 | 1.083 |
| `6_all` | all | 1.961 | 1.900 | 2.025 | 1.699 | 1.891 | 1.854 |
| `6_all` | egress | 1.019 | 1.592 | 1.706 | 1.143 | 1.191 | 1.087 |
| `6_all__seed42__diffusion` | all | 1.993 | 1.873 | 1.960 | 1.681 | 1.866 | 1.860 |
| `6_all__seed42__diffusion` | egress | 1.046 | 1.596 | 1.737 | 1.144 | 1.189 | 1.100 |
| `6_all__seed42__weighted` | all | 2.032 | 1.943 | 2.017 | 1.707 | 1.901 | 1.879 |
| `6_all__seed42__weighted` | egress | 1.045 | 1.620 | 1.706 | 1.173 | 1.207 | 1.112 |
| `6_all__seed43__diffusion` | all | 1.947 | 1.901 | 2.000 | 1.700 | 1.877 | 1.861 |
| `6_all__seed43__diffusion` | egress | 1.066 | 1.698 | 1.753 | 1.157 | 1.200 | 1.108 |
| `6_all__seed43__weighted` | all | 1.926 | 1.902 | 1.958 | 1.680 | 1.884 | 1.833 |
| `6_all__seed43__weighted` | egress | 1.021 | 1.617 | 1.714 | 1.134 | 1.194 | 1.113 |
| `6_all__seed44__diffusion` | all | 2.002 | 1.895 | 1.995 | 1.698 | 1.882 | 1.870 |
| `6_all__seed44__diffusion` | egress | 1.021 | 1.625 | 1.773 | 1.143 | 1.205 | 1.100 |
| `6_all__seed44__weighted` | all | 1.931 | 1.912 | 2.016 | 1.678 | 1.893 | 1.867 |
| `6_all__seed44__weighted` | egress | 1.035 | 1.633 | 1.727 | 1.159 | 1.213 | 1.115 |
| `historical_average` | all | 2.742 | 2.953 | 3.281 | 2.475 | 2.932 | 2.913 |
| `historical_average` | egress | 1.432 | 2.395 | 2.914 | 1.873 | 2.062 | 2.143 |
| `persistence` | all | 2.338 | 2.344 | 2.342 | 2.084 | 2.333 | 2.271 |
| `persistence` | egress | 1.103 | 1.663 | 1.734 | 1.263 | 1.299 | 1.156 |

## Congestion onset: lead time over a reactive rule

The reactive baseline acts once congestion is OBSERVED (< 45.0 mph for 15 min); the model acts when it PREDICTS that. Positive lead means the model was earlier. Sensors within 2 km of a venue.

A median lead of 0 for the trivial baselines is expected and is the control: persistence copies the last observation, so it cannot see an onset before it happens, and the historical average has no idea which day it is. Any positive lead a trained model shows is measured against this floor.

| model | onsets | detected | recall | false alarms | lead (median, min) |
|---|---|---|---|---|---|
| `0_speed` | 17,281 | 11,708 | 0.68 | 1,487 | 0.0 |
| `0_speed__seed42__diffusion` | 17,281 | 12,273 | 0.71 | 1,740 | 0.0 |
| `0_speed__seed42__weighted` | 17,281 | 13,271 | 0.77 | 3,166 | 0.0 |
| `0_speed__seed43__diffusion` | 17,281 | 12,111 | 0.70 | 1,779 | 0.0 |
| `0_speed__seed43__weighted` | 17,281 | 12,594 | 0.73 | 2,859 | 0.0 |
| `0_speed__seed44__diffusion` | 17,281 | 12,124 | 0.70 | 1,663 | 0.0 |
| `0_speed__seed44__weighted` | 17,281 | 12,743 | 0.74 | 2,722 | 0.0 |
| `1_traffic` | 17,281 | 12,318 | 0.71 | 1,961 | 0.0 |
| `3_weather` | 17,281 | 12,477 | 0.72 | 2,286 | 0.0 |
| `5_event_geo_att` | 17,281 | 12,385 | 0.72 | 1,851 | 0.0 |
| `6_all` | 17,281 | 12,364 | 0.72 | 1,753 | 0.0 |
| `6_all__seed42__diffusion` | 17,281 | 12,746 | 0.74 | 2,170 | 0.0 |
| `6_all__seed42__weighted` | 17,281 | 13,888 | 0.80 | 3,745 | 0.0 |
| `6_all__seed43__diffusion` | 17,281 | 12,538 | 0.73 | 1,891 | 0.0 |
| `6_all__seed43__weighted` | 17,281 | 13,208 | 0.76 | 2,275 | 0.0 |
| `6_all__seed44__diffusion` | 17,281 | 12,195 | 0.71 | 1,691 | 0.0 |
| `6_all__seed44__weighted` | 17,281 | 12,641 | 0.73 | 1,800 | 0.0 |
| `historical_average` | 17,281 | 12,553 | 0.73 | 4,391 | 0.0 |
| `persistence` | 17,281 | 11,308 | 0.65 | 1,975 | 0.0 |
