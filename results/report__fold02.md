# Stratified results - split `fold02`

## Aggregate (the number the field reports)

| model | 15min | 30min | 45min |
|---|---|---|---|
| `0_speed` | 1.355±0.001 | 1.785±0.002 | 2.095±0.003 |
| `0_speed__seed42__diffusion` | 1.356 | 1.788 | 2.101 |
| `0_speed__seed42__weighted` | 1.376 | 1.832 | 2.166 |
| `0_speed__seed43__diffusion` | 1.350 | 1.778 | 2.084 |
| `0_speed__seed43__weighted` | 1.375 | 1.822 | 2.154 |
| `0_speed__seed44__diffusion` | 1.352 | 1.784 | 2.094 |
| `0_speed__seed44__weighted` | 1.385 | 1.837 | 2.171 |
| `1_traffic` | 1.330±0.001 | 1.721±0.004 | 1.984±0.009 |
| `3_weather` | 1.336±0.001 | 1.733±0.002 | 2.000±0.004 |
| `5_event_geo_att` | 1.325±0.003 | 1.693±0.005 | 1.930±0.009 |
| `6_all` | 1.324±0.002 | 1.674±0.008 | 1.879±0.014 |
| `6_all__seed42__diffusion` | 1.325 | 1.676 | 1.882 |
| `6_all__seed42__weighted` | 1.347 | 1.709 | 1.917 |
| `6_all__seed43__diffusion` | 1.325 | 1.673 | 1.874 |
| `6_all__seed43__weighted` | 1.335 | 1.693 | 1.909 |
| `6_all__seed44__diffusion` | 1.315 | 1.656 | 1.860 |
| `6_all__seed44__weighted` | 1.344 | 1.715 | 1.940 |
| `historical_average` | 2.506 | 2.506 | 2.506 |
| `persistence` | 1.548 | 2.099 | 2.535 |

MAE in mph, mean±sd over seeds.

## Window: `adverse_weather`

12 episodes, 460 test samples (5.7% of the split).

| model | 15min in | 15min out | 30min in | 30min out | 45min in | 45min out |
|---|---|---|---|---|---|---|
| `0_speed` | 1.271±0.001 | 1.360±0.001 | 1.647±0.005 | 1.794±0.003 | 1.875±0.013 | 2.109±0.004 |
| `0_speed__seed42__diffusion` | 1.280 | 1.361 | 1.662 | 1.796 | 1.898 | 2.114 |
| `0_speed__seed42__weighted` | 1.299 | 1.380 | 1.749 | 1.837 | 2.064 | 2.173 |
| `0_speed__seed43__diffusion` | 1.272 | 1.355 | 1.648 | 1.786 | 1.876 | 2.096 |
| `0_speed__seed43__weighted` | 1.284 | 1.380 | 1.682 | 1.831 | 1.948 | 2.167 |
| `0_speed__seed44__diffusion` | 1.278 | 1.357 | 1.656 | 1.792 | 1.900 | 2.106 |
| `0_speed__seed44__weighted` | 1.310 | 1.390 | 1.740 | 1.843 | 2.033 | 2.179 |
| `1_traffic` | 1.268±0.002 | 1.334±0.001 | 1.618±0.002 | 1.727±0.004 | 1.805±0.002 | 1.995±0.009 |
| `3_weather` | 1.283±0.001 | 1.339±0.001 | 1.645±0.002 | 1.738±0.002 | 1.850±0.008 | 2.009±0.004 |
| `5_event_geo_att` | 1.276±0.005 | 1.328±0.003 | 1.622±0.009 | 1.697±0.005 | 1.805±0.012 | 1.937±0.009 |
| `6_all` | 1.300±0.002 | 1.326±0.002 | 1.640±0.005 | 1.676±0.008 | 1.824±0.008 | 1.883±0.015 |
| `6_all__seed42__diffusion` | 1.300 | 1.327 | 1.652 | 1.678 | 1.841 | 1.885 |
| `6_all__seed42__weighted` | 1.296 | 1.350 | 1.629 | 1.714 | 1.815 | 1.923 |
| `6_all__seed43__diffusion` | 1.308 | 1.326 | 1.638 | 1.675 | 1.825 | 1.877 |
| `6_all__seed43__weighted` | 1.287 | 1.337 | 1.631 | 1.697 | 1.818 | 1.915 |
| `6_all__seed44__diffusion` | 1.286 | 1.317 | 1.619 | 1.658 | 1.797 | 1.864 |
| `6_all__seed44__weighted` | 1.307 | 1.346 | 1.639 | 1.719 | 1.818 | 1.947 |
| `historical_average` | 2.337 | 2.516 | 2.287 | 2.519 | 2.229 | 2.522 |
| `persistence` | 1.473 | 1.553 | 1.988 | 2.106 | 2.382 | 2.544 |

n = 12 episodes, not 460 samples: neighbouring 5-min steps and sensors are correlated, so any error bar must be quoted per episode.

## Window: `weather_commute`

2 episodes, 35 test samples (0.43% of the split).

| model | 15min in | 15min out | 30min in | 30min out | 45min in | 45min out |
|---|---|---|---|---|---|---|
| `0_speed` | 2.563±0.007 | 1.349±0.001 | 3.863±0.016 | 1.776±0.002 | 4.517±0.044 | 2.085±0.003 |
| `0_speed__seed42__diffusion` | 2.586 | 1.351 | 3.954 | 1.779 | 4.755 | 2.090 |
| `0_speed__seed42__weighted` | 2.576 | 1.371 | 4.057 | 1.822 | 5.144 | 2.153 |
| `0_speed__seed43__diffusion` | 2.543 | 1.345 | 3.925 | 1.768 | 4.644 | 2.073 |
| `0_speed__seed43__weighted` | 2.572 | 1.370 | 4.013 | 1.813 | 4.949 | 2.142 |
| `0_speed__seed44__diffusion` | 2.575 | 1.347 | 3.923 | 1.775 | 4.665 | 2.083 |
| `0_speed__seed44__weighted` | 2.592 | 1.380 | 4.047 | 1.827 | 4.975 | 2.159 |
| `1_traffic` | 2.554±0.006 | 1.325±0.001 | 3.891±0.029 | 1.711±0.004 | 4.475±0.037 | 1.974±0.009 |
| `3_weather` | 2.651±0.003 | 1.330±0.001 | 4.090±0.014 | 1.723±0.002 | 4.822±0.019 | 1.988±0.004 |
| `5_event_geo_att` | 2.576±0.043 | 1.320±0.003 | 3.880±0.069 | 1.683±0.005 | 4.435±0.063 | 1.919±0.009 |
| `6_all` | 2.543±0.023 | 1.319±0.002 | 3.758±0.070 | 1.665±0.008 | 4.346±0.136 | 1.869±0.014 |
| `6_all__seed42__diffusion` | 2.595 | 1.320 | 3.998 | 1.666 | 4.650 | 1.870 |
| `6_all__seed42__weighted` | 2.529 | 1.342 | 3.746 | 1.700 | 4.407 | 1.906 |
| `6_all__seed43__diffusion` | 2.626 | 1.319 | 3.971 | 1.663 | 4.749 | 1.861 |
| `6_all__seed43__weighted` | 2.551 | 1.329 | 3.886 | 1.684 | 4.621 | 1.897 |
| `6_all__seed44__diffusion` | 2.623 | 1.309 | 3.906 | 1.646 | 4.479 | 1.849 |
| `6_all__seed44__weighted` | 2.555 | 1.338 | 3.710 | 1.706 | 4.180 | 1.930 |
| `historical_average` | 5.944 | 2.491 | 5.745 | 2.491 | 5.434 | 2.493 |
| `persistence` | 3.045 | 1.541 | 5.272 | 2.085 | 7.360 | 2.514 |

n = 2 episodes, not 35 samples: neighbouring 5-min steps and sensors are correlated, so any error bar must be quoted per episode.

## Window: `event_egress`

8 episodes, 184 test samples (2.28% of the split).

| model | 15min in | 15min out | 30min in | 30min out | 45min in | 45min out |
|---|---|---|---|---|---|---|
| `0_speed` | 0.897±0.009 | 1.365±0.001 | 1.042±0.015 | 1.803±0.002 | 1.099±0.018 | 2.119±0.003 |
| `0_speed__seed42__diffusion` | 0.912 | 1.366 | 1.050 | 1.805 | 1.098 | 2.125 |
| `0_speed__seed42__weighted` | 0.910 | 1.387 | 1.072 | 1.850 | 1.131 | 2.191 |
| `0_speed__seed43__diffusion` | 0.894 | 1.361 | 1.042 | 1.795 | 1.100 | 2.107 |
| `0_speed__seed43__weighted` | 0.930 | 1.385 | 1.085 | 1.839 | 1.152 | 2.178 |
| `0_speed__seed44__diffusion` | 0.904 | 1.363 | 1.061 | 1.801 | 1.126 | 2.117 |
| `0_speed__seed44__weighted` | 0.935 | 1.396 | 1.080 | 1.855 | 1.137 | 2.195 |
| `1_traffic` | 0.880±0.002 | 1.341±0.001 | 1.008±0.005 | 1.738±0.004 | 1.053±0.005 | 2.006±0.009 |
| `3_weather` | 0.895±0.005 | 1.346±0.001 | 1.032±0.008 | 1.749±0.002 | 1.086±0.011 | 2.021±0.004 |
| `5_event_geo_att` | 0.891±0.006 | 1.335±0.003 | 1.014±0.008 | 1.708±0.005 | 1.055±0.008 | 1.950±0.009 |
| `6_all` | 0.888±0.004 | 1.334±0.002 | 1.007±0.009 | 1.690±0.008 | 1.049±0.013 | 1.899±0.015 |
| `6_all__seed42__diffusion` | 0.889 | 1.335 | 1.002 | 1.692 | 1.037 | 1.902 |
| `6_all__seed42__weighted` | 0.907 | 1.357 | 1.033 | 1.725 | 1.077 | 1.936 |
| `6_all__seed43__diffusion` | 0.886 | 1.335 | 1.005 | 1.689 | 1.043 | 1.893 |
| `6_all__seed43__weighted` | 0.886 | 1.345 | 1.000 | 1.709 | 1.039 | 1.930 |
| `6_all__seed44__diffusion` | 0.890 | 1.325 | 1.002 | 1.671 | 1.041 | 1.879 |
| `6_all__seed44__weighted` | 0.900 | 1.354 | 1.017 | 1.731 | 1.047 | 1.961 |
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
| `0_speed` | all | 1.724 | 1.677 | 1.932 | 1.605 | 1.838 | 1.747 |
| `0_speed` | egress | 0.946 | 1.011 | 1.240 | 0.894 | 1.025 | 0.936 |
| `0_speed__seed42__diffusion` | all | 1.732 | 1.673 | 1.918 | 1.610 | 1.853 | 1.744 |
| `0_speed__seed42__diffusion` | egress | 0.957 | 1.050 | 1.282 | 0.915 | 1.041 | 0.952 |
| `0_speed__seed42__weighted` | all | 1.787 | 1.793 | 1.963 | 1.640 | 1.897 | 1.791 |
| `0_speed__seed42__weighted` | egress | 0.972 | 1.108 | 1.331 | 0.918 | 1.061 | 0.962 |
| `0_speed__seed43__diffusion` | all | 1.727 | 1.662 | 1.905 | 1.594 | 1.835 | 1.734 |
| `0_speed__seed43__diffusion` | egress | 0.956 | 1.032 | 1.290 | 0.904 | 1.036 | 0.944 |
| `0_speed__seed43__weighted` | all | 1.778 | 1.723 | 1.967 | 1.641 | 1.889 | 1.781 |
| `0_speed__seed43__weighted` | egress | 1.021 | 1.095 | 1.322 | 0.951 | 1.071 | 0.990 |
| `0_speed__seed44__diffusion` | all | 1.734 | 1.678 | 1.916 | 1.594 | 1.847 | 1.738 |
| `0_speed__seed44__diffusion` | egress | 0.962 | 1.058 | 1.295 | 0.923 | 1.059 | 0.961 |
| `0_speed__seed44__weighted` | all | 1.809 | 1.770 | 1.988 | 1.673 | 1.905 | 1.783 |
| `0_speed__seed44__weighted` | egress | 1.021 | 1.082 | 1.350 | 0.938 | 1.066 | 0.981 |
| `1_traffic` | all | 1.640 | 1.607 | 1.802 | 1.526 | 1.761 | 1.665 |
| `1_traffic` | egress | 0.901 | 0.999 | 1.176 | 0.860 | 1.010 | 0.913 |
| `3_weather` | all | 1.658 | 1.637 | 1.837 | 1.541 | 1.776 | 1.694 |
| `3_weather` | egress | 0.954 | 1.040 | 1.246 | 0.890 | 1.029 | 0.937 |
| `5_event_geo_att` | all | 1.618 | 1.568 | 1.764 | 1.484 | 1.733 | 1.632 |
| `5_event_geo_att` | egress | 0.896 | 1.031 | 1.184 | 0.862 | 1.006 | 0.918 |
| `6_all` | all | 1.618 | 1.585 | 1.779 | 1.483 | 1.700 | 1.621 |
| `6_all` | egress | 0.930 | 1.099 | 1.220 | 0.887 | 1.010 | 0.919 |
| `6_all__seed42__diffusion` | all | 1.594 | 1.552 | 1.738 | 1.457 | 1.712 | 1.610 |
| `6_all__seed42__diffusion` | egress | 0.893 | 1.030 | 1.180 | 0.861 | 0.996 | 0.915 |
| `6_all__seed42__weighted` | all | 1.637 | 1.576 | 1.736 | 1.467 | 1.733 | 1.660 |
| `6_all__seed42__weighted` | egress | 0.919 | 1.059 | 1.215 | 0.893 | 1.023 | 0.946 |
| `6_all__seed43__diffusion` | all | 1.607 | 1.558 | 1.720 | 1.451 | 1.700 | 1.611 |
| `6_all__seed43__diffusion` | egress | 0.930 | 1.045 | 1.182 | 0.869 | 0.998 | 0.914 |
| `6_all__seed43__weighted` | all | 1.615 | 1.588 | 1.755 | 1.480 | 1.727 | 1.636 |
| `6_all__seed43__weighted` | egress | 0.922 | 1.038 | 1.169 | 0.862 | 1.000 | 0.912 |
| `6_all__seed44__diffusion` | all | 1.575 | 1.547 | 1.708 | 1.446 | 1.699 | 1.591 |
| `6_all__seed44__diffusion` | egress | 0.893 | 1.045 | 1.168 | 0.869 | 1.004 | 0.909 |
| `6_all__seed44__weighted` | all | 1.653 | 1.623 | 1.774 | 1.480 | 1.750 | 1.663 |
| `6_all__seed44__weighted` | egress | 0.907 | 1.051 | 1.178 | 0.868 | 1.012 | 0.926 |
| `historical_average` | all | 2.379 | 2.599 | 2.626 | 2.183 | 2.593 | 2.524 |
| `historical_average` | egress | 1.186 | 1.801 | 1.762 | 1.409 | 1.584 | 1.568 |
| `persistence` | all | 2.114 | 2.022 | 2.237 | 1.884 | 2.204 | 2.070 |
| `persistence` | egress | 1.080 | 1.166 | 1.401 | 0.988 | 1.170 | 1.055 |

## Congestion onset: lead time over a reactive rule

The reactive baseline acts once congestion is OBSERVED (< 45.0 mph for 15 min); the model acts when it PREDICTS that. Positive lead means the model was earlier. Sensors within 2 km of a venue.

A median lead of 0 for the trivial baselines is expected and is the control: persistence copies the last observation, so it cannot see an onset before it happens, and the historical average has no idea which day it is. Any positive lead a trained model shows is measured against this floor.

| model | onsets | detected | recall | false alarms | lead (median, min) |
|---|---|---|---|---|---|
| `0_speed` | 12,501 | 8,394 | 0.67 | 934 | 0.0 |
| `0_speed__seed42__diffusion` | 12,501 | 8,659 | 0.69 | 1,064 | 0.0 |
| `0_speed__seed42__weighted` | 12,501 | 9,692 | 0.78 | 2,512 | 0.0 |
| `0_speed__seed43__diffusion` | 12,501 | 8,675 | 0.69 | 1,084 | 0.0 |
| `0_speed__seed43__weighted` | 12,501 | 9,148 | 0.73 | 1,623 | 0.0 |
| `0_speed__seed44__diffusion` | 12,501 | 8,722 | 0.70 | 1,114 | 0.0 |
| `0_speed__seed44__weighted` | 12,501 | 9,259 | 0.74 | 1,991 | 0.0 |
| `1_traffic` | 12,501 | 9,342 | 0.75 | 1,612 | 0.0 |
| `3_weather` | 12,501 | 8,690 | 0.70 | 1,065 | 0.0 |
| `5_event_geo_att` | 12,501 | 9,068 | 0.73 | 1,178 | 0.0 |
| `6_all` | 12,501 | 9,600 | 0.77 | 1,621 | 0.0 |
| `6_all__seed42__diffusion` | 12,501 | 9,446 | 0.76 | 1,241 | 0.0 |
| `6_all__seed42__weighted` | 12,501 | 10,242 | 0.82 | 2,178 | 0.0 |
| `6_all__seed43__diffusion` | 12,501 | 10,116 | 0.81 | 2,002 | 0.0 |
| `6_all__seed43__weighted` | 12,501 | 9,672 | 0.77 | 1,475 | 0.0 |
| `6_all__seed44__diffusion` | 12,501 | 9,829 | 0.79 | 1,563 | 0.0 |
| `6_all__seed44__weighted` | 12,501 | 10,442 | 0.84 | 2,315 | 0.0 |
| `historical_average` | 12,501 | 10,436 | 0.83 | 5,420 | 0.0 |
| `persistence` | 12,501 | 7,996 | 0.64 | 1,448 | 0.0 |
