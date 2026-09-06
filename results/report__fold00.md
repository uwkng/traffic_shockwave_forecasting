# Stratified results - split `fold00`

## Aggregate (the number the field reports)

| model | 15min | 30min | 45min |
|---|---|---|---|
| `0_speed` | 1.358±0.006 | 1.826±0.007 | 2.173±0.006 |
| `0_speed__seed42__diffusion` | 1.355 | 1.825 | 2.175 |
| `0_speed__seed42__weighted` | 1.362 | 1.839 | 2.199 |
| `0_speed__seed43__diffusion` | 1.355 | 1.826 | 2.175 |
| `0_speed__seed43__weighted` | 1.376 | 1.869 | 2.253 |
| `0_speed__seed44__diffusion` | 1.355 | 1.826 | 2.177 |
| `0_speed__seed44__weighted` | 1.366 | 1.843 | 2.203 |
| `1_traffic` | 1.343±0.006 | 1.780±0.007 | 2.090±0.007 |
| `3_weather` | 1.407±0.006 | 1.900±0.015 | 2.274±0.026 |
| `5_event_geo_att` | 1.340±0.001 | 1.764±0.003 | 2.055±0.005 |
| `6_all` | 1.409±0.006 | 1.907±0.020 | 2.283±0.035 |
| `6_all__seed42__diffusion` | 1.414 | 1.912 | 2.285 |
| `6_all__seed42__weighted` | 1.396 | 1.875 | 2.231 |
| `6_all__seed43__diffusion` | 1.394 | 1.880 | 2.233 |
| `6_all__seed43__weighted` | 1.405 | 1.876 | 2.226 |
| `6_all__seed44__diffusion` | 1.408 | 1.890 | 2.241 |
| `6_all__seed44__weighted` | 1.419 | 1.905 | 2.262 |
| `historical_average` | 2.937 | 2.937 | 2.937 |
| `persistence` | 1.520 | 2.099 | 2.566 |

MAE in mph, mean±sd over seeds.

## Window: `adverse_weather`

27 episodes, 1,155 test samples (14.32% of the split).

| model | 15min in | 15min out | 30min in | 30min out | 45min in | 45min out |
|---|---|---|---|---|---|---|
| `0_speed` | 1.471 | 1.339±0.007 | 2.044±0.002 | 1.790±0.008 | 2.487±0.005 | 2.121±0.007 |
| `0_speed__seed42__diffusion` | 1.468 | 1.336 | 2.038 | 1.789 | 2.475 | 2.125 |
| `0_speed__seed42__weighted` | 1.485 | 1.342 | 2.076 | 1.800 | 2.545 | 2.141 |
| `0_speed__seed43__diffusion` | 1.473 | 1.336 | 2.048 | 1.789 | 2.488 | 2.123 |
| `0_speed__seed43__weighted` | 1.497 | 1.356 | 2.096 | 1.831 | 2.581 | 2.198 |
| `0_speed__seed44__diffusion` | 1.470 | 1.335 | 2.041 | 1.790 | 2.480 | 2.126 |
| `0_speed__seed44__weighted` | 1.483 | 1.347 | 2.071 | 1.804 | 2.538 | 2.147 |
| `1_traffic` | 1.458±0.004 | 1.323±0.006 | 1.996±0.006 | 1.744±0.007 | 2.392±0.009 | 2.039±0.007 |
| `3_weather` | 1.574±0.015 | 1.379±0.005 | 2.204±0.020 | 1.849±0.014 | 2.694±0.043 | 2.203±0.024 |
| `5_event_geo_att` | 1.460±0.003 | 1.320±0.001 | 1.993±0.003 | 1.726±0.003 | 2.377±0.004 | 2.001±0.005 |
| `6_all` | 1.605±0.029 | 1.376±0.003 | 2.250±0.044 | 1.850±0.017 | 2.722±0.049 | 2.209±0.033 |
| `6_all__seed42__diffusion` | 1.683 | 1.369 | 2.382 | 1.833 | 2.898 | 2.182 |
| `6_all__seed42__weighted` | 1.586 | 1.364 | 2.155 | 1.828 | 2.606 | 2.168 |
| `6_all__seed43__diffusion` | 1.576 | 1.363 | 2.227 | 1.822 | 2.706 | 2.154 |
| `6_all__seed43__weighted` | 1.578 | 1.376 | 2.196 | 1.822 | 2.675 | 2.151 |
| `6_all__seed44__diffusion` | 1.615 | 1.373 | 2.259 | 1.828 | 2.739 | 2.158 |
| `6_all__seed44__weighted` | 1.627 | 1.385 | 2.223 | 1.852 | 2.664 | 2.195 |
| `historical_average` | 3.767 | 2.798 | 3.762 | 2.799 | 3.754 | 2.801 |
| `persistence` | 1.611 | 1.505 | 2.284 | 2.069 | 2.834 | 2.521 |

n = 27 episodes, not 1,155 samples: neighbouring 5-min steps and sensors are correlated, so any error bar must be quoted per episode.

## Window: `weather_commute`

10 episodes, 341 test samples (4.23% of the split).

| model | 15min in | 15min out | 30min in | 30min out | 45min in | 45min out |
|---|---|---|---|---|---|---|
| `0_speed` | 2.168±0.011 | 1.322±0.006 | 3.236±0.011 | 1.764±0.007 | 4.056±0.018 | 2.090±0.006 |
| `0_speed__seed42__diffusion` | 2.162 | 1.319 | 3.220 | 1.763 | 4.012 | 2.094 |
| `0_speed__seed42__weighted` | 2.211 | 1.325 | 3.329 | 1.773 | 4.216 | 2.110 |
| `0_speed__seed43__diffusion` | 2.176 | 1.319 | 3.259 | 1.762 | 4.073 | 2.092 |
| `0_speed__seed43__weighted` | 2.225 | 1.339 | 3.322 | 1.805 | 4.215 | 2.167 |
| `0_speed__seed44__diffusion` | 2.164 | 1.319 | 3.224 | 1.764 | 4.024 | 2.095 |
| `0_speed__seed44__weighted` | 2.195 | 1.330 | 3.301 | 1.778 | 4.173 | 2.116 |
| `1_traffic` | 2.170±0.007 | 1.306±0.006 | 3.208±0.021 | 1.717±0.006 | 3.988±0.034 | 2.006±0.006 |
| `3_weather` | 2.414±0.035 | 1.363±0.005 | 3.602±0.056 | 1.824±0.013 | 4.505±0.111 | 2.175±0.022 |
| `5_event_geo_att` | 2.175±0.010 | 1.303±0.001 | 3.207±0.018 | 1.701±0.003 | 3.956±0.025 | 1.971±0.006 |
| `6_all` | 2.515±0.079 | 1.360±0.003 | 3.815±0.126 | 1.823±0.015 | 4.758±0.145 | 2.173±0.030 |
| `6_all__seed42__diffusion` | 2.710 | 1.357 | 4.158 | 1.813 | 5.213 | 2.156 |
| `6_all__seed42__weighted` | 2.429 | 1.350 | 3.512 | 1.802 | 4.393 | 2.136 |
| `6_all__seed43__diffusion` | 2.438 | 1.347 | 3.778 | 1.796 | 4.738 | 2.122 |
| `6_all__seed43__weighted` | 2.453 | 1.359 | 3.711 | 1.795 | 4.690 | 2.117 |
| `6_all__seed44__diffusion` | 2.573 | 1.357 | 3.919 | 1.800 | 4.887 | 2.124 |
| `6_all__seed44__weighted` | 2.586 | 1.368 | 3.804 | 1.821 | 4.697 | 2.155 |
| `historical_average` | 6.305 | 2.788 | 6.373 | 2.785 | 6.388 | 2.785 |
| `persistence` | 2.280 | 1.487 | 3.437 | 2.040 | 4.361 | 2.487 |

n = 10 episodes, not 341 samples: neighbouring 5-min steps and sensors are correlated, so any error bar must be quoted per episode.

## Window: `event_egress`

4 episodes, 92 test samples (1.14% of the split).

| model | 15min in | 15min out | 30min in | 30min out | 45min in | 45min out |
|---|---|---|---|---|---|---|
| `0_speed` | 0.931±0.016 | 1.363±0.006 | 1.092±0.024 | 1.835±0.006 | 1.182±0.032 | 2.185±0.005 |
| `0_speed__seed42__diffusion` | 0.929 | 1.360 | 1.081 | 1.833 | 1.168 | 2.187 |
| `0_speed__seed42__weighted` | 0.928 | 1.367 | 1.093 | 1.848 | 1.180 | 2.210 |
| `0_speed__seed43__diffusion` | 0.923 | 1.360 | 1.075 | 1.834 | 1.157 | 2.187 |
| `0_speed__seed43__weighted` | 0.925 | 1.381 | 1.110 | 1.878 | 1.230 | 2.265 |
| `0_speed__seed44__diffusion` | 0.917 | 1.360 | 1.064 | 1.835 | 1.149 | 2.189 |
| `0_speed__seed44__weighted` | 0.934 | 1.371 | 1.098 | 1.851 | 1.189 | 2.215 |
| `1_traffic` | 0.921±0.009 | 1.348±0.006 | 1.066±0.013 | 1.788±0.007 | 1.140±0.014 | 2.101±0.007 |
| `3_weather` | 0.936±0.004 | 1.412±0.006 | 1.097±0.017 | 1.909±0.015 | 1.201±0.028 | 2.286±0.026 |
| `5_event_geo_att` | 0.917±0.003 | 1.345±0.001 | 1.059±0.003 | 1.772±0.003 | 1.129±0.004 | 2.066±0.005 |
| `6_all` | 1.006±0.014 | 1.413±0.007 | 1.263±0.042 | 1.915±0.021 | 1.478±0.083 | 2.292±0.036 |
| `6_all__seed42__diffusion` | 0.990 | 1.419 | 1.234 | 1.920 | 1.424 | 2.295 |
| `6_all__seed42__weighted` | 0.945 | 1.401 | 1.097 | 1.884 | 1.172 | 2.243 |
| `6_all__seed43__diffusion` | 0.975 | 1.398 | 1.169 | 1.888 | 1.299 | 2.244 |
| `6_all__seed43__weighted` | 0.991 | 1.410 | 1.178 | 1.884 | 1.354 | 2.236 |
| `6_all__seed44__diffusion` | 0.991 | 1.413 | 1.191 | 1.898 | 1.288 | 2.252 |
| `6_all__seed44__weighted` | 0.973 | 1.424 | 1.132 | 1.914 | 1.228 | 2.274 |
| `historical_average` | 2.139 | 2.946 | 2.047 | 2.947 | 1.955 | 2.948 |
| `persistence` | 0.992 | 1.527 | 1.180 | 2.110 | 1.270 | 2.581 |

n = 4 episodes, not 92 samples: neighbouring 5-min steps and sensors are correlated, so any error bar must be quoted per episode.

## Window: `holiday`

1 episodes, 299 test samples (3.71% of the split).

| model | 15min in | 15min out | 30min in | 30min out | 45min in | 45min out |
|---|---|---|---|---|---|---|
| `0_speed` | 1.204±0.002 | 1.364±0.006 | 1.572±0.006 | 1.836±0.007 | 1.841±0.008 | 2.186±0.006 |
| `0_speed__seed42__diffusion` | 1.200 | 1.361 | 1.559 | 1.835 | 1.821 | 2.189 |
| `0_speed__seed42__weighted` | 1.216 | 1.368 | 1.598 | 1.849 | 1.890 | 2.211 |
| `0_speed__seed43__diffusion` | 1.203 | 1.361 | 1.558 | 1.836 | 1.823 | 2.189 |
| `0_speed__seed43__weighted` | 1.221 | 1.382 | 1.610 | 1.879 | 1.913 | 2.266 |
| `0_speed__seed44__diffusion` | 1.198 | 1.361 | 1.552 | 1.837 | 1.813 | 2.191 |
| `0_speed__seed44__weighted` | 1.216 | 1.372 | 1.601 | 1.852 | 1.895 | 2.215 |
| `1_traffic` | 1.191±0.003 | 1.348±0.006 | 1.538±0.007 | 1.789±0.007 | 1.783±0.010 | 2.102±0.007 |
| `3_weather` | 1.349±0.035 | 1.409±0.005 | 1.809±0.050 | 1.903±0.014 | 2.175±0.088 | 2.277±0.025 |
| `5_event_geo_att` | 1.196±0.003 | 1.345±0.001 | 1.539±0.004 | 1.773±0.003 | 1.781±0.008 | 2.065±0.005 |
| `6_all` | 1.427±0.066 | 1.408±0.004 | 1.940±0.114 | 1.906±0.017 | 2.327±0.147 | 2.281±0.032 |
| `6_all__seed42__diffusion` | 1.583 | 1.407 | 2.153 | 1.903 | 2.612 | 2.272 |
| `6_all__seed42__weighted` | 1.511 | 1.391 | 1.900 | 1.874 | 2.273 | 2.230 |
| `6_all__seed43__diffusion` | 1.341 | 1.396 | 1.861 | 1.881 | 2.251 | 2.232 |
| `6_all__seed43__weighted` | 1.336 | 1.407 | 1.741 | 1.881 | 2.065 | 2.232 |
| `6_all__seed44__diffusion` | 1.399 | 1.408 | 1.851 | 1.891 | 2.188 | 2.243 |
| `6_all__seed44__weighted` | 1.368 | 1.421 | 1.764 | 1.910 | 2.066 | 2.270 |
| `historical_average` | 3.788 | 2.904 | 3.787 | 2.904 | 3.786 | 2.904 |
| `persistence` | 1.264 | 1.530 | 1.652 | 2.117 | 1.930 | 2.590 |

n = 1 episodes, not 299 samples: neighbouring 5-min steps and sensors are correlated, so any error bar must be quoted per episode.

## MAE by distance from the nearest venue

If the error grows towards a venue during egress but not otherwise, the error is spatially anchored to the crowd. If it is flat, it is not.

| model | window | 0-1km | 1-2km | 2-5km | 5-10km | 10-20km | 20-50km |
|---|---|---|---|---|---|---|---|
| `0_speed` | all | 1.782 | 1.827 | 1.906 | 1.625 | 1.880 | 1.791 |
| `0_speed` | egress | 0.959 | 1.277 | 1.214 | 0.928 | 1.095 | 1.013 |
| `0_speed__seed42__diffusion` | all | 1.781 | 1.842 | 1.924 | 1.631 | 1.884 | 1.792 |
| `0_speed__seed42__diffusion` | egress | 0.967 | 1.267 | 1.211 | 0.917 | 1.089 | 1.019 |
| `0_speed__seed42__weighted` | all | 1.802 | 1.859 | 1.935 | 1.639 | 1.902 | 1.815 |
| `0_speed__seed42__weighted` | egress | 0.981 | 1.304 | 1.240 | 0.918 | 1.104 | 1.013 |
| `0_speed__seed43__diffusion` | all | 1.777 | 1.829 | 1.923 | 1.622 | 1.887 | 1.796 |
| `0_speed__seed43__diffusion` | egress | 0.965 | 1.275 | 1.194 | 0.911 | 1.083 | 1.009 |
| `0_speed__seed43__weighted` | all | 1.858 | 1.911 | 1.984 | 1.691 | 1.933 | 1.845 |
| `0_speed__seed43__weighted` | egress | 0.993 | 1.351 | 1.243 | 0.940 | 1.129 | 1.050 |
| `0_speed__seed44__diffusion` | all | 1.788 | 1.838 | 1.917 | 1.628 | 1.883 | 1.798 |
| `0_speed__seed44__diffusion` | egress | 0.969 | 1.264 | 1.190 | 0.902 | 1.073 | 1.000 |
| `0_speed__seed44__weighted` | all | 1.814 | 1.859 | 1.938 | 1.650 | 1.900 | 1.816 |
| `0_speed__seed44__weighted` | egress | 1.008 | 1.323 | 1.232 | 0.922 | 1.099 | 1.031 |
| `1_traffic` | all | 1.722 | 1.786 | 1.854 | 1.559 | 1.812 | 1.744 |
| `1_traffic` | egress | 0.953 | 1.279 | 1.171 | 0.877 | 1.059 | 0.992 |
| `3_weather` | all | 1.893 | 1.954 | 1.989 | 1.688 | 1.943 | 1.874 |
| `3_weather` | egress | 0.954 | 1.291 | 1.230 | 0.934 | 1.115 | 1.027 |
| `5_event_geo_att` | all | 1.705 | 1.778 | 1.865 | 1.532 | 1.795 | 1.722 |
| `5_event_geo_att` | egress | 0.904 | 1.299 | 1.167 | 0.871 | 1.060 | 1.000 |
| `6_all` | all | 1.971 | 1.997 | 2.030 | 1.727 | 1.975 | 1.879 |
| `6_all` | egress | 1.216 | 1.780 | 1.635 | 1.267 | 1.228 | 1.071 |
| `6_all__seed42__diffusion` | all | 1.925 | 1.985 | 2.040 | 1.704 | 1.952 | 1.884 |
| `6_all__seed42__diffusion` | egress | 1.328 | 2.022 | 1.766 | 1.204 | 1.180 | 1.043 |
| `6_all__seed42__weighted` | all | 1.910 | 1.964 | 1.991 | 1.660 | 1.914 | 1.853 |
| `6_all__seed42__weighted` | egress | 1.003 | 1.409 | 1.236 | 0.987 | 1.082 | 1.012 |
| `6_all__seed43__diffusion` | all | 1.860 | 1.909 | 1.970 | 1.643 | 1.921 | 1.862 |
| `6_all__seed43__diffusion` | egress | 1.049 | 1.584 | 1.447 | 1.042 | 1.166 | 1.062 |
| `6_all__seed43__weighted` | all | 1.894 | 1.948 | 2.018 | 1.661 | 1.921 | 1.847 |
| `6_all__seed43__weighted` | egress | 1.167 | 1.725 | 1.423 | 1.052 | 1.198 | 1.109 |
| `6_all__seed44__diffusion` | all | 1.929 | 1.966 | 1.986 | 1.692 | 1.937 | 1.850 |
| `6_all__seed44__diffusion` | egress | 1.072 | 1.585 | 1.406 | 1.124 | 1.167 | 1.038 |
| `6_all__seed44__weighted` | all | 1.979 | 1.991 | 2.032 | 1.688 | 1.941 | 1.873 |
| `6_all__seed44__weighted` | egress | 1.028 | 1.466 | 1.358 | 1.014 | 1.131 | 1.033 |
| `historical_average` | all | 2.767 | 3.001 | 3.311 | 2.620 | 3.023 | 2.910 |
| `historical_average` | egress | 1.744 | 2.625 | 2.510 | 1.979 | 2.074 | 1.844 |
| `persistence` | all | 2.141 | 2.174 | 2.186 | 1.851 | 2.203 | 2.088 |
| `persistence` | egress | 1.059 | 1.450 | 1.264 | 0.977 | 1.205 | 1.079 |

## Congestion onset: lead time over a reactive rule

The reactive baseline acts once congestion is OBSERVED (< 45.0 mph for 15 min); the model acts when it PREDICTS that. Positive lead means the model was earlier. Sensors within 2 km of a venue.

A median lead of 0 for the trivial baselines is expected and is the control: persistence copies the last observation, so it cannot see an onset before it happens, and the historical average has no idea which day it is. Any positive lead a trained model shows is measured against this floor.

| model | onsets | detected | recall | false alarms | lead (median, min) |
|---|---|---|---|---|---|
| `0_speed` | 16,743 | 11,760 | 0.70 | 1,374 | 0.0 |
| `0_speed__seed42__diffusion` | 16,743 | 11,982 | 0.72 | 1,390 | 0.0 |
| `0_speed__seed42__weighted` | 16,743 | 12,427 | 0.74 | 1,965 | 0.0 |
| `0_speed__seed43__diffusion` | 16,743 | 11,644 | 0.70 | 1,232 | 0.0 |
| `0_speed__seed43__weighted` | 16,743 | 12,767 | 0.76 | 2,358 | 0.0 |
| `0_speed__seed44__diffusion` | 16,743 | 11,726 | 0.70 | 1,350 | 0.0 |
| `0_speed__seed44__weighted` | 16,743 | 12,367 | 0.74 | 1,988 | 0.0 |
| `1_traffic` | 16,743 | 12,002 | 0.72 | 1,429 | 0.0 |
| `3_weather` | 16,743 | 11,289 | 0.67 | 1,301 | 0.0 |
| `5_event_geo_att` | 16,743 | 12,472 | 0.74 | 1,490 | 0.0 |
| `6_all` | 16,743 | 12,211 | 0.73 | 1,907 | 0.0 |
| `6_all__seed42__diffusion` | 16,743 | 12,168 | 0.73 | 2,158 | 0.0 |
| `6_all__seed42__weighted` | 16,743 | 12,506 | 0.75 | 2,445 | 0.0 |
| `6_all__seed43__diffusion` | 16,743 | 11,807 | 0.71 | 1,529 | 0.0 |
| `6_all__seed43__weighted` | 16,743 | 13,347 | 0.80 | 3,023 | 0.0 |
| `6_all__seed44__diffusion` | 16,743 | 11,835 | 0.71 | 1,810 | 0.0 |
| `6_all__seed44__weighted` | 16,743 | 12,461 | 0.74 | 2,749 | 0.0 |
| `historical_average` | 16,743 | 12,513 | 0.75 | 4,507 | 0.0 |
| `persistence` | 16,743 | 11,204 | 0.67 | 1,822 | 0.0 |
