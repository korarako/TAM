# Corrected DW4: fixed FM seed0 and AM seeds 1/2/3

| Metric | Shared FM | AM seed1 | AM seed2 | AM seed3 | AM mean ± sample SD | Wins | Reference floor mean ± sample SD |
|---|---:|---:|---:|---:|---:|---:|---:|
| Energy W2 (20k) | 0.748119 | 0.360077 | 0.358199 | 0.359625 | 0.359301 ± 0.000981 | 3/3 | 0.040157 ± 0.010158 |
| Pair-distance W2 (20k) | 0.055866 | 0.036065 | 0.035873 | 0.035981 | 0.035973 ± 0.000096 | 3/3 | 0.003481 ± 0.000979 |
| Geometric W2 (2k) | 0.155135 | 0.137920 | 0.137813 | 0.137852 | 0.137861 ± 0.000054 | 3/3 | 0.128256 ± 0.006028 |

The FM column is one shared checkpoint, not three independent FM runs. The AM standard deviation describes only AM-training stochasticity conditional on this FM checkpoint and the fixed corrected dataset.

Reference floors compare independent corrected-SMC train and eval splits under the same metric-specific sample protocol. They are empirical discrepancies, not iid Monte Carlo standard errors.
