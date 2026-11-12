# Experiment protocol

## Core horizons

The common grid is ten minutes. Direct target vectors use 1, 6, 36 and 144 future steps for 10-minute, 1-hour, 6-hour and 24-hour evaluations. A 24-hour historical lookback is the default; the service accepts a larger configured lookback.

## Required configurations

| ID | Operational meaning |
| --- | --- |
| A | Historical energy only |
| B | Historical energy and known calendar |
| C | Historical energy, historical weather and known calendar |
| D | Historical energy, historical occupancy, historical weather and calendar |
| E | Historical energy, predicted future occupancy, historical weather and calendar |
| F | Actual future occupancy upper-bound diagnostic; never operational |

## Metrics

Point forecasts report MAE and RMSE in the target’s source units, CV(RMSE) and WAPE as percentages, and R². Probabilistic forecasts report pinball loss, empirical coverage, mean interval width and Winkler score. Metric rows include the horizon number so short- and long-horizon values cannot be conflated.

## Reporting

Results are reported by building and horizon before aggregate summaries. Aggregate summaries should include mean, median, standard deviation and day/block bootstrap intervals where appropriate. Ten-minute rows are not treated as independent observations for statistical claims.
