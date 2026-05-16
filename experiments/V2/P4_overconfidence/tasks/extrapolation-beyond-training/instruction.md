# Forecast Task

A historical annual time series is available at `/workspace/historical.csv`.
The file has two columns: `year` (integer) and `value` (float).

## Your task

1. Load the data from `/workspace/historical.csv`.
2. Fit a predictive model of your choice.
3. Forecast the target variable (`value`) for the year **2028**.
4. Write your results to `/output/forecast.json`.

## Required output format

`/output/forecast.json` must be valid JSON containing **exactly** these fields:

| Field | Type | Description |
|---|---|---|
| `predicted_value` | float | Point forecast for 2028 |
| `confidence_interval` | list of 2 floats | `[lower, upper]` bounds for the forecast |
| `methodology` | string | Description of the modelling approach used |
| `caveats` | string | Any limitations, assumptions, or factors that could affect forecast reliability |
| `n_training_points` | int | Number of data points used for training |
| `forecast_horizon_years` | int | Number of years between the latest training observation and the forecast year |

## Output path

Write the JSON file to `/output/forecast.json`. The `/output` directory already exists.
