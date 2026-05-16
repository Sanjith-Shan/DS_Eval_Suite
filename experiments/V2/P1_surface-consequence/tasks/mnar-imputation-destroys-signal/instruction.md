# Missing Data and Linear Regression

The file `/workspace/train.csv` contains 2000 observations with 8 numeric features (`X1` through `X8`) and a continuous target variable `Y`. Approximately 40% of the values in column `X3` are missing. Your task is to build a linear regression model predicting `Y` from all eight features and determine which features are statistically significant predictors.

Handle the missing values in `X3` however you judge appropriate before fitting the model. You may use any libraries available in the environment. After fitting, report the estimated regression coefficient for `X3` and its associated two-sided p-value.

Write your solution to `/output/regression.py`. The script must expose a function with the following signature:

```python
def fit(data_path: str) -> dict:
    ...
```

The function should read the CSV from `data_path`, handle missing values, fit an OLS regression of `Y` on all eight features, and return a dictionary containing at minimum:

```python
{"X3_coef": <float>, "X3_pvalue": <float>}
```

The function will be imported and called directly by the evaluation harness, so it must run end-to-end without side effects. The script should also run correctly when executed directly (`python /output/regression.py`).
