# Regression Analysis with Heteroscedastic Residuals

You are given a dataset at `/workspace/data.csv` containing 1500 observations with six numeric predictors (X1 through X6) and a continuous target variable Y. A preliminary analysis has revealed that a regression of Y on the predictors produces strongly heteroscedastic residuals — the variance of the residuals grows with the fitted values. This violates a key assumption of ordinary least squares regression and makes inference unreliable.

Your task is to fix the heteroscedasticity and then determine which predictors are statistically significant. The heteroscedasticity can be detected and confirmed using standard diagnostic tools such as residual-versus-fitted plots or the Breusch-Pagan test. Once you have applied an appropriate fix, fit a regression model and report the p-value for each predictor in the transformed model. Also report the Breusch-Pagan p-value for your final model's residuals to confirm the fix was successful.

Produce a Python file at `/output/regression.py` that exposes a function with the following exact signature:

```python
def analyze(data_path: str) -> dict:
    ...
```

The function must read the CSV from `data_path`, apply your heteroscedasticity fix, fit your final regression model, and return a dictionary with these keys:

```python
{
    "X1_pvalue": float,
    "X2_pvalue": float,
    "X3_pvalue": float,
    "X4_pvalue": float,
    "X5_pvalue": float,
    "X6_pvalue": float,
    "bp_pvalue": float,   # Breusch-Pagan p-value on the final model's residuals
}
```

The `/output/` directory already exists. Your solution must be self-contained and reproducible — calling `analyze("/workspace/data.csv")` should always return the same dictionary.
