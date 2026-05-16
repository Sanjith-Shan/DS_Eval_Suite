"""Generate synthetic data for the confounder-identification task.

Temperature drives both ice cream sales and drowning deaths. The marginal
ice-cream/drowning correlation should be strong (~0.85); the partial correlation
controlling for temperature should be near zero.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path

RNG_SEED = 20260515
OUT = Path(__file__).resolve().parents[1] / "samples" / "confounder-identification" / "environment" / "data.csv"


def main() -> None:
    rng = np.random.default_rng(RNG_SEED)
    n_days = 730  # two years of daily observations

    day = np.arange(n_days)
    # Seasonal temperature: mean ~60F, swing ~25F, plus weather noise.
    temp = 60 + 25 * np.sin(2 * np.pi * day / 365.25) + rng.normal(0, 4, n_days)

    # Ice cream sales scale with temperature.
    ice_cream_sales = 50 + 8.0 * (temp - 60) + rng.normal(0, 25, n_days)
    ice_cream_sales = np.clip(ice_cream_sales, 0, None).round().astype(int)

    # Drowning deaths also scale with temperature (people swim when warm),
    # but they are NOT caused by ice cream. We deliberately add independent
    # noise so that the partial correlation with ice cream sales is ~0.
    # Higher baseline + scale keeps integer rounding noise small relative to signal.
    drowning_deaths = 8.0 + 0.45 * (temp - 50) + rng.normal(0, 6.0, n_days)
    drowning_deaths = np.clip(drowning_deaths, 0, None).round().astype(int)

    dates = pd.date_range("2023-01-01", periods=n_days, freq="D")
    df = pd.DataFrame(
        {
            "date": dates.strftime("%Y-%m-%d"),
            "temperature_f": temp.round(2),
            "ice_cream_sales": ice_cream_sales,
            "drowning_deaths": drowning_deaths,
        }
    )

    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT, index=False)

    # Sanity: print correlations so we can confirm the structure.
    marginal = df["ice_cream_sales"].corr(df["drowning_deaths"])
    # Partial correlation controlling for temperature.
    from numpy.linalg import lstsq

    def residual(y, x):
        X = np.column_stack([np.ones_like(x), x])
        beta, *_ = lstsq(X, y, rcond=None)
        return y - X @ beta

    rx = residual(df["ice_cream_sales"].values, df["temperature_f"].values)
    ry = residual(df["drowning_deaths"].values, df["temperature_f"].values)
    partial = np.corrcoef(rx, ry)[0, 1]
    print(f"rows={len(df)} marginal_corr={marginal:.3f} partial_corr={partial:.3f}")


if __name__ == "__main__":
    main()
