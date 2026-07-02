"""Portfolio construction: target weights -> daily return series and
drifted-weight paths, under monthly rebalancing or buy-and-hold."""

import numpy as np
import pandas as pd


def build_portfolio(
    asset_returns: pd.DataFrame,
    weights: dict[str, float],
    rebalance: str = "Monthly",
) -> tuple[pd.Series, pd.DataFrame]:
    """Compute the portfolio daily return series and drifted weights.

    weights: target weights (fractions summing to 1), keys = tickers.
    rebalance: "Monthly" resets to targets on each month's first trading
    day; "Buy & hold" sets weights once at the window start.

    Returns (portfolio_returns, drifted_weights) where drifted_weights[t]
    holds the start-of-day weights used for day t's return.
    """
    tickers = list(asset_returns.columns)
    w_target = np.array([weights[t] for t in tickers], dtype=float)
    r = asset_returns.to_numpy()
    n = len(asset_returns)

    if rebalance == "Monthly":
        months = asset_returns.index.to_period("M")
        rebalance_day = np.r_[True, months[1:] != months[:-1]]
    else:
        rebalance_day = np.zeros(n, dtype=bool)
        rebalance_day[0] = True

    w = np.empty_like(r)
    current = w_target.copy()
    port_ret = np.empty(n)
    for i in range(n):
        if rebalance_day[i]:
            current = w_target.copy()
        w[i] = current
        port_ret[i] = current @ r[i]
        grown = current * (1 + r[i])
        current = grown / grown.sum()

    return (
        pd.Series(port_ret, index=asset_returns.index, name="portfolio"),
        pd.DataFrame(w, index=asset_returns.index, columns=tickers),
    )
