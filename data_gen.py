"""Seeded synthetic daily price generator for the 7-ETF universe.

Prices follow correlated log-return dynamics with Student-t fat tails and
regime overlays (drift overrides + vol multipliers) plus a handful of
deterministic jump days. Fully deterministic for a given SEED.
"""

import os

import numpy as np
import pandas as pd

from config import (
    ASSETS, CORR, DRIFT, END_DATE, JUMPS, REGIMES, SEED, START_DATE, T_DF,
    TICKERS, TRADING_DAYS, VOL,
)

DATA_PATH = os.path.join(os.path.dirname(__file__), "data", "prices.parquet")


def _innovations(rng: np.random.Generator, n_days: int) -> np.ndarray:
    """Correlated, unit-variance multivariate Student-t innovations."""
    eig_min = np.linalg.eigvalsh(CORR).min()
    if eig_min < -1e-8:
        raise ValueError(f"Correlation matrix is not PSD (min eig {eig_min:.2e})")
    L = np.linalg.cholesky(CORR)
    z = rng.standard_normal((n_days, len(TICKERS))) @ L.T
    g = rng.chisquare(T_DF, n_days)
    t = z * np.sqrt(T_DF / g)[:, None]
    t *= np.sqrt((T_DF - 2) / T_DF)  # standardize back to unit variance
    return t - t.mean(axis=0)        # center cumulative path wander


def _daily_params(dates: pd.DatetimeIndex) -> tuple[np.ndarray, np.ndarray]:
    """Per-day annualized drift and vol arrays after regime overlays."""
    mu = np.tile([DRIFT[t] for t in TICKERS], (len(dates), 1)).astype(float)
    sig = np.tile([VOL[t] for t in TICKERS], (len(dates), 1)).astype(float)
    for start, end, drift_over, vol_mult in REGIMES:
        mask = (dates >= start) & (dates <= end)
        for ticker, value in drift_over.items():
            mu[mask, TICKERS.index(ticker)] = value
        for ticker, value in vol_mult.items():
            sig[mask, TICKERS.index(ticker)] *= value
    return mu, sig


def generate_prices() -> pd.DataFrame:
    """Generate the full daily price panel (index=dates, columns=tickers)."""
    rng = np.random.default_rng(SEED)
    dates = pd.bdate_range(START_DATE, END_DATE)
    n = len(dates)

    eps = _innovations(rng, n)
    mu, sig = _daily_params(dates)
    mu_d = mu / TRADING_DAYS
    sig_d = sig / np.sqrt(TRADING_DAYS)

    log_ret = (mu_d - 0.5 * sig_d**2) + sig_d * eps

    for day, shocks in JUMPS.items():
        idx = dates.searchsorted(pd.Timestamp(day))
        if idx < n and dates[idx] == pd.Timestamp(day):
            for ticker, shock in shocks.items():
                log_ret[idx, TICKERS.index(ticker)] += shock

    # Calibrate: spread a tiny constant per-day adjustment over the sample so
    # each asset's full-period CAGR lands on its DRIFT target (removes seed
    # luck, ~1bp/day — crashes and daily randomness are untouched).
    target_total = (n / TRADING_DAYS) * np.log1p([DRIFT[t] for t in TICKERS])
    log_ret += (target_total - log_ret.sum(axis=0)) / n

    prices = 100.0 * np.exp(np.cumsum(log_ret, axis=0))
    df = pd.DataFrame(prices, index=dates, columns=TICKERS)
    df.index.name = "date"
    assert (df > 0).all().all()
    return df


def load_prices(force: bool = False) -> pd.DataFrame:
    """Load cached prices from parquet, regenerating if missing."""
    if not force and os.path.exists(DATA_PATH):
        return pd.read_parquet(DATA_PATH)
    df = generate_prices()
    os.makedirs(os.path.dirname(DATA_PATH), exist_ok=True)
    df.to_parquet(DATA_PATH)
    return df


if __name__ == "__main__":
    # Sanity report: realized moments vs targets, regime checks, determinism.
    df = generate_prices()
    rets = df.pct_change().dropna()
    print(f"{len(df)} days  {df.index[0].date()} -> {df.index[-1].date()}\n")
    print(f"{'':6}{'drift*':>8}{'CAGR':>8}{'vol*':>8}{'vol':>8}{'maxDD':>8}")
    for t in TICKERS:
        yrs = len(df) / TRADING_DAYS
        cagr = (df[t].iloc[-1] / df[t].iloc[0]) ** (1 / yrs) - 1
        vol = rets[t].std() * np.sqrt(TRADING_DAYS)
        dd = (df[t] / df[t].cummax() - 1).min()
        print(f"{t:6}{DRIFT[t]:8.1%}{cagr:8.1%}{VOL[t]:8.1%}{vol:8.1%}{dd:8.1%}")
    covid = df.loc["2020-02-15":"2020-04-01"]
    print(f"\nCOVID window SPY drawdown: {(covid['SPY'].min() / covid['SPY'].iloc[0] - 1):.1%}")
    y2022 = df.loc["2022-01-01":"2022-12-31"]
    for t in ("QQQ", "AGG", "DBC"):
        print(f"2022 {t}: {(y2022[t].iloc[-1] / y2022[t].iloc[0] - 1):+.1%}")
    corr = rets[["VT", "SPY", "QQQ", "GLD", "AGG"]].corr().round(2)
    print(f"\nRealized correlations:\n{corr}")
    assert generate_prices().equals(df), "generator is not deterministic"
    print("\nDeterminism check: OK")
