"""Performance / risk / exposure metrics. Pure pandas/numpy functions —
no Streamlit imports. All inputs are daily return Series/DataFrames;
rates are annualized decimals (e.g. rf=0.03)."""

import numpy as np
import pandas as pd
from scipy import stats

A = 252  # trading days per year


# ---------------------------------------------------------------------------
# Performance
# ---------------------------------------------------------------------------

def total_return(r: pd.Series) -> float:
    return float((1 + r).prod() - 1)


def cagr(r: pd.Series) -> float:
    if len(r) == 0:
        return np.nan
    return float((1 + total_return(r)) ** (A / len(r)) - 1)


def ann_vol(r: pd.Series) -> float:
    return float(r.std(ddof=1) * np.sqrt(A))


def sharpe(r: pd.Series, rf: float = 0.0) -> float:
    ex = r - rf / A
    sd = ex.std(ddof=1)
    return float(ex.mean() / sd * np.sqrt(A)) if sd > 0 else np.nan


def sortino(r: pd.Series, rf: float = 0.0) -> float:
    ex = r - rf / A
    downside = np.sqrt((np.minimum(ex, 0) ** 2).mean()) * np.sqrt(A)
    return float(ex.mean() * A / downside) if downside > 0 else np.nan


def calmar(r: pd.Series) -> float:
    mdd = max_drawdown(r)
    return float(cagr(r) / abs(mdd)) if mdd < 0 else np.nan


def hit_rate(r: pd.Series) -> float:
    return float((r > 0).mean())


def skew_kurt(r: pd.Series) -> tuple[float, float]:
    return float(stats.skew(r)), float(stats.kurtosis(r))  # excess kurtosis


def monthly_returns_table(r: pd.Series) -> pd.DataFrame:
    """Pivot of monthly returns: rows=year, cols=month (1-12) + YTD."""
    m = (1 + r).resample("ME").prod() - 1
    tbl = pd.DataFrame({"year": m.index.year, "month": m.index.month, "ret": m.values})
    pivot = tbl.pivot(index="year", columns="month", values="ret")
    pivot["YTD"] = (1 + pivot.fillna(0)).prod(axis=1) - 1
    return pivot


def annual_returns(r: pd.Series) -> pd.Series:
    return (1 + r).resample("YE").prod() - 1


def capture_ratios(r: pd.Series, rb: pd.Series) -> tuple[float, float]:
    """(up capture, down capture) vs benchmark, on benchmark up/down days."""
    up, down = rb > 0, rb < 0
    upcap = r[up].mean() / rb[up].mean() if up.any() and rb[up].mean() != 0 else np.nan
    dncap = r[down].mean() / rb[down].mean() if down.any() and rb[down].mean() != 0 else np.nan
    return float(upcap), float(dncap)


def best_worst_days(r: pd.Series, n: int = 10) -> tuple[pd.Series, pd.Series]:
    return r.nlargest(n), r.nsmallest(n)


# ---------------------------------------------------------------------------
# Drawdowns
# ---------------------------------------------------------------------------

def drawdown_series(r: pd.Series) -> pd.Series:
    wealth = (1 + r).cumprod()
    return wealth / wealth.cummax() - 1


def max_drawdown(r: pd.Series) -> float:
    return float(drawdown_series(r).min())


def drawdown_table(r: pd.Series, top: int = 5) -> pd.DataFrame:
    """Top drawdown episodes: peak, trough, depth, recovery, duration."""
    dd = drawdown_series(r)
    episodes, in_dd, start = [], False, None
    for date, val in dd.items():
        if val < 0 and not in_dd:
            in_dd, start = True, date
        elif val == 0 and in_dd:
            seg = dd.loc[start:date]
            episodes.append((start, seg.idxmin(), float(seg.min()), date))
            in_dd = False
    if in_dd:  # ongoing drawdown
        seg = dd.loc[start:]
        episodes.append((start, seg.idxmin(), float(seg.min()), pd.NaT))
    df = pd.DataFrame(episodes, columns=["peak", "trough", "depth", "recovery"])
    df["days"] = [
        int(len(dd.loc[p: rec if pd.notna(rec) else dd.index[-1]]))
        for p, rec in zip(df["peak"], df["recovery"])
    ]
    return df.nsmallest(top, "depth").reset_index(drop=True)


# ---------------------------------------------------------------------------
# Tail risk
# ---------------------------------------------------------------------------

def var_historical(r: pd.Series, level: float = 0.95) -> float:
    return float(-np.quantile(r, 1 - level))


def var_parametric(r: pd.Series, level: float = 0.95) -> float:
    z = stats.norm.ppf(level)
    return float(-(r.mean() - z * r.std(ddof=1)))


def cvar(r: pd.Series, level: float = 0.95) -> float:
    cutoff = -var_historical(r, level)
    tail = r[r <= cutoff]
    return float(-tail.mean()) if len(tail) else np.nan


# ---------------------------------------------------------------------------
# Benchmark-relative
# ---------------------------------------------------------------------------

def beta(r: pd.Series, rb: pd.Series) -> float:
    var_b = rb.var(ddof=1)
    return float(r.cov(rb) / var_b) if var_b > 0 else np.nan


def jensen_alpha(r: pd.Series, rb: pd.Series, rf: float = 0.0) -> float:
    return float(cagr(r) - (rf + beta(r, rb) * (cagr(rb) - rf)))


def tracking_error(r: pd.Series, rb: pd.Series) -> float:
    return float((r - rb).std(ddof=1) * np.sqrt(A))


def information_ratio(r: pd.Series, rb: pd.Series) -> float:
    diff = r - rb
    sd = diff.std(ddof=1)
    return float(diff.mean() / sd * np.sqrt(A)) if sd > 0 else np.nan


# ---------------------------------------------------------------------------
# Rolling
# ---------------------------------------------------------------------------

def rolling_vol(r: pd.Series, window: int = 63) -> pd.Series:
    return r.rolling(window).std(ddof=1) * np.sqrt(A)


def rolling_sharpe(r: pd.Series, rf: float = 0.0, window: int = 126) -> pd.Series:
    ex = r - rf / A
    return ex.rolling(window).mean() / ex.rolling(window).std(ddof=1) * np.sqrt(A)


def rolling_beta(r: pd.Series, rb: pd.Series, window: int = 63) -> pd.Series:
    cov = r.rolling(window).cov(rb)
    return cov / rb.rolling(window).var(ddof=1)


def rolling_corr(a: pd.Series, b: pd.Series, window: int = 63) -> pd.Series:
    return a.rolling(window).corr(b)


# ---------------------------------------------------------------------------
# Portfolio structure
# ---------------------------------------------------------------------------

def risk_contribution(asset_returns: pd.DataFrame, weights: dict[str, float]) -> pd.Series:
    """Fraction of total portfolio variance contributed by each asset:
    RC_i = w_i * (Sigma w)_i / (w' Sigma w). Sums to 1."""
    tickers = list(asset_returns.columns)
    w = np.array([weights[t] for t in tickers])
    sigma = asset_returns.cov().to_numpy() * A
    port_var = w @ sigma @ w
    rc = w * (sigma @ w) / port_var if port_var > 0 else np.full(len(w), np.nan)
    return pd.Series(rc, index=tickers)


def diversification_ratio(asset_returns: pd.DataFrame, weights: dict[str, float]) -> float:
    tickers = list(asset_returns.columns)
    w = np.array([weights[t] for t in tickers])
    vols = asset_returns.std(ddof=1).to_numpy() * np.sqrt(A)
    sigma = asset_returns.cov().to_numpy() * A
    port_vol = np.sqrt(w @ sigma @ w)
    return float((w @ vols) / port_vol) if port_vol > 0 else np.nan
