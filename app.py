"""Portfolio Risk Dashboard — Streamlit entry point."""

import numpy as np
import pandas as pd
import streamlit as st

import charts
import metrics as m
from config import (
    ASSETS, COLORS, DEFAULT_PRESET, PORTFOLIO_COLOR, BENCHMARK_COLOR,
    PRESETS, REGIONS, TICKERS,
)
from data_gen import load_prices
from portfolio import build_portfolio

st.set_page_config(page_title="Portfolio Risk Dashboard", page_icon="📊",
                   layout="wide")


@st.cache_data(show_spinner="Generating market data…")
def get_prices() -> pd.DataFrame:
    return load_prices()


prices = get_prices()
all_returns = prices.pct_change().dropna()

# ---------------------------------------------------------------------------
# Sidebar — portfolio weights & settings
# ---------------------------------------------------------------------------

for t in TICKERS:
    st.session_state.setdefault(f"w_{t}", PRESETS[DEFAULT_PRESET][t])


def apply_preset(name: str) -> None:
    for ticker, value in PRESETS[name].items():
        st.session_state[f"w_{ticker}"] = value


with st.sidebar:
    st.title("📊 Portfolio")

    st.caption("Preset strategies")
    preset_cols = st.columns(2)
    for i, name in enumerate(PRESETS):
        preset_cols[i % 2].button(name, on_click=apply_preset, args=(name,),
                                  width='stretch')

    st.caption("Weights (%)")
    for t in TICKERS:
        st.slider(f"{t} · {ASSETS[t]['name']}", 0, 100, key=f"w_{t}")

    raw = {t: st.session_state[f"w_{t}"] for t in TICKERS}
    total = sum(raw.values())
    if total == 0:
        st.error("All weights are zero — set at least one asset.")
        st.stop()
    if total != 100:
        st.warning(f"Weights sum to {total}% — normalized to 100%.")
    else:
        st.caption(f"Sum: {total}% ✓")
    weights = {t: v / total for t, v in raw.items()}

    st.divider()
    window = st.radio("Date range", ["1Y", "3Y", "5Y", "Max"], index=3,
                      horizontal=True)
    rebalance = st.selectbox("Rebalancing", ["Monthly", "Buy & hold"])
    benchmark = st.selectbox("Benchmark", ["SPY", "VT"],
                             format_func=lambda t: f"{t} · {ASSETS[t]['name']}")
    rf = st.number_input("Risk-free rate (%)", 0.0, 10.0, 3.0, 0.25) / 100
    value = st.number_input("Portfolio value ($)", 10_000, 1_000_000_000,
                            1_000_000, 50_000, format="%d")
    roll_window = st.radio("Rolling window", [63, 126], index=0, horizontal=True,
                           format_func=lambda d: f"{d}d (~{d // 21}M)")

# ---------------------------------------------------------------------------
# Window slicing & portfolio construction
# ---------------------------------------------------------------------------

end = all_returns.index[-1]
start = all_returns.index[0] if window == "Max" else end - pd.DateOffset(
    years=int(window[0]))
asset_returns = all_returns.loc[start:end]
window_prices = prices.loc[asset_returns.index]

port_ret, drifted = build_portfolio(asset_returns, weights, rebalance)
bench_ret = asset_returns[benchmark]
bench_label = f"{benchmark} · {ASSETS[benchmark]['name']}"

wealth = value * (1 + port_ret).cumprod()
bench_wealth = value * (1 + bench_ret).cumprod()

var95 = m.var_historical(port_ret, 0.95)
var99 = m.var_historical(port_ret, 0.99)
cvar95 = m.cvar(port_ret, 0.95)
cvar99 = m.cvar(port_ret, 0.99)
mdd = m.max_drawdown(port_ret)

st.title("Portfolio Risk Dashboard")
st.caption(
    f"{asset_returns.index[0]:%b %d, %Y} → {end:%b %d, %Y} · "
    f"{rebalance} rebalancing · benchmark {bench_label} · "
    f"risk-free {rf:.2%} · synthetic data"
)

tab_overview, tab_perf, tab_risk, tab_expo, tab_corr = st.tabs(
    ["Overview", "Performance", "Risk", "Exposure", "Correlations"])

# ---------------------------------------------------------------------------
# Overview
# ---------------------------------------------------------------------------

with tab_overview:
    tr, tr_b = m.total_return(port_ret), m.total_return(bench_ret)
    cg, cg_b = m.cagr(port_ret), m.cagr(bench_ret)
    vol, vol_b = m.ann_vol(port_ret), m.ann_vol(bench_ret)
    sh, sh_b = m.sharpe(port_ret, rf), m.sharpe(bench_ret, rf)
    mdd_b = m.max_drawdown(bench_ret)

    k = st.columns(6)
    k[0].metric("Total return", f"{tr:.1%}", f"{tr - tr_b:+.1%} vs bench")
    k[1].metric("CAGR", f"{cg:.2%}", f"{cg - cg_b:+.2%} vs bench")
    k[2].metric("Volatility (ann.)", f"{vol:.2%}", f"{vol - vol_b:+.2%} vs bench",
                delta_color="inverse")
    k[3].metric("Sharpe", f"{sh:.2f}", f"{sh - sh_b:+.2f} vs bench")
    k[4].metric("Max drawdown", f"{mdd:.1%}", f"{mdd - mdd_b:+.1%} vs bench")
    k[5].metric("VaR 95 (1-day)", f"${var95 * value:,.0f}", f"{var95:.2%} of value",
                delta_color="off")

    st.plotly_chart(charts.equity_curve(wealth, bench_wealth, bench_label, value),
                    width='stretch')
    st.plotly_chart(charts.underwater(m.drawdown_series(port_ret)),
                    width='stretch')

    skew, kurt = m.skew_kurt(port_ret)
    skew_b, kurt_b = m.skew_kurt(bench_ret)
    upcap, dncap = m.capture_ratios(port_ret, bench_ret)
    stats_rows = {
        "Total return": (f"{tr:.2%}", f"{tr_b:.2%}"),
        "CAGR": (f"{cg:.2%}", f"{cg_b:.2%}"),
        "Volatility (ann.)": (f"{vol:.2%}", f"{vol_b:.2%}"),
        "Sharpe": (f"{sh:.2f}", f"{sh_b:.2f}"),
        "Sortino": (f"{m.sortino(port_ret, rf):.2f}", f"{m.sortino(bench_ret, rf):.2f}"),
        "Calmar": (f"{m.calmar(port_ret):.2f}", f"{m.calmar(bench_ret):.2f}"),
        "Max drawdown": (f"{mdd:.2%}", f"{mdd_b:.2%}"),
        "Hit rate (daily)": (f"{m.hit_rate(port_ret):.1%}", f"{m.hit_rate(bench_ret):.1%}"),
        "Skew": (f"{skew:.2f}", f"{skew_b:.2f}"),
        "Excess kurtosis": (f"{kurt:.2f}", f"{kurt_b:.2f}"),
        "Up capture": (f"{upcap:.0%}", "100%"),
        "Down capture": (f"{dncap:.0%}", "100%"),
        "Best day": (f"{port_ret.max():.2%}", f"{bench_ret.max():.2%}"),
        "Worst day": (f"{port_ret.min():.2%}", f"{bench_ret.min():.2%}"),
    }
    st.dataframe(
        pd.DataFrame(stats_rows, index=["Portfolio", bench_label]).T,
        width='stretch', height=527)

# ---------------------------------------------------------------------------
# Performance
# ---------------------------------------------------------------------------

with tab_perf:
    st.plotly_chart(charts.monthly_heatmap(m.monthly_returns_table(port_ret)),
                    width='stretch')
    c1, c2 = st.columns(2)
    c1.plotly_chart(
        charts.annual_bars(m.annual_returns(port_ret), m.annual_returns(bench_ret),
                           bench_label),
        width='stretch')
    c2.plotly_chart(
        charts.line_chart(
            {"Portfolio": (m.rolling_sharpe(port_ret, rf, roll_window * 2),
                           PORTFOLIO_COLOR),
             bench_label: (m.rolling_sharpe(bench_ret, rf, roll_window * 2),
                           BENCHMARK_COLOR)},
            f"Rolling Sharpe ({roll_window * 2}d)", height=360),
        width='stretch')
    st.plotly_chart(charts.asset_price_lines(window_prices),
                    width='stretch')

    best, worst = m.best_worst_days(port_ret)
    c1, c2 = st.columns(2)
    c1.caption("Best 10 days")
    c1.dataframe(best.rename("return").map("{:+.2%}".format)
                 .rename_axis("date").reset_index()
                 .assign(date=lambda d: d["date"].dt.strftime("%Y-%m-%d")),
                 width='stretch', hide_index=True)
    c2.caption("Worst 10 days")
    c2.dataframe(worst.rename("return").map("{:+.2%}".format)
                 .rename_axis("date").reset_index()
                 .assign(date=lambda d: d["date"].dt.strftime("%Y-%m-%d")),
                 width='stretch', hide_index=True)

# ---------------------------------------------------------------------------
# Risk
# ---------------------------------------------------------------------------

with tab_risk:
    k = st.columns(6)
    k[0].metric("Sortino", f"{m.sortino(port_ret, rf):.2f}")
    k[1].metric("Calmar", f"{m.calmar(port_ret):.2f}")
    k[2].metric("Beta vs bench", f"{m.beta(port_ret, bench_ret):.2f}")
    k[3].metric("Tracking error", f"{m.tracking_error(port_ret, bench_ret):.2%}")
    k[4].metric("CVaR 99 (1-day)", f"${cvar99 * value:,.0f}",
                f"{cvar99:.2%} of value", delta_color="off")
    k[5].metric("Diversification", f"{m.diversification_ratio(asset_returns, weights):.2f}×")

    st.plotly_chart(charts.return_histogram(port_ret, var95, var99, cvar95),
                    width='stretch')

    c1, c2 = st.columns(2)
    c1.plotly_chart(
        charts.line_chart(
            {"Portfolio": (m.rolling_vol(port_ret, roll_window), PORTFOLIO_COLOR),
             bench_label: (m.rolling_vol(bench_ret, roll_window), BENCHMARK_COLOR)},
            f"Rolling volatility ({roll_window}d, ann.)", yfmt=".0%"),
        width='stretch')
    c2.plotly_chart(
        charts.line_chart(
            {"Portfolio": (m.rolling_beta(port_ret, bench_ret, roll_window),
                           PORTFOLIO_COLOR)},
            f"Rolling beta vs {benchmark} ({roll_window}d)"),
        width='stretch')

    st.plotly_chart(
        charts.risk_contribution_bars(pd.Series(weights),
                                      m.risk_contribution(asset_returns, weights)),
        width='stretch')
    st.caption(
        "Share of risk = fraction of total portfolio variance each asset is "
        "responsible for (marginal contribution × weight)."
    )

    st.caption("Five deepest drawdowns")
    dd_tbl = m.drawdown_table(port_ret)
    dd_view = pd.DataFrame({
        "Peak": dd_tbl["peak"].dt.strftime("%Y-%m-%d"),
        "Trough": dd_tbl["trough"].dt.strftime("%Y-%m-%d"),
        "Depth": dd_tbl["depth"].map("{:.1%}".format),
        "Recovery": dd_tbl["recovery"].dt.strftime("%Y-%m-%d").fillna("ongoing"),
        "Length (days)": dd_tbl["days"],
    })
    st.dataframe(dd_view, width='stretch', hide_index=True)

# ---------------------------------------------------------------------------
# Exposure
# ---------------------------------------------------------------------------

with tab_expo:
    c1, c2 = st.columns([1, 1])
    c1.plotly_chart(charts.weights_donut(weights), width='stretch')

    class_expo, region_expo, theme_expo = {}, {}, {}
    for t, w in weights.items():
        class_expo[ASSETS[t]["class"]] = class_expo.get(ASSETS[t]["class"], 0) + w
        theme_expo[ASSETS[t]["theme"]] = theme_expo.get(ASSETS[t]["theme"], 0) + w
        for region, frac in REGIONS[t].items():
            region_expo[region] = region_expo.get(region, 0) + w * frac
    c2.plotly_chart(charts.breakdown_bar(class_expo, "By asset class"),
                    width='stretch')
    c2.plotly_chart(charts.breakdown_bar(region_expo, "By region",
                                         color="#199e70"),
                    width='stretch')
    c2.plotly_chart(charts.breakdown_bar(theme_expo, "By theme",
                                         color="#9085e9"),
                    width='stretch')

    st.plotly_chart(charts.drifted_weights_area(drifted), width='stretch')

    current = drifted.iloc[-1]
    expo_tbl = pd.DataFrame({
        "ETF": [f"{t} · {ASSETS[t]['name']}" for t in TICKERS],
        "Class": [ASSETS[t]["class"] for t in TICKERS],
        "Theme": [ASSETS[t]["theme"] for t in TICKERS],
        "Target": [f"{weights[t]:.1%}" for t in TICKERS],
        "Current (drifted)": [f"{current[t]:.1%}" for t in TICKERS],
        "Notional": [f"${weights[t] * value:,.0f}" for t in TICKERS],
    })
    st.dataframe(expo_tbl, width='stretch', hide_index=True)

# ---------------------------------------------------------------------------
# Correlations
# ---------------------------------------------------------------------------

with tab_corr:
    c1, c2 = st.columns([1.2, 1])
    c1.plotly_chart(charts.corr_heatmap(asset_returns.corr()),
                    width='stretch')
    stats_df = pd.DataFrame({
        "vol": asset_returns.std(ddof=1) * np.sqrt(252),
        "ret": (1 + asset_returns).prod() ** (252 / len(asset_returns)) - 1,
    })
    c2.plotly_chart(
        charts.risk_return_scatter(stats_df, m.ann_vol(port_ret), m.cagr(port_ret)),
        width='stretch')

    others = [t for t in TICKERS if t != benchmark]
    st.plotly_chart(
        charts.line_chart(
            {f"{t} · {ASSETS[t]['name']}":
                 (m.rolling_corr(asset_returns[t], bench_ret, roll_window), COLORS[t])
             for t in others},
            f"Rolling correlation vs {benchmark} ({roll_window}d)",
            yfmt=".1f", height=420),
        width='stretch')
