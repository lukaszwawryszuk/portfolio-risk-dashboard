"""Plotly figure factories sharing the `risk_dark` template.

Rules applied throughout (dataviz method): one fixed color slot per ETF,
never reassigned; single y-axis everywhere; diverging scales centered on a
neutral gray midpoint; legends on every multi-series chart; unified hover
on time series; text in ink colors, never series colors."""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
from scipy import stats

from config import (
    ASSETS, BENCHMARK_COLOR, COLORS, GRID_COLOR, PANE_BG, PORTFOLIO_COLOR,
    TEXT_MUTED, TICKERS,
)

# Diverging scale: red (negative) <-> neutral gray <-> blue (positive)
DIVERGING = [(0.0, "#e66767"), (0.5, "#383835"), (1.0, "#3987e5")]

_template = go.layout.Template(
    layout=dict(
        paper_bgcolor=PANE_BG,
        plot_bgcolor=PANE_BG,
        font=dict(family='system-ui, -apple-system, "Segoe UI", sans-serif',
                  color="#c3c2b7", size=12),
        xaxis=dict(gridcolor=GRID_COLOR, zerolinecolor="#383835",
                   tickcolor=GRID_COLOR, tickfont=dict(color=TEXT_MUTED)),
        yaxis=dict(gridcolor=GRID_COLOR, zerolinecolor="#383835",
                   tickcolor=GRID_COLOR, tickfont=dict(color=TEXT_MUTED)),
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color="#c3c2b7"),
                    orientation="h", yanchor="bottom", y=1.02, x=0),
        margin=dict(l=50, r=20, t=48, b=40),
        hoverlabel=dict(bgcolor="#0d0d0d", font=dict(color="#ffffff")),
        colorway=[COLORS[t] for t in TICKERS],
    )
)
pio.templates["risk_dark"] = _template


def _base(title: str, height: int = 400) -> go.Figure:
    fig = go.Figure()
    fig.update_layout(template="risk_dark", height=height,
                      title=dict(text=title, font=dict(color="#ffffff", size=15)))
    return fig


def _label(t: str) -> str:
    return f"{t} · {ASSETS[t]['name']}"


# ---------------------------------------------------------------------------
# Time series
# ---------------------------------------------------------------------------

def equity_curve(port: pd.Series, bench: pd.Series, bench_name: str,
                 start_value: float) -> go.Figure:
    fig = _base(f"Growth of ${start_value:,.0f}", height=430)
    fig.add_scatter(x=bench.index, y=bench, name=bench_name, mode="lines",
                    line=dict(color=BENCHMARK_COLOR, width=2))
    fig.add_scatter(x=port.index, y=port, name="Portfolio", mode="lines",
                    line=dict(color=PORTFOLIO_COLOR, width=2.4))
    fig.update_layout(hovermode="x unified", yaxis_tickformat="$,.0f")
    return fig


def underwater(dd: pd.Series) -> go.Figure:
    fig = _base("Drawdown (underwater)", height=300)
    fig.add_scatter(x=dd.index, y=dd, mode="lines", name="Drawdown",
                    line=dict(color="#e66767", width=1.5),
                    fill="tozeroy", fillcolor="rgba(230,103,103,0.25)",
                    showlegend=False)
    fig.update_layout(hovermode="x unified", yaxis_tickformat=".0%")
    return fig


def line_chart(series_map: dict[str, tuple[pd.Series, str]], title: str,
               yfmt: str = ".2f", height: int = 340) -> go.Figure:
    """Generic multi-line chart. series_map: name -> (series, color)."""
    fig = _base(title, height=height)
    for name, (s, color) in series_map.items():
        fig.add_scatter(x=s.index, y=s, name=name, mode="lines",
                        line=dict(color=color, width=2))
    fig.update_layout(hovermode="x unified", yaxis_tickformat=yfmt,
                      showlegend=len(series_map) > 1)
    return fig


def asset_price_lines(prices: pd.DataFrame) -> go.Figure:
    normalized = prices / prices.iloc[0] * 100
    fig = _base("Assets, indexed to 100", height=430)
    for t in prices.columns:
        fig.add_scatter(x=normalized.index, y=normalized[t], name=_label(t),
                        mode="lines", line=dict(color=COLORS[t], width=1.8))
    fig.update_layout(hovermode="x unified", yaxis_tickformat=",.0f")
    return fig


def drifted_weights_area(weights: pd.DataFrame) -> go.Figure:
    monthly = weights.resample("ME").last()
    fig = _base("Actual allocation over time (weight drift)", height=400)
    for t in weights.columns:
        fig.add_scatter(x=monthly.index, y=monthly[t], name=_label(t),
                        mode="lines", stackgroup="one",
                        line=dict(color=COLORS[t], width=0.8),
                        fillcolor=COLORS[t])
    fig.update_layout(hovermode="x unified", yaxis_tickformat=".0%",
                      yaxis_range=[0, 1])
    return fig


# ---------------------------------------------------------------------------
# Heatmaps
# ---------------------------------------------------------------------------

_MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
           "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]


def monthly_heatmap(pivot: pd.DataFrame) -> go.Figure:
    cols = [c for c in range(1, 13) if c in pivot.columns] + ["YTD"]
    z = pivot[cols].to_numpy(dtype=float)
    xlabels = [_MONTHS[c - 1] if isinstance(c, int) else c for c in cols]
    lim = np.nanmax(np.abs(z))
    fig = _base("Monthly returns", height=380)
    fig.add_heatmap(z=z, x=xlabels, y=pivot.index.astype(str),
                    colorscale=DIVERGING, zmid=0, zmin=-lim, zmax=lim,
                    texttemplate="%{z:.1%}", textfont=dict(size=10),
                    hovertemplate="%{y} %{x}: %{z:.2%}<extra></extra>",
                    xgap=2, ygap=2, showscale=False)
    fig.update_layout(yaxis_autorange="reversed")
    return fig


def corr_heatmap(corr: pd.DataFrame) -> go.Figure:
    fig = _base("Asset correlation matrix (daily returns)", height=460)
    fig.add_heatmap(z=corr.to_numpy(), x=list(corr.columns), y=list(corr.index),
                    colorscale=DIVERGING, zmid=0, zmin=-1, zmax=1,
                    texttemplate="%{z:.2f}", textfont=dict(size=11),
                    hovertemplate="%{y} vs %{x}: %{z:.2f}<extra></extra>",
                    xgap=2, ygap=2,
                    colorbar=dict(outlinewidth=0, tickfont=dict(color=TEXT_MUTED)))
    fig.update_layout(yaxis_autorange="reversed")
    return fig


# ---------------------------------------------------------------------------
# Bars & distributions
# ---------------------------------------------------------------------------

def annual_bars(port: pd.Series, bench: pd.Series, bench_name: str) -> go.Figure:
    fig = _base("Annual returns", height=360)
    years = port.index.year
    fig.add_bar(x=years, y=port.values, name="Portfolio",
                marker=dict(color=PORTFOLIO_COLOR))
    fig.add_bar(x=bench.index.year, y=bench.values, name=bench_name,
                marker=dict(color=BENCHMARK_COLOR))
    fig.update_layout(barmode="group", bargap=0.25, bargroupgap=0.08,
                      yaxis_tickformat=".0%", hovermode="x unified")
    return fig


def return_histogram(r: pd.Series, var95: float, var99: float,
                     cvar95: float) -> go.Figure:
    fig = _base("Daily return distribution", height=420)
    fig.add_histogram(x=r, histnorm="probability density", name="Daily returns",
                      marker=dict(color="rgba(57,135,229,0.55)",
                                  line=dict(color=PANE_BG, width=1)),
                      nbinsx=120)
    xs = np.linspace(r.min(), r.max(), 300)
    fig.add_scatter(x=xs, y=stats.norm.pdf(xs, r.mean(), r.std()),
                    name="Normal fit", mode="lines",
                    line=dict(color=TEXT_MUTED, width=2, dash="dot"))
    for value, label in ((-var95, "VaR 95"), (-var99, "VaR 99"), (-cvar95, "CVaR 95")):
        fig.add_vline(x=value, line=dict(color="#e66767", width=1.5, dash="dash"),
                      annotation_text=f"{label} {value:.1%}",
                      annotation_font=dict(color="#e66767", size=11),
                      annotation_position="top")
    fig.update_layout(xaxis_tickformat=".1%", yaxis_title="density")
    return fig


def risk_contribution_bars(weights_pct: pd.Series, rc_pct: pd.Series) -> go.Figure:
    order = rc_pct.sort_values().index
    fig = _base("Capital weight vs risk contribution", height=420)
    fig.add_bar(y=[_label(t) for t in order], x=weights_pct[order], name="Weight",
                orientation="h", marker=dict(color="#3987e5"))
    fig.add_bar(y=[_label(t) for t in order], x=rc_pct[order], name="Share of risk",
                orientation="h", marker=dict(color="#e66767"))
    fig.update_layout(barmode="group", bargap=0.3, bargroupgap=0.06,
                      xaxis_tickformat=".0%")
    return fig


def weights_donut(weights: dict[str, float]) -> go.Figure:
    labels = [_label(t) for t in TICKERS]
    values = [weights[t] for t in TICKERS]
    fig = _base("Target allocation", height=420)
    fig.add_pie(labels=labels, values=values, hole=0.55, sort=False,
                marker=dict(colors=[COLORS[t] for t in TICKERS],
                            line=dict(color=PANE_BG, width=2)),
                textinfo="label+percent", textfont=dict(size=11),
                hovertemplate="%{label}: %{percent}<extra></extra>")
    fig.update_layout(showlegend=False)
    return fig


def breakdown_bar(breakdown: dict[str, float], title: str,
                  color: str = "#3987e5") -> go.Figure:
    items = sorted(breakdown.items(), key=lambda kv: kv[1])
    fig = _base(title, height=64 + 42 * len(items))
    fig.add_bar(y=[k for k, _ in items], x=[v for _, v in items],
                orientation="h", marker=dict(color=color),
                text=[f"{v:.1%}" for _, v in items], textposition="outside",
                textfont=dict(color="#c3c2b7"), showlegend=False)
    fig.update_layout(xaxis_tickformat=".0%",
                      xaxis_range=[0, max(v for _, v in items) * 1.18],
                      margin=dict(t=48, b=20))
    return fig


def risk_return_scatter(asset_stats: pd.DataFrame, port_vol: float,
                        port_ret: float) -> go.Figure:
    """asset_stats: index=tickers, columns=['vol', 'ret']."""
    fig = _base("Risk vs return (annualized)", height=460)
    for t in asset_stats.index:
        fig.add_scatter(x=[asset_stats.loc[t, "vol"]], y=[asset_stats.loc[t, "ret"]],
                        mode="markers+text", name=_label(t), text=[t],
                        textposition="top center",
                        textfont=dict(color="#c3c2b7", size=11),
                        marker=dict(color=COLORS[t], size=13,
                                    line=dict(color=PANE_BG, width=2)))
    fig.add_scatter(x=[port_vol], y=[port_ret], mode="markers+text",
                    name="Portfolio", text=["Portfolio"],
                    textposition="top center",
                    textfont=dict(color="#ffffff", size=12),
                    marker=dict(color=PORTFOLIO_COLOR, size=17, symbol="star",
                                line=dict(color=PANE_BG, width=2)))
    fig.update_layout(xaxis_title="Annualized volatility",
                      yaxis_title="Annualized return",
                      xaxis_tickformat=".0%", yaxis_tickformat=".0%")
    return fig
