# Portfolio Risk Dashboard

An interactive **portfolio risk-management dashboard** built with Streamlit and Plotly.
Build a portfolio from seven asset-class ETFs, adjust the weights, and watch exposure,
performance, and risk analytics recompute instantly.

> **Data note:** prices are **synthetic** — ~10 years of daily data generated
> deterministically and calibrated per asset class (realistic drift, volatility,
> cross-asset correlations, fat tails, and embedded stress regimes such as a
> COVID-style crash and a 2022-style joint stock/bond drawdown). No live market
> feed is required, so the app runs fully offline and reproducibly.

## The universe

| Ticker | Asset class | Exposure |
|--------|-------------|----------|
| **VT**   | Global Equity   | Broad developed + emerging markets |
| **SPY**  | US Equity       | S&P 500 |
| **QQQ**  | US Equity       | Nasdaq-100 (tech / growth) |
| **GLD**  | Commodity       | Gold |
| **DBC**  | Commodity       | Broad commodities |
| **AGG**  | Fixed income    | US aggregate bonds |
| **BOTZ** | Thematic Equity | AI & robotics |

## Features

**Interactive controls (sidebar)**
- Per-ETF weight sliders with automatic normalization to 100%
- Preset strategies — Balanced, Growth, Defensive, All-Weather
- Date range (1Y / 3Y / 5Y / Max), rebalancing mode (monthly / buy & hold)
- Benchmark selection, risk-free rate, portfolio value, rolling-window length

**Five analytics tabs**
- **Overview** — KPI cards (return, CAGR, vol, Sharpe, max drawdown, VaR), growth-of-capital vs benchmark, underwater drawdown chart, full stats table
- **Performance** — monthly-returns heatmap, annual-return bars, rolling Sharpe, indexed asset prices, best/worst days
- **Risk** — return distribution with VaR/CVaR markers, rolling volatility & beta, **capital weight vs risk contribution**, deepest-drawdown episodes
- **Exposure** — allocation donut, breakdowns by asset class / region / theme, weight-drift over time, notional exposure table
- **Correlations** — correlation matrix, rolling correlations vs benchmark, risk-vs-return scatter

**Metrics catalog** — total return, CAGR, annualized volatility, Sharpe, Sortino,
Calmar, historical & parametric VaR/CVaR (95 & 99), beta, Jensen's alpha, tracking
error, information ratio, up/down capture, risk contribution, diversification ratio,
drawdown depth & duration, skew & kurtosis, and more.

## Quick start

```bash
git clone https://github.com/lukaszwawryszuk/portfolio-risk-dashboard.git
cd portfolio-risk-dashboard

# with uv (recommended — https://docs.astral.sh/uv/)
uv venv .venv --python 3.14
uv pip install -r requirements.txt --python .venv/bin/python
.venv/bin/streamlit run app.py

# or with standard pip
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

Then open **http://localhost:8501**. On first run the synthetic price panel is
generated and cached to `data/prices.parquet` (a few seconds); later runs load
instantly.

## Project layout

```
config.py        Asset universe, simulation parameters, correlation matrix,
                 market regimes, preset strategies, chart palette
data_gen.py      Seeded synthetic price generator (Student-t innovations + regimes)
portfolio.py     Weights -> portfolio return series (monthly rebalance / buy & hold)
metrics.py       Pure performance / risk / exposure metric functions
charts.py        Plotly figure factories on a shared dark template
app.py           Streamlit entry point — sidebar + five tabs
```

## Stack

Python · Streamlit · Plotly · NumPy · pandas · SciPy

---

*Synthetic data is for demonstration and educational use only — this is not
investment advice.*
