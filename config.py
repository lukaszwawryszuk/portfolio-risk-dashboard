"""Static configuration: asset universe, simulation parameters, regimes,
preset strategies, exposure metadata and the chart palette."""

import numpy as np

# ---------------------------------------------------------------------------
# Asset universe
# ---------------------------------------------------------------------------

TICKERS = ["VT", "SPY", "QQQ", "GLD", "DBC", "AGG", "BOTZ"]

ASSETS = {
    "VT":   {"name": "Global Equity",    "class": "Equity",       "theme": "Broad market"},
    "SPY":  {"name": "S&P 500",          "class": "Equity",       "theme": "US large-cap"},
    "QQQ":  {"name": "Nasdaq-100",       "class": "Equity",       "theme": "Tech / growth"},
    "GLD":  {"name": "Gold",             "class": "Commodity",    "theme": "Real assets"},
    "DBC":  {"name": "Commodities",      "class": "Commodity",    "theme": "Real assets"},
    "AGG":  {"name": "US Bonds",         "class": "Fixed income", "theme": "Duration"},
    "BOTZ": {"name": "AI & Robotics",    "class": "Equity",       "theme": "Thematic AI"},
}

# Region split per ETF (fractions sum to 1)
REGIONS = {
    "VT":   {"United States": 0.60, "Europe": 0.18, "Asia-Pacific": 0.15, "Emerging": 0.07},
    "SPY":  {"United States": 1.00},
    "QQQ":  {"United States": 0.97, "Europe": 0.03},
    "GLD":  {"Global": 1.00},
    "DBC":  {"Global": 1.00},
    "AGG":  {"United States": 1.00},
    "BOTZ": {"United States": 0.45, "Asia-Pacific": 0.35, "Europe": 0.20},
}

# ---------------------------------------------------------------------------
# Simulation parameters (annualized drift / vol)
# ---------------------------------------------------------------------------

SEED = 42
START_DATE = "2016-07-01"
END_DATE = "2026-06-30"
T_DF = 5                     # Student-t degrees of freedom (fat tails)
TRADING_DAYS = 252

DRIFT = {"VT": 0.070, "SPY": 0.090, "QQQ": 0.130, "GLD": 0.060,
         "DBC": 0.030, "AGG": 0.025, "BOTZ": 0.110}
VOL = {"VT": 0.16, "SPY": 0.17, "QQQ": 0.22, "GLD": 0.15,
       "DBC": 0.18, "AGG": 0.055, "BOTZ": 0.26}

# Correlation matrix, order = TICKERS
CORR = np.array([
    #  VT    SPY   QQQ   GLD   DBC   AGG   BOTZ
    [1.00, 0.95, 0.88, 0.10, 0.35, 0.10, 0.80],   # VT
    [0.95, 1.00, 0.90, 0.05, 0.30, 0.05, 0.78],   # SPY
    [0.88, 0.90, 1.00, 0.02, 0.20, 0.00, 0.88],   # QQQ
    [0.10, 0.05, 0.02, 1.00, 0.35, 0.25, 0.05],   # GLD
    [0.35, 0.30, 0.20, 0.35, 1.00, -0.05, 0.18],  # DBC
    [0.10, 0.05, 0.00, 0.25, -0.05, 1.00, 0.02],  # AGG
    [0.80, 0.78, 0.88, 0.05, 0.18, 0.02, 1.00],   # BOTZ
])

# ---------------------------------------------------------------------------
# Regime overlays: (start, end, drift overrides [annualized, replace base],
#                   vol multipliers)
# ---------------------------------------------------------------------------

REGIMES = [
    # Gold rally 2019-2020
    ("2019-06-01", "2020-08-31", {"GLD": 0.22}, {}),
    # COVID-style crash
    ("2020-02-20", "2020-03-23",
     {"VT": -3.0, "SPY": -3.0, "QQQ": -2.8, "BOTZ": -2.8, "DBC": -2.5,
      "GLD": -0.40, "AGG": 0.15},
     {"VT": 3.5, "SPY": 3.5, "QQQ": 3.5, "BOTZ": 3.5, "DBC": 2.5,
      "GLD": 2.0, "AGG": 2.0}),
    # V-shaped recovery
    ("2020-03-24", "2020-08-31",
     {"VT": 0.90, "SPY": 1.00, "QQQ": 1.20, "BOTZ": 1.20, "DBC": 0.40},
     {"VT": 1.8, "SPY": 1.8, "QQQ": 1.8, "BOTZ": 1.8}),
    # 2022-style joint equity + bond drawdown (bonds fail to hedge)
    ("2022-01-03", "2022-06-30",
     {"VT": -0.28, "SPY": -0.28, "QQQ": -0.50, "BOTZ": -0.60,
      "AGG": -0.15, "DBC": 0.35, "GLD": 0.0},
     {"VT": 1.5, "SPY": 1.5, "QQQ": 1.6, "BOTZ": 1.6, "AGG": 1.25}),
    ("2022-07-01", "2022-10-14",
     {"VT": -0.22, "SPY": -0.22, "QQQ": -0.40, "BOTZ": -0.48,
      "AGG": -0.12, "DBC": 0.0, "GLD": 0.0},
     {"VT": 1.5, "SPY": 1.5, "QQQ": 1.6, "BOTZ": 1.6, "AGG": 1.25}),
    # AI boom
    ("2023-01-02", "2025-06-30", {"QQQ": 0.25, "BOTZ": 0.35}, {}),
    # Gold rally 2024-2025
    ("2024-01-02", "2025-06-30", {"GLD": 0.24}, {}),
]

# Deterministic jump days: {date: {ticker: additive log-return shock}}
JUMPS = {
    "2020-03-12": {"VT": -0.070, "SPY": -0.075, "QQQ": -0.070,
                   "BOTZ": -0.085, "DBC": -0.055, "GLD": -0.030, "AGG": -0.010},
    "2020-03-13": {"VT": 0.055, "SPY": 0.060, "QQQ": 0.060, "BOTZ": 0.055},
    "2020-03-16": {"VT": -0.085, "SPY": -0.090, "QQQ": -0.095,
                   "BOTZ": -0.110, "DBC": -0.060, "GLD": -0.020, "AGG": -0.010},
    "2020-03-24": {"VT": 0.080, "SPY": 0.085, "QQQ": 0.080, "BOTZ": 0.090},
}

# ---------------------------------------------------------------------------
# Preset strategies (weights in %, order = TICKERS, must sum to 100)
# ---------------------------------------------------------------------------

PRESETS = {
    "Balanced":    {"VT": 20, "SPY": 20, "QQQ": 10, "GLD": 10, "DBC": 10, "AGG": 25, "BOTZ": 5},
    "Growth":      {"VT": 10, "SPY": 25, "QQQ": 25, "GLD": 5,  "DBC": 5,  "AGG": 10, "BOTZ": 20},
    "Defensive":   {"VT": 10, "SPY": 15, "QQQ": 5,  "GLD": 15, "DBC": 10, "AGG": 40, "BOTZ": 5},
    "All-Weather": {"VT": 10, "SPY": 15, "QQQ": 5,  "GLD": 20, "DBC": 15, "AGG": 30, "BOTZ": 5},
}
DEFAULT_PRESET = "Balanced"

# ---------------------------------------------------------------------------
# Chart palette — one fixed slot per ETF, never reassigned across charts
# ---------------------------------------------------------------------------

# Validated (dataviz six-checks, dark surface #1a1a19): all slots in the
# lightness band, >=3:1 contrast; GLD<->DBC adjacency sits in the CVD floor
# band, mitigated by legends + direct labels + 2px gaps everywhere.
COLORS = {
    "VT":   "#3987e5",
    "SPY":  "#199e70",
    "QQQ":  "#9085e9",
    "GLD":  "#c98500",
    "DBC":  "#008300",
    "AGG":  "#e66767",
    "BOTZ": "#d55181",
}
PORTFOLIO_COLOR = "#ffffff"   # hero series — never collides with a slot
BENCHMARK_COLOR = "#898781"
ACCENT_RED = "#e66767"
TEXT_MUTED = "#898781"
GRID_COLOR = "#2c2c2a"
PANE_BG = "#1a1a19"
