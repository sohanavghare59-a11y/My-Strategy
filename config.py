"""
Configuration for the Swing Trading Research Agent v3.
Daily timeframe. NIFTY stocks + all major NSE/BSE indices.
Parallel fetching for speed.

NOTE: Screening 500 stocks takes 15-20 min even with parallel fetching.
Default is NIFTY 200 for a good balance of speed and coverage.
Change NIFTY_LIST below to nifty500list for full coverage.
"""

import os

# ─── Stock Universe ──────────────────────────────────────────────
# Which NSE list to fetch: "ind_nifty200list.csv" or "ind_nifty500list.csv"
# nifty200 = ~200 stocks, takes ~3-5 min
# nifty500 = ~500 stocks, takes ~10-20 min
NIFTY_LIST = "ind_nifty200list.csv"
FALLBACK_WATCHLIST = [
    "RELIANCE", "TCS", "HDFCBANK", "INFY", "ICICIBANK",
    "HINDUNILVR", "SBIN", "BHARTIARTL", "ITC", "KOTAKBANK",
    "LT", "AXISBANK", "MARUTI", "SUNPHARMA", "TMPV",
    "WIPRO", "ADANIENT", "TATASTEEL", "BAJFINANCE", "HCLTECH",
    "ASIANPAINT", "DMART", "TITAN", "TECHM", "ULTRACEMCO",
    "NESTLEIND", "POWERGRID", "NTPC", "JSWSTEEL", "COALINDIA",
    "ONGC", "M&M", "BAJAJFINSV", "GRASIM", "HDFCLIFE",
    "SBILIFE", "BRITANNIA", "DIVISLAB", "CIPLA", "DRREDDY",
    "EICHERMOT", "HEROMOTOCO", "BAJAJ-AUTO", "INDUSINDBK", "SHRIRAMFIN",
    "ETERNAL", "PIDILITIND", "SIEMENS", "ABB", "DLF",
]

# ─── Parallel Fetching ──────────────────────────────────────────
MAX_WORKERS = 8              # number of parallel threads for fetching
FETCH_DELAY = 0.1            # delay between requests within a thread

# ─── Timeframe ────────────────────────────────────────────────────
TIMEFRAME_INTERVAL = "1d"
TIMEFRAME_PERIOD = "2y"

# ─── EMA Ribbon Settings ─────────────────────────────────────────
EMA_FAST = 5
EMA_MID = 13
EMA_SLOW = 26

# ─── MACD Settings ────────────────────────────────────────────────
MACD_FAST = 12
MACD_SLOW = 26
MACD_SIGNAL = 9

# ─── RSI 60-40 Settings ──────────────────────────────────────────
RSI_PERIOD = 14
RSI_BULL_THRESHOLD = 60
RSI_BEAR_THRESHOLD = 40
RSI_PULLBACK_LOW = 40
RSI_PULLBACK_HIGH = 55

# ─── ADX Trend Filter ────────────────────────────────────────────
# Only allows BUY (long) signals when ADX >= this value.
# ADX < 20 = choppy/sideways market, crossovers will fail
# ADX >= 25 = trending market, crossovers have higher success rate
ADX_PERIOD = 14
ADX_TREND_THRESHOLD = 25

# ─── Volume Confirmation ─────────────────────────────────────────
VOLUME_AVG_PERIOD = 20
VOLUME_SPIKE_MULT = 1.3

# ─── Swing Trade Risk Management ─────────────────────────────────
STOP_LOSS_PCT = 0.025
TARGET_1_PCT = 0.05
TARGET_2_PCT = 0.10
MIN_RISK_REWARD = 2.0

# ─── Scoring Weights (must sum to 1.0) ────────────────────────────
WEIGHTS = {
    "ema_ribbon":  0.30,
    "macd":        0.25,
    "rsi":         0.20,
    "volume":      0.15,
    "structure":   0.10,
}

# ─── Output ───────────────────────────────────────────────────────
TOP_N_STOCKS = 20             # show top 20 stocks (increased from 10)
TOP_N_INDICES = 10            # show top indices in dashboard
CACHE_HOURS = 4

# ─── Optional: LLM Agent ─────────────────────────────────────────
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
LLM_MODEL = "gpt-4o-mini"
USE_LLM_SUMMARY = bool(OPENAI_API_KEY)
