"""
Configuration for the Daily NSE/BSE Swing Trading Agent.

Strategy
--------
Daily timeframe.

LONG:
    EMA 5 crosses above EMA 13
    EMA 5 crosses above EMA 26
    MACD crosses above Signal
    RSI crosses above 60
    ADX >= 25 (trending market)

SHORT:
    EMA 5 crosses below EMA 13
    EMA 5 crosses below EMA 26
    MACD crosses below Signal
    RSI crosses below 40
    (No ADX filter on shorts)

The crossover events can occur on different daily candles,
but they must occur within SIGNAL_CONFIRMATION_WINDOW candles.
"""

import os


# ============================================================
# TIMEFRAME
# ============================================================

TIMEFRAME_INTERVAL = "1d"

# Two years gives enough history for indicators and research.
TIMEFRAME_PERIOD = "2y"


# ============================================================
# EMA SETTINGS
# ============================================================

EMA_FAST = 5
EMA_MID = 13
EMA_SLOW = 26


# ============================================================
# MACD SETTINGS
# ============================================================

MACD_FAST = 12
MACD_SLOW = 26
MACD_SIGNAL = 9


# ============================================================
# RSI SETTINGS
# ============================================================

RSI_PERIOD = 14

RSI_BULL_THRESHOLD = 60
RSI_BEAR_THRESHOLD = 40


# ============================================================
# ADX TREND FILTER
# ============================================================

# Only allows BUY (long) signals when ADX >= this value.
# ADX < 20 = choppy/sideways market, crossovers will fail
# ADX >= 25 = trending market, crossovers have higher success rate
ADX_PERIOD = 14
ADX_TREND_THRESHOLD = 25


# ============================================================
# SIGNAL SETTINGS
# ============================================================

# The four crossover events do not have to happen
# on exactly the same candle.
#
# Example:
#
# Day 1 -> EMA crossover
# Day 2 -> MACD crossover
# Day 3 -> RSI crossover
#
# This is still considered one setup if all events
# happen inside this window.

SIGNAL_CONFIRMATION_WINDOW = 5

# After all crossover conditions occur, the current
# indicator state must still be aligned.
REQUIRE_CURRENT_TREND_ALIGNMENT = True


# ============================================================
# RISK MANAGEMENT
# ============================================================

STOP_LOSS_PCT = 0.025

TARGET_1_PCT = 0.05

TARGET_2_PCT = 0.10

MIN_RISK_REWARD = 2.0


# ============================================================
# VOLUME
# ============================================================

VOLUME_AVG_PERIOD = 20

VOLUME_SPIKE_MULT = 1.3


# ============================================================
# UNIVERSE
# ============================================================

SUPPORTED_EXCHANGES = [
    "NSE",
    "BSE",
]

SUPPORTED_INSTRUMENT_TYPES = [
    "STOCK",
    "INDEX",
]

# Eventually the scanner will use the complete
# available NSE + BSE universe.
SCAN_ALL_LISTED = True


# ============================================================
# SCANNER
# ============================================================

MAX_WORKERS = 8

FETCH_DELAY = 0.1

# IMPORTANT:
# Keep this True while we test the system.
#
# Once everything works correctly, we will change
# this to False and scan the full universe.

TEST_MODE = True

# Number of instruments during testing.
TEST_SYMBOL_LIMIT = 5


# ============================================================
# OUTPUT
# ============================================================

OUTPUT_DIRECTORY = "output"

SIGNAL_OUTPUT_FILE = (
    "output/daily_signals.csv"
)

MASTER_UNIVERSE_FILE = (
    "output/master_universe.csv"
)


# ============================================================
# DATA CACHE
# ============================================================

CACHE_HOURS = 4


# ============================================================
# OPTIONAL AI SUMMARY
# ============================================================

OPENAI_API_KEY = os.environ.get(
    "OPENAI_API_KEY",
    "",
)

LLM_MODEL = "gpt-4o-mini"

USE_LLM_SUMMARY = bool(
    OPENAI_API_KEY
)
# ============================================================
# DASHBOARD OUTPUT
# ============================================================

TOP_N_STOCKS = 20
TOP_N_INDICES = 10

# ============================================================
# STOCK UNIVERSE
# ============================================================

NIFTY_LIST = "ind_nifty1000list.csv"

FALLBACK_WATCHLIST = [
    "RELIANCE", "TCS", "HDFCBANK", "INFY", "ICICIBANK",
    "HINDUNILVR", "SBIN", "BHARTIARTL", "ITC", "KOTAKBANK",
    "LT", "AXISBANK", "MARUTI", "SUNPHARMA", "WIPRO",
    "ADANIENT", "TATASTEEL", "BAJFINANCE", "HCLTECH",
    "ASIANPAINT", "DMART", "TITAN", "TECHM", "ULTRACEMCO",
    "NESTLEIND", "POWERGRID", "NTPC", "JSWSTEEL", "COALINDIA",
    "ONGC", "M&M", "BAJAJFINSV", "GRASIM", "HDFCLIFE",
    "SBILIFE", "BRITANNIA", "DIVISLAB", "CIPLA", "DRREDDY",
    "EICHERMOT", "HEROMOTOCO", "BAJAJ-AUTO", "INDUSINDBK", "SHRIRAMFIN",
]

# ============================================================
# SCORING WEIGHTS (must sum to 1.0)
# ============================================================

WEIGHTS = {
    "ema_ribbon":  0.30,
    "macd":        0.25,
    "rsi":         0.20,
    "volume":      0.15,
    "structure":   0.10,
}

# ============================================================
# RSI PULLBACK ZONE
# ============================================================

RSI_PULLBACK_LOW = 40
RSI_PULLBACK_HIGH = 55
