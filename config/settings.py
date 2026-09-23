"""
Configuration for the Daily NSE/BSE Swing Trading Agent — improved.

Strategy
--------
Daily timeframe.

BUY (ALL conditions required):
    EMA 5 crosses above EMA 13
    EMA 5 crosses above EMA 26
    MACD crosses above Signal
    RSI crosses above 60
    ADX >= 25 (trending market)
    NIFTY 50 above its 200 EMA (market regime)
    Stock above its own 200 EMA (long-term trend)
    Volume ratio >= 1.2 (participation)

SELL (ALL conditions required):
    EMA 5 crosses below EMA 13
    EMA 5 crosses below EMA 26
    MACD crosses below Signal
    RSI crosses below 40
    NIFTY 50 below its 200 EMA
    Stock below its own 200 EMA
    Volume ratio >= 1.2

Risk management:
    ATR-based stop: 1.5 x ATR(14), clamped between 1% and 3%
    Target 1 = 2x risk  (1:2 R:R)
    Target 2 = 4x risk  (1:4 R:R)
    Position size risks 1% of account per trade
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
# ADX TREND FILTER (BUY only)
# ============================================================

ADX_PERIOD = 14
ADX_TREND_THRESHOLD = 25


# ============================================================
# MARKET REGIME FILTER
# ============================================================
# BUY signals only fire when NIFTY 50 is above its own
# 200-day EMA (bullish regime).
# SELL signals only fire when NIFTY 50 is below it.
# This blocks counter-trend trades — the main source of
# losing longs in the backtests.

REGIME_EMA_PERIOD = 200


# ============================================================
# VOLUME CONFIRMATION (required for entry)
# ============================================================
# Latest day's volume must be at least this multiple of the
# 20-day average volume. Low-volume crossovers fail often.

VOLUME_CONFIRM_MIN = 1.2


# ============================================================
# ATR VOLATILITY STOP
# ============================================================
# Stop loss distance = ATR_STOP_MULT x ATR(14), clamped
# between STOP_MIN_PCT and STOP_MAX_PCT of the entry price.
# Targets are set as multiples of the actual risk, so the
# R:R stays fixed at 1:2 (T1) and 1:4 (T2) on every trade.

ATR_PERIOD = 14
ATR_STOP_MULT = 1.5
STOP_MIN_PCT = 0.01     # never tighter than 1%
STOP_MAX_PCT = 0.03     # never wider  than 3%
TARGET_1_RR = 2.0       # Target 1 = 2 x risk
TARGET_2_RR = 4.0       # Target 2 = 4 x risk


# ============================================================
# POSITION SIZING
# ============================================================
# Suggested quantity risks 1% of the account on one trade.

ACCOUNT_SIZE = 100000   # rupees — change to your capital
RISK_PER_TRADE = 0.01   # 1% of account risked per trade


# ============================================================
# SIGNAL SETTINGS
# ============================================================

SIGNAL_CONFIRMATION_WINDOW = 5

REQUIRE_CURRENT_TREND_ALIGNMENT = True


# ============================================================
# LEGACY FIXED PERCENTAGE RISK (kept for compatibility)
# ============================================================

STOP_LOSS_PCT = 0.025

TARGET_1_PCT = 0.05

TARGET_2_PCT = 0.10

MIN_RISK_REWARD = 2.0


# ============================================================
# VOLUME AVERAGE (display / ratio)
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

SCAN_ALL_LISTED = True


# ============================================================
# SCANNER
# ============================================================

MAX_WORKERS = 8

FETCH_DELAY = 0.1

TEST_MODE = True

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

NIFTY_LIST = "ind_nifty500list.csv"

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
# SCORING WEIGHTS (legacy, unused by signal screener)
# ============================================================

WEIGHTS = {
    "ema_ribbon":  0.30,
    "macd":        0.25,
    "rsi":         0.20,
    "volume":      0.15,
    "structure":   0.10,
}

# ============================================================
# RSI PULLBACK ZONE (legacy)
# ============================================================

RSI_PULLBACK_LOW = 40
RSI_PULLBACK_HIGH = 55