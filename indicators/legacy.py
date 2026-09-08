"""
Indicators module — EMA ribbon, MACD, RSI, volume, support/resistance.
Includes crossover detection for signal-based trading.
"""

import pandas as pd
import numpy as np


def ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False, min_periods=period).mean()


def ema_ribbon(close: pd.Series, fast=5, mid=13, slow=26):
    return {
        "ema_fast": ema(close, fast),
        "ema_mid":  ema(close, mid),
        "ema_slow": ema(close, slow),
    }


def ribbon_state(ribbon: dict, close: pd.Series, lookback=3):
    """
    Check EMA ribbon state and detect crossovers within last N bars.
    Returns:
      - state: current stack (bullish/bearish/transition)
      - fast_crossed_above_mid: EMA5 crossed above EMA13 within lookback bars
      - fast_crossed_above_slow: EMA5 crossed above EMA26 within lookback bars
      - fast_crossed_below_mid: EMA5 crossed below EMA13 within lookback bars
      - fast_crossed_below_slow: EMA5 crossed below EMA26 within lookback bars
    """
    ef = ribbon["ema_fast"]
    em = ribbon["ema_mid"]
    es = ribbon["ema_slow"]

    ef_v, em_v, es_v = float(ef.iloc[-1]), float(em.iloc[-1]), float(es.iloc[-1])
    price = float(close.iloc[-1])

    if ef_v > em_v > es_v:
        state = "bullish_stack"
    elif ef_v < em_v < es_v:
        state = "bearish_stack"
    else:
        state = "transition"

    fast_crossed_above_mid = False
    fast_crossed_above_slow = False
    fast_crossed_below_mid = False
    fast_crossed_below_slow = False

    if len(ef) >= lookback + 1:
        for i in range(-lookback, 0):
            # Above crossovers
            if ef.iloc[i - 1] <= em.iloc[i - 1] and ef.iloc[i] > em.iloc[i]:
                fast_crossed_above_mid = True
            if ef.iloc[i - 1] <= es.iloc[i - 1] and ef.iloc[i] > es.iloc[i]:
                fast_crossed_above_slow = True
            # Below crossovers
            if ef.iloc[i - 1] >= em.iloc[i - 1] and ef.iloc[i] < em.iloc[i]:
                fast_crossed_below_mid = True
            if ef.iloc[i - 1] >= es.iloc[i - 1] and ef.iloc[i] < es.iloc[i]:
                fast_crossed_below_slow = True

    spread_pct = ((ef_v - es_v) / es_v) * 100 if es_v > 0 else 0.0

    return {
        "state": state,
        "ema_fast": round(ef_v, 2),
        "ema_mid": round(em_v, 2),
        "ema_slow": round(es_v, 2),
        "spread_pct": round(spread_pct, 2),
        "fast_crossed_above_mid": fast_crossed_above_mid,
        "fast_crossed_above_slow": fast_crossed_above_slow,
        "fast_crossed_below_mid": fast_crossed_below_mid,
        "fast_crossed_below_slow": fast_crossed_below_slow,
    }


def macd(close: pd.Series, fast=12, slow=26, signal=9):
    ema_fast = ema(close, fast)
    ema_slow = ema(close, slow)
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False, min_periods=signal).mean()
    histogram = macd_line - signal_line
    return {
        "macd_line": macd_line,
        "signal_line": signal_line,
        "histogram": histogram,
        "latest": {
            "macd": round(float(macd_line.iloc[-1]), 4),
            "signal": round(float(signal_line.iloc[-1]), 4),
            "histogram": round(float(histogram.iloc[-1]), 4),
        },
    }


def macd_state(macd_data: dict, lookback=3):
    """
    Detect MACD crossovers within last N bars.
    Returns:
      - signal: bullish/bearish/neutral
      - bullish_crossover: MACD crossed above signal within lookback bars
      - bearish_crossover: MACD crossed below signal within lookback bars
      - histogram_rising: momentum increasing
    """
    ml = macd_data["macd_line"]
    sl = macd_data["signal_line"]
    hist = macd_data["histogram"]

    ml_v, sl_v, hist_v = float(ml.iloc[-1]), float(sl.iloc[-1]), float(hist.iloc[-1])

    if ml_v > sl_v and hist_v > 0:
        signal = "bullish"
    elif ml_v < sl_v and hist_v < 0:
        signal = "bearish"
    else:
        signal = "neutral"

    bullish_crossover = False
    bearish_crossover = False

    if len(ml) >= lookback + 1:
        for i in range(-lookback, 0):
            if ml.iloc[i - 1] <= sl.iloc[i - 1] and ml.iloc[i] > sl.iloc[i]:
                bullish_crossover = True
            if ml.iloc[i - 1] >= sl.iloc[i - 1] and ml.iloc[i] < sl.iloc[i]:
                bearish_crossover = True

    hist_rising = len(hist) >= 2 and hist.iloc[-1] > hist.iloc[-2]

    return {
        "signal": signal,
        "bullish_crossover": bullish_crossover,
        "bearish_crossover": bearish_crossover,
        "histogram_rising": bool(hist_rising),
        "histogram": round(hist_v, 4),
    }


def rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss
    rsi_series = 100 - (100 / (1 + rs))
    return rsi_series


def rsi_state(rsi_series: pd.Series, bull_threshold=60, bear_threshold=40,
              lookback=3):
    """
    Detect RSI crossing above 60 or below 40 within last N bars.
    Returns:
      - value: current RSI
      - zone: current zone
      - crossed_above_60: RSI crossed above 60 within lookback bars
      - crossed_below_40: RSI crossed below 40 within lookback bars
    """
    rsi_v = float(rsi_series.iloc[-1])
    if rsi_v != rsi_v:
        rsi_v = 50.0

    if rsi_v >= bull_threshold:
        zone = "bullish"
    elif rsi_v >= bear_threshold:
        zone = "neutral"
    else:
        zone = "bearish"

    crossed_above_60 = False
    crossed_below_40 = False

    if len(rsi_series) >= lookback + 1:
        for i in range(-lookback, 0):
            if rsi_series.iloc[i - 1] <= bull_threshold and rsi_series.iloc[i] > bull_threshold:
                crossed_above_60 = True
            if rsi_series.iloc[i - 1] >= bear_threshold and rsi_series.iloc[i] < bear_threshold:
                crossed_below_40 = True

    return {
        "value": round(rsi_v, 1),
        "zone": zone,
        "crossed_above_60": crossed_above_60,
        "crossed_below_40": crossed_below_40,
    }


def volume_state(volume: pd.Series, period=20, spike_mult=1.3):
    if len(volume) < period:
        period = max(len(volume) // 2, 5)

    avg_vol = float(volume.iloc[-period:].mean())
    latest_vol = float(volume.iloc[-1])
    ratio = latest_vol / avg_vol if avg_vol > 0 else 0.0

    if len(volume) >= 10:
        recent = volume.iloc[-5:].mean()
        prior = volume.iloc[-10:-5].mean()
        if recent > prior * 1.1:
            trend = "rising"
        elif recent < prior * 0.9:
            trend = "falling"
        else:
            trend = "flat"
    else:
        trend = "flat"

    return {
        "avg_volume": int(avg_vol),
        "latest_volume": int(latest_vol),
        "ratio": round(ratio, 2),
        "confirmed": ratio >= spike_mult,
        "trend": trend,
    }


def support_resistance(hist: pd.DataFrame, lookback=20):
    recent = hist.tail(lookback)
    close = float(hist["Close"].iloc[-1])

    resistance = float(recent["High"].max())
    support = float(recent["Low"].min())

    if close >= resistance:
        resistance = float(hist["High"].tail(lookback * 3).max())
    if close <= support:
        support = float(hist["Low"].tail(lookback * 3).min())

    resistance_pct = ((resistance - close) / close) * 100 if resistance > 0 else 0
    support_pct = ((close - support) / close) * 100 if support > 0 else 0

    return {
        "resistance": round(resistance, 2),
        "support": round(support, 2),
        "resistance_pct": round(resistance_pct, 2),
        "support_pct": round(support_pct, 2),
    }


def candle_signal(hist: pd.DataFrame):
    if len(hist) < 2:
        return {"pattern": "unknown", "bullish": False}

    o = float(hist["Open"].iloc[-1])
    c = float(hist["Close"].iloc[-1])
    h = float(hist["High"].iloc[-1])
    l = float(hist["Low"].iloc[-1])

    body = abs(c - o)
    range_ = h - l if h > l else 0.01
    upper_wick = h - max(o, c)
    lower_wick = min(o, c) - l
    body_pct = (body / range_) * 100

    if c > o and body_pct > 60:
        return {"pattern": "bullish_marubozu", "bullish": True}
    elif c < o and body_pct > 60:
        return {"pattern": "bearish_marubozu", "bullish": False}
    elif lower_wick > body * 2 and c > o:
        return {"pattern": "hammer", "bullish": True}
    elif upper_wick > body * 2 and c < o:
        return {"pattern": "shooting_star", "bullish": False}
    elif body_pct < 30:
        return {"pattern": "doji", "bullish": False}
    else:
        return {"pattern": "neutral", "bullish": c > o}
