"""
Signal-based swing trading logic with ADX trend filter.

Gives BUY signal when ALL conditions are met:
  1. EMA 5 crosses above EMA 13 AND EMA 26
  2. MACD bullish crossover
  3. RSI crosses above 60
  4. ADX >= 25 (trending market — filters out choppy sideways longs)

Gives SELL signal when ALL conditions are met:
  1. EMA 5 crosses below EMA 13 AND EMA 26
  2. MACD bearish crossover
  3. RSI crosses below 40
  (No ADX filter on shorts — breakdowns work even in weak trend markets)

Stop loss uses a FIXED percentage (2.5%) for consistent 1:2 R:R.
No scoring — just clear BUY / SELL / NO SIGNAL.
"""

from config import (
    STOP_LOSS_PCT, TARGET_1_PCT, TARGET_2_PCT,
    RSI_BULL_THRESHOLD, RSI_BEAR_THRESHOLD,
    ADX_TREND_THRESHOLD,
)


def check_buy_signal(data):
    """
    Check if ALL buy conditions are met:
      1. EMA 5 crossed above EMA 13 AND EMA 26 (within last 3 bars)
      2. MACD bullish crossover (within last 3 bars)
      3. RSI crossed above 60 (within last 3 bars)
      4. ADX >= 25 (market is trending, not choppy)
    """
    ema = data["ema"]
    macd = data["macd"]
    rsi = data["rsi"]
    adx = data.get("adx", {})

    signals = []
    all_conditions_met = True

    # Condition 1: EMA 5 crossed above EMA 13 AND EMA 26
    if ema["fast_crossed_above_mid"] and ema["fast_crossed_above_slow"]:
        signals.append("✅ EMA 5 crossed above EMA 13 AND EMA 26 (golden cross)")
    else:
        all_conditions_met = False
        if not ema["fast_crossed_above_mid"]:
            signals.append("❌ EMA 5 has NOT crossed above EMA 13")
        if not ema["fast_crossed_above_slow"]:
            signals.append("❌ EMA 5 has NOT crossed above EMA 26")

    # Condition 2: MACD bullish crossover
    if macd["bullish_crossover"]:
        signals.append("✅ MACD bullish crossover (MACD above signal)")
    else:
        all_conditions_met = False
        signals.append("❌ MACD has NOT crossed above signal line")

    # Condition 3: RSI crossed above 60
    if rsi["crossed_above_60"]:
        signals.append(f"✅ RSI crossed above {RSI_BULL_THRESHOLD} (currently {rsi['value']})")
    else:
        all_conditions_met = False
        signals.append(f"❌ RSI has NOT crossed above {RSI_BULL_THRESHOLD} (currently {rsi['value']})")

    # Condition 4: ADX >= 25 (trend strength filter)
    adx_val = adx.get("value", 0)
    adx_trending = adx.get("trending", False)
    if adx_trending:
        signals.append(f"✅ ADX = {adx_val} (trending market, >= {ADX_TREND_THRESHOLD})")
    else:
        all_conditions_met = False
        signals.append(f"❌ ADX = {adx_val} (< {ADX_TREND_THRESHOLD}, market is choppy/sideways)")

    return all_conditions_met, signals


def check_sell_signal(data):
    """
    Check if ALL sell conditions are met:
      1. EMA 5 crossed below EMA 13 AND EMA 26 (within last 3 bars)
      2. MACD bearish crossover (within last 3 bars)
      3. RSI crossed below 40 (within last 3 bars)
    (No ADX filter on shorts — breakdowns work in any market condition)
    """
    ema = data["ema"]
    macd = data["macd"]
    rsi = data["rsi"]

    signals = []
    all_conditions_met = True

    # Condition 1: EMA 5 crossed below EMA 13 AND EMA 26
    if ema["fast_crossed_below_mid"] and ema["fast_crossed_below_slow"]:
        signals.append("✅ EMA 5 crossed below EMA 13 AND EMA 26 (death cross)")
    else:
        all_conditions_met = False
        if not ema["fast_crossed_below_mid"]:
            signals.append("❌ EMA 5 has NOT crossed below EMA 13")
        if not ema["fast_crossed_below_slow"]:
            signals.append("❌ EMA 5 has NOT crossed below EMA 26")

    # Condition 2: MACD bearish crossover
    if macd["bearish_crossover"]:
        signals.append("✅ MACD bearish crossover (MACD below signal)")
    else:
        all_conditions_met = False
        signals.append("❌ MACD has NOT crossed below signal line")

    # Condition 3: RSI crossed below 40
    if rsi["crossed_below_40"]:
        signals.append(f"✅ RSI crossed below {RSI_BEAR_THRESHOLD} (currently {rsi['value']})")
    else:
        all_conditions_met = False
        signals.append(f"❌ RSI has NOT crossed below {RSI_BEAR_THRESHOLD} (currently {rsi['value']})")

    return all_conditions_met, signals


def calculate_trade_setup(data, direction):
    """
    Calculate entry, stop-loss, and targets for a BUY or SELL signal.
    
    Uses FIXED percentage stop loss for consistent R:R ratio:
      Stop Loss = 2.5% from entry
      Target 1 = 5.0% from entry  →  R:R = 1:2
      Target 2 = 10.0% from entry →  R:R = 1:4
    """
    entry = data["latest_close"]

    if direction == "BUY":
        stop_loss = entry * (1 - STOP_LOSS_PCT)
        target1 = entry * (1 + TARGET_1_PCT)
        target2 = entry * (1 + TARGET_2_PCT)
    else:  # SELL
        stop_loss = entry * (1 + STOP_LOSS_PCT)
        target1 = entry * (1 - TARGET_1_PCT)
        target2 = entry * (1 - TARGET_2_PCT)

    risk = abs(entry - stop_loss)
    reward1 = abs(target1 - entry)
    reward2 = abs(target2 - entry)

    rr1 = reward1 / risk if risk > 0 else 0
    rr2 = reward2 / risk if risk > 0 else 0

    return {
        "entry": round(entry, 2),
        "stop_loss": round(stop_loss, 2),
        "target1": round(target1, 2),
        "target2": round(target2, 2),
        "risk": round(risk, 2),
        "reward1": round(reward1, 2),
        "reward2": round(reward2, 2),
        "rr1": round(rr1, 2),
        "rr2": round(rr2, 2),
        "risk_pct": round((risk / entry) * 100, 2),
        "reward1_pct": round((reward1 / entry) * 100, 2),
        "reward2_pct": round((reward2 / entry) * 100, 2),
    }


def determine_trend(data):
    """Determine overall trend from EMA stack."""
    state = data["ema"]["state"]
    if state == "bullish_stack":
        return "bullish"
    elif state == "bearish_stack":
        return "bearish"
    return "neutral"
