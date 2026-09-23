"""
Signal-based swing trading logic — v4 (data-optimized).

Selected by a 5-year backtest (2022-2026, 82 stocks, train/test split).
The only variant that was profitable in BOTH the training years
(2022-2024: +44%, PF 1.21) and the validation years (2025-2026: +838%, PF 1.73).

BUY signal when ALL conditions are met:
  1. EMA 5 crossed above EMA 13 AND EMA 26 (within last 3 bars)
  2. MACD bullish crossover (within last 3 bars)
  3. RSI crossed above 60 (within last 3 bars)
  4. Stock is above its own 200 EMA (long-term uptrend)
  5. Volume ratio >= 1.2 (real participation behind the move)

SELL signal when ALL conditions are met:
  1. EMA 5 crossed below EMA 13 AND EMA 26 (within last 3 bars)
  2. MACD bearish crossover (within last 3 bars)
  3. RSI crossed below 40 (within last 3 bars)
  4. Stock is below its own 200 EMA (long-term downtrend)
  5. Volume ratio >= 1.2

REMOVED (they hurt performance in every 5-year test):
  - ADX >= 25 filter  (selected 17.7% win-rate trades — momentum exhaustion)
  - NIFTY 200-EMA regime filter (blocked profitable trades in both directions)

Risk management (5-year tested):
  - Stop loss distance = 1.5 x ATR(14), clamped between 1% and 3%
  - Target 1 = 3x the risk -> 1:3 R:R  (3R beat 2R at every filter setting)
  - Target 2 = 4x the risk -> 1:4 R:R
  - Suggested quantity risks 1% of the account

Backtest reference (2022-2026, 304 trades):
  win rate 36.5%, profit factor 1.51, positive in 4 of 5 years,
  avg holding 7 sessions, ~1.4 signals per week.
"""

from config import (
    RSI_BULL_THRESHOLD, RSI_BEAR_THRESHOLD,
    VOLUME_CONFIRM_MIN,
    ATR_STOP_MULT, STOP_MIN_PCT, STOP_MAX_PCT,
    TARGET_1_RR, TARGET_2_RR,
    ACCOUNT_SIZE, RISK_PER_TRADE,
)


def check_buy_signal(data, market_regime=None):
    """
    Check if ALL buy conditions are met.
    market_regime is accepted for compatibility but intentionally unused —
    the NIFTY regime filter was removed after 5-year testing showed it
    hurt performance in both directions.
    """
    ema = data["ema"]
    macd = data["macd"]
    rsi = data["rsi"]

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

    # Condition 4: Stock above its own 200 EMA (long-term trend)
    above200 = data.get("above_ema200")
    if above200 is True:
        signals.append("✅ Stock is above its 200 EMA (long-term uptrend)")
    elif above200 is False:
        all_conditions_met = False
        signals.append("❌ Stock is below its 200 EMA (long-term downtrend)")
    else:
        all_conditions_met = False
        signals.append("❌ Not enough history for 200 EMA trend check")

    # Condition 5: Volume confirmation
    vol_ratio = data.get("volume", {}).get("ratio", 0)
    if vol_ratio >= VOLUME_CONFIRM_MIN:
        signals.append(f"✅ Volume {vol_ratio}x its 20-day average (participation confirmed)")
    else:
        all_conditions_met = False
        signals.append(f"❌ Volume only {vol_ratio}x average (< {VOLUME_CONFIRM_MIN}x required)")

    return all_conditions_met, signals


def check_sell_signal(data, market_regime=None):
    """
    Check if ALL sell conditions are met.
    market_regime is accepted for compatibility but intentionally unused.
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

    # Condition 4: Stock below its own 200 EMA (long-term trend)
    above200 = data.get("above_ema200")
    if above200 is False:
        signals.append("✅ Stock is below its 200 EMA (long-term downtrend)")
    elif above200 is True:
        all_conditions_met = False
        signals.append("❌ Stock is above its 200 EMA (long-term uptrend)")
    else:
        all_conditions_met = False
        signals.append("❌ Not enough history for 200 EMA trend check")

    # Condition 5: Volume confirmation
    vol_ratio = data.get("volume", {}).get("ratio", 0)
    if vol_ratio >= VOLUME_CONFIRM_MIN:
        signals.append(f"✅ Volume {vol_ratio}x its 20-day average (participation confirmed)")
    else:
        all_conditions_met = False
        signals.append(f"❌ Volume only {vol_ratio}x average (< {VOLUME_CONFIRM_MIN}x required)")

    return all_conditions_met, signals


def calculate_trade_setup(data, direction):
    """
    Calculate entry, stop-loss, and targets using ATR volatility.

    Stop distance = 1.5 x ATR(14), clamped between 1% and 3% of entry.
    Target 1 = 3x the stop distance  -> 1:3 R:R (backtested optimum)
    Target 2 = 4x the stop distance  -> 1:4 R:R (stretch)

    The 3R target is deliberate: over 5 years of data, letting winners run
    to 3x risk beat taking profit at 2x risk at every filter setting
    (profit factor 1.51 vs 1.14 for the same rules at 2R).
    """
    entry = data["latest_close"]
    atr_v = data.get("atr") or entry * 0.02

    # ATR-based stop distance, clamped to 1%..3%
    risk_pct = (ATR_STOP_MULT * atr_v) / entry if entry > 0 else 0.02
    risk_pct = min(max(risk_pct, STOP_MIN_PCT), STOP_MAX_PCT)

    if direction == "BUY":
        stop_loss = entry * (1 - risk_pct)
        target1 = entry * (1 + TARGET_1_RR * risk_pct)
        target2 = entry * (1 + TARGET_2_RR * risk_pct)
    else:  # SELL
        stop_loss = entry * (1 + risk_pct)
        target1 = entry * (1 - TARGET_1_RR * risk_pct)
        target2 = entry * (1 - TARGET_2_RR * risk_pct)

    risk = abs(entry - stop_loss)
    reward1 = abs(target1 - entry)
    reward2 = abs(target2 - entry)

    rr1 = reward1 / risk if risk > 0 else 0
    rr2 = reward2 / risk if risk > 0 else 0

    # Position size: risk only 1% of the account on this trade
    quantity = int((ACCOUNT_SIZE * RISK_PER_TRADE) / risk) if risk > 0 else 0
    if quantity < 1:
        quantity = 1

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
        "risk_pct": round(risk_pct * 100, 2),
        "reward1_pct": round((reward1 / entry) * 100, 2),
        "reward2_pct": round((reward2 / entry) * 100, 2),
        "atr": round(atr_v, 2),
        "quantity": quantity,
        "position_value": round(quantity * entry, 2),
    }


def determine_trend(data):
    """Determine overall trend from EMA stack."""
    state = data["ema"]["state"]
    if state == "bullish_stack":
        return "bullish"
    elif state == "bearish_stack":
        return "bearish"
    return "neutral"
