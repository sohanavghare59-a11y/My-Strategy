"""
Signal-based screener — checks for exact BUY/SELL conditions.
No scoring. Just clear signals.

Now includes:
  - ADX trend filter for BUY signals
  - NIFTY market regime filter (200 EMA)
  - Stock long-term trend filter (200 EMA)
  - Required volume confirmation
"""

from config import TOP_N_STOCKS
from screener_logic import (
    check_buy_signal, check_sell_signal,
    calculate_trade_setup, determine_trend,
)


def screen_stock(data, news_items, market_regime=None):
    """
    Screen a single stock for exact BUY/SELL signals.
    market_regime: "bullish" / "bearish" / None.
    Returns the signal + which conditions were met/not met.
    """
    # Check for BUY signal
    is_buy, buy_signals = check_buy_signal(data, market_regime)

    # Check for SELL signal
    is_sell, sell_signals = check_sell_signal(data, market_regime)

    # Determine the signal
    if is_buy:
        signal = "BUY"
        trade = calculate_trade_setup(data, "BUY")
        direction = "LONG"
        trend = "bullish"
        signals = buy_signals
    elif is_sell:
        signal = "SELL"
        trade = calculate_trade_setup(data, "SELL")
        direction = "SHORT"
        trend = "bearish"
        signals = sell_signals
    else:
        signal = "NO SIGNAL"
        trade = None
        direction = "WAIT"
        trend = determine_trend(data)
        # Show whichever conditions were closer
        if buy_signals and any("✅" in s for s in buy_signals):
            signals = buy_signals
        elif sell_signals and any("✅" in s for s in sell_signals):
            signals = sell_signals
        else:
            signals = buy_signals

    return {
        "ticker": data["ticker"],
        "name": data.get("name", data["ticker"]),
        "sector": data.get("sector", "Unknown"),
        "is_index": data.get("is_index", False),
        "latest_close": data["latest_close"],
        "price_change_pct": data["price_change_pct"],
        "signal": signal,
        "trade_direction": direction,
        "trend": trend,
        "trade": trade,
        "signals": signals,
        "indicators": {
            "ema": data["ema"],
            "macd": data["macd"],
            "rsi": data["rsi"],
            "adx": data.get("adx", {"value": 0, "zone": "unknown", "trending": False, "rising": False}),
            "volume": data["volume"],
            "support_resistance": data["support_resistance"],
            "candle": data["candle"],
        },
        "data": data,
        "news": news_items[:5] if news_items else [],
    }


def screen_all(stock_data, news_data, index_data=None):
    """
    Screen all stocks with the NIFTY market regime applied.
    """
    from data_fetcher import get_market_regime

    market_regime = get_market_regime(index_data)

    results = []
    signal_stocks = []

    for ticker, data in stock_data.items():
        news = news_data.get(ticker, [])
        result = screen_stock(data, news, market_regime)
        if result:
            results.append(result)
            if result["signal"] != "NO SIGNAL":
                signal_stocks.append(result)

    # Signal stocks first (BUY first, then SELL), then no-signal stocks by name
    signal_stocks.sort(key=lambda x: (0 if x["signal"] == "BUY" else 1, x["ticker"]))
    results.sort(key=lambda x: (0 if x["signal"] == "BUY" else 1 if x["signal"] == "SELL" else 2, x["ticker"]))

    return results


def screen_indices(index_data):
    """Screen indices with the same signal logic (no regime filter on themselves)."""
    results = []
    for ticker, data in index_data.items():
        result = screen_stock(data, [], market_regime=None)
        if result:
            results.append(result)

    results.sort(key=lambda x: (0 if x["signal"] == "BUY" else 1 if x["signal"] == "SELL" else 2, x["ticker"]))
    return results
