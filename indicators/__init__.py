"""Technical indicator package."""

from .technical import (
    calculate_indicators,
    get_latest_indicators,
)

from .legacy import (
    ema,
    ema_ribbon,
    ribbon_state,
    macd,
    macd_state,
    rsi,
    rsi_state,
    volume_state,
    support_resistance,
    candle_signal,
    adx,
    adx_state,
)

__all__ = [
    "calculate_indicators",
    "get_latest_indicators",
    "ema",
    "ema_ribbon",
    "ribbon_state",
    "macd",
    "macd_state",
    "rsi",
    "rsi_state",
    "volume_state",
    "support_resistance",
    "candle_signal",
    "adx",
    "adx_state",
]
