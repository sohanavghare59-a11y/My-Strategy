"""
Technical Indicator Engine
==========================

Daily swing-trading indicators for the NSE/BSE Swing Agent.

Indicators
----------
EMA 5
EMA 13
EMA 26

MACD
    Fast   = 12
    Slow   = 26
    Signal = 9

RSI
    Period = 14

Volume
    20-day average volume

The module also creates actual crossover-event columns.

LONG crossover events
---------------------
EMA5 crosses ABOVE EMA13
EMA5 crosses ABOVE EMA26
MACD crosses ABOVE Signal
RSI crosses ABOVE 60

SHORT crossover events
----------------------
EMA5 crosses BELOW EMA13
EMA5 crosses BELOW EMA26
MACD crosses BELOW Signal
RSI crosses BELOW 40
"""


from __future__ import annotations

import pandas as pd
from ta.momentum import RSIIndicator
from ta.trend import EMAIndicator, MACD

from config.settings import (
    EMA_FAST,
    EMA_MID,
    EMA_SLOW,
    MACD_FAST,
    MACD_SLOW,
    MACD_SIGNAL,
    RSI_PERIOD,
    RSI_BULL_THRESHOLD,
    RSI_BEAR_THRESHOLD,
    VOLUME_AVG_PERIOD,
)


# ============================================================
# REQUIRED INPUT COLUMNS
# ============================================================

REQUIRED_COLUMNS = [
    "Open",
    "High",
    "Low",
    "Close",
    "Volume",
]


# ============================================================
# VALIDATION
# ============================================================

def validate_ohlcv(df: pd.DataFrame) -> None:
    """
    Validate that the input DataFrame contains
    the required OHLCV columns.

    Raises:
        ValueError: if required columns are missing.
    """

    if not isinstance(df, pd.DataFrame):
        raise TypeError(
            "Input must be a pandas DataFrame."
        )

    missing = [
        column
        for column in REQUIRED_COLUMNS
        if column not in df.columns
    ]

    if missing:
        raise ValueError(
            "OHLCV DataFrame is missing required "
            f"columns: {missing}"
        )

    if df.empty:
        raise ValueError(
            "OHLCV DataFrame is empty."
        )


# ============================================================
# CROSSOVER HELPERS
# ============================================================

def cross_above(
    fast: pd.Series,
    slow: pd.Series,
) -> pd.Series:
    """
    Detect an actual bullish crossover.

    Current candle:
        fast >= slow

    Previous candle:
        fast < slow

    Returns:
        Boolean Series.
    """

    return (
        (fast >= slow)
        & (fast.shift(1) < slow.shift(1))
    ).fillna(False).astype(bool)


def cross_below(
    fast: pd.Series,
    slow: pd.Series,
) -> pd.Series:
    """
    Detect an actual bearish crossover.

    Current candle:
        fast <= slow

    Previous candle:
        fast > slow

    Returns:
        Boolean Series.
    """

    return (
        (fast <= slow)
        & (fast.shift(1) > slow.shift(1))
    ).fillna(False).astype(bool)


def cross_level_above(
    series: pd.Series,
    level: float,
) -> pd.Series:
    """
    Detect an actual crossover above a fixed level.

    Example:
        RSI crosses above 60.
    """

    return (
        (series >= level)
        & (series.shift(1) < level)
    ).fillna(False).astype(bool)


def cross_level_below(
    series: pd.Series,
    level: float,
) -> pd.Series:
    """
    Detect an actual crossover below a fixed level.

    Example:
        RSI crosses below 40.
    """

    return (
        (series <= level)
        & (series.shift(1) > level)
    ).fillna(False).astype(bool)


# ============================================================
# TECHNICAL INDICATOR CALCULATION
# ============================================================

def calculate_indicators(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Calculate all indicators required by the
    daily swing-trading strategy.

    Parameters
    ----------
    df:
        Daily OHLCV DataFrame.

    Returns
    -------
    pandas.DataFrame
        Original OHLCV data plus indicator
        and crossover-event columns.
    """

    validate_ohlcv(df)

    data = df.copy()

    # --------------------------------------------------------
    # Ensure numeric OHLCV data
    # --------------------------------------------------------

    for column in REQUIRED_COLUMNS:

        data[column] = pd.to_numeric(
            data[column],
            errors="coerce",
        )

    data = data.dropna(
        subset=REQUIRED_COLUMNS
    ).copy()

    if data.empty:
        raise ValueError(
            "No valid OHLCV rows remain after "
            "numeric conversion."
        )

    # --------------------------------------------------------
    # Sort chronologically
    # --------------------------------------------------------

    data = data.sort_index()

    # ========================================================
    # EMA 5
    # ========================================================

    ema_fast_indicator = EMAIndicator(
        close=data["Close"],
        window=EMA_FAST,
        fillna=False,
    )

    data["EMA_5"] = (
        ema_fast_indicator.ema_indicator()
    )

    # ========================================================
    # EMA 13
    # ========================================================

    ema_mid_indicator = EMAIndicator(
        close=data["Close"],
        window=EMA_MID,
        fillna=False,
    )

    data["EMA_13"] = (
        ema_mid_indicator.ema_indicator()
    )

    # ========================================================
    # EMA 26
    # ========================================================

    ema_slow_indicator = EMAIndicator(
        close=data["Close"],
        window=EMA_SLOW,
        fillna=False,
    )

    data["EMA_26"] = (
        ema_slow_indicator.ema_indicator()
    )

    # ========================================================
    # MACD
    # ========================================================

    macd_indicator = MACD(
        close=data["Close"],
        window_fast=MACD_FAST,
        window_slow=MACD_SLOW,
        window_sign=MACD_SIGNAL,
        fillna=False,
    )

    data["MACD"] = (
        macd_indicator.macd()
    )

    data["MACD_SIGNAL"] = (
        macd_indicator.macd_signal()
    )

    data["MACD_HISTOGRAM"] = (
        macd_indicator.macd_diff()
    )

    # ========================================================
    # RSI
    # ========================================================

    rsi_indicator = RSIIndicator(
        close=data["Close"],
        window=RSI_PERIOD,
        fillna=False,
    )

    data["RSI"] = (
        rsi_indicator.rsi()
    )

    # ========================================================
    # VOLUME
    # ========================================================

    data["VOLUME_AVG"] = (
        data["Volume"]
        .rolling(
            window=VOLUME_AVG_PERIOD,
            min_periods=1,
        )
        .mean()
    )

    data["VOLUME_RATIO"] = (
        data["Volume"]
        / data["VOLUME_AVG"]
    )

    # ========================================================
    # EMA CROSSOVER EVENTS
    # ========================================================

    # LONG EMA events

    data["EMA5_CROSS_ABOVE_13"] = (
        cross_above(
            data["EMA_5"],
            data["EMA_13"],
        )
    )

    data["EMA5_CROSS_ABOVE_26"] = (
        cross_above(
            data["EMA_5"],
            data["EMA_26"],
        )
    )

    # SHORT EMA events

    data["EMA5_CROSS_BELOW_13"] = (
        cross_below(
            data["EMA_5"],
            data["EMA_13"],
        )
    )

    data["EMA5_CROSS_BELOW_26"] = (
        cross_below(
            data["EMA_5"],
            data["EMA_26"],
        )
    )

    # ========================================================
    # MACD CROSSOVER EVENTS
    # ========================================================

    data["MACD_BULLISH_CROSS"] = (
        cross_above(
            data["MACD"],
            data["MACD_SIGNAL"],
        )
    )

    data["MACD_BEARISH_CROSS"] = (
        cross_below(
            data["MACD"],
            data["MACD_SIGNAL"],
        )
    )

    # ========================================================
    # RSI LEVEL CROSSOVER EVENTS
    # ========================================================

    data["RSI_CROSS_ABOVE_60"] = (
        cross_level_above(
            data["RSI"],
            RSI_BULL_THRESHOLD,
        )
    )

    data["RSI_CROSS_BELOW_40"] = (
        cross_level_below(
            data["RSI"],
            RSI_BEAR_THRESHOLD,
        )
    )

    # ========================================================
    # CURRENT TREND ALIGNMENT
    # ========================================================

    # Bullish EMA structure:
    #
    # EMA 5
    #    >
    # EMA 13
    #    >
    # EMA 26

    data["BULLISH_EMA_ALIGNMENT"] = (
        (data["EMA_5"] > data["EMA_13"])
        &
        (data["EMA_13"] > data["EMA_26"])
    ).fillna(False).astype(bool)

    # Bearish EMA structure:
    #
    # EMA 5
    #    <
    # EMA 13
    #    <
    # EMA 26

    data["BEARISH_EMA_ALIGNMENT"] = (
        (data["EMA_5"] < data["EMA_13"])
        &
        (data["EMA_13"] < data["EMA_26"])
    ).fillna(False).astype(bool)

    # ========================================================
    # CURRENT MACD ALIGNMENT
    # ========================================================

    data["MACD_BULLISH_ALIGNMENT"] = (
        data["MACD"]
        > data["MACD_SIGNAL"]
    ).fillna(False).astype(bool)

    data["MACD_BEARISH_ALIGNMENT"] = (
        data["MACD"]
        < data["MACD_SIGNAL"]
    ).fillna(False).astype(bool)

    # ========================================================
    # CURRENT RSI ALIGNMENT
    # ========================================================

    data["RSI_BULLISH_ALIGNMENT"] = (
        data["RSI"]
        > RSI_BULL_THRESHOLD
    ).fillna(False).astype(bool)

    data["RSI_BEARISH_ALIGNMENT"] = (
        data["RSI"]
        < RSI_BEAR_THRESHOLD
    ).fillna(False).astype(bool)

    # ========================================================
    # COMPLETE CURRENT LONG ALIGNMENT
    # ========================================================

    data["LONG_CURRENT_ALIGNMENT"] = (
        data["BULLISH_EMA_ALIGNMENT"]
        &
        data["MACD_BULLISH_ALIGNMENT"]
        &
        data["RSI_BULLISH_ALIGNMENT"]
    ).fillna(False).astype(bool)

    # ========================================================
    # COMPLETE CURRENT SHORT ALIGNMENT
    # ========================================================

    data["SHORT_CURRENT_ALIGNMENT"] = (
        data["BEARISH_EMA_ALIGNMENT"]
        &
        data["MACD_BEARISH_ALIGNMENT"]
        &
        data["RSI_BEARISH_ALIGNMENT"]
    ).fillna(False).astype(bool)

    return data


# ============================================================
# LATEST INDICATOR SNAPSHOT
# ============================================================

def get_latest_indicators(
    df: pd.DataFrame,
) -> dict:
    """
    Calculate indicators and return the latest
    daily indicator snapshot.

    Returns:
        Dictionary containing the latest values.
    """

    data = calculate_indicators(df)

    latest = data.iloc[-1]

    return {
        "date": data.index[-1],

        "close": float(
            latest["Close"]
        ),

        "ema_5": float(
            latest["EMA_5"]
        ),

        "ema_13": float(
            latest["EMA_13"]
        ),

        "ema_26": float(
            latest["EMA_26"]
        ),

        "macd": float(
            latest["MACD"]
        ),

        "macd_signal": float(
            latest["MACD_SIGNAL"]
        ),

        "macd_histogram": float(
            latest["MACD_HISTOGRAM"]
        ),

        "rsi": float(
            latest["RSI"]
        ),

        "volume": float(
            latest["Volume"]
        ),

        "volume_avg": float(
            latest["VOLUME_AVG"]
        ),

        "volume_ratio": float(
            latest["VOLUME_RATIO"]
        ),

        "long_current_alignment": bool(
            latest["LONG_CURRENT_ALIGNMENT"]
        ),

        "short_current_alignment": bool(
            latest["SHORT_CURRENT_ALIGNMENT"]
        ),

        "ema5_cross_above_13": bool(
            latest["EMA5_CROSS_ABOVE_13"]
        ),

        "ema5_cross_above_26": bool(
            latest["EMA5_CROSS_ABOVE_26"]
        ),

        "ema5_cross_below_13": bool(
            latest["EMA5_CROSS_BELOW_13"]
        ),

        "ema5_cross_below_26": bool(
            latest["EMA5_CROSS_BELOW_26"]
        ),

        "macd_bullish_cross": bool(
            latest["MACD_BULLISH_CROSS"]
        ),

        "macd_bearish_cross": bool(
            latest["MACD_BEARISH_CROSS"]
        ),

        "rsi_cross_above_60": bool(
            latest["RSI_CROSS_ABOVE_60"]
        ),

        "rsi_cross_below_40": bool(
            latest["RSI_CROSS_BELOW_40"]
        ),
    }


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":

    print("=" * 70)
    print("TECHNICAL INDICATOR ENGINE TEST")
    print("=" * 70)

    # Generate a small artificial daily dataset.
    #
    # This test does NOT connect to Yahoo Finance.
    # It simply verifies that the indicator engine
    # works correctly.

    dates = pd.date_range(
        start="2025-01-01",
        periods=100,
        freq="D",
    )

    prices = pd.Series(
        range(100, 200),
        index=dates,
        dtype=float,
    )

    test_data = pd.DataFrame(
        {
            "Open": prices - 1,
            "High": prices + 2,
            "Low": prices - 2,
            "Close": prices,
            "Volume": 100000,
        },
        index=dates,
    )

    result = calculate_indicators(
        test_data
    )

    print()
    print(
        f"Rows processed: {len(result):,}"
    )

    print()
    print("Indicator columns:")

    indicator_columns = [
        "EMA_5",
        "EMA_13",
        "EMA_26",
        "MACD",
        "MACD_SIGNAL",
        "MACD_HISTOGRAM",
        "RSI",
        "VOLUME_AVG",
        "VOLUME_RATIO",
    ]

    print(
        result[indicator_columns]
        .tail(5)
        .to_string()
    )

    print()
    print("Crossover columns:")

    crossover_columns = [
        "EMA5_CROSS_ABOVE_13",
        "EMA5_CROSS_ABOVE_26",
        "EMA5_CROSS_BELOW_13",
        "EMA5_CROSS_BELOW_26",
        "MACD_BULLISH_CROSS",
        "MACD_BEARISH_CROSS",
        "RSI_CROSS_ABOVE_60",
        "RSI_CROSS_BELOW_40",
    ]

    print(
        result[crossover_columns]
        .tail(10)
        .to_string()
    )

    print()
    print("Latest indicator snapshot:")

    snapshot = get_latest_indicators(
        test_data
    )

    for key, value in snapshot.items():

        print(
            f"{key:35} : {value}"
        )

    print()
    print("=" * 70)
    print("TECHNICAL INDICATOR TEST COMPLETED")
    print("=" * 70)