"""
Daily Swing Trading Signal Engine
=================================

Daily swing strategy:

LONG:
    1. EMA 5 crosses above EMA 13
    2. EMA 5 crosses above EMA 26
    3. MACD crosses above Signal
    4. RSI crosses above 60
    5. All events occur within the confirmation window
    6. Current trend is bullish

SHORT:
    1. EMA 5 crosses below EMA 13
    2. EMA 5 crosses below EMA 26
    3. MACD crosses below Signal
    4. RSI crosses below 40
    5. All events occur within the confirmation window
    6. Current trend is bearish
"""

from __future__ import annotations

import pandas as pd

from config.settings import (
    SIGNAL_CONFIRMATION_WINDOW,
    REQUIRE_CURRENT_TREND_ALIGNMENT,
)

from indicators.technical import calculate_indicators


# ============================================================
# REQUIRED INDICATOR COLUMNS
# ============================================================

REQUIRED_INDICATOR_COLUMNS = [
    "EMA_5",
    "EMA_13",
    "EMA_26",
    "MACD",
    "MACD_SIGNAL",
    "RSI",
]


# ============================================================
# LONG EVENTS
# ============================================================

LONG_EVENT_COLUMNS = [
    "EMA5_CROSS_ABOVE_13",
    "EMA5_CROSS_ABOVE_26",
    "MACD_BULLISH_CROSS",
    "RSI_CROSS_ABOVE_60",
]


# ============================================================
# SHORT EVENTS
# ============================================================

SHORT_EVENT_COLUMNS = [
    "EMA5_CROSS_BELOW_13",
    "EMA5_CROSS_BELOW_26",
    "MACD_BEARISH_CROSS",
    "RSI_CROSS_BELOW_40",
]


# ============================================================
# VALIDATION
# ============================================================

def validate_indicator_data(df: pd.DataFrame) -> None:
    """
    Make sure the required indicator columns exist.
    """

    missing = [
        column
        for column in REQUIRED_INDICATOR_COLUMNS
        if column not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing required indicator columns: {missing}"
        )


# ============================================================
# EVENT WINDOW HELPER
# ============================================================

def _event_dates_in_window(
    df: pd.DataFrame,
    event_columns: list[str],
    window: int,
) -> dict[str, pd.Timestamp]:
    """
    Find the most recent occurrence of every required event
    inside the trailing confirmation window.
    """

    if len(df) < window:
        return {}

    recent = df.tail(window)

    event_dates: dict[str, pd.Timestamp] = {}

    for column in event_columns:

        if column not in recent.columns:
            return {}

        matches = recent.index[
            recent[column].fillna(False)
        ]

        if len(matches) == 0:
            return {}

        event_dates[column] = matches[-1]

    return event_dates


# ============================================================
# LONG EVENT CHECK
# ============================================================

def check_long_events(
    df: pd.DataFrame,
    window: int = SIGNAL_CONFIRMATION_WINDOW,
) -> tuple[bool, dict[str, pd.Timestamp]]:
    """
    Check whether all LONG events occurred inside
    the confirmation window.
    """

    events = _event_dates_in_window(
        df=df,
        event_columns=LONG_EVENT_COLUMNS,
        window=window,
    )

    complete = (
        len(events) == len(LONG_EVENT_COLUMNS)
    )

    return complete, events


# ============================================================
# SHORT EVENT CHECK
# ============================================================

def check_short_events(
    df: pd.DataFrame,
    window: int = SIGNAL_CONFIRMATION_WINDOW,
) -> tuple[bool, dict[str, pd.Timestamp]]:
    """
    Check whether all SHORT events occurred inside
    the confirmation window.
    """

    events = _event_dates_in_window(
        df=df,
        event_columns=SHORT_EVENT_COLUMNS,
        window=window,
    )

    complete = (
        len(events) == len(SHORT_EVENT_COLUMNS)
    )

    return complete, events


# ============================================================
# BUILD SETUP COLUMNS
# ============================================================

def build_setup_columns(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build LONG_SETUP and SHORT_SETUP.
    """

    validate_indicator_data(df)

    result = df.copy()

    window = SIGNAL_CONFIRMATION_WINDOW

    # --------------------------------------------------------
    # LONG EVENT WINDOW
    # --------------------------------------------------------

    long_event_counts = (
        result[LONG_EVENT_COLUMNS]
        .astype(int)
        .rolling(
            window=window,
            min_periods=window,
        )
        .sum()
    )

    result["LONG_EVENTS_COMPLETE"] = (
        long_event_counts
        .eq(1)
        .all(axis=1)
    )

    # --------------------------------------------------------
    # SHORT EVENT WINDOW
    # --------------------------------------------------------

    short_event_counts = (
        result[SHORT_EVENT_COLUMNS]
        .astype(int)
        .rolling(
            window=window,
            min_periods=window,
        )
        .sum()
    )

    result["SHORT_EVENTS_COMPLETE"] = (
        short_event_counts
        .eq(1)
        .all(axis=1)
    )

    # --------------------------------------------------------
    # CURRENT ALIGNMENT
    # --------------------------------------------------------

    if REQUIRE_CURRENT_TREND_ALIGNMENT:

        result["LONG_SETUP"] = (
            result["LONG_EVENTS_COMPLETE"]
            & result[
                "LONG_CURRENT_ALIGNMENT"
            ].fillna(False)
        )

        result["SHORT_SETUP"] = (
            result["SHORT_EVENTS_COMPLETE"]
            & result[
                "SHORT_CURRENT_ALIGNMENT"
            ].fillna(False)
        )

    else:

        result["LONG_SETUP"] = (
            result["LONG_EVENTS_COMPLETE"]
        )

        result["SHORT_SETUP"] = (
            result["SHORT_EVENTS_COMPLETE"]
        )

    return result


# ============================================================
# FRESH SIGNAL LOGIC
# ============================================================

def build_fresh_signal_columns(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Generate a signal only when a setup changes
    from False to True.

    This prevents repeated signals while the setup
    remains active.
    """

    result = df.copy()

    previous_long_setup = (
        result["LONG_SETUP"]
        .shift(1)
        .fillna(False)
        .astype(bool)
    )

    previous_short_setup = (
        result["SHORT_SETUP"]
        .shift(1)
        .fillna(False)
        .astype(bool)
    )

    result["LONG_SIGNAL"] = (
        result["LONG_SETUP"].astype(bool)
        & ~previous_long_setup
    )

    result["SHORT_SIGNAL"] = (
        result["SHORT_SETUP"].astype(bool)
        & ~previous_short_setup
    )

    return result


# ============================================================
# COMPLETE SIGNAL EVALUATION
# ============================================================

def evaluate_signals(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Complete signal pipeline.

    1. Calculate indicators if necessary.
    2. Build setup conditions.
    3. Build fresh signals.
    """

    result = df.copy()

    # --------------------------------------------------------
    # Check indicators
    # --------------------------------------------------------

    missing_indicators = [
        column
        for column in REQUIRED_INDICATOR_COLUMNS
        if column not in result.columns
    ]

    # --------------------------------------------------------
    # Calculate indicators if required
    # --------------------------------------------------------

    if missing_indicators:
        result = calculate_indicators(result)

    # --------------------------------------------------------
    # Build setup
    # --------------------------------------------------------

    result = build_setup_columns(result)

    # --------------------------------------------------------
    # Build fresh signal
    # --------------------------------------------------------

    result = build_fresh_signal_columns(result)

    return result


# ============================================================
# GET LATEST SIGNAL
# ============================================================

def get_latest_signal(
    df: pd.DataFrame,
) -> dict:
    """
    Return the latest LONG, SHORT, or NO_SIGNAL state.
    """

    result = evaluate_signals(df)

    latest = result.iloc[-1]

    if bool(latest["LONG_SIGNAL"]):

        signal = "LONG"

        reason = (
            "EMA 5 crossed above EMA 13 and EMA 26, "
            "MACD crossed bullish, and RSI crossed above 60 "
            "within the confirmation window."
        )

    elif bool(latest["SHORT_SIGNAL"]):

        signal = "SHORT"

        reason = (
            "EMA 5 crossed below EMA 13 and EMA 26, "
            "MACD crossed bearish, and RSI crossed below 40 "
            "within the confirmation window."
        )

    else:

        signal = "NO_SIGNAL"

        reason = (
            "The complete LONG or SHORT confirmation "
            "conditions are not currently satisfied."
        )

    return {
        "signal": signal,
        "date": result.index[-1],
        "close": latest.get("Close"),
        "reason": reason,
    }


# ============================================================
# SYNTHETIC TEST DATA
# ============================================================

def _create_test_data(
    periods: int = 220,
) -> pd.DataFrame:
    """
    Create synthetic daily OHLCV data for testing.
    """

    dates = pd.date_range(
        end="2025-08-08",
        periods=periods,
        freq="D",
    )

    close_prices = []

    price = 100.0

    for i in range(periods):

        if i < 35:
            change = 0.35

        elif i < 75:
            change = -0.90

        elif i < 115:
            change = 1.10

        elif i < 155:
            change = -1.20

        elif i < 195:
            change = 1.30

        else:
            change = 0.25

        price += change

        close_prices.append(price)

    close = pd.Series(
        close_prices,
        index=dates,
        dtype=float,
    )

    data = pd.DataFrame(index=dates)

    data["Open"] = close - 0.20
    data["High"] = close + 0.50
    data["Low"] = close - 0.50
    data["Close"] = close
    data["Volume"] = 1000000

    return data


# ============================================================
# DETERMINISTIC TEST
# ============================================================

def _create_deterministic_signal_test() -> pd.DataFrame:
    """
    Construct a deterministic LONG setup.

    Events:

        Day 6:
            EMA5 crosses above EMA13

        Day 7:
            EMA5 crosses above EMA26

        Day 8:
            MACD crosses bullish

        Day 9:
            RSI crosses above 60

    All four events occur inside the five-candle window.

    Expected:

        Day 9:
            LONG_EVENTS_COMPLETE = True
            LONG_SETUP = True
            LONG_SIGNAL = True

        Day 10:
            LONG_EVENTS_COMPLETE = True
            LONG_SETUP = True
            LONG_SIGNAL = False
    """

    dates = pd.date_range(
        end="2025-09-04",
        periods=10,
        freq="D",
    )

    df = pd.DataFrame(index=dates)

    # --------------------------------------------------------
    # OHLCV
    # --------------------------------------------------------

    df["Open"] = 100.0
    df["High"] = 101.0
    df["Low"] = 99.0
    df["Close"] = 100.0
    df["Volume"] = 1000000

    # --------------------------------------------------------
    # Indicators
    # --------------------------------------------------------

    df["EMA_5"] = 10.0
    df["EMA_13"] = 9.0
    df["EMA_26"] = 8.0

    df["MACD"] = 2.0
    df["MACD_SIGNAL"] = 1.0

    df["RSI"] = 65.0

    # --------------------------------------------------------
    # Event columns
    # --------------------------------------------------------

    for column in (
        LONG_EVENT_COLUMNS
        + SHORT_EVENT_COLUMNS
    ):
        df[column] = False

    # --------------------------------------------------------
    # LONG events
    # --------------------------------------------------------

    df.iloc[
        5,
        df.columns.get_loc(
            "EMA5_CROSS_ABOVE_13"
        ),
    ] = True

    df.iloc[
        6,
        df.columns.get_loc(
            "EMA5_CROSS_ABOVE_26"
        ),
    ] = True

    df.iloc[
        7,
        df.columns.get_loc(
            "MACD_BULLISH_CROSS"
        ),
    ] = True

    df.iloc[
        8,
        df.columns.get_loc(
            "RSI_CROSS_ABOVE_60"
        ),
    ] = True

    # --------------------------------------------------------
    # Current alignment
    # --------------------------------------------------------

    df["LONG_CURRENT_ALIGNMENT"] = True
    df["SHORT_CURRENT_ALIGNMENT"] = False

    return df


# ============================================================
# RUN TESTS
# ============================================================

def _run_tests() -> None:

    print("=" * 70)
    print("DAILY SIGNAL ENGINE TEST")
    print("=" * 70)

    # ========================================================
    # TEST 1
    # ========================================================

    test_data = _create_test_data()

    print()
    print("Test 1: Indicator + Signal Pipeline")
    print("-" * 70)

    result = evaluate_signals(test_data)

    print(
        f"Test rows: {len(result)}"
    )

    # --------------------------------------------------------
    # Signal summary
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("SIGNAL SUMMARY")
    print("=" * 70)

    print(
        "LONG signals :",
        int(result["LONG_SIGNAL"].sum()),
    )

    print(
        "SHORT signals:",
        int(result["SHORT_SIGNAL"].sum()),
    )

    # --------------------------------------------------------
    # Event counts
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("CROSSOVER EVENT COUNTS")
    print("=" * 70)

    event_columns = (
        LONG_EVENT_COLUMNS
        + SHORT_EVENT_COLUMNS
    )

    for column in event_columns:

        print(
            f"{column:<32}: "
            f"{int(result[column].sum())}"
        )

    # --------------------------------------------------------
    # Latest signal
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("LATEST SIGNAL")
    print("=" * 70)

    latest = get_latest_signal(test_data)

    print(
        f"{'signal':<21}: "
        f"{latest['signal']}"
    )

    print(
        f"{'date':<21}: "
        f"{latest['date']}"
    )

    print(
        f"{'close':<21}: "
        f"{latest['close']}"
    )

    print(
        f"{'reason':<21}: "
        f"{latest['reason']}"
    )

    # ========================================================
    # TEST 2
    # ========================================================

    print()
    print("=" * 70)
    print("DETERMINISTIC SIGNAL LOGIC TEST")
    print("=" * 70)

    deterministic = (
        _create_deterministic_signal_test()
    )

    deterministic_result = (
        evaluate_signals(deterministic)
    )

    # --------------------------------------------------------
    # Find all fresh LONG signals
    # --------------------------------------------------------

    long_signal_rows = deterministic_result[
        deterministic_result["LONG_SIGNAL"]
    ]

    # --------------------------------------------------------
    # Latest state
    # --------------------------------------------------------

    latest_deterministic = (
        deterministic_result.iloc[-1]
    )

    print()
    print(
        "LATEST LONG_EVENTS_COMPLETE:",
        bool(
            latest_deterministic[
                "LONG_EVENTS_COMPLETE"
            ]
        ),
    )

    print(
        "LATEST LONG_SETUP:",
        bool(
            latest_deterministic[
                "LONG_SETUP"
            ]
        ),
    )

    print(
        "LATEST LONG_SIGNAL:",
        bool(
            latest_deterministic[
                "LONG_SIGNAL"
            ]
        ),
    )

    # --------------------------------------------------------
    # Verify exactly one signal
    # --------------------------------------------------------

    print()

    if len(long_signal_rows) == 1:

        signal_date = long_signal_rows.index[0]

        print(
            "SUCCESS: LONG signal logic is working."
        )

        print(
            "Signal generated on:",
            signal_date,
        )

    elif len(long_signal_rows) == 0:

        print(
            "ERROR: No LONG signal was generated."
        )

    else:

        print(
            "ERROR: Multiple LONG signals were generated."
        )

    # ========================================================
    # TEST 3
    # ========================================================

    print()
    print("=" * 70)
    print("DETERMINISTIC TEST STATES")
    print("=" * 70)

    deterministic_columns = [
        "EMA5_CROSS_ABOVE_13",
        "EMA5_CROSS_ABOVE_26",
        "MACD_BULLISH_CROSS",
        "RSI_CROSS_ABOVE_60",
        "LONG_EVENTS_COMPLETE",
        "LONG_SETUP",
        "LONG_SIGNAL",
    ]

    # IMPORTANT:
    # These columns are in deterministic_result,
    # not deterministic.

    print(
        deterministic_result[
            deterministic_columns
        ].to_string()
    )

    # ========================================================
    # TEST 4
    # ========================================================

    print()
    print("=" * 70)
    print("LAST 10 SIGNAL STATES")
    print("=" * 70)

    columns_to_show = [
        "Close",
        "EMA_5",
        "EMA_13",
        "EMA_26",
        "MACD",
        "MACD_SIGNAL",
        "RSI",
        "LONG_EVENTS_COMPLETE",
        "SHORT_EVENTS_COMPLETE",
        "LONG_SETUP",
        "SHORT_SETUP",
        "LONG_SIGNAL",
        "SHORT_SIGNAL",
    ]

    print(
        result[
            columns_to_show
        ].tail(10).to_string()
    )

    # ========================================================
    # COMPLETE
    # ========================================================

    print()
    print("=" * 70)
    print("DAILY SIGNAL ENGINE TEST COMPLETED")
    print("=" * 70)


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    _run_tests()