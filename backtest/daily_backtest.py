"""
Regime-Filtered Daily Swing Trading Backtest

Purpose
-------
Test whether a broad-market regime filter improves the existing
daily swing-trading strategy without changing the core indicators.

BASE STRATEGY
-------------
LONG:
    EMA 5 crosses above EMA 13
    EMA 5 crosses above EMA 26
    MACD crosses above Signal
    RSI crosses above 60

SHORT:
    EMA 5 crosses below EMA 13
    EMA 5 crosses below EMA 26
    MACD crosses below Signal
    RSI crosses below 40

Events can occur on different candles but must occur within
SIGNAL_CONFIRMATION_WINDOW candles.

REGIME FILTER
-------------
LONG trades are allowed only when:

    NIFTY 50 close > NIFTY 50 EMA 200
    AND
    NIFTY 50 EMA 20 > NIFTY 50 EMA 50

SHORT trades are allowed only when:

    NIFTY 50 close < NIFTY 50 EMA 200
    AND
    NIFTY 50 EMA 20 < NIFTY 50 EMA 50

The regime is calculated using data available on the signal date.
Entry remains the next trading-day open.

EXECUTION
---------
Entry slippage : 0.05%
Exit slippage  : 0.05%
Transaction cost: 0.15% once per trade

Stop loss      : 2.5%
Target         : 5.0%
Maximum hold   : 30 trading sessions

Same-candle stop/target:
    STOP LOSS first

This experiment does NOT overwrite the baseline backtest results.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf


# ============================================================================
# CONFIGURATION
# ============================================================================

LIQUID_UNIVERSE_FILE = Path("output/liquid_universe.csv")

SELECTED_UNIVERSE_FILE = Path(
    "output/backtest_regime_selected_universe.csv"
)

TRADES_OUTPUT_FILE = Path(
    "output/backtest_regime_filtered_trades.csv"
)

SUMMARY_OUTPUT_FILE = Path(
    "output/backtest_regime_filtered_summary.csv"
)

REGIME_OUTPUT_FILE = Path(
    "output/nifty50_regime.csv"
)

TEST_UNIVERSE_SIZE = 100
BALANCE_EXCHANGES = True

PERIOD = "2y"
INTERVAL = "1d"

MIN_DATA_ROWS = 50

EMA_FAST = 5
EMA_MID = 13
EMA_SLOW = 26

MACD_FAST = 12
MACD_SLOW = 26
MACD_SIGNAL = 9

RSI_PERIOD = 14
RSI_BULL_THRESHOLD = 60
RSI_BEAR_THRESHOLD = 40

SIGNAL_CONFIRMATION_WINDOW = 5

STOP_LOSS_PCT = 0.025
TARGET_PCT = 0.05

MAX_HOLDING_SESSIONS = 30

ENTRY_SLIPPAGE_PCT = 0.0005
EXIT_SLIPPAGE_PCT = 0.0005
TRANSACTION_COST_PCT = 0.0015

NIFTY_SYMBOL = "^NSEI"

REGIME_EMA_FAST = 20
REGIME_EMA_SLOW = 50
REGIME_EMA_TREND = 200


# ============================================================================
# HELPERS
# ============================================================================

def normalize_bool_series(series):
    """
    Convert a pandas Series to reliable boolean values.
    """
    if series.dtype == bool:
        return series.fillna(False)

    return (
        series.astype(str)
        .str.strip()
        .str.upper()
        .isin(["TRUE", "1", "YES", "PASS"])
    )


def safe_float(value):
    """
    Convert a value to float safely.
    """
    try:
        return float(value)
    except (TypeError, ValueError):
        return np.nan


# ============================================================================
# LIQUID UNIVERSE
# ============================================================================

def load_test_universe():
    """
    Load only PASS stocks from the liquidity scanner.

    Select the top stocks by average volume, balanced between NSE and BSE
    where possible.
    """

    print("=" * 70)
    print("LIQUID UNIVERSE + REGIME BACKTEST SELECTION")
    print("=" * 70)

    if not LIQUID_UNIVERSE_FILE.exists():
        raise FileNotFoundError(
            f"Missing liquidity file: {LIQUID_UNIVERSE_FILE}"
        )

    df = pd.read_csv(LIQUID_UNIVERSE_FILE)

    print(f"Original input rows       : {len(df)}")

    required_columns = [
        "exchange",
        "instrument_type",
        "symbol",
        "data_symbol",
        "latest_close",
        "avg_volume",
        "liquidity_status",
    ]

    missing = [
        column for column in required_columns
        if column not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing required columns in liquid universe: {missing}"
        )

    # ----------------------------------------------------------------------
    # Normalize columns
    # ----------------------------------------------------------------------

    df["exchange"] = (
        df["exchange"]
        .astype(str)
        .str.upper()
        .str.strip()
    )

    df["instrument_type"] = (
        df["instrument_type"]
        .astype(str)
        .str.upper()
        .str.strip()
    )

    df["liquidity_status"] = (
        df["liquidity_status"]
        .astype(str)
        .str.upper()
        .str.strip()
    )

    df["symbol"] = (
        df["symbol"]
        .astype(str)
        .str.strip()
    )

    df["data_symbol"] = (
        df["data_symbol"]
        .astype(str)
        .str.strip()
    )

    df["avg_volume"] = pd.to_numeric(
        df["avg_volume"],
        errors="coerce",
    )

    df["latest_close"] = pd.to_numeric(
        df["latest_close"],
        errors="coerce",
    )

    # ----------------------------------------------------------------------
    # Keep actual liquid PASS stocks only
    # ----------------------------------------------------------------------

    df = df[
        (df["liquidity_status"] == "PASS")
        & (df["instrument_type"] == "STOCK")
        & (df["exchange"].isin(["NSE", "BSE"]))
    ].copy()

    print(f"PASS stock rows           : {len(df)}")

    # Remove invalid symbols.
    df = df[
        df["data_symbol"].notna()
        & (df["data_symbol"] != "")
        & (df["data_symbol"].str.upper() != "NAN")
    ].copy()

    # Remove duplicate Yahoo data symbols.
    df = (
        df.sort_values(
            ["avg_volume", "symbol"],
            ascending=[False, True],
        )
        .drop_duplicates(
            subset=["data_symbol"],
            keep="first",
        )
        .copy()
    )

    nse = df[df["exchange"] == "NSE"].copy()
    bse = df[df["exchange"] == "BSE"].copy()

    print(f"Available NSE PASS stocks : {len(nse)}")
    print(f"Available BSE PASS stocks : {len(bse)}")

    # ----------------------------------------------------------------------
    # Select top liquid stocks
    # ----------------------------------------------------------------------

    if BALANCE_EXCHANGES:
        nse_count = min(TEST_UNIVERSE_SIZE // 2, len(nse))
        bse_count = min(TEST_UNIVERSE_SIZE // 2, len(bse))

        selected_nse = (
            nse.sort_values(
                ["avg_volume", "symbol"],
                ascending=[False, True],
            )
            .head(nse_count)
        )

        selected_bse = (
            bse.sort_values(
                ["avg_volume", "symbol"],
                ascending=[False, True],
            )
            .head(bse_count)
        )

        selected = pd.concat(
            [selected_nse, selected_bse],
            ignore_index=True,
        )

        # If one exchange did not have enough stocks, fill remaining
        # positions from the other exchange.
        remaining = TEST_UNIVERSE_SIZE - len(selected)

        if remaining > 0:
            selected_symbols = set(
                selected["data_symbol"].astype(str)
            )

            additional = (
                df[
                    ~df["data_symbol"].astype(str).isin(
                        selected_symbols
                    )
                ]
                .sort_values(
                    ["avg_volume", "symbol"],
                    ascending=[False, True],
                )
                .head(remaining)
            )

            selected = pd.concat(
                [selected, additional],
                ignore_index=True,
            )

    else:
        selected = (
            df.sort_values(
                ["avg_volume", "symbol"],
                ascending=[False, True],
            )
            .head(TEST_UNIVERSE_SIZE)
            .copy()
        )

    selected = selected.reset_index(drop=True)

    # Final safety check.
    selected = selected.drop_duplicates(
        subset=["data_symbol"],
        keep="first",
    ).reset_index(drop=True)

    print()
    print(f"Selected stocks           : {len(selected)}")
    print(
        f"Selected NSE              : "
        f"{(selected['exchange'] == 'NSE').sum()}"
    )
    print(
        f"Selected BSE              : "
        f"{(selected['exchange'] == 'BSE').sum()}"
    )

    print()
    print("TOP SELECTED STOCKS")
    print("-" * 70)

    display_columns = [
        "exchange",
        "symbol",
        "name",
        "data_symbol",
        "latest_close",
        "avg_volume",
        "liquidity_status",
    ]

    print(
        selected[display_columns]
        .to_string(index=False)
    )

    SELECTED_UNIVERSE_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    selected.to_csv(
        SELECTED_UNIVERSE_FILE,
        index=False,
    )

    print()
    print(
        f"Saved: {SELECTED_UNIVERSE_FILE}"
    )

    return selected


# ============================================================================
# DATA DOWNLOAD
# ============================================================================

def download_data(symbol):
    """
    Download daily OHLCV data.
    """

    try:
        data = yf.Ticker(symbol).history(
            period=PERIOD,
            interval=INTERVAL,
            auto_adjust=False,
            actions=False,
        )

    except Exception as exc:
        print(f"  ERROR downloading data: {exc}")
        return pd.DataFrame()

    if data is None or data.empty:
        return pd.DataFrame()

    # Handle MultiIndex columns defensively.
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = [
            column[0]
            if isinstance(column, tuple)
            else column
            for column in data.columns
        ]

    data.columns = [
        str(column).lower()
        for column in data.columns
    ]

    required = [
        "open",
        "high",
        "low",
        "close",
        "volume",
    ]

    missing = [
        column for column in required
        if column not in data.columns
    ]

    if missing:
        print(
            f"  ERROR missing OHLCV columns: {missing}"
        )
        return pd.DataFrame()

    data = data[required].copy()

    for column in required:
        data[column] = pd.to_numeric(
            data[column],
            errors="coerce",
        )

    data = data.dropna(
        subset=required
    ).copy()

    # Remove timezone from index.
    try:
        if getattr(data.index, "tz", None) is not None:
            data.index = data.index.tz_localize(None)
    except Exception:
        pass

    data = data.sort_index()

    return data


# ============================================================================
# INDICATORS
# ============================================================================

def calculate_indicators(data):
    """
    Calculate the exact strategy indicators.
    """

    df = data.copy()

    df["ema_5"] = (
        df["close"]
        .ewm(
            span=EMA_FAST,
            adjust=False,
            min_periods=EMA_FAST,
        )
        .mean()
    )

    df["ema_13"] = (
        df["close"]
        .ewm(
            span=EMA_MID,
            adjust=False,
            min_periods=EMA_MID,
        )
        .mean()
    )

    df["ema_26"] = (
        df["close"]
        .ewm(
            span=EMA_SLOW,
            adjust=False,
            min_periods=EMA_SLOW,
        )
        .mean()
    )

    ema_fast = (
        df["close"]
        .ewm(
            span=MACD_FAST,
            adjust=False,
            min_periods=MACD_FAST,
        )
        .mean()
    )

    ema_slow = (
        df["close"]
        .ewm(
            span=MACD_SLOW,
            adjust=False,
            min_periods=MACD_SLOW,
        )
        .mean()
    )

    df["macd"] = ema_fast - ema_slow

    df["macd_signal"] = (
        df["macd"]
        .ewm(
            span=MACD_SIGNAL,
            adjust=False,
            min_periods=MACD_SIGNAL,
        )
        .mean()
    )

    delta = df["close"].diff()

    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = (
        gain.ewm(
            alpha=1 / RSI_PERIOD,
            adjust=False,
            min_periods=RSI_PERIOD,
        )
        .mean()
    )

    avg_loss = (
        loss.ewm(
            alpha=1 / RSI_PERIOD,
            adjust=False,
            min_periods=RSI_PERIOD,
        )
        .mean()
    )

    rs = avg_gain / avg_loss

    df["rsi"] = 100 - (
        100 / (1 + rs)
    )

    # Exact crossover events.
    df["ema5_cross_above_13"] = (
        (df["ema_5"] > df["ema_13"])
        & (df["ema_5"].shift(1) <= df["ema_13"].shift(1))
    )

    df["ema5_cross_above_26"] = (
        (df["ema_5"] > df["ema_26"])
        & (df["ema_5"].shift(1) <= df["ema_26"].shift(1))
    )

    df["macd_bullish_cross"] = (
        (df["macd"] > df["macd_signal"])
        & (
            df["macd"].shift(1)
            <= df["macd_signal"].shift(1)
        )
    )

    df["rsi_cross_above_60"] = (
        (df["rsi"] > RSI_BULL_THRESHOLD)
        & (
            df["rsi"].shift(1)
            <= RSI_BULL_THRESHOLD
        )
    )

    df["ema5_cross_below_13"] = (
        (df["ema_5"] < df["ema_13"])
        & (df["ema_5"].shift(1) >= df["ema_13"].shift(1))
    )

    df["ema5_cross_below_26"] = (
        (df["ema_5"] < df["ema_26"])
        & (df["ema_5"].shift(1) >= df["ema_26"].shift(1))
    )

    df["macd_bearish_cross"] = (
        (df["macd"] < df["macd_signal"])
        & (
            df["macd"].shift(1)
            >= df["macd_signal"].shift(1)
        )
    )

    df["rsi_cross_below_40"] = (
        (df["rsi"] < RSI_BEAR_THRESHOLD)
        & (
            df["rsi"].shift(1)
            >= RSI_BEAR_THRESHOLD
        )
    )

    # ----------------------------------------------------------------------
    # Rolling event counts
    # ----------------------------------------------------------------------

    window = SIGNAL_CONFIRMATION_WINDOW

    df["long_ema13_count"] = (
        df["ema5_cross_above_13"]
        .astype(int)
        .rolling(window)
        .sum()
    )

    df["long_ema26_count"] = (
        df["ema5_cross_above_26"]
        .astype(int)
        .rolling(window)
        .sum()
    )

    df["long_macd_count"] = (
        df["macd_bullish_cross"]
        .astype(int)
        .rolling(window)
        .sum()
    )

    df["long_rsi_count"] = (
        df["rsi_cross_above_60"]
        .astype(int)
        .rolling(window)
        .sum()
    )

    df["short_ema13_count"] = (
        df["ema5_cross_below_13"]
        .astype(int)
        .rolling(window)
        .sum()
    )

    df["short_ema26_count"] = (
        df["ema5_cross_below_26"]
        .astype(int)
        .rolling(window)
        .sum()
    )

    df["short_macd_count"] = (
        df["macd_bearish_cross"]
        .astype(int)
        .rolling(window)
        .sum()
    )

    df["short_rsi_count"] = (
        df["rsi_cross_below_40"]
        .astype(int)
        .rolling(window)
        .sum()
    )

    # ----------------------------------------------------------------------
    # Exact event requirement: every event exactly once
    # ----------------------------------------------------------------------

    df["long_events_valid"] = (
        (df["long_ema13_count"] == 1)
        & (df["long_ema26_count"] == 1)
        & (df["long_macd_count"] == 1)
        & (df["long_rsi_count"] == 1)
    )

    df["short_events_valid"] = (
        (df["short_ema13_count"] == 1)
        & (df["short_ema26_count"] == 1)
        & (df["short_macd_count"] == 1)
        & (df["short_rsi_count"] == 1)
    )

    # ----------------------------------------------------------------------
    # Current alignment
    # ----------------------------------------------------------------------

    df["long_alignment"] = (
        (df["ema_5"] > df["ema_13"])
        & (df["ema_13"] > df["ema_26"])
        & (df["macd"] > df["macd_signal"])
        & (df["rsi"] > RSI_BULL_THRESHOLD)
    )

    df["short_alignment"] = (
        (df["ema_5"] < df["ema_13"])
        & (df["ema_13"] < df["ema_26"])
        & (df["macd"] < df["macd_signal"])
        & (df["rsi"] < RSI_BEAR_THRESHOLD)
    )

    df["long_setup"] = (
        df["long_events_valid"]
        & df["long_alignment"]
    )

    df["short_setup"] = (
        df["short_events_valid"]
        & df["short_alignment"]
    )

    # ----------------------------------------------------------------------
    # Fresh setup logic
    # ----------------------------------------------------------------------

    df["long_signal"] = (
        df["long_setup"]
        & ~df["long_setup"].shift(
            1,
            fill_value=False,
        )
    )

    df["short_signal"] = (
        df["short_setup"]
        & ~df["short_setup"].shift(
            1,
            fill_value=False,
        )
    )

    return df


# ============================================================================
# NIFTY MARKET REGIME
# ============================================================================

def build_nifty_regime():
    """
    Download NIFTY 50 and calculate the market regime.

    Important:
        Regime is calculated using the NIFTY close on each date.

    LONG regime:
        close > EMA200
        AND EMA20 > EMA50

    SHORT regime:
        close < EMA200
        AND EMA20 < EMA50
    """

    print()
    print("=" * 70)
    print("NIFTY 50 MARKET REGIME")
    print("=" * 70)

    print()
    print(f"Downloading NIFTY 50: {NIFTY_SYMBOL}")

    nifty = download_data(NIFTY_SYMBOL)

    if nifty.empty:
        raise RuntimeError(
            "Unable to download NIFTY 50 data."
        )

    print(
        f"NIFTY raw data rows      : {len(nifty)}"
    )

    regime = nifty.copy()

    regime["ema_20"] = (
        regime["close"]
        .ewm(
            span=REGIME_EMA_FAST,
            adjust=False,
            min_periods=REGIME_EMA_FAST,
        )
        .mean()
    )

    regime["ema_50"] = (
        regime["close"]
        .ewm(
            span=REGIME_EMA_SLOW,
            adjust=False,
            min_periods=REGIME_EMA_SLOW,
        )
        .mean()
    )

    regime["ema_200"] = (
        regime["close"]
        .ewm(
            span=REGIME_EMA_TREND,
            adjust=False,
            min_periods=REGIME_EMA_TREND,
        )
        .mean()
    )

    regime["bull_regime"] = (
        (regime["close"] > regime["ema_200"])
        & (regime["ema_20"] > regime["ema_50"])
    )

    regime["bear_regime"] = (
        (regime["close"] < regime["ema_200"])
        & (regime["ema_20"] < regime["ema_50"])
    )

    regime["neutral_regime"] = ~(
        regime["bull_regime"]
        | regime["bear_regime"]
    )

    regime["regime"] = np.select(
        [
            regime["bull_regime"],
            regime["bear_regime"],
        ],
        [
            "BULL",
            "BEAR",
        ],
        default="NEUTRAL",
    )

    output = regime[
        [
            "close",
            "ema_20",
            "ema_50",
            "ema_200",
            "bull_regime",
            "bear_regime",
            "neutral_regime",
            "regime",
        ]
    ].copy()

    output.index.name = "date"

    REGIME_OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    output.to_csv(REGIME_OUTPUT_FILE)

    print(
        f"Saved NIFTY regime data: {REGIME_OUTPUT_FILE}"
    )

    valid = output[
        output["ema_200"].notna()
    ].copy()

    if not valid.empty:
        regime_counts = (
            valid["regime"]
            .value_counts()
        )

        print()
        print("REGIME DISTRIBUTION")
        print("-" * 70)

        for regime_name in [
            "BULL",
            "BEAR",
            "NEUTRAL",
        ]:
            count = int(
                regime_counts.get(
                    regime_name,
                    0,
                )
            )

            pct = (
                count / len(valid) * 100
                if len(valid) > 0
                else 0
            )

            print(
                f"{regime_name:<10} "
                f"{count:>5} days "
                f"({pct:>6.2f}%)"
            )

    return output


# ============================================================================
# TRADE EXECUTION
# ============================================================================

def execute_trade(
    df,
    signal_index,
    direction,
    symbol,
    exchange,
    regime,
):
    """
    Execute a single signal.

    Signal occurs at daily close.
    Entry occurs on next trading-day open.
    """

    if signal_index + 1 >= len(df):
        return None

    signal_row = df.iloc[signal_index]

    signal_date = df.index[signal_index]

    entry_index = signal_index + 1

    entry_row = df.iloc[entry_index]

    entry_date = df.index[entry_index]

    raw_entry_price = safe_float(
        entry_row["open"]
    )

    if not np.isfinite(raw_entry_price):
        return None

    # Adverse entry slippage.
    if direction == "LONG":
        entry_price = (
            raw_entry_price
            * (1 + ENTRY_SLIPPAGE_PCT)
        )

        stop_loss = (
            entry_price
            * (1 - STOP_LOSS_PCT)
        )

        target = (
            entry_price
            * (1 + TARGET_PCT)
        )

    else:
        entry_price = (
            raw_entry_price
            * (1 - ENTRY_SLIPPAGE_PCT)
        )

        stop_loss = (
            entry_price
            * (1 + STOP_LOSS_PCT)
        )

        target = (
            entry_price
            * (1 - TARGET_PCT)
        )

    exit_index = None
    raw_exit_price = None
    exit_reason = None

    # ----------------------------------------------------------------------
    # Walk forward through future trading sessions.
    # ----------------------------------------------------------------------

    max_index = min(
        len(df) - 1,
        entry_index + MAX_HOLDING_SESSIONS - 1,
    )

    for i in range(
        entry_index,
        max_index + 1,
    ):
        row = df.iloc[i]

        high = safe_float(row["high"])
        low = safe_float(row["low"])
        close = safe_float(row["close"])

        if not (
            np.isfinite(high)
            and np.isfinite(low)
            and np.isfinite(close)
        ):
            continue

        # --------------------------------------------------------------
        # LONG
        # --------------------------------------------------------------

        if direction == "LONG":

            # STOP FIRST when both stop and target are touched.
            if low <= stop_loss:
                exit_index = i
                raw_exit_price = stop_loss
                exit_reason = "STOP_LOSS"
                break

            if high >= target:
                exit_index = i
                raw_exit_price = target
                exit_reason = "TARGET"
                break

        # --------------------------------------------------------------
        # SHORT
        # --------------------------------------------------------------

        else:

            # STOP FIRST when both stop and target are touched.
            if high >= stop_loss:
                exit_index = i
                raw_exit_price = stop_loss
                exit_reason = "STOP_LOSS"
                break

            if low <= target:
                exit_index = i
                raw_exit_price = target
                exit_reason = "TARGET"
                break

    # ----------------------------------------------------------------------
    # Time exit
    # ----------------------------------------------------------------------

    if exit_index is None:

        # If max holding session exists inside the data,
        # close there.
        if max_index >= entry_index:
            exit_index = max_index

            raw_exit_price = safe_float(
                df.iloc[exit_index]["close"]
            )

            exit_reason = "TIME_EXIT"

        else:
            return None

    if not np.isfinite(raw_exit_price):
        return None

    exit_date = df.index[exit_index]

    # ----------------------------------------------------------------------
    # Adverse exit slippage.
    # ----------------------------------------------------------------------

    if direction == "LONG":
        exit_price = (
            raw_exit_price
            * (1 - EXIT_SLIPPAGE_PCT)
        )

        gross_pnl_pct = (
            (exit_price - entry_price)
            / entry_price
            * 100
        )

    else:
        exit_price = (
            raw_exit_price
            * (1 + EXIT_SLIPPAGE_PCT)
        )

        gross_pnl_pct = (
            (entry_price - exit_price)
            / entry_price
            * 100
        )

    net_pnl_pct = (
        gross_pnl_pct
        - TRANSACTION_COST_PCT * 100
    )

    holding_days = (
        exit_index
        - entry_index
        + 1
    )

    # ----------------------------------------------------------------------
    # Extract signal indicators.
    # ----------------------------------------------------------------------

    signal_close = safe_float(
        signal_row["close"]
    )

    rsi = safe_float(
        signal_row["rsi"]
    )

    ema_5 = safe_float(
        signal_row["ema_5"]
    )

    ema_13 = safe_float(
        signal_row["ema_13"]
    )

    ema_26 = safe_float(
        signal_row["ema_26"]
    )

    macd = safe_float(
        signal_row["macd"]
    )

    macd_signal = safe_float(
        signal_row["macd_signal"]
    )

    macd_histogram = (
        macd - macd_signal
        if np.isfinite(macd)
        and np.isfinite(macd_signal)
        else np.nan
    )

    extension_pct = (
        abs(signal_close - ema_5)
        / ema_5
        * 100
        if np.isfinite(signal_close)
        and np.isfinite(ema_5)
        and ema_5 != 0
        else np.nan
    )

    risk_reward = (
        TARGET_PCT
        / STOP_LOSS_PCT
    )

    return {
        "symbol": symbol,
        "exchange": exchange,
        "direction": direction,
        "signal_date": signal_date.strftime("%Y-%m-%d"),
        "entry_date": entry_date.strftime("%Y-%m-%d"),
        "exit_date": exit_date.strftime("%Y-%m-%d"),
        "entry_price": entry_price,
        "stop_loss": stop_loss,
        "target": target,
        "exit_price": exit_price,
        "raw_exit_price": raw_exit_price,
        "exit_reason": exit_reason,
        "holding_days": holding_days,
        "gross_pnl_pct": gross_pnl_pct,
        "transaction_cost_pct": TRANSACTION_COST_PCT * 100,
        "net_pnl_pct": net_pnl_pct,
        "pnl_pct": net_pnl_pct,
        "signal_close": signal_close,
        "rsi": rsi,
        "ema_5": ema_5,
        "ema_13": ema_13,
        "ema_26": ema_26,
        "macd": macd,
        "macd_signal": macd_signal,
        "macd_histogram": macd_histogram,
        "extension_pct": extension_pct,
        "risk_reward": risk_reward,
        "market_regime": regime,
    }


# ============================================================================
# BACKTEST ONE STOCK
# ============================================================================

def backtest_stock(
    stock_row,
    nifty_regime,
):
    """
    Backtest one stock.
    """

    exchange = str(
        stock_row["exchange"]
    )

    symbol = str(
        stock_row["symbol"]
    )

    data_symbol = str(
        stock_row["data_symbol"]
    )

    print(
        f"\n{exchange} {symbol} "
        f"({data_symbol})"
    )

    data = download_data(
        data_symbol
    )

    print(
        f"  Raw data rows: {len(data)}"
    )

    if len(data) < MIN_DATA_ROWS:
        print(
            "  SKIP: insufficient data"
        )
        return []

    df = calculate_indicators(data)

    usable_columns = [
        "open",
        "high",
        "low",
        "close",
        "ema_5",
        "ema_13",
        "ema_26",
        "macd",
        "macd_signal",
        "rsi",
    ]

    df = df.dropna(
        subset=usable_columns
    ).copy()

    print(
        f"  Usable indicator rows: {len(df)}"
    )

    trades = []

    for i in range(len(df)):

        signal_date = df.index[i]

        if signal_date not in nifty_regime.index:
            continue

        regime_row = nifty_regime.loc[
            signal_date
        ]

        regime = regime_row.get(
            "regime",
            "NEUTRAL",
        )

        # --------------------------------------------------------------
        # LONG
        # --------------------------------------------------------------

        if bool(df.iloc[i]["long_signal"]):

            # Regime filter:
            # LONG only in BULL market.
            if regime == "BULL":

                trade = execute_trade(
                    df=df,
                    signal_index=i,
                    direction="LONG",
                    symbol=data_symbol,
                    exchange=exchange,
                    regime=regime,
                )

                if trade is not None:
                    trade["display_symbol"] = symbol
                    trade["data_symbol"] = data_symbol
                    trades.append(trade)

        # --------------------------------------------------------------
        # SHORT
        # --------------------------------------------------------------

        if bool(df.iloc[i]["short_signal"]):

            # Regime filter:
            # SHORT only in BEAR market.
            if regime == "BEAR":

                trade = execute_trade(
                    df=df,
                    signal_index=i,
                    direction="SHORT",
                    symbol=data_symbol,
                    exchange=exchange,
                    regime=regime,
                )

                if trade is not None:
                    trade["display_symbol"] = symbol
                    trade["data_symbol"] = data_symbol
                    trades.append(trade)

    print(
        f"  Trades generated: {len(trades)}"
    )

    return trades


# ============================================================================
# PERFORMANCE
# ============================================================================

def calculate_max_drawdown(returns):
    """
    Calculate compounded-equity maximum drawdown.
    """

    if not returns:
        return 0.0

    equity = 1.0
    peak = 1.0
    max_drawdown = 0.0

    for ret in returns:
        equity *= 1 + ret / 100

        if equity > peak:
            peak = equity

        drawdown = (
            equity - peak
        ) / peak

        if drawdown < max_drawdown:
            max_drawdown = drawdown

    return max_drawdown * 100


def calculate_profit_factor(winners, losers):
    """
    Profit factor = gross profit / gross loss.
    """

    gross_profit = sum(
        r for r in winners
        if r > 0
    )

    gross_loss = abs(
        sum(
            r for r in losers
            if r < 0
        )
    )

    if gross_loss == 0:
        if gross_profit > 0:
            return float("inf")
        return 0.0

    return gross_profit / gross_loss


# ============================================================================
# SUMMARY
# ============================================================================

def build_summary(trades):
    """
    Build summary metrics.
    """

    if not trades:
        return {
            "total_trades": 0,
            "winning_trades": 0,
            "losing_trades": 0,
            "win_rate_pct": 0.0,
            "average_net_return_pct": 0.0,
            "gross_average_return_pct": 0.0,
            "profit_factor": 0.0,
            "compounded_return_pct": 0.0,
            "max_drawdown_pct": 0.0,
            "average_holding_sessions": 0.0,
        }

    returns = [
        float(
            trade["net_pnl_pct"]
        )
        for trade in trades
    ]

    gross_returns = [
        float(
            trade["gross_pnl_pct"]
        )
        for trade in trades
    ]

    winners = [
        r for r in returns
        if r > 0
    ]

    losers = [
        r for r in returns
        if r <= 0
    ]

    equity = 1.0

    for ret in returns:
        equity *= 1 + ret / 100

    compounded_return = (
        equity - 1
    ) * 100

    return {
        "total_trades": len(trades),
        "winning_trades": len(winners),
        "losing_trades": len(losers),
        "win_rate_pct": (
            len(winners)
            / len(trades)
            * 100
        ),
        "average_net_return_pct": (
            np.mean(returns)
        ),
        "gross_average_return_pct": (
            np.mean(gross_returns)
        ),
        "profit_factor": (
            calculate_profit_factor(
                winners,
                losers,
            )
        ),
        "compounded_return_pct": (
            compounded_return
        ),
        "max_drawdown_pct": (
            calculate_max_drawdown(
                returns
            )
        ),
        "average_holding_sessions": (
            np.mean(
                [
                    trade["holding_days"]
                    for trade in trades
                ]
            )
        ),
    }


# ============================================================================
# MAIN
# ============================================================================

def main():

    print("=" * 70)
    print("REGIME-FILTERED DAILY SWING TRADING BACKTEST")
    print("=" * 70)

    print()
    print("REGIME FILTER")
    print(
        "LONG  : NIFTY close > EMA200 AND EMA20 > EMA50"
    )
    print(
        "SHORT : NIFTY close < EMA200 AND EMA20 < EMA50"
    )

    print()
    print("BACKTEST CONFIGURATION")
    print(f"Stocks selected          : {TEST_UNIVERSE_SIZE}")
    print(f"Period                   : {PERIOD}")
    print(f"Interval                 : {INTERVAL}")
    print(
        f"Confirmation window      : "
        f"{SIGNAL_CONFIRMATION_WINDOW}"
    )
    print(
        f"Stop loss                : "
        f"{STOP_LOSS_PCT * 100:.2f}%"
    )
    print(
        f"Target                   : "
        f"{TARGET_PCT * 100:.2f}%"
    )
    print(
        f"Maximum holding sessions : "
        f"{MAX_HOLDING_SESSIONS}"
    )

    print()
    print("EXECUTION COST ASSUMPTIONS")
    print(
        f"Entry slippage           : "
        f"{ENTRY_SLIPPAGE_PCT * 100:.3f}%"
    )
    print(
        f"Exit slippage            : "
        f"{EXIT_SLIPPAGE_PCT * 100:.3f}%"
    )
    print(
        f"Transaction cost         : "
        f"{TRANSACTION_COST_PCT * 100:.3f}%"
    )

    print()
    print("SIGNAL EXECUTION")
    print("Signal                   : daily close")
    print("Entry                    : next trading-day open")
    print("Fresh setup logic        : ENABLED")
    print("Same-candle stop/target  : STOP first")
    print("Holding period           : trading sessions")
    print("Market regime filter     : ENABLED")

    # ----------------------------------------------------------------------
    # Universe
    # ----------------------------------------------------------------------

    universe = load_test_universe()

    # ----------------------------------------------------------------------
    # NIFTY regime
    # ----------------------------------------------------------------------

    nifty_regime = build_nifty_regime()

    # Make sure dates are normalized consistently.
    nifty_regime.index = pd.to_datetime(
        nifty_regime.index
    ).normalize()

    # ----------------------------------------------------------------------
    # Backtest
    # ----------------------------------------------------------------------

    all_trades = []

    stocks_with_trades = 0
    stocks_without_trades = 0

    print()
    print("=" * 70)
    print("RUNNING REGIME-FILTERED BACKTEST")
    print("=" * 70)

    total_stocks = len(universe)

    for number, (_, stock_row) in enumerate(
        universe.iterrows(),
        start=1,
    ):

        print(
            f"\n[{number}/{total_stocks}]",
            end=" ",
        )

        trades = backtest_stock(
            stock_row,
            nifty_regime,
        )

        if trades:
            stocks_with_trades += 1
            all_trades.extend(trades)
        else:
            stocks_without_trades += 1

    # ----------------------------------------------------------------------
    # Save trades
    # ----------------------------------------------------------------------

    if all_trades:

        trades_df = pd.DataFrame(
            all_trades
        )

        trades_df["signal_date"] = pd.to_datetime(
            trades_df["signal_date"]
        )

        trades_df = trades_df.sort_values(
            "signal_date"
        ).reset_index(drop=True)

        trades_df["signal_date"] = (
            trades_df["signal_date"]
            .dt.strftime("%Y-%m-%d")
        )

    else:

        trades_df = pd.DataFrame(
            columns=[
                "symbol",
                "exchange",
                "direction",
                "signal_date",
                "entry_date",
                "exit_date",
                "entry_price",
                "stop_loss",
                "target",
                "exit_price",
                "raw_exit_price",
                "exit_reason",
                "holding_days",
                "gross_pnl_pct",
                "transaction_cost_pct",
                "net_pnl_pct",
                "pnl_pct",
                "signal_close",
                "rsi",
                "ema_5",
                "ema_13",
                "ema_26",
                "macd",
                "macd_signal",
                "macd_histogram",
                "extension_pct",
                "risk_reward",
                "market_regime",
                "display_symbol",
                "data_symbol",
            ]
        )

    TRADES_OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    trades_df.to_csv(
        TRADES_OUTPUT_FILE,
        index=False,
    )

    # ----------------------------------------------------------------------
    # Summary
    # ----------------------------------------------------------------------

    summary = build_summary(
        all_trades
    )

    summary_row = {
        "stocks_tested": len(universe),
        "stocks_with_trades": stocks_with_trades,
        "stocks_without_trades": stocks_without_trades,
        "total_trades": summary["total_trades"],
        "winning_trades": summary["winning_trades"],
        "losing_trades": summary["losing_trades"],
        "win_rate_pct": summary["win_rate_pct"],
        "average_net_return_pct": summary[
            "average_net_return_pct"
        ],
        "gross_average_return_pct": summary[
            "gross_average_return_pct"
        ],
        "profit_factor": summary[
            "profit_factor"
        ],
        "compounded_return_pct": summary[
            "compounded_return_pct"
        ],
        "max_drawdown_pct": summary[
            "max_drawdown_pct"
        ],
        "average_holding_sessions": summary[
            "average_holding_sessions"
        ],
        "stop_loss_pct": STOP_LOSS_PCT * 100,
        "target_pct": TARGET_PCT * 100,
        "entry_slippage_pct": ENTRY_SLIPPAGE_PCT * 100,
        "exit_slippage_pct": EXIT_SLIPPAGE_PCT * 100,
        "transaction_cost_pct": TRANSACTION_COST_PCT * 100,
        "confirmation_window": SIGNAL_CONFIRMATION_WINDOW,
        "regime_filter": "NIFTY_BULL_LONG_BEAR_SHORT",
    }

    pd.DataFrame(
        [summary_row]
    ).to_csv(
        SUMMARY_OUTPUT_FILE,
        index=False,
    )

    # ----------------------------------------------------------------------
    # Print results
    # ----------------------------------------------------------------------

    print()
    print("=" * 70)
    print("REGIME-FILTERED RESULTS")
    print("=" * 70)

    print(
        f"Stocks tested       : "
        f"{summary_row['stocks_tested']}"
    )

    print(
        f"Total trades        : "
        f"{summary_row['total_trades']}"
    )

    print(
        f"Winning trades      : "
        f"{summary_row['winning_trades']}"
    )

    print(
        f"Losing trades       : "
        f"{summary_row['losing_trades']}"
    )

    print(
        f"Win rate            : "
        f"{summary_row['win_rate_pct']:.2f}%"
    )

    print(
        f"Average net return  : "
        f"{summary_row['average_net_return_pct']:.2f}%"
    )

    print(
        f"Total compounded    : "
        f"{summary_row['compounded_return_pct']:.2f}%"
    )

    pf = summary_row["profit_factor"]

    if np.isinf(pf):
        pf_text = "INF"
    else:
        pf_text = f"{pf:.2f}"

    print(
        f"Profit factor       : "
        f"{pf_text}"
    )

    print(
        f"Max drawdown        : "
        f"{summary_row['max_drawdown_pct']:.2f}%"
    )

    print(
        f"Avg holding         : "
        f"{summary_row['average_holding_sessions']:.2f} "
        f"trading sessions"
    )

    # ----------------------------------------------------------------------
    # Gross vs net
    # ----------------------------------------------------------------------

    if all_trades:

        gross_returns = [
            trade["gross_pnl_pct"]
            for trade in all_trades
        ]

        net_returns = [
            trade["net_pnl_pct"]
            for trade in all_trades
        ]

        gross_equity = 1.0
        net_equity = 1.0

        for ret in gross_returns:
            gross_equity *= (
                1 + ret / 100
            )

        for ret in net_returns:
            net_equity *= (
                1 + ret / 100
            )

        gross_compounded = (
            gross_equity - 1
        ) * 100

        net_compounded = (
            net_equity - 1
        ) * 100

        print()
        print("GROSS VS NET")
        print("-" * 70)

        print(
            f"Gross average return : "
            f"{np.mean(gross_returns):.2f}%"
        )

        print(
            f"Net average return   : "
            f"{np.mean(net_returns):.2f}%"
        )

        print(
            f"Gross compounded     : "
            f"{gross_compounded:.2f}%"
        )

        print(
            f"Net compounded       : "
            f"{net_compounded:.2f}%"
        )

        print(
            f"Performance drag     : "
            f"{gross_compounded - net_compounded:.2f}%"
        )

    # ----------------------------------------------------------------------
    # Direction breakdown
    # ----------------------------------------------------------------------

    if all_trades:

        print()
        print("DIRECTION")
        print("-" * 70)

        for direction in [
            "LONG",
            "SHORT",
        ]:

            direction_trades = [
                trade
                for trade in all_trades
                if trade["direction"] == direction
            ]

            if not direction_trades:
                continue

            direction_returns = [
                trade["net_pnl_pct"]
                for trade in direction_trades
            ]

            direction_winners = [
                r
                for r in direction_returns
                if r > 0
            ]

            direction_losers = [
                r
                for r in direction_returns
                if r <= 0
            ]

            direction_pf = (
                calculate_profit_factor(
                    direction_winners,
                    direction_losers,
                )
            )

            if np.isinf(direction_pf):
                pf_text = "INF"
            else:
                pf_text = (
                    f"{direction_pf:.2f}"
                )

            print(
                f"{direction:<6} "
                f"trades={len(direction_trades)} "
                f"win_rate="
                f"{len(direction_winners) / len(direction_trades) * 100:.2f}% "
                f"avg_net="
                f"{np.mean(direction_returns):.2f}% "
                f"PF={pf_text}"
            )

    # ----------------------------------------------------------------------
    # Exchange breakdown
    # ----------------------------------------------------------------------

    if all_trades:

        print()
        print("EXCHANGE")
        print("-" * 70)

        for exchange in [
            "NSE",
            "BSE",
        ]:

            exchange_trades = [
                trade
                for trade in all_trades
                if trade["exchange"] == exchange
            ]

            if not exchange_trades:
                continue

            exchange_returns = [
                trade["net_pnl_pct"]
                for trade in exchange_trades
            ]

            exchange_winners = [
                r
                for r in exchange_returns
                if r > 0
            ]

            exchange_losers = [
                r
                for r in exchange_returns
                if r <= 0
            ]

            exchange_pf = (
                calculate_profit_factor(
                    exchange_winners,
                    exchange_losers,
                )
            )

            if np.isinf(exchange_pf):
                pf_text = "INF"
            else:
                pf_text = (
                    f"{exchange_pf:.2f}"
                )

            print(
                f"{exchange:<6} "
                f"trades={len(exchange_trades)} "
                f"win_rate="
                f"{len(exchange_winners) / len(exchange_trades) * 100:.2f}% "
                f"avg_net="
                f"{np.mean(exchange_returns):.2f}% "
                f"PF={pf_text}"
            )

    # ----------------------------------------------------------------------
    # Regime breakdown
    # ----------------------------------------------------------------------

    if all_trades:

        print()
        print("MARKET REGIME")
        print("-" * 70)

        for regime_name in [
            "BULL",
            "BEAR",
        ]:

            regime_trades = [
                trade
                for trade in all_trades
                if trade["market_regime"] == regime_name
            ]

            if not regime_trades:
                continue

            regime_returns = [
                trade["net_pnl_pct"]
                for trade in regime_trades
            ]

            regime_winners = [
                r
                for r in regime_returns
                if r > 0
            ]

            regime_losers = [
                r
                for r in regime_returns
                if r <= 0
            ]

            regime_pf = (
                calculate_profit_factor(
                    regime_winners,
                    regime_losers,
                )
            )

            if np.isinf(regime_pf):
                pf_text = "INF"
            else:
                pf_text = (
                    f"{regime_pf:.2f}"
                )

            print(
                f"{regime_name:<10} "
                f"trades={len(regime_trades)} "
                f"win_rate="
                f"{len(regime_winners) / len(regime_trades) * 100:.2f}% "
                f"avg_net="
                f"{np.mean(regime_returns):.2f}% "
                f"PF={pf_text}"
            )

    # ----------------------------------------------------------------------
    # Exit breakdown
    # ----------------------------------------------------------------------

    if all_trades:

        print()
        print("EXIT")
        print("-" * 70)

        exit_counts = {}

        for trade in all_trades:

            reason = trade[
                "exit_reason"
            ]

            exit_counts[reason] = (
                exit_counts.get(
                    reason,
                    0,
                )
                + 1
            )

        for reason, count in sorted(
            exit_counts.items()
        ):
            print(
                f"{reason:<15} {count}"
            )

    # ----------------------------------------------------------------------
    # Coverage
    # ----------------------------------------------------------------------

    print()
    print("TRADE COVERAGE")
    print("-" * 70)

    print(
        f"Stocks tested       : "
        f"{stocks_with_trades + stocks_without_trades}"
    )

    print(
        f"Stocks with trades  : "
        f"{stocks_with_trades}"
    )

    print(
        f"Stocks without trade: "
        f"{stocks_without_trades}"
    )

    # ----------------------------------------------------------------------
    # Files
    # ----------------------------------------------------------------------

    print()
    print("=" * 70)
    print("FILES SAVED")
    print("=" * 70)

    print(
        f"Selected universe : "
        f"{SELECTED_UNIVERSE_FILE}"
    )

    print(
        f"NIFTY regime      : "
        f"{REGIME_OUTPUT_FILE}"
    )

    print(
        f"Trades            : "
        f"{TRADES_OUTPUT_FILE}"
    )

    print(
        f"Summary           : "
        f"{SUMMARY_OUTPUT_FILE}"
    )

    print()
    print(
        "REGIME-FILTERED BACKTEST COMPLETE."
    )


if __name__ == "__main__":
    main()