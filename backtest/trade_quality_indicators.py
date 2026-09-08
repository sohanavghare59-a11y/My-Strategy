"""
Trade Quality Indicator Analysis
================================

Research-only analysis.

Reconstructs indicator values at each historical signal date from the
same daily OHLCV data used by the controlled RSI experiment.

Outputs:
    output/trade_quality_indicators.csv
    output/trade_quality_buckets.csv
    output/trade_quality_rsi.csv
    output/trade_quality_macd.csv
    output/trade_quality_ema.csv
    output/trade_quality_volume.csv
    output/trade_quality_freshness.csv
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from data.market_data import get_daily_data


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

TRADES_FILE = BASE_DIR / "output" / "controlled_rsi_trades.csv"
UNIVERSE_FILE = BASE_DIR / "output" / "backtest_selected_universe.csv"
REGIME_FILE = BASE_DIR / "output" / "nifty50_regime.csv"

OUTPUT_DIR = BASE_DIR / "output"

FEATURE_TRADES_FILE = OUTPUT_DIR / "trade_quality_indicators.csv"
BUCKET_FILE = OUTPUT_DIR / "trade_quality_buckets.csv"
RSI_FILE = OUTPUT_DIR / "trade_quality_rsi.csv"
MACD_FILE = OUTPUT_DIR / "trade_quality_macd.csv"
EMA_FILE = OUTPUT_DIR / "trade_quality_ema.csv"
VOLUME_FILE = OUTPUT_DIR / "trade_quality_volume.csv"
FRESHNESS_FILE = OUTPUT_DIR / "trade_quality_freshness.csv"


# ============================================================
# SETTINGS
# ============================================================

MIN_HISTORY_ROWS = 60

RSI_PERIOD = 14
RSI_LONG_LEVEL = 60
RSI_SHORT_LEVEL = 40

EMA_FAST = 5
EMA_MID = 13
EMA_SLOW = 26

MACD_FAST = 12
MACD_SLOW = 26
MACD_SIGNAL = 9

VOLUME_PERIOD = 20
SIGNAL_CONFIRMATION_WINDOW = 5


# ============================================================
# CROSSOVER HELPERS
# ============================================================

def crossed_above(a, b):
    return (
        (a > b)
        & (a.shift(1) <= b.shift(1))
    ).fillna(False)


def crossed_below(a, b):
    return (
        (a < b)
        & (a.shift(1) >= b.shift(1))
    ).fillna(False)


def recent_event(event):
    return (
        event.astype(int)
        .rolling(
            SIGNAL_CONFIRMATION_WINDOW,
            min_periods=1,
        )
        .sum()
        > 0
    )


# ============================================================
# RSI
# ============================================================

def calculate_rsi(close, period=RSI_PERIOD):

    delta = close.diff()

    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.ewm(
        alpha=1 / period,
        adjust=False,
        min_periods=period,
    ).mean()

    avg_loss = loss.ewm(
        alpha=1 / period,
        adjust=False,
        min_periods=period,
    ).mean()

    rs = avg_gain / avg_loss

    return 100 - (100 / (1 + rs))


# ============================================================
# PREPARE DATA
# ============================================================

def prepare_data(data):

    df = data.copy()

    df.columns = [
        str(c).lower()
        for c in df.columns
    ]

    required = [
        "open",
        "high",
        "low",
        "close",
        "volume",
    ]

    missing = [
        column
        for column in required
        if column not in df.columns
    ]

    if missing:
        raise ValueError(
            "Missing OHLCV columns: "
            + ", ".join(missing)
        )

    for column in required:
        df[column] = pd.to_numeric(
            df[column],
            errors="coerce",
        )

    df = df.dropna(
        subset=required
    )

    df.index = pd.to_datetime(
        df.index,
        errors="coerce",
    )

    df = df[
        ~df.index.isna()
    ]

    df = df.sort_index()

    df = df[
        ~df.index.duplicated(
            keep="last"
        )
    ]

    if len(df) < MIN_HISTORY_ROWS:
        raise ValueError(
            f"Only {len(df)} rows"
        )

    # --------------------------------------------------------
    # EMA
    # --------------------------------------------------------

    df["ema_5"] = df["close"].ewm(
        span=EMA_FAST,
        adjust=False,
        min_periods=EMA_FAST,
    ).mean()

    df["ema_13"] = df["close"].ewm(
        span=EMA_MID,
        adjust=False,
        min_periods=EMA_MID,
    ).mean()

    df["ema_26"] = df["close"].ewm(
        span=EMA_SLOW,
        adjust=False,
        min_periods=EMA_SLOW,
    ).mean()

    # --------------------------------------------------------
    # MACD
    # --------------------------------------------------------

    fast = df["close"].ewm(
        span=MACD_FAST,
        adjust=False,
        min_periods=MACD_FAST,
    ).mean()

    slow = df["close"].ewm(
        span=MACD_SLOW,
        adjust=False,
        min_periods=MACD_SLOW,
    ).mean()

    df["macd"] = fast - slow

    df["macd_signal"] = df["macd"].ewm(
        span=MACD_SIGNAL,
        adjust=False,
        min_periods=MACD_SIGNAL,
    ).mean()

    df["macd_histogram"] = (
        df["macd"]
        - df["macd_signal"]
    )

    # --------------------------------------------------------
    # RSI
    # --------------------------------------------------------

    df["rsi"] = calculate_rsi(
        df["close"],
        RSI_PERIOD,
    )

    # --------------------------------------------------------
    # VOLUME
    # --------------------------------------------------------

    df["volume_avg"] = (
        df["volume"]
        .rolling(
            VOLUME_PERIOD,
            min_periods=VOLUME_PERIOD,
        )
        .mean()
    )

    df["volume_ratio"] = (
        df["volume"]
        / df["volume_avg"]
    )

    df = df.dropna(
        subset=[
            "ema_5",
            "ema_13",
            "ema_26",
            "macd",
            "macd_signal",
            "rsi",
        ]
    )

    # --------------------------------------------------------
    # CROSSOVER EVENTS
    # --------------------------------------------------------

    df["ema_up_13"] = crossed_above(
        df["ema_5"],
        df["ema_13"],
    )

    df["ema_up_26"] = crossed_above(
        df["ema_5"],
        df["ema_26"],
    )

    df["ema_down_13"] = crossed_below(
        df["ema_5"],
        df["ema_13"],
    )

    df["ema_down_26"] = crossed_below(
        df["ema_5"],
        df["ema_26"],
    )

    df["macd_up"] = crossed_above(
        df["macd"],
        df["macd_signal"],
    )

    df["macd_down"] = crossed_below(
        df["macd"],
        df["macd_signal"],
    )

    df["rsi_up"] = (
        (df["rsi"] > RSI_LONG_LEVEL)
        & (
            df["rsi"].shift(1)
            <= RSI_LONG_LEVEL
        )
    ).fillna(False)

    df["rsi_down"] = (
        (df["rsi"] < RSI_SHORT_LEVEL)
        & (
            df["rsi"].shift(1)
            >= RSI_SHORT_LEVEL
        )
    ).fillna(False)

    # --------------------------------------------------------
    # EMA SETUPS
    # --------------------------------------------------------

    df["ema_long"] = (
        recent_event(df["ema_up_13"])
        & recent_event(df["ema_up_26"])
        & (df["ema_5"] > df["ema_13"])
        & (df["ema_13"] > df["ema_26"])
    )

    df["ema_short"] = (
        recent_event(df["ema_down_13"])
        & recent_event(df["ema_down_26"])
        & (df["ema_5"] < df["ema_13"])
        & (df["ema_13"] < df["ema_26"])
    )

    # --------------------------------------------------------
    # MACD SETUPS
    # --------------------------------------------------------

    df["macd_long"] = (
        recent_event(df["macd_up"])
        & (df["macd"] > df["macd_signal"])
    )

    df["macd_short"] = (
        recent_event(df["macd_down"])
        & (df["macd"] < df["macd_signal"])
    )

    # --------------------------------------------------------
    # RSI SETUPS
    # --------------------------------------------------------

    df["rsi_long"] = (
        recent_event(df["rsi_up"])
        & (df["rsi"] > RSI_LONG_LEVEL)
    )

    df["rsi_short"] = (
        recent_event(df["rsi_down"])
        & (df["rsi"] < RSI_SHORT_LEVEL)
    )

    # --------------------------------------------------------
    # COMPLETE SETUPS
    # --------------------------------------------------------

    df["current_long"] = (
        df["ema_long"]
        & df["macd_long"]
        & df["rsi_long"]
    )

    df["current_short"] = (
        df["ema_short"]
        & df["macd_short"]
        & df["rsi_short"]
    )

    df["ema_macd_long"] = (
        df["ema_long"]
        & df["macd_long"]
    )

    df["ema_macd_short"] = (
        df["ema_short"]
        & df["macd_short"]
    )

    return df


# ============================================================
# EVENT AGE
# ============================================================

def event_age(event_series):

    ages = np.full(
        len(event_series),
        np.nan,
    )

    last_event = None

    values = event_series.to_numpy()

    for i, value in enumerate(values):

        if bool(value):
            last_event = i

        if last_event is not None:
            ages[i] = i - last_event

    return pd.Series(
        ages,
        index=event_series.index,
    )


# ============================================================
# SIGNAL POSITION
# ============================================================

def find_signal_position(
    df,
    signal_date,
):

    target = pd.Timestamp(
        signal_date
    )

    exact = np.where(
        df.index == target
    )[0]

    if len(exact):
        return int(exact[0])

    normalized = (
        df.index.normalize()
        == target.normalize()
    )

    positions = np.where(
        normalized
    )[0]

    if len(positions):
        return int(positions[0])

    return None


# ============================================================
# FEATURE EXTRACTION
# ============================================================

def extract_features(
    df,
    signal_position,
    trade,
):

    row = df.iloc[
        signal_position
    ]

    direction = str(
        trade["direction"]
    ).upper()

    close = float(row["close"])
    ema5 = float(row["ema_5"])
    ema13 = float(row["ema_13"])
    ema26 = float(row["ema_26"])

    macd = float(row["macd"])
    macd_signal = float(
        row["macd_signal"]
    )

    histogram = float(
        row["macd_histogram"]
    )

    rsi = float(row["rsi"])

    volume_ratio = (
        float(row["volume_ratio"])
        if pd.notna(row["volume_ratio"])
        else np.nan
    )

    # --------------------------------------------------------
    # EMA spreads
    # --------------------------------------------------------

    ema_5_13_signed_pct = (
        (ema5 - ema13)
        / abs(ema13)
        * 100
    )

    ema_13_26_signed_pct = (
        (ema13 - ema26)
        / abs(ema26)
        * 100
    )

    ema_5_26_signed_pct = (
        (ema5 - ema26)
        / abs(ema26)
        * 100
    )

    ema_5_13_abs_pct = abs(
        ema_5_13_signed_pct
    )

    ema_13_26_abs_pct = abs(
        ema_13_26_signed_pct
    )

    ema_5_26_abs_pct = abs(
        ema_5_26_signed_pct
    )

    # --------------------------------------------------------
    # MACD
    # --------------------------------------------------------

    macd_histogram_pct = (
        histogram
        / close
        * 100
        if close != 0
        else np.nan
    )

    macd_histogram_abs_pct = abs(
        macd_histogram_pct
    )

    # --------------------------------------------------------
    # Extension from EMA5
    # --------------------------------------------------------

    extension_pct = (
        (close - ema5)
        / abs(ema5)
        * 100
        if ema5 != 0
        else np.nan
    )

    extension_abs_pct = abs(
        extension_pct
    )

    # --------------------------------------------------------
    # Event ages
    # --------------------------------------------------------

    ema13_event = (
        df["ema_up_13"]
        if direction == "LONG"
        else df["ema_down_13"]
    )

    ema26_event = (
        df["ema_up_26"]
        if direction == "LONG"
        else df["ema_down_26"]
    )

    macd_event = (
        df["macd_up"]
        if direction == "LONG"
        else df["macd_down"]
    )

    rsi_event = (
        df["rsi_up"]
        if direction == "LONG"
        else df["rsi_down"]
    )

    ema13_age = event_age(
        ema13_event
    ).iloc[
        signal_position
    ]

    ema26_age = event_age(
        ema26_event
    ).iloc[
        signal_position
    ]

    macd_age = event_age(
        macd_event
    ).iloc[
        signal_position
    ]

    rsi_age = event_age(
        rsi_event
    ).iloc[
        signal_position
    ]

    ages = [
        ema13_age,
        ema26_age,
        macd_age,
        rsi_age,
    ]

    valid_ages = [
        value
        for value in ages
        if pd.notna(value)
    ]

    max_event_age = (
        max(valid_ages)
        if valid_ages
        else np.nan
    )

    min_event_age = (
        min(valid_ages)
        if valid_ages
        else np.nan
    )

    result = dict(trade)

    result.update(
        {
            "signal_close": close,
            "rsi": rsi,
            "ema_5": ema5,
            "ema_13": ema13,
            "ema_26": ema26,
            "ema_5_13_signed_pct":
                ema_5_13_signed_pct,
            "ema_13_26_signed_pct":
                ema_13_26_signed_pct,
            "ema_5_26_signed_pct":
                ema_5_26_signed_pct,
            "ema_5_13_abs_pct":
                ema_5_13_abs_pct,
            "ema_13_26_abs_pct":
                ema_13_26_abs_pct,
            "ema_5_26_abs_pct":
                ema_5_26_abs_pct,
            "macd": macd,
            "macd_signal": macd_signal,
            "macd_histogram":
                histogram,
            "macd_histogram_pct":
                macd_histogram_pct,
            "macd_histogram_abs_pct":
                macd_histogram_abs_pct,
            "volume_ratio":
                volume_ratio,
            "extension_pct":
                extension_pct,
            "extension_abs_pct":
                extension_abs_pct,
            "ema13_event_age":
                ema13_age,
            "ema26_event_age":
                ema26_age,
            "macd_event_age":
                macd_age,
            "rsi_event_age":
                rsi_age,
            "max_event_age":
                max_event_age,
            "min_event_age":
                min_event_age,
        }
    )

    return result


# ============================================================
# LOAD UNIVERSE
# ============================================================

def load_universe():

    universe = pd.read_csv(
        UNIVERSE_FILE
    )

    universe["symbol"] = (
        universe["symbol"]
        .astype(str)
    )

    universe["exchange"] = (
        universe["exchange"]
        .astype(str)
        .str.upper()
    )

    universe["data_symbol"] = (
        universe["data_symbol"]
        .astype(str)
    )

    if "instrument_type" in universe.columns:

        universe = universe[
            universe[
                "instrument_type"
            ]
            .astype(str)
            .str.upper()
            .eq("STOCK")
        ]

    universe = universe[
        universe["exchange"].isin(
            ["NSE", "BSE"]
        )
    ]

    universe = universe.drop_duplicates(
        "data_symbol"
    )

    return universe.reset_index(
        drop=True
    )


# ============================================================
# LOAD REGIME
# ============================================================

def load_regime():

    if not REGIME_FILE.exists():
        return None

    regime = pd.read_csv(
        REGIME_FILE
    )

    if (
        "date" not in regime.columns
        or "regime" not in regime.columns
    ):
        return None

    regime["date"] = pd.to_datetime(
        regime["date"],
        errors="coerce",
    )

    regime = regime.dropna(
        subset=["date"]
    )

    regime["regime"] = (
        regime["regime"]
        .astype(str)
        .str.upper()
    )

    return (
        regime[
            ["date", "regime"]
        ]
        .drop_duplicates("date")
        .set_index("date")["regime"]
    )


# ============================================================
# METRICS
# ============================================================

def calculate_metrics(df):

    if df.empty:

        return {
            "trades": 0,
            "wins": 0,
            "losses": 0,
            "win_rate_pct": 0.0,
            "avg_net_pct": 0.0,
            "median_net_pct": 0.0,
            "profit_factor": 0.0,
            "avg_winner_pct": 0.0,
            "avg_loser_pct": 0.0,
            "avg_holding_days": 0.0,
        }

    net = pd.to_numeric(
        df["net_pnl_pct"],
        errors="coerce",
    ).dropna()

    winners = net[net > 0]
    losers = net[net <= 0]

    gross_profit = winners.sum()
    gross_loss = abs(losers.sum())

    profit_factor = (
        gross_profit / gross_loss
        if gross_loss > 0
        else np.inf
    )

    return {
        "trades": len(net),
        "wins": int((net > 0).sum()),
        "losses": int((net <= 0).sum()),
        "win_rate_pct":
            (net > 0).mean() * 100,
        "avg_net_pct":
            net.mean(),
        "median_net_pct":
            net.median(),
        "profit_factor":
            profit_factor,
        "avg_winner_pct":
            winners.mean()
            if len(winners)
            else 0.0,
        "avg_loser_pct":
            losers.mean()
            if len(losers)
            else 0.0,
        "avg_holding_days":
            pd.to_numeric(
                df["holding_days"],
                errors="coerce",
            ).mean(),
    }


# ============================================================
# BUCKET ANALYSIS
# ============================================================

def bucket_analysis(
    df,
    feature,
    bins,
    labels,
):

    work = df.copy()

    numeric_values = pd.to_numeric(
        work[feature],
        errors="coerce",
    )

    work["bucket"] = pd.cut(
        numeric_values,
        bins=bins,
        labels=labels,
        include_lowest=True,
    )

    rows = []

    for bucket, subset in work.groupby(
        "bucket",
        observed=False,
    ):

        result = calculate_metrics(
            subset
        )

        result["feature"] = feature
        result["bucket"] = str(bucket)

        rows.append(result)

    return pd.DataFrame(rows)


# ============================================================
# PRINT BUCKET TABLE
# ============================================================

def print_bucket_table(
    title,
    dataframe,
):

    print()
    print("=" * 78)
    print(title)
    print("=" * 78)

    if dataframe.empty:

        print("No data.")

        return

    columns = [
        "bucket",
        "trades",
        "win_rate_pct",
        "avg_net_pct",
        "profit_factor",
    ]

    print(
        dataframe[
            columns
        ].to_string(
            index=False
        )
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 78)
    print("TRADE QUALITY INDICATOR ANALYSIS")
    print("=" * 78)

    if not TRADES_FILE.exists():

        raise FileNotFoundError(
            f"Missing trades file:\n"
            f"{TRADES_FILE}"
        )

    if not UNIVERSE_FILE.exists():

        raise FileNotFoundError(
            f"Missing universe file:\n"
            f"{UNIVERSE_FILE}"
        )

    trades = pd.read_csv(
        TRADES_FILE
    )

    if trades.empty:

        print(
            "No trades available."
        )

        return

    trades["signal_date"] = pd.to_datetime(
        trades["signal_date"],
        errors="coerce",
    )

    trades = trades.dropna(
        subset=["signal_date"]
    ).copy()

    print()
    print(
        f"Trades loaded : {len(trades)}"
    )

    universe = load_universe()

    print(
        f"Universe rows : {len(universe)}"
    )

    # --------------------------------------------------------
    # Regime mapping belongs to the trade table.
    # We keep it there and copy it into each feature record.
    # --------------------------------------------------------

    regime_series = load_regime()

    if regime_series is not None:

        trades["mapped_regime"] = (
            trades["signal_date"]
            .map(regime_series)
            .fillna("UNKNOWN")
        )

    else:

        trades["mapped_regime"] = "UNKNOWN"

    # --------------------------------------------------------
    # Reconstruct indicators.
    # --------------------------------------------------------

    print()
    print("=" * 78)
    print("RECONSTRUCTING SIGNAL FEATURES")
    print("=" * 78)

    all_features = []

    data_cache = {}

    grouped = trades.groupby(
        [
            "symbol",
            "exchange",
        ]
    )

    total_groups = len(grouped)
    processed_groups = 0
    matched = 0
    missing = 0

    for (
        symbol,
        exchange,
    ), group in grouped:

        processed_groups += 1

        print(
            f"\rProcessing "
            f"{processed_groups}/{total_groups} "
            f"{exchange}:{symbol:<18}",
            end="",
            flush=True,
        )

        universe_match = universe[
            (
                universe["symbol"]
                == str(symbol)
            )
            & (
                universe["exchange"]
                == str(exchange).upper()
            )
        ]

        if universe_match.empty:

            missing += len(group)

            continue

        data_symbol = str(
            universe_match.iloc[0][
                "data_symbol"
            ]
        )

        try:

            if data_symbol not in data_cache:

                data = get_daily_data(
                    data_symbol,
                    period="2y",
                    interval="1d",
                )

                if data is None or data.empty:

                    raise ValueError(
                        "No daily data"
                    )

                data_cache[data_symbol] = (
                    prepare_data(data)
                )

            df = data_cache[
                data_symbol
            ]

        except Exception as exc:

            print(
                f"\nWARNING: "
                f"{exchange}:{symbol} "
                f"feature reconstruction "
                f"failed: {exc}"
            )

            missing += len(group)

            continue

        for _, trade in group.iterrows():

            position = find_signal_position(
                df,
                trade["signal_date"],
            )

            if position is None:

                missing += 1

                continue

            try:

                feature_row = extract_features(
                    df,
                    position,
                    trade.to_dict(),
                )

                # ------------------------------------------------
                # IMPORTANT FIX:
                # Copy mapped_regime from the ORIGINAL trade row.
                # Do not expect it to exist inside feature_trades.
                # ------------------------------------------------

                feature_row["regime"] = (
                    trade["mapped_regime"]
                )

                feature_row["data_symbol"] = (
                    data_symbol
                )

                all_features.append(
                    feature_row
                )

                matched += 1

            except Exception as exc:

                print(
                    f"\nWARNING: "
                    f"{exchange}:{symbol} "
                    f"{trade['signal_date']} "
                    f"feature extraction failed: "
                    f"{exc}"
                )

                missing += 1

    print()

    feature_trades = pd.DataFrame(
        all_features
    )

    if feature_trades.empty:

        print(
            "No feature records were created."
        )

        return

    # --------------------------------------------------------
    # Save feature-rich trades.
    # --------------------------------------------------------

    feature_trades.to_csv(
        FEATURE_TRADES_FILE,
        index=False,
    )

    # --------------------------------------------------------
    # Reconstruction report.
    # --------------------------------------------------------

    print()
    print("=" * 78)
    print("FEATURE RECONSTRUCTION")
    print("=" * 78)

    print(
        f"Original trades : {len(trades)}"
    )

    print(
        f"Features matched: {matched}"
    )

    print(
        f"Features missing: {missing}"
    )

    print(
        f"Match rate      : "
        f"{matched / len(trades) * 100:.2f}%"
    )

    # ========================================================
    # OVERALL
    # ========================================================

    print()
    print("=" * 78)
    print("OVERALL TRADE QUALITY")
    print("=" * 78)

    print(
        f"{'Variant':15s}"
        f"{'Trades':>9s}"
        f"{'Win %':>10s}"
        f"{'Avg Net':>12s}"
        f"{'PF':>8s}"
        f"{'Winner':>10s}"
        f"{'Loser':>10s}"
    )

    print("-" * 78)

    for variant in [
        "CURRENT",
        "EMA_MACD",
    ]:

        subset = feature_trades[
            feature_trades["variant"]
            == variant
        ]

        result = calculate_metrics(
            subset
        )

        pf = result["profit_factor"]

        pf_text = (
            "INF"
            if np.isinf(pf)
            else f"{pf:.2f}"
        )

        print(
            f"{variant:15s}"
            f"{result['trades']:9d}"
            f"{result['win_rate_pct']:9.2f}%"
            f"{result['avg_net_pct']:+11.2f}%"
            f"{pf_text:>8s}"
            f"{result['avg_winner_pct']:+9.2f}%"
            f"{result['avg_loser_pct']:+9.2f}%"
        )

    # ========================================================
    # RSI
    # ========================================================

    rsi_frames = []

    long_trades = feature_trades[
        feature_trades["direction"]
        == "LONG"
    ]

    short_trades = feature_trades[
        feature_trades["direction"]
        == "SHORT"
    ]

    if not long_trades.empty:

        long_rsi = bucket_analysis(
            long_trades,
            "rsi",
            [
                0,
                60,
                65,
                70,
                75,
                100,
            ],
            [
                "<60",
                "60-65",
                "65-70",
                "70-75",
                ">75",
            ],
        )

        long_rsi["direction"] = "LONG"

        rsi_frames.append(
            long_rsi
        )

    if not short_trades.empty:

        short_rsi = bucket_analysis(
            short_trades,
            "rsi",
            [
                0,
                25,
                30,
                35,
                40,
                100,
            ],
            [
                "<25",
                "25-30",
                "30-35",
                "35-40",
                ">40",
            ],
        )

        short_rsi["direction"] = "SHORT"

        rsi_frames.append(
            short_rsi
        )

    if rsi_frames:

        rsi_result = pd.concat(
            rsi_frames,
            ignore_index=True,
        )

    else:

        rsi_result = pd.DataFrame()

    rsi_result.to_csv(
        RSI_FILE,
        index=False,
    )

    # ========================================================
    # MACD
    # ========================================================

    macd_result = bucket_analysis(
        feature_trades,
        "macd_histogram_abs_pct",
        [
            0,
            0.05,
            0.10,
            0.20,
            0.50,
            np.inf,
        ],
        [
            "<0.05%",
            "0.05-0.10%",
            "0.10-0.20%",
            "0.20-0.50%",
            ">0.50%",
        ],
    )

    macd_result.to_csv(
        MACD_FILE,
        index=False,
    )

    # ========================================================
    # EMA
    # ========================================================

    ema_result = bucket_analysis(
        feature_trades,
        "ema_5_26_abs_pct",
        [
            0,
            0.25,
            0.50,
            1.00,
            2.00,
            np.inf,
        ],
        [
            "<0.25%",
            "0.25-0.50%",
            "0.50-1.00%",
            "1.00-2.00%",
            ">2.00%",
        ],
    )

    ema_result.to_csv(
        EMA_FILE,
        index=False,
    )

    # ========================================================
    # VOLUME
    # ========================================================

    volume_result = bucket_analysis(
        feature_trades,
        "volume_ratio",
        [
            0,
            0.50,
            1.00,
            1.30,
            2.00,
            np.inf,
        ],
        [
            "<0.50x",
            "0.50-1.00x",
            "1.00-1.30x",
            "1.30-2.00x",
            ">2.00x",
        ],
    )

    volume_result.to_csv(
        VOLUME_FILE,
        index=False,
    )

    # ========================================================
    # FRESHNESS
    # ========================================================

    freshness_result = bucket_analysis(
        feature_trades,
        "max_event_age",
        [
            -1,
            1,
            2,
            3,
            4,
            np.inf,
        ],
        [
            "0-1",
            "2",
            "3",
            "4",
            "5+",
        ],
    )

    freshness_result.to_csv(
        FRESHNESS_FILE,
        index=False,
    )

    # ========================================================
    # COMBINED BUCKET FILE
    # ========================================================

    combined_frames = [
        rsi_result,
        macd_result,
        ema_result,
        volume_result,
        freshness_result,
    ]

    combined_frames = [
        frame
        for frame in combined_frames
        if not frame.empty
    ]

    if combined_frames:

        combined = pd.concat(
            combined_frames,
            ignore_index=True,
        )

    else:

        combined = pd.DataFrame()

    combined.to_csv(
        BUCKET_FILE,
        index=False,
    )

    # ========================================================
    # PRINT BUCKETS
    # ========================================================

    print_bucket_table(
        "RSI BUCKET ANALYSIS",
        rsi_result,
    )

    print_bucket_table(
        "MACD HISTOGRAM STRENGTH",
        macd_result,
    )

    print_bucket_table(
        "EMA SEPARATION",
        ema_result,
    )

    print_bucket_table(
        "VOLUME RATIO",
        volume_result,
    )

    print_bucket_table(
        "SIGNAL FRESHNESS",
        freshness_result,
    )

    # ========================================================
    # DIRECTION
    # ========================================================

    print()
    print("=" * 78)
    print("QUALITY BY DIRECTION")
    print("=" * 78)

    for direction in [
        "LONG",
        "SHORT",
    ]:

        subset = feature_trades[
            feature_trades["direction"]
            == direction
        ]

        result = calculate_metrics(
            subset
        )

        print(
            f"{direction:8s} "
            f"trades={result['trades']:4d} "
            f"win={result['win_rate_pct']:6.2f}% "
            f"avg={result['avg_net_pct']:+6.2f}% "
            f"PF={result['profit_factor']:.2f}"
        )

    # ========================================================
    # REGIME
    # ========================================================

    print()
    print("=" * 78)
    print("QUALITY BY REGIME")
    print("=" * 78)

    for regime_name in [
        "BULL",
        "BEAR",
        "NEUTRAL",
        "UNKNOWN",
    ]:

        subset = feature_trades[
            feature_trades["regime"]
            == regime_name
        ]

        if subset.empty:
            continue

        result = calculate_metrics(
            subset
        )

        print(
            f"{regime_name:9s} "
            f"trades={result['trades']:4d} "
            f"win={result['win_rate_pct']:6.2f}% "
            f"avg={result['avg_net_pct']:+6.2f}% "
            f"PF={result['profit_factor']:.2f}"
        )

    # ========================================================
    # WINNERS VS LOSERS
    # ========================================================

    print()
    print("=" * 78)
    print("WINNER VS LOSER FEATURE MEANS")
    print("=" * 78)

    numeric_net = pd.to_numeric(
        feature_trades["net_pnl_pct"],
        errors="coerce",
    )

    winners = feature_trades[
        numeric_net > 0
    ]

    losers = feature_trades[
        numeric_net <= 0
    ]

    feature_columns = [
        "rsi",
        "ema_5_13_abs_pct",
        "ema_13_26_abs_pct",
        "ema_5_26_abs_pct",
        "macd_histogram_abs_pct",
        "volume_ratio",
        "extension_abs_pct",
        "max_event_age",
    ]

    print(
        f"{'Feature':28s}"
        f"{'Winner':>14s}"
        f"{'Loser':>14s}"
        f"{'Difference':>14s}"
    )

    print("-" * 72)

    for feature in feature_columns:

        winner_mean = pd.to_numeric(
            winners[feature],
            errors="coerce",
        ).mean()

        loser_mean = pd.to_numeric(
            losers[feature],
            errors="coerce",
        ).mean()

        difference = (
            winner_mean
            - loser_mean
        )

        print(
            f"{feature:28s}"
            f"{winner_mean:14.4f}"
            f"{loser_mean:14.4f}"
            f"{difference:14.4f}"
        )

    # ========================================================
    # VARIANT FEATURE MEANS
    # ========================================================

    print()
    print("=" * 78)
    print("CURRENT VS EMA_MACD FEATURE MEANS")
    print("=" * 78)

    comparison_features = [
        "rsi",
        "ema_5_13_abs_pct",
        "ema_13_26_abs_pct",
        "ema_5_26_abs_pct",
        "macd_histogram_abs_pct",
        "volume_ratio",
        "extension_abs_pct",
        "max_event_age",
    ]

    print(
        f"{'Feature':28s}"
        f"{'CURRENT':>14s}"
        f"{'EMA_MACD':>14s}"
    )

    print("-" * 58)

    for feature in comparison_features:

        current_mean = pd.to_numeric(
            feature_trades[
                feature_trades["variant"]
                == "CURRENT"
            ][feature],
            errors="coerce",
        ).mean()

        ema_macd_mean = pd.to_numeric(
            feature_trades[
                feature_trades["variant"]
                == "EMA_MACD"
            ][feature],
            errors="coerce",
        ).mean()

        print(
            f"{feature:28s}"
            f"{current_mean:14.4f}"
            f"{ema_macd_mean:14.4f}"
        )

    # ========================================================
    # OUTPUT FILES
    # ========================================================

    print()
    print("=" * 78)
    print("FILES SAVED")
    print("=" * 78)

    print(
        f"Features  : {FEATURE_TRADES_FILE}"
    )

    print(
        f"Buckets   : {BUCKET_FILE}"
    )

    print(
        f"RSI       : {RSI_FILE}"
    )

    print(
        f"MACD      : {MACD_FILE}"
    )

    print(
        f"EMA       : {EMA_FILE}"
    )

    print(
        f"Volume    : {VOLUME_FILE}"
    )

    print(
        f"Freshness : {FRESHNESS_FILE}"
    )

    print()
    print("=" * 78)
    print("TRADE QUALITY ANALYSIS COMPLETE")
    print("=" * 78)


if __name__ == "__main__":
    main()