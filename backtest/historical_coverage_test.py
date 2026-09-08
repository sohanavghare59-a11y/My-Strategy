from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from data.market_data import get_daily_data


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

UNIVERSE_FILE = (
    BASE_DIR
    / "output"
    / "backtest_selected_universe.csv"
)

OUTPUT_COMPARISON = (
    BASE_DIR
    / "output"
    / "historical_coverage_comparison.csv"
)

OUTPUT_UNIVERSE = (
    BASE_DIR
    / "output"
    / "historical_coverage_universe.csv"
)

OUTPUT_TRADES = (
    BASE_DIR
    / "output"
    / "historical_coverage_trades.csv"
)


# ============================================================
# STRATEGY SETTINGS
# ============================================================

EMA_FAST = 5
EMA_MID = 13
EMA_SLOW = 26

MACD_FAST = 12
MACD_SLOW = 26
MACD_SIGNAL = 9

RSI_PERIOD = 14

RSI_BULL_MIN = 60
RSI_BULL_MAX = 65

RSI_BEAR_THRESHOLD = 40

CONFIRMATION_WINDOW = 5

STOP_LOSS_PCT = 0.025
TARGET_PCT = 0.05

TRANSACTION_COST_PCT = 0.15

MAX_HOLDING_SESSIONS = 10

MIN_HISTORY = 400

BACKTEST_PERIOD = "2y"

START_DATE = None
END_DATE = None


# ============================================================
# HELPERS
# ============================================================

def calculate_indicators(df):

    data = df.copy()

    close = pd.to_numeric(
        data["Close"],
        errors="coerce",
    )

    data["ema_5"] = (
        close.ewm(
            span=EMA_FAST,
            adjust=False,
        ).mean()
    )

    data["ema_13"] = (
        close.ewm(
            span=EMA_MID,
            adjust=False,
        ).mean()
    )

    data["ema_26"] = (
        close.ewm(
            span=EMA_SLOW,
            adjust=False,
        ).mean()
    )

    ema_fast = (
        close.ewm(
            span=MACD_FAST,
            adjust=False,
        ).mean()
    )

    ema_slow = (
        close.ewm(
            span=MACD_SLOW,
            adjust=False,
        ).mean()
    )

    data["macd"] = (
        ema_fast - ema_slow
    )

    data["macd_signal"] = (
        data["macd"]
        .ewm(
            span=MACD_SIGNAL,
            adjust=False,
        )
        .mean()
    )

    delta = close.diff()

    gain = (
        delta.clip(lower=0)
        .rolling(RSI_PERIOD)
        .mean()
    )

    loss = (
        -delta.clip(upper=0)
        .rolling(RSI_PERIOD)
        .mean()
    )

    rs = gain / loss.replace(
        0,
        np.nan,
    )

    data["rsi"] = (
        100
        - (
            100
            / (
                1
                + rs
            )
        )
    )

    return data


def prepare_data(df):

    data = df.copy()

    if isinstance(
        data.columns,
        pd.MultiIndex,
    ):

        data.columns = [
            column[0]
            for column in data.columns
        ]

    required = [
        "Open",
        "High",
        "Low",
        "Close",
    ]

    for column in required:

        if column not in data.columns:
            return None

        data[column] = pd.to_numeric(
            data[column],
            errors="coerce",
        )

    data = data.dropna(
        subset=required
    ).copy()

    if data.empty:
        return None

    if not isinstance(
        data.index,
        pd.DatetimeIndex,
    ):

        data.index = pd.to_datetime(
            data.index
        )

    if data.index.tz is not None:

        data.index = (
            data.index.tz_localize(
                None
            )
        )

    data = data.sort_index()

    data = calculate_indicators(
        data
    )

    data = data.dropna(
        subset=[
            "ema_5",
            "ema_13",
            "ema_26",
            "macd",
            "macd_signal",
            "rsi",
        ]
    ).copy()

    return data


def crossover_above(
    current_a,
    current_b,
    previous_a,
    previous_b,
):

    return (
        current_a > current_b
        and previous_a <= previous_b
    )


def crossover_below(
    current_a,
    current_b,
    previous_a,
    previous_b,
):

    return (
        current_a < current_b
        and previous_a >= previous_b
    )


def recent_event(
    values,
    index,
    window,
):

    start = max(
        0,
        index - window + 1,
    )

    return any(
        values[start:index + 1]
    )


def generate_signals(
    data,
    variant,
):

    df = data.copy()

    long_ema_13 = []
    long_ema_26 = []
    long_macd = []
    long_rsi = []

    short_ema_13 = []
    short_ema_26 = []
    short_macd = []
    short_rsi = []

    for i in range(
        len(df)
    ):

        if i == 0:

            long_ema_13.append(False)
            long_ema_26.append(False)
            long_macd.append(False)
            long_rsi.append(False)

            short_ema_13.append(False)
            short_ema_26.append(False)
            short_macd.append(False)
            short_rsi.append(False)

            continue

        current = df.iloc[i]
        previous = df.iloc[i - 1]

        long_ema_13.append(
            crossover_above(
                current["ema_5"],
                current["ema_13"],
                previous["ema_5"],
                previous["ema_13"],
            )
        )

        long_ema_26.append(
            crossover_above(
                current["ema_5"],
                current["ema_26"],
                previous["ema_5"],
                previous["ema_26"],
            )
        )

        long_macd.append(
            crossover_above(
                current["macd"],
                current["macd_signal"],
                previous["macd"],
                previous["macd_signal"],
            )
        )

        long_rsi.append(
            (
                previous["rsi"]
                <= RSI_BULL_MIN
                and current["rsi"]
                > RSI_BULL_MIN
            )
        )

        short_ema_13.append(
            crossover_below(
                current["ema_5"],
                current["ema_13"],
                previous["ema_5"],
                previous["ema_13"],
            )
        )

        short_ema_26.append(
            crossover_below(
                current["ema_5"],
                current["ema_26"],
                previous["ema_5"],
                previous["ema_26"],
            )
        )

        short_macd.append(
            crossover_below(
                current["macd"],
                current["macd_signal"],
                previous["macd"],
                previous["macd_signal"],
            )
        )

        short_rsi.append(
            (
                previous["rsi"]
                >= RSI_BEAR_THRESHOLD
                and current["rsi"]
                < RSI_BEAR_THRESHOLD
            )
        )

    long_ema_13 = np.asarray(
        long_ema_13,
        dtype=bool,
    )

    long_ema_26 = np.asarray(
        long_ema_26,
        dtype=bool,
    )

    long_macd = np.asarray(
        long_macd,
        dtype=bool,
    )

    long_rsi = np.asarray(
        long_rsi,
        dtype=bool,
    )

    short_ema_13 = np.asarray(
        short_ema_13,
        dtype=bool,
    )

    short_ema_26 = np.asarray(
        short_ema_26,
        dtype=bool,
    )

    short_macd = np.asarray(
        short_macd,
        dtype=bool,
    )

    short_rsi = np.asarray(
        short_rsi,
        dtype=bool,
    )

    long_setup = []

    short_setup = []

    for i in range(
        len(df)
    ):

        long_events = (
            recent_event(
                long_ema_13,
                i,
                CONFIRMATION_WINDOW,
            )
            and recent_event(
                long_ema_26,
                i,
                CONFIRMATION_WINDOW,
            )
            and recent_event(
                long_macd,
                i,
                CONFIRMATION_WINDOW,
            )
        )

        short_events = (
            recent_event(
                short_ema_13,
                i,
                CONFIRMATION_WINDOW,
            )
            and recent_event(
                short_ema_26,
                i,
                CONFIRMATION_WINDOW,
            )
            and recent_event(
                short_macd,
                i,
                CONFIRMATION_WINDOW,
            )
        )

        if variant == "CURRENT":

            long_events = (
                long_events
                and recent_event(
                    long_rsi,
                    i,
                    CONFIRMATION_WINDOW,
                )
            )

            short_events = (
                short_events
                and recent_event(
                    short_rsi,
                    i,
                    CONFIRMATION_WINDOW,
                )
            )

        elif variant == "LONG_RSI_ZONE":

            long_events = (
                long_events
                and (
                    df.iloc[i]["rsi"]
                    >= RSI_BULL_MIN
                )
                and (
                    df.iloc[i]["rsi"]
                    <= RSI_BULL_MAX
                )
            )

            short_events = (
                short_events
                and recent_event(
                    short_rsi,
                    i,
                    CONFIRMATION_WINDOW,
                )
            )

        else:

            raise ValueError(
                f"Unknown variant: {variant}"
            )

        long_alignment = (
            df.iloc[i]["ema_5"]
            > df.iloc[i]["ema_13"]
            > df.iloc[i]["ema_26"]
            and df.iloc[i]["macd"]
            > df.iloc[i]["macd_signal"]
        )

        short_alignment = (
            df.iloc[i]["ema_5"]
            < df.iloc[i]["ema_13"]
            < df.iloc[i]["ema_26"]
            and df.iloc[i]["macd"]
            < df.iloc[i]["macd_signal"]
        )

        long_setup.append(
            bool(
                long_events
                and long_alignment
            )
        )

        short_setup.append(
            bool(
                short_events
                and short_alignment
            )
        )

    long_setup = np.asarray(
        long_setup,
        dtype=bool,
    )

    short_setup = np.asarray(
        short_setup,
        dtype=bool,
    )

    long_fresh = (
        long_setup
        & ~pd.Series(
            long_setup
        ).shift(
            1,
            fill_value=False,
        ).to_numpy(
            dtype=bool
        )
    )

    short_fresh = (
        short_setup
        & ~pd.Series(
            short_setup
        ).shift(
            1,
            fill_value=False,
        ).to_numpy(
            dtype=bool
        )
    )

    return long_fresh, short_fresh


def simulate_trade(
    data,
    signal_index,
    direction,
):

    if signal_index + 1 >= len(data):
        return None

    signal_row = data.iloc[
        signal_index
    ]

    entry_row = data.iloc[
        signal_index + 1
    ]

    entry_price = float(
        entry_row["Open"]
    )

    if not np.isfinite(
        entry_price
    ) or entry_price <= 0:
        return None

    if direction == "LONG":

        stop_loss = (
            entry_price
            * (
                1
                - STOP_LOSS_PCT
            )
        )

        target = (
            entry_price
            * (
                1
                + TARGET_PCT
            )
        )

    else:

        stop_loss = (
            entry_price
            * (
                1
                + STOP_LOSS_PCT
            )
        )

        target = (
            entry_price
            * (
                1
                - TARGET_PCT
            )
        )

    last_index = min(
        len(data) - 1,
        signal_index
        + 1
        + MAX_HOLDING_SESSIONS,
    )

    exit_index = last_index
    exit_price = float(
        data.iloc[
            last_index
        ]["Close"]
    )
    exit_reason = "TIME_EXIT"

    for j in range(
        signal_index + 1,
        last_index + 1,
    ):

        row = data.iloc[j]

        high = float(
            row["High"]
        )

        low = float(
            row["Low"]
        )

        if direction == "LONG":

            stop_hit = (
                low
                <= stop_loss
            )

            target_hit = (
                high
                >= target
            )

            if (
                stop_hit
                and target_hit
            ):

                exit_index = j
                exit_price = stop_loss
                exit_reason = "STOP_LOSS"
                break

            if stop_hit:

                exit_index = j
                exit_price = stop_loss
                exit_reason = "STOP_LOSS"
                break

            if target_hit:

                exit_index = j
                exit_price = target
                exit_reason = "TARGET"
                break

        else:

            stop_hit = (
                high
                >= stop_loss
            )

            target_hit = (
                low
                <= target
            )

            if (
                stop_hit
                and target_hit
            ):

                exit_index = j
                exit_price = stop_loss
                exit_reason = "STOP_LOSS"
                break

            if stop_hit:

                exit_index = j
                exit_price = stop_loss
                exit_reason = "STOP_LOSS"
                break

            if target_hit:

                exit_index = j
                exit_price = target
                exit_reason = "TARGET"
                break

    if direction == "LONG":

        gross_return = (
            (
                exit_price
                - entry_price
            )
            / entry_price
        ) * 100

    else:

        gross_return = (
            (
                entry_price
                - exit_price
            )
            / entry_price
        ) * 100

    net_return = (
        gross_return
        - TRANSACTION_COST_PCT
    )

    holding_days = (
        exit_index
        - (
            signal_index + 1
        )
    )

    return {
        "signal_date":
            data.index[
                signal_index
            ].strftime(
                "%Y-%m-%d"
            ),
        "entry_date":
            data.index[
                signal_index + 1
            ].strftime(
                "%Y-%m-%d"
            ),
        "exit_date":
            data.index[
                exit_index
            ].strftime(
                "%Y-%m-%d"
            ),
        "direction":
            direction,
        "entry_price":
            entry_price,
        "stop_loss":
            stop_loss,
        "target":
            target,
        "exit_price":
            exit_price,
        "exit_reason":
            exit_reason,
        "holding_days":
            holding_days,
        "gross_pnl_pct":
            gross_return,
        "net_pnl_pct":
            net_return,
    }


def run_variant(
    data,
    symbol,
    exchange,
    variant,
):

    long_signals, short_signals = (
        generate_signals(
            data,
            variant,
        )
    )

    trades = []

    next_available_index = 0

    for i in range(
        len(data)
    ):

        if i < next_available_index:
            continue

        direction = None

        if long_signals[i]:

            direction = "LONG"

        elif short_signals[i]:

            direction = "SHORT"

        if direction is None:
            continue

        trade = simulate_trade(
            data,
            i,
            direction,
        )

        if trade is None:
            continue

        trade["variant"] = variant
        trade["symbol"] = symbol
        trade["exchange"] = exchange

        trades.append(
            trade
        )

        next_available_index = (
            i
            + 1
            + trade[
                "holding_days"
            ]
            + 1
        )

    return trades


def calculate_metrics(
    trades,
):

    if not trades:

        return {
            "trades": 0,
            "wins": 0,
            "losses": 0,
            "win_rate": 0.0,
            "avg_net_pct": 0.0,
            "profit_factor": 0.0,
            "compounded_return_pct": 0.0,
            "max_drawdown_pct": 0.0,
            "avg_holding_days": 0.0,
            "target_exits": 0,
            "stop_exits": 0,
            "time_exits": 0,
        }

    returns = np.asarray(
        [
            float(
                trade[
                    "net_pnl_pct"
                ]
            )
            for trade in trades
        ]
    )

    wins = (
        returns > 0
    )

    losses = (
        returns <= 0
    )

    gross_profit = returns[
        wins
    ].sum()

    gross_loss = abs(
        returns[
            losses
        ].sum()
    )

    if gross_loss > 0:

        profit_factor = (
            gross_profit
            / gross_loss
        )

    else:

        profit_factor = np.inf

    equity = (
        np.cumprod(
            1
            + (
                returns
                / 100
            )
        )
    )

    running_max = np.maximum.accumulate(
        equity
    )

    drawdown = (
        equity
        / running_max
        - 1
    )

    return {
        "trades":
            len(returns),
        "wins":
            int(wins.sum()),
        "losses":
            int(losses.sum()),
        "win_rate":
            float(
                wins.mean()
                * 100
            ),
        "avg_net_pct":
            float(
                returns.mean()
            ),
        "profit_factor":
            float(
                profit_factor
            ),
        "compounded_return_pct":
            float(
                (
                    equity[-1]
                    - 1
                )
                * 100
            ),
        "max_drawdown_pct":
            float(
                drawdown.min()
                * 100
            ),
        "avg_holding_days":
            float(
                np.mean(
                    [
                        trade[
                            "holding_days"
                        ]
                        for trade in trades
                    ]
                )
            ),
        "target_exits":
            sum(
                trade[
                    "exit_reason"
                ]
                == "TARGET"
                for trade in trades
            ),
        "stop_exits":
            sum(
                trade[
                    "exit_reason"
                ]
                == "STOP_LOSS"
                for trade in trades
            ),
        "time_exits":
            sum(
                trade[
                    "exit_reason"
                ]
                == "TIME_EXIT"
                for trade in trades
            ),
    }


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 95)
    print(
        "HISTORICAL-COVERAGE CONTROLLED TEST"
    )
    print("=" * 95)

    if not UNIVERSE_FILE.exists():

        print()
        print(
            "ERROR: Universe file missing:"
        )
        print(UNIVERSE_FILE)
        return

    universe = pd.read_csv(
        UNIVERSE_FILE
    )

    universe = universe[
        universe[
            "exchange"
        ].isin(
            [
                "NSE",
                "BSE",
            ]
        )
    ].copy()

    universe = universe[
        universe[
            "instrument_type"
        ].eq(
            "STOCK"
        )
    ].copy()

    universe = universe.drop_duplicates(
        subset=[
            "data_symbol"
        ]
    )

    print()
    print(
        f"Original selected stocks:"
        f" {len(universe)}"
    )

    # ========================================================
    # DOWNLOAD / VALIDATE HISTORY
    # ========================================================

    eligible_rows = []
    skipped_rows = []

    for position, row in universe.iterrows():

        symbol = str(
            row[
                "symbol"
            ]
        )

        exchange = str(
            row[
                "exchange"
            ]
        )

        data_symbol = str(
            row[
                "data_symbol"
            ]
        )

        try:

            data = get_daily_data(
                data_symbol,
                period=BACKTEST_PERIOD,
                force_refresh=False,
            )

            if data is None:

                skipped_rows.append(
                    {
                        "symbol":
                            symbol,
                        "exchange":
                            exchange,
                        "data_symbol":
                            data_symbol,
                        "reason":
                            "NO_DATA",
                    }
                )

                continue

            prepared = prepare_data(
                data
            )

            if prepared is None:

                skipped_rows.append(
                    {
                        "symbol":
                            symbol,
                        "exchange":
                            exchange,
                        "data_symbol":
                            data_symbol,
                        "reason":
                            "INVALID_DATA",
                    }
                )

                continue

            raw_rows = len(
                prepared
            )

            if raw_rows < MIN_HISTORY:

                skipped_rows.append(
                    {
                        "symbol":
                            symbol,
                        "exchange":
                            exchange,
                        "data_symbol":
                            data_symbol,
                        "reason":
                            (
                                f"SHORT_HISTORY_"
                                f"{raw_rows}"
                            ),
                    }
                )

                continue

            eligible_rows.append(
                {
                    "exchange":
                        exchange,
                    "instrument_type":
                        "STOCK",
                    "symbol":
                        symbol,
                    "name":
                        row.get(
                            "name",
                            "",
                        ),
                    "isin":
                        row.get(
                            "isin",
                            "",
                        ),
                    "security_id":
                        row.get(
                            "security_id",
                            "",
                        ),
                    "series":
                        row.get(
                            "series",
                            "",
                        ),
                    "data_symbol":
                        data_symbol,
                    "history_rows":
                        raw_rows,
                    "first_date":
                        prepared.index[
                            0
                        ].strftime(
                            "%Y-%m-%d"
                        ),
                    "last_date":
                        prepared.index[
                            -1
                        ].strftime(
                            "%Y-%m-%d"
                        ),
                }
            )

        except Exception as exc:

            skipped_rows.append(
                {
                    "symbol":
                        symbol,
                    "exchange":
                        exchange,
                    "data_symbol":
                        data_symbol,
                    "reason":
                        (
                            "ERROR_"
                            + str(exc)[:120]
                        ),
                }
            )

    eligible = pd.DataFrame(
        eligible_rows
    )

    skipped = pd.DataFrame(
        skipped_rows
    )

    # ========================================================
    # COVERAGE REPORT
    # ========================================================

    print()
    print("=" * 95)
    print(
        "HISTORICAL COVERAGE"
    )
    print("=" * 95)

    print(
        f"Original selected:"
        f" {len(universe)}"
    )

    print(
        f"Eligible:"
        f" {len(eligible)}"
    )

    print(
        f"Excluded:"
        f" {len(skipped)}"
    )

    if not eligible.empty:

        print(
            f"NSE eligible:"
            f" {(
                eligible['exchange']
                == 'NSE'
            ).sum()}"
        )

        print(
            f"BSE eligible:"
            f" {(
                eligible['exchange']
                == 'BSE'
            ).sum()}"
        )

        print(
            f"Minimum history:"
            f" {eligible['history_rows'].min()}"
        )

        print(
            f"Median history:"
            f" {eligible['history_rows'].median():.0f}"
        )

        print(
            f"Maximum history:"
            f" {eligible['history_rows'].max()}"
        )

    eligible.to_csv(
        OUTPUT_UNIVERSE,
        index=False,
    )

    # ========================================================
    # RUN BACKTEST
    # ========================================================

    all_trades = []

    if eligible.empty:

        print()
        print(
            "ERROR: No stocks passed"
        )
        print(
            "historical coverage."
        )
        return

    for _, row in eligible.iterrows():

        symbol = str(
            row[
                "symbol"
            ]
        )

        exchange = str(
            row[
                "exchange"
            ]
        )

        data_symbol = str(
            row[
                "data_symbol"
            ]
        )

        try:

            raw = get_daily_data(
                data_symbol,
                period=BACKTEST_PERIOD,
                force_refresh=False,
            )

            data = prepare_data(
                raw
            )

            if data is None:
                continue

            variants = [
                "CURRENT",
                "LONG_RSI_ZONE",
            ]

            for variant in variants:

                trades = run_variant(
                    data,
                    symbol,
                    exchange,
                    variant,
                )

                all_trades.extend(
                    trades
                )

        except Exception as exc:

            print(
                f"ERROR {exchange} "
                f"{symbol}: {exc}"
            )

    trades_df = pd.DataFrame(
        all_trades
    )

    if trades_df.empty:

        print()
        print(
            "ERROR: No trades generated."
        )
        return

    # ========================================================
    # RESULTS
    # ========================================================

    comparison = []

    for variant in [
        "CURRENT",
        "LONG_RSI_ZONE",
    ]:

        variant_trades = trades_df[
            trades_df[
                "variant"
            ]
            == variant
        ].copy()

        metrics = calculate_metrics(
            variant_trades.to_dict(
                "records"
            )
        )

        metrics[
            "variant"
        ] = variant

        comparison.append(
            metrics
        )

    comparison_df = pd.DataFrame(
        comparison
    )

    comparison_df = comparison_df[
        [
            "variant",
            "trades",
            "wins",
            "losses",
            "win_rate",
            "avg_net_pct",
            "profit_factor",
            "compounded_return_pct",
            "max_drawdown_pct",
            "avg_holding_days",
            "target_exits",
            "stop_exits",
            "time_exits",
        ]
    ]

    trades_df.to_csv(
        OUTPUT_TRADES,
        index=False,
    )

    comparison_df.to_csv(
        OUTPUT_COMPARISON,
        index=False,
    )

    # ========================================================
    # PRINT
    # ========================================================

    print()
    print("=" * 95)
    print(
        "CONTROLLED BACKTEST RESULTS"
    )
    print("=" * 95)

    print(
        f"{'Variant':<22s}"
        f"{'Trades':>10s}"
        f"{'Win %':>10s}"
        f"{'Avg Net':>12s}"
        f"{'PF':>10s}"
        f"{'Comp.':>12s}"
        f"{'Max DD':>12s}"
    )

    print("-" * 95)

    for _, row in comparison_df.iterrows():

        print(
            f"{row['variant']:<22s}"
            f"{int(row['trades']):>10d}"
            f"{row['win_rate']:>9.2f}%"
            f"{row['avg_net_pct']:>+11.2f}%"
            f"{row['profit_factor']:>10.2f}"
            f"{row['compounded_return_pct']:>+11.2f}%"
            f"{row['max_drawdown_pct']:>+11.2f}%"
        )

    current = comparison_df[
        comparison_df[
            "variant"
        ]
        == "CURRENT"
    ].iloc[0]

    zone = comparison_df[
        comparison_df[
            "variant"
        ]
        == "LONG_RSI_ZONE"
    ].iloc[0]

    print()
    print("=" * 95)
    print(
        "LONG_RSI_ZONE IMPACT"
    )
    print("=" * 95)

    print(
        f"Trade count:"
        f" {int(current['trades'])}"
        f" -> {int(zone['trades'])}"
    )

    print(
        f"Win rate:"
        f" {current['win_rate']:.2f}%"
        f" -> {zone['win_rate']:.2f}%"
    )

    print(
        f"Average net:"
        f" {current['avg_net_pct']:+.2f}%"
        f" -> {zone['avg_net_pct']:+.2f}%"
    )

    print(
        f"Profit factor:"
        f" {current['profit_factor']:.2f}"
        f" -> {zone['profit_factor']:.2f}"
    )

    print(
        f"Compounded:"
        f" {current['compounded_return_pct']:+.2f}%"
        f" -> {zone['compounded_return_pct']:+.2f}%"
    )

    print(
        f"Max drawdown:"
        f" {current['max_drawdown_pct']:+.2f}%"
        f" -> {zone['max_drawdown_pct']:+.2f}%"
    )

    print()
    print("=" * 95)
    print(
        "FILES SAVED"
    )
    print("=" * 95)

    print(
        f"Universe:"
        f" {OUTPUT_UNIVERSE}"
    )

    print(
        f"Trades:"
        f" {OUTPUT_TRADES}"
    )

    print(
        f"Comparison:"
        f" {OUTPUT_COMPARISON}"
    )

    print()
    print("=" * 95)
    print(
        "HISTORICAL-COVERAGE TEST COMPLETE"
    )
    print("=" * 95)


if __name__ == "__main__":
    main()