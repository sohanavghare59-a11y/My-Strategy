from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from data.market_data import get_daily_data


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

LIQUID_UNIVERSE_FILE = (
    BASE_DIR
    / "output"
    / "liquid_universe.csv"
)

OUTPUT_UNIVERSE = (
    BASE_DIR
    / "output"
    / "historical_oos_universe.csv"
)

OUTPUT_TRADES = (
    BASE_DIR
    / "output"
    / "historical_oos_trades.csv"
)

OUTPUT_COMPARISON = (
    BASE_DIR
    / "output"
    / "historical_oos_comparison.csv"
)


# ============================================================
# SETTINGS
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

PERIOD = "2y"

# First half is used ONLY for universe selection.
# Second half is the genuine OOS trading period.
SELECTION_FRACTION = 0.50

STOCKS_PER_EXCHANGE = 50


# ============================================================
# DATA PREPARATION
# ============================================================

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
        "Volume",
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

        data.index = data.index.tz_localize(
            None
        )

    data = data.sort_index()

    close = data["Close"]

    data["ema_5"] = close.ewm(
        span=EMA_FAST,
        adjust=False,
    ).mean()

    data["ema_13"] = close.ewm(
        span=EMA_MID,
        adjust=False,
    ).mean()

    data["ema_26"] = close.ewm(
        span=EMA_SLOW,
        adjust=False,
    ).mean()

    ema_fast = close.ewm(
        span=MACD_FAST,
        adjust=False,
    ).mean()

    ema_slow = close.ewm(
        span=MACD_SLOW,
        adjust=False,
    ).mean()

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
            / (1 + rs)
        )
    )

    data["volume"] = data["Volume"]

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


# ============================================================
# SIGNAL HELPERS
# ============================================================

def recent_event(
    events,
    index,
):

    start = max(
        0,
        index - CONFIRMATION_WINDOW + 1,
    )

    return bool(
        np.any(
            events[
                start:index + 1
            ]
        )
    )


def generate_signals(
    data,
    variant,
):

    n = len(data)

    long_ema13 = np.zeros(
        n,
        dtype=bool,
    )

    long_ema26 = np.zeros(
        n,
        dtype=bool,
    )

    long_macd = np.zeros(
        n,
        dtype=bool,
    )

    long_rsi = np.zeros(
        n,
        dtype=bool,
    )

    short_ema13 = np.zeros(
        n,
        dtype=bool,
    )

    short_ema26 = np.zeros(
        n,
        dtype=bool,
    )

    short_macd = np.zeros(
        n,
        dtype=bool,
    )

    short_rsi = np.zeros(
        n,
        dtype=bool,
    )

    for i in range(
        1,
        n,
    ):

        current = data.iloc[i]
        previous = data.iloc[i - 1]

        long_ema13[i] = (
            current["ema_5"]
            > current["ema_13"]
            and previous["ema_5"]
            <= previous["ema_13"]
        )

        long_ema26[i] = (
            current["ema_5"]
            > current["ema_26"]
            and previous["ema_5"]
            <= previous["ema_26"]
        )

        long_macd[i] = (
            current["macd"]
            > current["macd_signal"]
            and previous["macd"]
            <= previous["macd_signal"]
        )

        long_rsi[i] = (
            previous["rsi"]
            <= RSI_BULL_MIN
            and current["rsi"]
            > RSI_BULL_MIN
        )

        short_ema13[i] = (
            current["ema_5"]
            < current["ema_13"]
            and previous["ema_5"]
            >= previous["ema_13"]
        )

        short_ema26[i] = (
            current["ema_5"]
            < current["ema_26"]
            and previous["ema_5"]
            >= previous["ema_26"]
        )

        short_macd[i] = (
            current["macd"]
            < current["macd_signal"]
            and previous["macd"]
            >= previous["macd_signal"]
        )

        short_rsi[i] = (
            previous["rsi"]
            >= RSI_BEAR_THRESHOLD
            and current["rsi"]
            < RSI_BEAR_THRESHOLD
        )

    long_setup = np.zeros(
        n,
        dtype=bool,
    )

    short_setup = np.zeros(
        n,
        dtype=bool,
    )

    for i in range(n):

        long_ok = (
            recent_event(
                long_ema13,
                i,
            )
            and recent_event(
                long_ema26,
                i,
            )
            and recent_event(
                long_macd,
                i,
            )
        )

        short_ok = (
            recent_event(
                short_ema13,
                i,
            )
            and recent_event(
                short_ema26,
                i,
            )
            and recent_event(
                short_macd,
                i,
            )
        )

        if variant == "CURRENT":

            long_ok = (
                long_ok
                and recent_event(
                    long_rsi,
                    i,
                )
            )

            short_ok = (
                short_ok
                and recent_event(
                    short_rsi,
                    i,
                )
            )

        elif variant == "LONG_RSI_ZONE":

            long_ok = (
                long_ok
                and data.iloc[i]["rsi"]
                >= RSI_BULL_MIN
                and data.iloc[i]["rsi"]
                <= RSI_BULL_MAX
            )

            short_ok = (
                short_ok
                and recent_event(
                    short_rsi,
                    i,
                )
            )

        long_alignment = (
            data.iloc[i]["ema_5"]
            > data.iloc[i]["ema_13"]
            > data.iloc[i]["ema_26"]
            and data.iloc[i]["macd"]
            > data.iloc[i]["macd_signal"]
        )

        short_alignment = (
            data.iloc[i]["ema_5"]
            < data.iloc[i]["ema_13"]
            < data.iloc[i]["ema_26"]
            and data.iloc[i]["macd"]
            < data.iloc[i]["macd_signal"]
        )

        long_setup[i] = (
            long_ok
            and long_alignment
        )

        short_setup[i] = (
            short_ok
            and short_alignment
        )

    long_fresh = (
        long_setup
        & ~pd.Series(
            long_setup
        )
        .shift(
            1,
            fill_value=False,
        )
        .to_numpy(
            dtype=bool
        )
    )

    short_fresh = (
        short_setup
        & ~pd.Series(
            short_setup
        )
        .shift(
            1,
            fill_value=False,
        )
        .to_numpy(
            dtype=bool
        )
    )

    return (
        long_fresh,
        short_fresh,
    )


# ============================================================
# TRADE SIMULATION
# ============================================================

def simulate_trade(
    data,
    signal_index,
    direction,
    oos_start,
):

    if signal_index + 1 >= len(data):
        return None

    signal_date = data.index[
        signal_index
    ]

    if signal_date < oos_start:
        return None

    entry_index = (
        signal_index + 1
    )

    if entry_index >= len(data):
        return None

    entry_row = data.iloc[
        entry_index
    ]

    entry_price = float(
        entry_row["Open"]
    )

    if (
        not np.isfinite(
            entry_price
        )
        or entry_price <= 0
    ):
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
        entry_index
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
        entry_index,
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

        gross = (
            (
                exit_price
                - entry_price
            )
            / entry_price
        ) * 100

    else:

        gross = (
            (
                entry_price
                - exit_price
            )
            / entry_price
        ) * 100

    net = (
        gross
        - TRANSACTION_COST_PCT
    )

    return {
        "signal_date":
            signal_date.strftime(
                "%Y-%m-%d"
            ),
        "entry_date":
            data.index[
                entry_index
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
            exit_index
            - entry_index,
        "gross_pnl_pct":
            gross,
        "net_pnl_pct":
            net,
    }


# ============================================================
# RUN VARIANT
# ============================================================

def run_variant(
    data,
    symbol,
    exchange,
    variant,
    oos_start,
):

    long_signals, short_signals = (
        generate_signals(
            data,
            variant,
        )
    )

    trades = []

    blocked_until = -1

    for i in range(
        len(data)
    ):

        if i <= blocked_until:
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
            oos_start,
        )

        if trade is None:
            continue

        trade["variant"] = variant
        trade["symbol"] = symbol
        trade["exchange"] = exchange

        trades.append(
            trade
        )

        blocked_until = (
            i
            + 1
            + trade[
                "holding_days"
            ]
        )

    return trades


# ============================================================
# METRICS
# ============================================================

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
        ],
        dtype=float,
    )

    wins = returns > 0
    losses = returns <= 0

    gross_profit = returns[
        wins
    ].sum()

    gross_loss = abs(
        returns[
            losses
        ].sum()
    )

    if gross_loss > 0:

        pf = (
            gross_profit
            / gross_loss
        )

    else:

        pf = np.inf

    equity = np.cumprod(
        1
        + returns / 100
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
            float(pf),
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
        "HISTORICAL UNIVERSE / OUT-OF-SAMPLE TEST"
    )
    print("=" * 95)

    if not LIQUID_UNIVERSE_FILE.exists():

        print()
        print(
            "ERROR: liquid universe file missing:"
        )
        print(
            LIQUID_UNIVERSE_FILE
        )
        return

    universe = pd.read_csv(
        LIQUID_UNIVERSE_FILE
    )

    universe = universe[
        universe[
            "liquidity_status"
        ]
        .eq("PASS")
    ].copy()

    universe = universe[
        universe[
            "instrument_type"
        ]
        .eq("STOCK")
    ].copy()

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

    universe = universe.drop_duplicates(
        subset=[
            "data_symbol"
        ]
    )

    print()
    print(
        f"Liquid candidate stocks:"
        f" {len(universe)}"
    )

    # ========================================================
    # LOAD HISTORY
    # ========================================================

    candidates = []

    print()
    print(
        "Loading historical data..."
    )

    for count, (_, row) in enumerate(
        universe.iterrows(),
        start=1,
    ):

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
                period=PERIOD,
                force_refresh=False,
            )

            data = prepare_data(
                raw
            )

            if data is None:
                continue

            if len(data) < MIN_HISTORY:
                continue

            midpoint = int(
                len(data)
                * SELECTION_FRACTION
            )

            if midpoint < 150:
                continue

            selection_data = data.iloc[
                :midpoint
            ].copy()

            oos_data = data.iloc[
                midpoint:
            ].copy()

            if len(oos_data) < 100:
                continue

            avg_volume = float(
                selection_data[
                    "volume"
                ].mean()
            )

            if (
                not np.isfinite(
                    avg_volume
                )
                or avg_volume <= 0
            ):
                continue

            candidates.append(
                {
                    "exchange":
                        exchange,
                    "symbol":
                        symbol,
                    "data_symbol":
                        data_symbol,
                    "history_rows":
                        len(data),
                    "selection_rows":
                        len(selection_data),
                    "oos_rows":
                        len(oos_data),
                    "selection_avg_volume":
                        avg_volume,
                    "first_date":
                        data.index[
                            0
                        ].strftime(
                            "%Y-%m-%d"
                        ),
                    "selection_end":
                        selection_data.index[
                            -1
                        ].strftime(
                            "%Y-%m-%d"
                        ),
                    "oos_start":
                        oos_data.index[
                            0
                        ].strftime(
                            "%Y-%m-%d"
                        ),
                    "last_date":
                        data.index[
                            -1
                        ].strftime(
                            "%Y-%m-%d"
                        ),
                }
            )

        except Exception as exc:

            print(
                f"SKIP {exchange} "
                f"{symbol}: "
                f"{str(exc)[:100]}"
            )

        if count % 100 == 0:

            print(
                f"Processed {count}/"
                f"{len(universe)} "
                f"| usable candidates:"
                f" {len(candidates)}"
            )

    candidates_df = pd.DataFrame(
        candidates
    )

    if candidates_df.empty:

        print()
        print(
            "ERROR: No historical candidates."
        )
        return

    # ========================================================
    # HISTORICAL SELECTION
    # ========================================================

    selected_parts = []

    for exchange in [
        "NSE",
        "BSE",
    ]:

        exchange_df = candidates_df[
            candidates_df[
                "exchange"
            ]
            == exchange
        ].copy()

        exchange_df = (
            exchange_df
            .sort_values(
                "selection_avg_volume",
                ascending=False,
            )
            .head(
                STOCKS_PER_EXCHANGE
            )
        )

        selected_parts.append(
            exchange_df
        )

    selected = pd.concat(
        selected_parts,
        ignore_index=True,
    )

    selected = selected.drop_duplicates(
        subset=[
            "data_symbol"
        ]
    )

    print()
    print("=" * 95)
    print(
        "HISTORICAL UNIVERSE SELECTION"
    )
    print("=" * 95)

    print(
        f"Usable historical candidates:"
        f" {len(candidates_df)}"
    )

    print(
        f"Selected stocks:"
        f" {len(selected)}"
    )

    print(
        f"NSE selected:"
        f" {(
            selected['exchange']
            == 'NSE'
        ).sum()}"
    )

    print(
        f"BSE selected:"
        f" {(
            selected['exchange']
            == 'BSE'
        ).sum()}"
    )

    print()
    print(
        "IMPORTANT:"
    )
    print(
        "Stocks were selected using"
    )
    print(
        "ONLY first-half historical volume."
    )

    selected.to_csv(
        OUTPUT_UNIVERSE,
        index=False,
    )

    # ========================================================
    # OOS BACKTEST
    # ========================================================

    all_trades = []

    print()
    print(
        "Running out-of-sample backtest..."
    )

    for count, (_, row) in enumerate(
        selected.iterrows(),
        start=1,
    ):

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

        oos_start = pd.Timestamp(
            row[
                "oos_start"
            ]
        )

        try:

            raw = get_daily_data(
                data_symbol,
                period=PERIOD,
                force_refresh=False,
            )

            data = prepare_data(
                raw
            )

            if data is None:
                continue

            for variant in [
                "CURRENT",
                "LONG_RSI_ZONE",
            ]:

                trades = run_variant(
                    data,
                    symbol,
                    exchange,
                    variant,
                    oos_start,
                )

                all_trades.extend(
                    trades
                )

        except Exception as exc:

            print(
                f"ERROR {exchange} "
                f"{symbol}: "
                f"{str(exc)[:100]}"
            )

        if count % 20 == 0:

            print(
                f"Backtested {count}/"
                f"{len(selected)}"
            )

    trades_df = pd.DataFrame(
        all_trades
    )

    if trades_df.empty:

        print()
        print(
            "ERROR: No OOS trades."
        )
        return

    # ========================================================
    # COMPARISON
    # ========================================================

    results = []

    for variant in [
        "CURRENT",
        "LONG_RSI_ZONE",
    ]:

        variant_df = trades_df[
            trades_df[
                "variant"
            ]
            == variant
        ]

        metrics = calculate_metrics(
            variant_df.to_dict(
                "records"
            )
        )

        metrics[
            "variant"
        ] = variant

        results.append(
            metrics
        )

    comparison = pd.DataFrame(
        results
    )

    comparison = comparison[
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

    comparison.to_csv(
        OUTPUT_COMPARISON,
        index=False,
    )

    # ========================================================
    # PRINT RESULTS
    # ========================================================

    print()
    print("=" * 95)
    print(
        "OUT-OF-SAMPLE RESULTS"
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

    for _, row in comparison.iterrows():

        print(
            f"{row['variant']:<22s}"
            f"{int(row['trades']):>10d}"
            f"{row['win_rate']:>9.2f}%"
            f"{row['avg_net_pct']:>+11.2f}%"
            f"{row['profit_factor']:>10.2f}"
            f"{row['compounded_return_pct']:>+11.2f}%"
            f"{row['max_drawdown_pct']:>+11.2f}%"
        )

    current = comparison[
        comparison[
            "variant"
        ]
        == "CURRENT"
    ].iloc[0]

    zone = comparison[
        comparison[
            "variant"
        ]
        == "LONG_RSI_ZONE"
    ].iloc[0]

    print()
    print("=" * 95)
    print(
        "LONG_RSI_ZONE OOS IMPACT"
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
        "HISTORICAL OOS TEST COMPLETE"
    )
    print("=" * 95)


if __name__ == "__main__":
    main()