"""
Signal Component Backtest
=========================

Research-only diagnostic.

Tests individual and combined components:

    EMA_ONLY
    MACD_ONLY
    RSI_ONLY
    EMA_MACD
    EMA_RSI
    MACD_RSI
    EMA_MACD_RSI

This does NOT modify the production strategy or baseline files.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from config.settings import (
    STOP_LOSS_PCT,
    TARGET_1_PCT,
    SIGNAL_CONFIRMATION_WINDOW,
)

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

SUMMARY_FILE = (
    BASE_DIR
    / "output"
    / "signal_component_backtest.csv"
)

TRADES_FILE = (
    BASE_DIR
    / "output"
    / "signal_component_trades.csv"
)

STOCK_FILE = (
    BASE_DIR
    / "output"
    / "signal_component_stock_results.csv"
)


# ============================================================
# SETTINGS
# ============================================================

TRANSACTION_COST_PCT = 0.15
MAX_HOLDING_DAYS = 10

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


def event_in_window(event):
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

def calculate_rsi(close, period=14):

    delta = close.diff()

    gain = delta.clip(
        lower=0
    )

    loss = -delta.clip(
        upper=0
    )

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

    rsi = 100 - (
        100 / (1 + rs)
    )

    return rsi


# ============================================================
# INDICATORS
# ============================================================

def prepare_data(data):

    df = data.copy()

    # --------------------------------------------------------
    # Normalize column names
    # --------------------------------------------------------

    df.columns = [
        str(column).lower()
        for column in df.columns
    ]

    required_price_columns = [
        "open",
        "high",
        "low",
        "close",
        "volume",
    ]

    missing = [
        column
        for column in required_price_columns
        if column not in df.columns
    ]

    if missing:

        raise ValueError(
            "Missing OHLCV columns: "
            + ", ".join(missing)
        )

    # --------------------------------------------------------
    # Numeric conversion
    # --------------------------------------------------------

    for column in required_price_columns:

        df[column] = pd.to_numeric(
            df[column],
            errors="coerce",
        )

    df = df.dropna(
        subset=required_price_columns
    )

    if len(df) < MIN_HISTORY_ROWS:

        raise ValueError(
            f"Only {len(df)} usable rows"
        )

    # --------------------------------------------------------
    # Sort and clean index
    # --------------------------------------------------------

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

    ema_macd_fast = df["close"].ewm(
        span=MACD_FAST,
        adjust=False,
        min_periods=MACD_FAST,
    ).mean()

    ema_macd_slow = df["close"].ewm(
        span=MACD_SLOW,
        adjust=False,
        min_periods=MACD_SLOW,
    ).mean()

    df["macd"] = (
        ema_macd_fast
        - ema_macd_slow
    )

    df["macd_signal"] = df[
        "macd"
    ].ewm(
        span=MACD_SIGNAL,
        adjust=False,
        min_periods=MACD_SIGNAL,
    ).mean()

    # --------------------------------------------------------
    # RSI
    # --------------------------------------------------------

    df["rsi"] = calculate_rsi(
        df["close"],
        RSI_PERIOD,
    )

    # --------------------------------------------------------
    # Remove indicator startup rows
    # --------------------------------------------------------

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

    if len(df) < MIN_HISTORY_ROWS:

        raise ValueError(
            "Insufficient rows after indicators: "
            f"{len(df)}"
        )

    # ========================================================
    # EVENTS
    # ========================================================

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

    # ========================================================
    # COMPONENT SETUPS
    # ========================================================

    window = SIGNAL_CONFIRMATION_WINDOW

    # --------------------------------------------------------
    # EMA
    # --------------------------------------------------------

    df["ema_long"] = (
        event_in_window(
            df["ema_up_13"]
        )
        & event_in_window(
            df["ema_up_26"]
        )
        & (
            df["ema_5"]
            > df["ema_13"]
        )
        & (
            df["ema_13"]
            > df["ema_26"]
        )
    )

    df["ema_short"] = (
        event_in_window(
            df["ema_down_13"]
        )
        & event_in_window(
            df["ema_down_26"]
        )
        & (
            df["ema_5"]
            < df["ema_13"]
        )
        & (
            df["ema_13"]
            < df["ema_26"]
        )
    )

    # --------------------------------------------------------
    # MACD
    # --------------------------------------------------------

    df["macd_long"] = (
        event_in_window(
            df["macd_up"]
        )
        & (
            df["macd"]
            > df["macd_signal"]
        )
    )

    df["macd_short"] = (
        event_in_window(
            df["macd_down"]
        )
        & (
            df["macd"]
            < df["macd_signal"]
        )
    )

    # --------------------------------------------------------
    # RSI
    # --------------------------------------------------------

    df["rsi_long"] = (
        event_in_window(
            df["rsi_up"]
        )
        & (
            df["rsi"]
            > RSI_LONG_LEVEL
        )
    )

    df["rsi_short"] = (
        event_in_window(
            df["rsi_down"]
        )
        & (
            df["rsi"]
            < RSI_SHORT_LEVEL
        )
    )

    # ========================================================
    # COMBINATIONS
    # ========================================================

    df["ema_macd_long"] = (
        df["ema_long"]
        & df["macd_long"]
    )

    df["ema_macd_short"] = (
        df["ema_short"]
        & df["macd_short"]
    )

    df["ema_rsi_long"] = (
        df["ema_long"]
        & df["rsi_long"]
    )

    df["ema_rsi_short"] = (
        df["ema_short"]
        & df["rsi_short"]
    )

    df["macd_rsi_long"] = (
        df["macd_long"]
        & df["rsi_long"]
    )

    df["macd_rsi_short"] = (
        df["macd_short"]
        & df["rsi_short"]
    )

    df["ema_macd_rsi_long"] = (
        df["ema_long"]
        & df["macd_long"]
        & df["rsi_long"]
    )

    df["ema_macd_rsi_short"] = (
        df["ema_short"]
        & df["macd_short"]
        & df["rsi_short"]
    )

    return df


# ============================================================
# EXPERIMENTS
# ============================================================

EXPERIMENTS = {
    "EMA_ONLY": (
        "ema_long",
        "ema_short",
    ),
    "MACD_ONLY": (
        "macd_long",
        "macd_short",
    ),
    "RSI_ONLY": (
        "rsi_long",
        "rsi_short",
    ),
    "EMA_MACD": (
        "ema_macd_long",
        "ema_macd_short",
    ),
    "EMA_RSI": (
        "ema_rsi_long",
        "ema_rsi_short",
    ),
    "MACD_RSI": (
        "macd_rsi_long",
        "macd_rsi_short",
    ),
    "EMA_MACD_RSI": (
        "ema_macd_rsi_long",
        "ema_macd_rsi_short",
    ),
}


# ============================================================
# FRESH SIGNAL
# ============================================================

def fresh_signal(series):

    return (
        series
        & ~series.shift(
            1,
            fill_value=False,
        )
    )


# ============================================================
# TRADE SIMULATION
# ============================================================

def simulate_trade(
    df,
    signal_index,
    direction,
    symbol,
    exchange,
    experiment,
):

    entry_index = signal_index + 1

    if entry_index >= len(df):
        return None

    entry = float(
        df.iloc[
            entry_index
        ]["open"]
    )

    if (
        not np.isfinite(entry)
        or entry <= 0
    ):
        return None

    if direction == "LONG":

        stop = (
            entry
            * (1 - STOP_LOSS_PCT)
        )

        target = (
            entry
            * (1 + TARGET_1_PCT)
        )

    else:

        stop = (
            entry
            * (1 + STOP_LOSS_PCT)
        )

        target = (
            entry
            * (1 - TARGET_1_PCT)
        )

    last_index = min(
        len(df) - 1,
        entry_index
        + MAX_HOLDING_DAYS,
    )

    exit_index = None
    exit_price = None
    exit_reason = None

    for i in range(
        entry_index,
        last_index + 1,
    ):

        row = df.iloc[i]

        high = float(
            row["high"]
        )

        low = float(
            row["low"]
        )

        if direction == "LONG":

            stop_hit = low <= stop
            target_hit = high >= target

        else:

            stop_hit = high >= stop
            target_hit = low <= target

        if stop_hit and target_hit:

            exit_index = i
            exit_price = stop
            exit_reason = "STOP_LOSS"
            break

        if stop_hit:

            exit_index = i
            exit_price = stop
            exit_reason = "STOP_LOSS"
            break

        if target_hit:

            exit_index = i
            exit_price = target
            exit_reason = "TARGET"
            break

    if exit_index is None:

        exit_index = last_index

        exit_price = float(
            df.iloc[
                exit_index
            ]["close"]
        )

        exit_reason = "TIME_EXIT"

    if direction == "LONG":

        gross = (
            exit_price / entry
            - 1
        ) * 100

    else:

        gross = (
            entry / exit_price
            - 1
        ) * 100

    net = (
        gross
        - TRANSACTION_COST_PCT
    )

    return {
        "experiment": experiment,
        "symbol": symbol,
        "exchange": exchange,
        "signal_date": df.index[
            signal_index
        ],
        "entry_date": df.index[
            entry_index
        ],
        "exit_date": df.index[
            exit_index
        ],
        "direction": direction,
        "entry_price": entry,
        "stop_loss": stop,
        "target": target,
        "exit_price": exit_price,
        "exit_reason": exit_reason,
        "holding_days": (
            exit_index
            - entry_index
            + 1
        ),
        "gross_pnl_pct": gross,
        "transaction_cost_pct": (
            TRANSACTION_COST_PCT
        ),
        "net_pnl_pct": net,
    }


# ============================================================
# RUN EXPERIMENT
# ============================================================

def run_experiment(
    df,
    symbol,
    exchange,
    experiment,
    long_column,
    short_column,
):

    long_signal = fresh_signal(
        df[long_column]
    )

    short_signal = fresh_signal(
        df[short_column]
    )

    trades = []

    next_available_index = 0

    for i in range(
        len(df) - 1
    ):

        if i < next_available_index:
            continue

        direction = None

        if bool(
            long_signal.iloc[i]
        ):

            direction = "LONG"

        elif bool(
            short_signal.iloc[i]
        ):

            direction = "SHORT"

        if direction is None:
            continue

        trade = simulate_trade(
            df,
            i,
            direction,
            symbol,
            exchange,
            experiment,
        )

        if trade is None:
            continue

        trades.append(trade)

        exit_date = trade[
            "exit_date"
        ]

        exit_positions = np.where(
            df.index == exit_date
        )[0]

        if len(exit_positions):

            next_available_index = (
                int(exit_positions[0])
                + 1
            )

    return trades


# ============================================================
# METRICS
# ============================================================

def calculate_metrics(trades):

    if trades.empty:

        return {
            "trades": 0,
            "wins": 0,
            "losses": 0,
            "win_rate_pct": 0,
            "avg_net_pct": 0,
            "avg_gross_pct": 0,
            "avg_winner_pct": 0,
            "avg_loser_pct": 0,
            "profit_factor": 0,
            "expectancy_pct": 0,
            "compounded_return_pct": 0,
            "max_drawdown_pct": 0,
            "avg_holding_days": 0,
            "target_exits": 0,
            "stop_exits": 0,
            "time_exits": 0,
        }

    net = pd.to_numeric(
        trades["net_pnl_pct"],
        errors="coerce",
    ).dropna()

    gross = pd.to_numeric(
        trades["gross_pnl_pct"],
        errors="coerce",
    ).dropna()

    winners = net[
        net > 0
    ]

    losers = net[
        net <= 0
    ]

    gross_profit = winners.sum()

    gross_loss = abs(
        losers.sum()
    )

    if gross_loss > 0:

        profit_factor = (
            gross_profit
            / gross_loss
        )

    else:

        profit_factor = np.inf

    equity = (
        1 + net / 100
    ).cumprod()

    compounded = (
        equity.iloc[-1]
        - 1
    ) * 100

    running_max = (
        equity.cummax()
    )

    drawdown = (
        equity / running_max
        - 1
    ) * 100

    return {
        "trades": len(net),
        "wins": int(
            (net > 0).sum()
        ),
        "losses": int(
            (net <= 0).sum()
        ),
        "win_rate_pct": (
            (net > 0).mean()
            * 100
        ),
        "avg_net_pct": net.mean(),
        "avg_gross_pct": gross.mean(),
        "avg_winner_pct": (
            winners.mean()
            if len(winners)
            else 0
        ),
        "avg_loser_pct": (
            losers.mean()
            if len(losers)
            else 0
        ),
        "profit_factor": (
            profit_factor
        ),
        "expectancy_pct": net.mean(),
        "compounded_return_pct": (
            compounded
        ),
        "max_drawdown_pct": (
            drawdown.min()
        ),
        "avg_holding_days": (
            trades["holding_days"]
            .mean()
        ),
        "target_exits": int(
            (
                trades["exit_reason"]
                == "TARGET"
            ).sum()
        ),
        "stop_exits": int(
            (
                trades["exit_reason"]
                == "STOP_LOSS"
            ).sum()
        ),
        "time_exits": int(
            (
                trades["exit_reason"]
                == "TIME_EXIT"
            ).sum()
        ),
    }


# ============================================================
# UNIVERSE
# ============================================================

def load_universe():

    if not UNIVERSE_FILE.exists():

        raise FileNotFoundError(
            f"Missing universe file:\n"
            f"{UNIVERSE_FILE}"
        )

    df = pd.read_csv(
        UNIVERSE_FILE
    )

    required = [
        "symbol",
        "exchange",
        "data_symbol",
    ]

    missing = [
        c
        for c in required
        if c not in df.columns
    ]

    if missing:

        raise ValueError(
            "Universe missing columns: "
            + ", ".join(missing)
        )

    if "instrument_type" in df.columns:

        df = df[
            df[
                "instrument_type"
            ]
            .astype(str)
            .str.upper()
            .eq("STOCK")
        ]

    df = df[
        df["exchange"]
        .astype(str)
        .str.upper()
        .isin(
            ["NSE", "BSE"]
        )
    ]

    df = df.drop_duplicates(
        subset=["data_symbol"]
    )

    return df.reset_index(
        drop=True
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 78)
    print("SIGNAL COMPONENT BACKTEST")
    print("=" * 78)

    print()
    print(
        f"Universe : {UNIVERSE_FILE}"
    )

    print(
        f"Transaction cost : "
        f"{TRANSACTION_COST_PCT:.2f}%"
    )

    print(
        f"Stop loss : "
        f"{STOP_LOSS_PCT * 100:.2f}%"
    )

    print(
        f"Target : "
        f"{TARGET_1_PCT * 100:.2f}%"
    )

    print(
        f"Confirmation window : "
        f"{SIGNAL_CONFIRMATION_WINDOW}"
    )

    print(
        f"Minimum history : "
        f"{MIN_HISTORY_ROWS} rows"
    )

    universe = load_universe()

    print()
    print(
        f"Stocks selected : "
        f"{len(universe)}"
    )

    all_trades = []

    usable_stocks = 0
    skipped_stocks = 0

    error_examples = []

    for number, (_, row) in enumerate(
        universe.iterrows(),
        start=1,
    ):

        symbol = str(
            row["symbol"]
        )

        exchange = str(
            row["exchange"]
        )

        data_symbol = str(
            row["data_symbol"]
        )

        print(
            f"\rProcessing "
            f"{number}/{len(universe)} "
            f"{exchange}:{symbol:<18}",
            end="",
            flush=True,
        )

        try:

            data = get_daily_data(
                data_symbol,
                period="2y",
                interval="1d",
            )

            if (
                data is None
                or data.empty
            ):

                raise ValueError(
                    "No market data"
                )

            prepared = prepare_data(
                data
            )

        except Exception as exc:

            skipped_stocks += 1

            if len(error_examples) < 10:

                error_examples.append(
                    (
                        exchange,
                        symbol,
                        data_symbol,
                        str(exc),
                    )
                )

            continue

        usable_stocks += 1

        for (
            experiment,
            columns,
        ) in EXPERIMENTS.items():

            trades = run_experiment(
                prepared,
                symbol,
                exchange,
                experiment,
                columns[0],
                columns[1],
            )

            all_trades.extend(
                trades
            )

    print()

    print()
    print("=" * 78)
    print("DATA COVERAGE")
    print("=" * 78)

    print(
        f"Stocks selected : "
        f"{len(universe)}"
    )

    print(
        f"Usable stocks   : "
        f"{usable_stocks}"
    )

    print(
        f"Skipped stocks  : "
        f"{skipped_stocks}"
    )

    print(
        f"Generated trades: "
        f"{len(all_trades)}"
    )

    # --------------------------------------------------------
    # Error diagnostics
    # --------------------------------------------------------

    if error_examples:

        print()
        print("=" * 78)
        print("FIRST SKIP REASONS")
        print("=" * 78)

        for (
            exchange,
            symbol,
            data_symbol,
            error,
        ) in error_examples:

            print(
                f"{exchange}:{symbol} "
                f"({data_symbol}) -> "
                f"{error}"
            )

    if not all_trades:

        print()
        print(
            "No trades were generated."
        )

        print(
            "The skip reasons above identify "
            "the actual failure."
        )

        return

    trades_df = pd.DataFrame(
        all_trades
    )

    # --------------------------------------------------------
    # Experiment summary
    # --------------------------------------------------------

    summary_rows = []

    for experiment in EXPERIMENTS:

        subset = trades_df[
            trades_df["experiment"]
            == experiment
        ]

        if subset.empty:
            continue

        result = calculate_metrics(
            subset
        )

        result["experiment"] = (
            experiment
        )

        summary_rows.append(
            result
        )

    summary = pd.DataFrame(
        summary_rows
    )

    # --------------------------------------------------------
    # Stock-level results
    # --------------------------------------------------------

    stock_rows = []

    for (
        experiment,
        exchange,
        symbol,
    ), subset in trades_df.groupby(
        [
            "experiment",
            "exchange",
            "symbol",
        ]
    ):

        result = calculate_metrics(
            subset
        )

        result["experiment"] = (
            experiment
        )

        result["exchange"] = (
            exchange
        )

        result["symbol"] = symbol

        stock_rows.append(
            result
        )

    stock_results = pd.DataFrame(
        stock_rows
    )

    # --------------------------------------------------------
    # Save files
    # --------------------------------------------------------

    trades_df.to_csv(
        TRADES_FILE,
        index=False,
    )

    summary.to_csv(
        SUMMARY_FILE,
        index=False,
    )

    stock_results.to_csv(
        STOCK_FILE,
        index=False,
    )

    # --------------------------------------------------------
    # Print summary
    # --------------------------------------------------------

    print()
    print("=" * 78)
    print("COMPONENT COMPARISON")
    print("=" * 78)

    print(
        f"{'Experiment':20s}"
        f"{'Trades':>8s}"
        f"{'Win %':>10s}"
        f"{'Avg Net':>12s}"
        f"{'PF':>8s}"
        f"{'Max DD':>12s}"
    )

    print("-" * 78)

    for _, row in summary.iterrows():

        pf = row[
            "profit_factor"
        ]

        pf_text = (
            "INF"
            if np.isinf(pf)
            else f"{pf:.2f}"
        )

        print(
            f"{row['experiment']:20s}"
            f"{int(row['trades']):8d}"
            f"{row['win_rate_pct']:9.2f}%"
            f"{row['avg_net_pct']:+11.2f}%"
            f"{pf_text:>8s}"
            f"{row['max_drawdown_pct']:11.2f}%"
        )

    print()
    print("=" * 78)
    print("EXISTING 417-TRADE BASELINE")
    print("=" * 78)

    print("Trades     : 417")
    print("Win rate   : 35.01%")
    print("Avg net    : -0.08%")
    print("PF         : 0.95")
    print("Max DD     : -80.33%")

    print()
    print("=" * 78)
    print("FILES SAVED")
    print("=" * 78)

    print(
        f"Summary : {SUMMARY_FILE}"
    )

    print(
        f"Trades  : {TRADES_FILE}"
    )

    print(
        f"Stocks  : {STOCK_FILE}"
    )

    print()
    print(
        "SIGNAL COMPONENT BACKTEST COMPLETE"
    )


if __name__ == "__main__":
    main()