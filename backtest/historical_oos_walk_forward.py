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
    / "historical_oos_universe.csv"
)

OUTPUT_TRADES = (
    BASE_DIR
    / "output"
    / "historical_oos_walk_forward_trades.csv"
)

OUTPUT_RESULTS = (
    BASE_DIR
    / "output"
    / "historical_oos_walk_forward.csv"
)

OUTPUT_SUMMARY = (
    BASE_DIR
    / "output"
    / "historical_oos_walk_forward_summary.csv"
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

PERIOD = "2y"


# ============================================================
# FOUR OOS PERIODS
#
# These are based on the historical OOS period produced by
# historical_universe_oos_test.py.
# ============================================================

WALK_FORWARD_PERIODS = [
    (
        "P1",
        "2025-09-04",
        "2025-12-31",
    ),
    (
        "P2",
        "2026-01-01",
        "2026-03-31",
    ),
    (
        "P3",
        "2026-04-01",
        "2026-06-30",
    ),
    (
        "P4",
        "2026-07-01",
        "2026-09-05",
    ),
]


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
# EVENT DETECTION
# ============================================================

def build_events(data):

    n = len(data)

    events = {
        "long_ema13": np.zeros(
            n,
            dtype=bool,
        ),
        "long_ema26": np.zeros(
            n,
            dtype=bool,
        ),
        "long_macd": np.zeros(
            n,
            dtype=bool,
        ),
        "long_rsi": np.zeros(
            n,
            dtype=bool,
        ),
        "short_ema13": np.zeros(
            n,
            dtype=bool,
        ),
        "short_ema26": np.zeros(
            n,
            dtype=bool,
        ),
        "short_macd": np.zeros(
            n,
            dtype=bool,
        ),
        "short_rsi": np.zeros(
            n,
            dtype=bool,
        ),
    }

    for i in range(
        1,
        n,
    ):

        current = data.iloc[i]
        previous = data.iloc[i - 1]

        events[
            "long_ema13"
        ][i] = (
            current["ema_5"]
            > current["ema_13"]
            and previous["ema_5"]
            <= previous["ema_13"]
        )

        events[
            "long_ema26"
        ][i] = (
            current["ema_5"]
            > current["ema_26"]
            and previous["ema_5"]
            <= previous["ema_26"]
        )

        events[
            "long_macd"
        ][i] = (
            current["macd"]
            > current["macd_signal"]
            and previous["macd"]
            <= previous["macd_signal"]
        )

        events[
            "long_rsi"
        ][i] = (
            previous["rsi"]
            <= RSI_BULL_MIN
            and current["rsi"]
            > RSI_BULL_MIN
        )

        events[
            "short_ema13"
        ][i] = (
            current["ema_5"]
            < current["ema_13"]
            and previous["ema_5"]
            >= previous["ema_13"]
        )

        events[
            "short_ema26"
        ][i] = (
            current["ema_5"]
            < current["ema_26"]
            and previous["ema_5"]
            >= previous["ema_26"]
        )

        events[
            "short_macd"
        ][i] = (
            current["macd"]
            < current["macd_signal"]
            and previous["macd"]
            >= previous["macd_signal"]
        )

        events[
            "short_rsi"
        ][i] = (
            previous["rsi"]
            >= RSI_BEAR_THRESHOLD
            and current["rsi"]
            < RSI_BEAR_THRESHOLD
        )

    return events


def recent_event(
    event_array,
    index,
):

    start = max(
        0,
        index - CONFIRMATION_WINDOW + 1,
    )

    return bool(
        np.any(
            event_array[
                start:index + 1
            ]
        )
    )


# ============================================================
# SIGNAL GENERATION
# ============================================================

def generate_signals(
    data,
    variant,
):

    events = build_events(
        data
    )

    n = len(data)

    long_setup = np.zeros(
        n,
        dtype=bool,
    )

    short_setup = np.zeros(
        n,
        dtype=bool,
    )

    for i in range(n):

        long_base = (
            recent_event(
                events["long_ema13"],
                i,
            )
            and recent_event(
                events["long_ema26"],
                i,
            )
            and recent_event(
                events["long_macd"],
                i,
            )
        )

        short_base = (
            recent_event(
                events["short_ema13"],
                i,
            )
            and recent_event(
                events["short_ema26"],
                i,
            )
            and recent_event(
                events["short_macd"],
                i,
            )
        )

        if variant == "CURRENT":

            long_condition = (
                long_base
                and recent_event(
                    events["long_rsi"],
                    i,
                )
            )

            short_condition = (
                short_base
                and recent_event(
                    events["short_rsi"],
                    i,
                )
            )

        elif variant == "LONG_RSI_ZONE":

            long_condition = (
                long_base
                and data.iloc[i]["rsi"]
                >= RSI_BULL_MIN
                and data.iloc[i]["rsi"]
                <= RSI_BULL_MAX
            )

            short_condition = (
                short_base
                and recent_event(
                    events["short_rsi"],
                    i,
                )
            )

        elif variant == "EMA_MACD":

            long_condition = (
                long_base
            )

            short_condition = (
                short_base
            )

        else:

            raise ValueError(
                f"Unknown variant: {variant}"
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
            long_condition
            and long_alignment
        )

        short_setup[i] = (
            short_condition
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
):

    if (
        signal_index + 1
        >= len(data)
    ):
        return None

    entry_index = (
        signal_index + 1
    )

    entry_price = float(
        data.iloc[
            entry_index
        ]["Open"]
    )

    if (
        not np.isfinite(
            entry_price
        )
        or entry_price <= 0
    ):
        return None

    direction = None

    if data.iloc[
        signal_index
    ].get(
        "_direction",
        None,
    ) == "LONG":

        direction = "LONG"

    else:

        direction = "SHORT"

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
            data.index[
                signal_index
            ].strftime(
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
# RUN ONE VARIANT
# ============================================================

def run_variant(
    data,
    symbol,
    exchange,
    variant,
    period_name,
    period_start,
    period_end,
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

        signal_date = data.index[i]

        if signal_date < oos_start:
            continue

        if (
            signal_date
            < period_start
            or signal_date
            > period_end
        ):
            continue

        direction = None

        if long_signals[i]:

            direction = "LONG"

        elif short_signals[i]:

            direction = "SHORT"

        if direction is None:
            continue

        data.iloc[
            i,
            data.columns.get_loc(
                "_direction"
            ),
        ] = direction

        trade = simulate_trade(
            data,
            i,
        )

        if trade is None:
            continue

        trade["variant"] = variant
        trade["symbol"] = symbol
        trade["exchange"] = exchange
        trade["period"] = period_name

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

        profit_factor = (
            gross_profit
            / gross_loss
        )

    else:

        profit_factor = np.inf

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

    print("=" * 100)
    print(
        "HISTORICAL OOS WALK-FORWARD TEST"
    )
    print("=" * 100)

    if not UNIVERSE_FILE.exists():

        print()
        print(
            "ERROR: Required universe file missing:"
        )
        print(
            UNIVERSE_FILE
        )
        return

    universe = pd.read_csv(
        UNIVERSE_FILE
    )

    print()
    print(
        f"Historical OOS universe:"
        f" {len(universe)} stocks"
    )

    print()
    print(
        "Variants:"
    )
    print(
        "  CURRENT"
    )
    print(
        "  LONG_RSI_ZONE"
    )
    print(
        "  EMA_MACD"
    )

    print()
    print(
        "Loading stock data..."
    )

    all_trades = []

    usable = 0
    skipped = 0

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
                skipped += 1
                continue

            if len(data) < MIN_HISTORY:
                skipped += 1
                continue

            # Temporary column used by simulate_trade.
            data["_direction"] = ""

            usable += 1

            for (
                period_name,
                start_date,
                end_date,
            ) in WALK_FORWARD_PERIODS:

                period_start = pd.Timestamp(
                    start_date
                )

                period_end = pd.Timestamp(
                    end_date
                )

                for variant in [
                    "CURRENT",
                    "LONG_RSI_ZONE",
                    "EMA_MACD",
                ]:

                    trades = run_variant(
                        data.copy(),
                        symbol,
                        exchange,
                        variant,
                        period_name,
                        period_start,
                        period_end,
                        oos_start,
                    )

                    all_trades.extend(
                        trades
                    )

        except Exception as exc:

            skipped += 1

            print(
                f"ERROR {exchange} "
                f"{symbol}: "
                f"{str(exc)[:120]}"
            )

        if count % 20 == 0:

            print(
                f"Processed {count}/"
                f"{len(universe)} "
                f"| usable: {usable}"
                f" | skipped: {skipped}"
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
    # PERIOD RESULTS
    # ========================================================

    results = []

    for period_name in [
        period[0]
        for period in WALK_FORWARD_PERIODS
    ]:

        for variant in [
            "CURRENT",
            "LONG_RSI_ZONE",
            "EMA_MACD",
        ]:

            subset = trades_df[
                (
                    trades_df[
                        "period"
                    ]
                    == period_name
                )
                & (
                    trades_df[
                        "variant"
                    ]
                    == variant
                )
            ]

            metrics = calculate_metrics(
                subset.to_dict(
                    "records"
                )
            )

            metrics[
                "period"
            ] = period_name

            metrics[
                "variant"
            ] = variant

            results.append(
                metrics
            )

    results_df = pd.DataFrame(
        results
    )

    results_df = results_df[
        [
            "period",
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

    # ========================================================
    # STABILITY SUMMARY
    # ========================================================

    summary_rows = []

    for variant in [
        "CURRENT",
        "LONG_RSI_ZONE",
        "EMA_MACD",
    ]:

        subset = results_df[
            results_df[
                "variant"
            ]
            == variant
        ].copy()

        active = subset[
            subset[
                "trades"
            ]
            > 0
        ]

        profitable = active[
            active[
                "avg_net_pct"
            ]
            > 0
        ]

        pf_above_one = active[
            active[
                "profit_factor"
            ]
            > 1
        ]

        summary_rows.append(
            {
                "variant":
                    variant,
                "periods_with_trades":
                    len(active),
                "profitable_periods":
                    len(profitable),
                "pf_above_1_periods":
                    len(pf_above_one),
                "profitable_period_pct":
                    (
                        len(profitable)
                        / len(active)
                        * 100
                    )
                    if len(active)
                    else 0.0,
                "avg_pf":
                    active[
                        "profit_factor"
                    ].mean()
                    if len(active)
                    else 0.0,
                "median_pf":
                    active[
                        "profit_factor"
                    ].median()
                    if len(active)
                    else 0.0,
                "avg_net_pct":
                    active[
                        "avg_net_pct"
                    ].mean()
                    if len(active)
                    else 0.0,
                "median_net_pct":
                    active[
                        "avg_net_pct"
                    ].median()
                    if len(active)
                    else 0.0,
                "total_trades":
                    active[
                        "trades"
                    ].sum(),
            }
        )

    summary_df = pd.DataFrame(
        summary_rows
    )

    # ========================================================
    # SAVE
    # ========================================================

    trades_df.to_csv(
        OUTPUT_TRADES,
        index=False,
    )

    results_df.to_csv(
        OUTPUT_RESULTS,
        index=False,
    )

    summary_df.to_csv(
        OUTPUT_SUMMARY,
        index=False,
    )

    # ========================================================
    # PRINT PERIOD RESULTS
    # ========================================================

    print()
    print("=" * 100)
    print(
        "WALK-FORWARD PERIOD RESULTS"
    )
    print("=" * 100)

    for period_name in [
        period[0]
        for period in WALK_FORWARD_PERIODS
    ]:

        period_df = results_df[
            results_df[
                "period"
            ]
            == period_name
        ]

        print()
        print(
            f"{period_name}"
        )

        print(
            "-" * 100
        )

        print(
            f"{'Variant':<22s}"
            f"{'Trades':>10s}"
            f"{'Win %':>10s}"
            f"{'Avg Net':>12s}"
            f"{'PF':>10s}"
            f"{'Comp.':>12s}"
            f"{'Max DD':>12s}"
        )

        for _, row in period_df.iterrows():

            print(
                f"{row['variant']:<22s}"
                f"{int(row['trades']):>10d}"
                f"{row['win_rate']:>9.2f}%"
                f"{row['avg_net_pct']:>+11.2f}%"
                f"{row['profit_factor']:>10.2f}"
                f"{row['compounded_return_pct']:>+11.2f}%"
                f"{row['max_drawdown_pct']:>+11.2f}%"
            )

    # ========================================================
    # PRINT STABILITY
    # ========================================================

    print()
    print("=" * 100)
    print(
        "STABILITY SUMMARY"
    )
    print("=" * 100)

    print(
        f"{'Variant':<22s}"
        f"{'Active':>10s}"
        f"{'Profitable':>12s}"
        f"{'PF > 1':>10s}"
        f"{'Avg PF':>10s}"
        f"{'Median PF':>12s}"
        f"{'Avg Net':>12s}"
        f"{'Median Net':>14s}"
    )

    print(
        "-" * 100
    )

    for _, row in summary_df.iterrows():

        print(
            f"{row['variant']:<22s}"
            f"{int(row['periods_with_trades']):>10d}"
            f"{int(row['profitable_periods']):>12d}"
            f"{int(row['pf_above_1_periods']):>10d}"
            f"{row['avg_pf']:>10.2f}"
            f"{row['median_pf']:>12.2f}"
            f"{row['avg_net_pct']:>+11.2f}%"
            f"{row['median_net_pct']:>+13.2f}%"
        )

    # ========================================================
    # FINAL INTERPRETATION
    # ========================================================

    print()
    print("=" * 100)
    print(
        "FILES SAVED"
    )
    print("=" * 100)

    print(
        f"Trades:"
        f" {OUTPUT_TRADES}"
    )

    print(
        f"Period results:"
        f" {OUTPUT_RESULTS}"
    )

    print(
        f"Stability summary:"
        f" {OUTPUT_SUMMARY}"
    )

    print()
    print("=" * 100)
    print(
        "HISTORICAL OOS WALK-FORWARD COMPLETE"
    )
    print("=" * 100)


if __name__ == "__main__":
    main()