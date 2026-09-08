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
    / "historical_oos_regime_trades.csv"
)

OUTPUT_RESULTS = (
    BASE_DIR
    / "output"
    / "historical_oos_regime_comparison.csv"
)

OUTPUT_SUMMARY = (
    BASE_DIR
    / "output"
    / "historical_oos_regime_summary.csv"
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
# NIFTY 50 REGIME SETTINGS
# ============================================================

REGIME_SYMBOL = "^NSEI"

REGIME_EMA_FAST = 20
REGIME_EMA_MID = 50
REGIME_EMA_SLOW = 200


# ============================================================
# DATA PREPARATION
# ============================================================

def prepare_stock_data(df):

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
# NIFTY REGIME
# ============================================================

def build_nifty_regime():

    print()
    print(
        "Loading NIFTY 50 regime data..."
    )

    raw = get_daily_data(
        REGIME_SYMBOL,
        period=PERIOD,
        force_refresh=False,
    )

    if raw is None:
        raise RuntimeError(
            "Unable to load NIFTY 50 data."
        )

    nifty = raw.copy()

    if isinstance(
        nifty.columns,
        pd.MultiIndex,
    ):
        nifty.columns = [
            column[0]
            for column in nifty.columns
        ]

    if "Close" not in nifty.columns:
        raise RuntimeError(
            "NIFTY data has no Close column."
        )

    nifty["Close"] = pd.to_numeric(
        nifty["Close"],
        errors="coerce",
    )

    nifty = nifty.dropna(
        subset=["Close"]
    ).copy()

    if not isinstance(
        nifty.index,
        pd.DatetimeIndex,
    ):
        nifty.index = pd.to_datetime(
            nifty.index
        )

    if nifty.index.tz is not None:
        nifty.index = nifty.index.tz_localize(
            None
        )

    nifty = nifty.sort_index()

    nifty["ema20"] = (
        nifty["Close"]
        .ewm(
            span=REGIME_EMA_FAST,
            adjust=False,
        )
        .mean()
    )

    nifty["ema50"] = (
        nifty["Close"]
        .ewm(
            span=REGIME_EMA_MID,
            adjust=False,
        )
        .mean()
    )

    nifty["ema200"] = (
        nifty["Close"]
        .ewm(
            span=REGIME_EMA_SLOW,
            adjust=False,
        )
        .mean()
    )

    nifty["regime"] = "NEUTRAL"

    bull = (
        (nifty["Close"] > nifty["ema200"])
        & (
            nifty["ema20"]
            > nifty["ema50"]
        )
    )

    bear = (
        (nifty["Close"] < nifty["ema200"])
        & (
            nifty["ema20"]
            < nifty["ema50"]
        )
    )

    nifty.loc[
        bull,
        "regime",
    ] = "BULL"

    nifty.loc[
        bear,
        "regime",
    ] = "BEAR"

    return nifty[
        [
            "Close",
            "ema20",
            "ema50",
            "ema200",
            "regime",
        ]
    ]


def get_regime(
    regime_data,
    signal_date,
):

    signal_date = pd.Timestamp(
        signal_date
    )

    eligible = regime_data[
        regime_data.index
        <= signal_date
    ]

    if eligible.empty:
        return "NEUTRAL"

    return str(
        eligible.iloc[-1]["regime"]
    )


# ============================================================
# SIGNAL EVENTS
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

        events["long_ema13"][i] = (
            current["ema_5"]
            > current["ema_13"]
            and previous["ema_5"]
            <= previous["ema_13"]
        )

        events["long_ema26"][i] = (
            current["ema_5"]
            > current["ema_26"]
            and previous["ema_5"]
            <= previous["ema_26"]
        )

        events["long_macd"][i] = (
            current["macd"]
            > current["macd_signal"]
            and previous["macd"]
            <= previous["macd_signal"]
        )

        events["long_rsi"][i] = (
            previous["rsi"]
            <= RSI_BULL_MIN
            and current["rsi"]
            > RSI_BULL_MIN
        )

        events["short_ema13"][i] = (
            current["ema_5"]
            < current["ema_13"]
            and previous["ema_5"]
            >= previous["ema_13"]
        )

        events["short_ema26"][i] = (
            current["ema_5"]
            < current["ema_26"]
            and previous["ema_5"]
            >= previous["ema_26"]
        )

        events["short_macd"][i] = (
            current["macd"]
            < current["macd_signal"]
            and previous["macd"]
            >= previous["macd_signal"]
        )

        events["short_rsi"][i] = (
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
    direction,
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

            if low <= stop_loss:

                exit_index = j
                exit_price = stop_loss
                exit_reason = "STOP_LOSS"
                break

            if high >= target:

                exit_index = j
                exit_price = target
                exit_reason = "TARGET"
                break

        else:

            if high >= stop_loss:

                exit_index = j
                exit_price = stop_loss
                exit_reason = "STOP_LOSS"
                break

            if low <= target:

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
            ],
        "entry_date":
            data.index[
                entry_index
            ],
        "exit_date":
            data.index[
                exit_index
            ],
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
    oos_start,
    regime_data,
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

        regime = get_regime(
            regime_data,
            signal_date,
        )

        trade["variant"] = variant
        trade["symbol"] = symbol
        trade["exchange"] = exchange
        trade["regime"] = regime

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
            int(
                wins.sum()
            ),

        "losses":
            int(
                losses.sum()
            ),

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
        "HISTORICAL OOS REGIME × STRATEGY TEST"
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
        "Regime definition:"
    )

    print(
        "  BULL    = NIFTY Close > EMA200"
        " AND EMA20 > EMA50"
    )

    print(
        "  BEAR    = NIFTY Close < EMA200"
        " AND EMA20 < EMA50"
    )

    print(
        "  NEUTRAL = everything else"
    )

    # ========================================================
    # BUILD REGIME
    # ========================================================

    regime_data = build_nifty_regime()

    print()

    regime_counts = (
        regime_data[
            "regime"
        ]
        .value_counts()
    )

    print(
        "NIFTY regime history:"
    )

    for regime in [
        "BULL",
        "BEAR",
        "NEUTRAL",
    ]:

        print(
            f"  {regime:<8s}"
            f"{int(regime_counts.get(regime, 0)):>6d} days"
        )

    # ========================================================
    # RUN STOCKS
    # ========================================================

    all_trades = []

    usable = 0
    skipped = 0

    print()
    print(
        "Loading historical OOS stocks..."
    )

    for count, (_, row) in enumerate(
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

        oos_start = pd.Timestamp(
            row["oos_start"]
        )

        try:

            raw = get_daily_data(
                data_symbol,
                period=PERIOD,
                force_refresh=False,
            )

            data = prepare_stock_data(
                raw
            )

            if data is None:

                skipped += 1
                continue

            if len(data) < MIN_HISTORY:

                skipped += 1
                continue

            usable += 1

            for variant in [
                "CURRENT",
                "LONG_RSI_ZONE",
                "EMA_MACD",
            ]:

                trades = run_variant(
                    data,
                    symbol,
                    exchange,
                    variant,
                    oos_start,
                    regime_data,
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
                f"{len(universe)}"
                f" | usable: {usable}"
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
    # REGIME × VARIANT RESULTS
    # ========================================================

    results = []

    for regime in [
        "BULL",
        "BEAR",
        "NEUTRAL",
    ]:

        for variant in [
            "CURRENT",
            "LONG_RSI_ZONE",
            "EMA_MACD",
        ]:

            subset = trades_df[
                (
                    trades_df[
                        "regime"
                    ]
                    == regime
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
                "regime"
            ] = regime

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
            "regime",
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
    # OVERALL VARIANT RESULTS
    # ========================================================

    overall_rows = []

    for variant in [
        "CURRENT",
        "LONG_RSI_ZONE",
        "EMA_MACD",
    ]:

        subset = trades_df[
            trades_df[
                "variant"
            ]
            == variant
        ]

        metrics = calculate_metrics(
            subset.to_dict(
                "records"
            )
        )

        metrics[
            "regime"
        ] = "ALL"

        metrics[
            "variant"
        ] = variant

        overall_rows.append(
            metrics
        )

    overall_df = pd.DataFrame(
        overall_rows
    )

    results_df = pd.concat(
        [
            overall_df,
            results_df,
        ],
        ignore_index=True,
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

    # ========================================================
    # REGIME SUMMARY
    # ========================================================

    summary_rows = []

    for variant in [
        "CURRENT",
        "LONG_RSI_ZONE",
        "EMA_MACD",
    ]:

        for regime in [
            "BULL",
            "BEAR",
            "NEUTRAL",
        ]:

            row = results_df[
                (
                    results_df[
                        "variant"
                    ]
                    == variant
                )
                & (
                    results_df[
                        "regime"
                    ]
                    == regime
                )
            ]

            if row.empty:
                continue

            row = row.iloc[0]

            summary_rows.append(
                {
                    "variant":
                        variant,
                    "regime":
                        regime,
                    "trades":
                        int(
                            row[
                                "trades"
                            ]
                        ),
                    "win_rate":
                        float(
                            row[
                                "win_rate"
                            ]
                        ),
                    "avg_net_pct":
                        float(
                            row[
                                "avg_net_pct"
                            ]
                        ),
                    "profit_factor":
                        float(
                            row[
                                "profit_factor"
                            ]
                        ),
                    "compounded_return_pct":
                        float(
                            row[
                                "compounded_return_pct"
                            ]
                        ),
                    "max_drawdown_pct":
                        float(
                            row[
                                "max_drawdown_pct"
                            ]
                        ),
                }
            )

    summary_df = pd.DataFrame(
        summary_rows
    )

    summary_df.to_csv(
        OUTPUT_SUMMARY,
        index=False,
    )

    # ========================================================
    # PRINT OVERALL
    # ========================================================

    print()
    print("=" * 100)
    print(
        "OVERALL OOS RESULTS"
    )
    print("=" * 100)

    print(
        f"{'Variant':<22s}"
        f"{'Trades':>10s}"
        f"{'Win %':>10s}"
        f"{'Avg Net':>12s}"
        f"{'PF':>10s}"
        f"{'Comp.':>12s}"
        f"{'Max DD':>12s}"
    )

    print(
        "-" * 100
    )

    for _, row in overall_df.iterrows():

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
    # PRINT REGIME RESULTS
    # ========================================================

    for regime in [
        "BULL",
        "BEAR",
        "NEUTRAL",
    ]:

        print()
        print("=" * 100)
        print(
            f"{regime} REGIME"
        )
        print("=" * 100)

        print(
            f"{'Variant':<22s}"
            f"{'Trades':>10s}"
            f"{'Win %':>10s}"
            f"{'Avg Net':>12s}"
            f"{'PF':>10s}"
            f"{'Comp.':>12s}"
            f"{'Max DD':>12s}"
        )

        print(
            "-" * 100
        )

        regime_df = results_df[
            results_df[
                "regime"
            ]
            == regime
        ]

        for _, row in regime_df.iterrows():

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
    # LONG / SHORT BY REGIME
    # ========================================================

    print()
    print("=" * 100)
    print(
        "DIRECTION × REGIME"
    )
    print("=" * 100)

    print(
        f"{'Variant':<20s}"
        f"{'Regime':<12s}"
        f"{'Direction':<10s}"
        f"{'Trades':>9s}"
        f"{'Win %':>10s}"
        f"{'Avg Net':>12s}"
        f"{'PF':>10s}"
    )

    print(
        "-" * 100
    )

    direction_rows = []

    for variant in [
        "CURRENT",
        "LONG_RSI_ZONE",
        "EMA_MACD",
    ]:

        for regime in [
            "BULL",
            "BEAR",
            "NEUTRAL",
        ]:

            for direction in [
                "LONG",
                "SHORT",
            ]:

                subset = trades_df[
                    (
                        trades_df[
                            "variant"
                        ]
                        == variant
                    )
                    & (
                        trades_df[
                            "regime"
                        ]
                        == regime
                    )
                    & (
                        trades_df[
                            "direction"
                        ]
                        == direction
                    )
                ]

                if subset.empty:
                    continue

                metrics = calculate_metrics(
                    subset.to_dict(
                        "records"
                    )
                )

                direction_rows.append(
                    {
                        "variant":
                            variant,
                        "regime":
                            regime,
                        "direction":
                            direction,
                        "trades":
                            metrics[
                                "trades"
                            ],
                        "win_rate":
                            metrics[
                                "win_rate"
                            ],
                        "avg_net_pct":
                            metrics[
                                "avg_net_pct"
                            ],
                        "profit_factor":
                            metrics[
                                "profit_factor"
                            ],
                    }
                )

                print(
                    f"{variant:<20s}"
                    f"{regime:<12s}"
                    f"{direction:<10s}"
                    f"{metrics['trades']:>9d}"
                    f"{metrics['win_rate']:>9.2f}%"
                    f"{metrics['avg_net_pct']:>+11.2f}%"
                    f"{metrics['profit_factor']:>10.2f}"
                )

    direction_df = pd.DataFrame(
        direction_rows
    )

    direction_output = (
        BASE_DIR
        / "output"
        / "historical_oos_regime_direction.csv"
    )

    direction_df.to_csv(
        direction_output,
        index=False,
    )

    # ========================================================
    # FILES
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
        f"Regime comparison:"
        f" {OUTPUT_RESULTS}"
    )

    print(
        f"Regime summary:"
        f" {OUTPUT_SUMMARY}"
    )

    print(
        f"Direction results:"
        f" {direction_output}"
    )

    print()
    print("=" * 100)
    print(
        "HISTORICAL OOS REGIME TEST COMPLETE"
    )
    print("=" * 100)


if __name__ == "__main__":
    main()