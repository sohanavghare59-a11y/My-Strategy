"""
Controlled RSI Test
===================

Research-only experiment.

Compares:

    CURRENT
        EMA + MACD + RSI

    EMA_MACD
        EMA + MACD
        RSI removed

Everything else is kept identical:

    Same universe
    Same 2-year data
    Same entry rule
    Same stop loss
    Same target
    Same transaction cost
    Same confirmation window
    Same maximum holding period
    Same no-overlap rule

This file does NOT modify the production strategy.
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
    / "controlled_rsi_comparison.csv"
)

TRADES_FILE = (
    BASE_DIR
    / "output"
    / "controlled_rsi_trades.csv"
)

REGIME_FILE = (
    BASE_DIR
    / "output"
    / "nifty50_regime.csv"
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
# CROSSOVERS
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

def calculate_rsi(
    close,
    period=14,
):

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

    rs = (
        avg_gain
        / avg_loss
    )

    return (
        100
        - (
            100
            / (1 + rs)
        )
    )


# ============================================================
# INDICATORS
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
        c
        for c in required
        if c not in df.columns
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

    df["macd"] = (
        fast - slow
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

    # ========================================================
    # EMA EVENTS
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

    # ========================================================
    # MACD EVENTS
    # ========================================================

    df["macd_up"] = crossed_above(
        df["macd"],
        df["macd_signal"],
    )

    df["macd_down"] = crossed_below(
        df["macd"],
        df["macd_signal"],
    )

    # ========================================================
    # RSI EVENTS
    # ========================================================

    df["rsi_up"] = (
        (
            df["rsi"]
            > RSI_LONG_LEVEL
        )
        & (
            df["rsi"].shift(1)
            <= RSI_LONG_LEVEL
        )
    ).fillna(False)

    df["rsi_down"] = (
        (
            df["rsi"]
            < RSI_SHORT_LEVEL
        )
        & (
            df["rsi"].shift(1)
            >= RSI_SHORT_LEVEL
        )
    ).fillna(False)

    window = SIGNAL_CONFIRMATION_WINDOW

    # ========================================================
    # EMA SETUP
    # ========================================================

    df["ema_long"] = (
        recent_event(
            df["ema_up_13"]
        )
        & recent_event(
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
        recent_event(
            df["ema_down_13"]
        )
        & recent_event(
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

    # ========================================================
    # MACD SETUP
    # ========================================================

    df["macd_long"] = (
        recent_event(
            df["macd_up"]
        )
        & (
            df["macd"]
            > df["macd_signal"]
        )
    )

    df["macd_short"] = (
        recent_event(
            df["macd_down"]
        )
        & (
            df["macd"]
            < df["macd_signal"]
        )
    )

    # ========================================================
    # RSI SETUP
    # ========================================================

    df["rsi_long"] = (
        recent_event(
            df["rsi_up"]
        )
        & (
            df["rsi"]
            > RSI_LONG_LEVEL
        )
    )

    df["rsi_short"] = (
        recent_event(
            df["rsi_down"]
        )
        & (
            df["rsi"]
            < RSI_SHORT_LEVEL
        )
    )

    # ========================================================
    # CURRENT STRATEGY
    # ========================================================

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

    # ========================================================
    # RSI REMOVED
    # ========================================================

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
    variant,
    symbol,
    exchange,
):

    entry_index = (
        signal_index + 1
    )

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

            stop_hit = (
                low <= stop
            )

            target_hit = (
                high >= target
            )

        else:

            stop_hit = (
                high >= stop
            )

            target_hit = (
                low <= target
            )

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
        "variant": variant,
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
        "net_pnl_pct": net,
    }


# ============================================================
# RUN VARIANT
# ============================================================

def run_variant(
    df,
    variant,
    long_column,
    short_column,
    symbol,
    exchange,
):

    long_signal = fresh_signal(
        df[long_column]
    )

    short_signal = fresh_signal(
        df[short_column]
    )

    trades = []

    next_available = 0

    for i in range(
        len(df) - 1
    ):

        if i < next_available:
            continue

        if bool(
            long_signal.iloc[i]
        ):

            direction = "LONG"

        elif bool(
            short_signal.iloc[i]
        ):

            direction = "SHORT"

        else:

            continue

        trade = simulate_trade(
            df,
            i,
            direction,
            variant,
            symbol,
            exchange,
        )

        if trade is None:
            continue

        trades.append(trade)

        exit_date = trade[
            "exit_date"
        ]

        positions = np.where(
            df.index == exit_date
        )[0]

        if len(positions):

            next_available = (
                int(positions[0])
                + 1
            )

    return trades


# ============================================================
# METRICS
# ============================================================

def calculate_metrics(
    trades,
):

    if trades.empty:

        return {
            "trades": 0,
            "wins": 0,
            "losses": 0,
            "win_rate_pct": 0,
            "avg_net_pct": 0,
            "avg_gross_pct": 0,
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

    if gross_loss:

        pf = (
            gross_profit
            / gross_loss
        )

    else:

        pf = np.inf

    equity = (
        1 + net / 100
    ).cumprod()

    compounded = (
        equity.iloc[-1]
        - 1
    ) * 100

    peak = equity.cummax()

    drawdown = (
        equity / peak
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
        "profit_factor": pf,
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
# REGIME MAPPING
# ============================================================

def load_regime():

    if not REGIME_FILE.exists():

        return None

    regime = pd.read_csv(
        REGIME_FILE
    )

    if "date" not in regime.columns:
        return None

    if "regime" not in regime.columns:
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

    return regime[
        ["date", "regime"]
    ].drop_duplicates(
        "date"
    )


def add_regime(
    trades,
    regime,
):

    if (
        trades.empty
        or regime is None
    ):

        trades["regime"] = "UNKNOWN"

        return trades

    mapping = regime.set_index(
        "date"
    )["regime"]

    dates = pd.to_datetime(
        trades["signal_date"]
    )

    trades["regime"] = (
        dates.map(mapping)
        .fillna("UNKNOWN")
    )

    return trades


# ============================================================
# UNIVERSE
# ============================================================

def load_universe():

    if not UNIVERSE_FILE.exists():

        raise FileNotFoundError(
            f"Missing universe:\n"
            f"{UNIVERSE_FILE}"
        )

    df = pd.read_csv(
        UNIVERSE_FILE
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
        "data_symbol"
    )

    return df.reset_index(
        drop=True
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 78)
    print("CONTROLLED RSI TEST")
    print("=" * 78)

    print()
    print("CURRENT  = EMA + MACD + RSI")
    print("TEST     = EMA + MACD")
    print()
    print(
        "Everything else is held constant."
    )

    universe = load_universe()

    print()
    print(
        f"Stocks selected : "
        f"{len(universe)}"
    )

    all_trades = []

    usable = 0
    skipped = 0

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
                    "No data"
                )

            df = prepare_data(
                data
            )

        except Exception:

            skipped += 1
            continue

        usable += 1

        current_trades = run_variant(
            df,
            "CURRENT",
            "current_long",
            "current_short",
            symbol,
            exchange,
        )

        ema_macd_trades = run_variant(
            df,
            "EMA_MACD",
            "ema_macd_long",
            "ema_macd_short",
            symbol,
            exchange,
        )

        all_trades.extend(
            current_trades
        )

        all_trades.extend(
            ema_macd_trades
        )

    print()

    trades = pd.DataFrame(
        all_trades
    )

    print()
    print("=" * 78)
    print("DATA COVERAGE")
    print("=" * 78)

    print(
        f"Selected : {len(universe)}"
    )

    print(
        f"Usable   : {usable}"
    )

    print(
        f"Skipped  : {skipped}"
    )

    print(
        f"Trades   : {len(trades)}"
    )

    if trades.empty:

        print(
            "No trades generated."
        )

        return

    # --------------------------------------------------------
    # Regime
    # --------------------------------------------------------

    regime = load_regime()

    trades = add_regime(
        trades,
        regime,
    )

    # --------------------------------------------------------
    # Overall summary
    # --------------------------------------------------------

    rows = []

    for variant in [
        "CURRENT",
        "EMA_MACD",
    ]:

        subset = trades[
            trades["variant"]
            == variant
        ]

        result = calculate_metrics(
            subset
        )

        result["variant"] = variant

        rows.append(
            result
        )

    summary = pd.DataFrame(
        rows
    )

    # --------------------------------------------------------
    # Direction summary
    # --------------------------------------------------------

    direction_rows = []

    for (
        variant,
        direction,
    ), subset in trades.groupby(
        ["variant", "direction"]
    ):

        result = calculate_metrics(
            subset
        )

        result["variant"] = variant
        result["direction"] = direction

        direction_rows.append(
            result
        )

    direction_summary = pd.DataFrame(
        direction_rows
    )

    # --------------------------------------------------------
    # Regime summary
    # --------------------------------------------------------

    regime_rows = []

    for (
        variant,
        market_regime,
    ), subset in trades.groupby(
        ["variant", "regime"]
    ):

        result = calculate_metrics(
            subset
        )

        result["variant"] = variant
        result["regime"] = market_regime

        regime_rows.append(
            result
        )

    regime_summary = pd.DataFrame(
        regime_rows
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    trades.to_csv(
        TRADES_FILE,
        index=False,
    )

    summary.to_csv(
        SUMMARY_FILE,
        index=False,
    )

    direction_file = (
        BASE_DIR
        / "output"
        / "controlled_rsi_direction.csv"
    )

    regime_file = (
        BASE_DIR
        / "output"
        / "controlled_rsi_regime.csv"
    )

    direction_summary.to_csv(
        direction_file,
        index=False,
    )

    regime_summary.to_csv(
        regime_file,
        index=False,
    )

    # ========================================================
    # PRINT OVERALL
    # ========================================================

    print()
    print("=" * 78)
    print("OVERALL COMPARISON")
    print("=" * 78)

    print(
        f"{'Variant':15s}"
        f"{'Trades':>9s}"
        f"{'Win %':>10s}"
        f"{'Avg Net':>12s}"
        f"{'PF':>8s}"
        f"{'Comp.':>12s}"
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
            f"{row['variant']:15s}"
            f"{int(row['trades']):9d}"
            f"{row['win_rate_pct']:9.2f}%"
            f"{row['avg_net_pct']:+11.2f}%"
            f"{pf_text:>8s}"
            f"{row['compounded_return_pct']:+11.2f}%"
            f"{row['max_drawdown_pct']:11.2f}%"
        )

    # ========================================================
    # DIRECTION
    # ========================================================

    print()
    print("=" * 78)
    print("DIRECTION")
    print("=" * 78)

    for _, row in direction_summary.iterrows():

        print(
            f"{row['variant']:12s}"
            f"{row['direction']:7s}"
            f"trades={int(row['trades']):4d} "
            f"win={row['win_rate_pct']:6.2f}% "
            f"avg={row['avg_net_pct']:+6.2f}% "
            f"PF={row['profit_factor']:.2f}"
        )

    # ========================================================
    # REGIME
    # ========================================================

    print()
    print("=" * 78)
    print("MARKET REGIME")
    print("=" * 78)

    for _, row in regime_summary.iterrows():

        pf = row[
            "profit_factor"
        ]

        pf_text = (
            "INF"
            if np.isinf(pf)
            else f"{pf:.2f}"
        )

        print(
            f"{row['variant']:12s}"
            f"{row['regime']:9s}"
            f"trades={int(row['trades']):4d} "
            f"win={row['win_rate_pct']:6.2f}% "
            f"avg={row['avg_net_pct']:+6.2f}% "
            f"PF={pf_text}"
        )

    # ========================================================
    # EXIT
    # ========================================================

    print()
    print("=" * 78)
    print("EXIT DISTRIBUTION")
    print("=" * 78)

    for variant in [
        "CURRENT",
        "EMA_MACD",
    ]:

        subset = trades[
            trades["variant"]
            == variant
        ]

        print(
            f"{variant:12s}"
            f"TARGET={int((subset['exit_reason'] == 'TARGET').sum()):4d} "
            f"STOP={int((subset['exit_reason'] == 'STOP_LOSS').sum()):4d} "
            f"TIME={int((subset['exit_reason'] == 'TIME_EXIT').sum()):4d}"
        )

    # ========================================================
    # DIFFERENCE
    # ========================================================

    current = summary[
        summary["variant"]
        == "CURRENT"
    ].iloc[0]

    test = summary[
        summary["variant"]
        == "EMA_MACD"
    ].iloc[0]

    print()
    print("=" * 78)
    print("RSI REMOVAL IMPACT")
    print("=" * 78)

    print(
        f"Trade count change : "
        f"{int(test['trades'] - current['trades']):+d}"
    )

    print(
        f"Win rate change    : "
        f"{test['win_rate_pct'] - current['win_rate_pct']:+.2f}%"
    )

    print(
        f"Avg net change     : "
        f"{test['avg_net_pct'] - current['avg_net_pct']:+.2f}%"
    )

    print(
        f"PF change          : "
        f"{test['profit_factor'] - current['profit_factor']:+.2f}"
    )

    print(
        f"Compounded change  : "
        f"{test['compounded_return_pct'] - current['compounded_return_pct']:+.2f}%"
    )

    print(
        f"Max DD change      : "
        f"{test['max_drawdown_pct'] - current['max_drawdown_pct']:+.2f}%"
    )

    # ========================================================
    # FILES
    # ========================================================

    print()
    print("=" * 78)
    print("FILES SAVED")
    print("=" * 78)

    print(
        f"Overall : {SUMMARY_FILE}"
    )

    print(
        f"Trades  : {TRADES_FILE}"
    )

    print(
        f"Direction: {direction_file}"
    )

    print(
        f"Regime  : {regime_file}"
    )

    print()
    print("=" * 78)
    print("CONTROLLED RSI TEST COMPLETE")
    print("=" * 78)


if __name__ == "__main__":
    main()