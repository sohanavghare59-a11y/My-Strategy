from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from data.market_data import get_daily_data


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

UNIVERSE_FILE = BASE_DIR / "output" / "backtest_selected_universe.csv"

OUTPUT_FILE = (
    BASE_DIR
    / "output"
    / "directional_rsi_stock_robustness.csv"
)

SUMMARY_FILE = (
    BASE_DIR
    / "output"
    / "directional_rsi_stock_robustness_summary.csv"
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

CONFIRMATION_WINDOW = 5

STOP_LOSS_PCT = 0.025
TARGET_PCT = 0.05

TRANSACTION_COST_PCT = 0.15

MAX_HOLDING_DAYS = 10
MIN_HISTORY_ROWS = 60


# ============================================================
# VARIANTS
# ============================================================

VARIANTS = {
    "CURRENT": {
        "long_rsi_mode": "threshold",
        "short_rsi_mode": "threshold",
    },
    "LONG_RSI_ZONE": {
        "long_rsi_mode": "zone",
        "short_rsi_mode": "threshold",
    },
}


# ============================================================
# CROSSOVER
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


def recent_event(series):

    return (
        series.astype(int)
        .rolling(
            CONFIRMATION_WINDOW,
            min_periods=1,
        )
        .sum()
        > 0
    )


# ============================================================
# RSI
# ============================================================

def calculate_rsi(close):

    delta = close.diff()

    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.ewm(
        alpha=1 / RSI_PERIOD,
        adjust=False,
        min_periods=RSI_PERIOD,
    ).mean()

    avg_loss = loss.ewm(
        alpha=1 / RSI_PERIOD,
        adjust=False,
        min_periods=RSI_PERIOD,
    ).mean()

    rs = avg_gain / avg_loss

    return 100 - (
        100 / (1 + rs)
    )


# ============================================================
# DATA
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

    for column in required:

        if column not in df.columns:
            raise ValueError(
                f"Missing column: {column}"
            )

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

    # --------------------------------------------------------
    # RSI
    # --------------------------------------------------------

    df["rsi"] = calculate_rsi(
        df["close"]
    )

    # --------------------------------------------------------
    # EVENTS
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

    df["recent_ema_up_13"] = recent_event(
        df["ema_up_13"]
    )

    df["recent_ema_up_26"] = recent_event(
        df["ema_up_26"]
    )

    df["recent_ema_down_13"] = recent_event(
        df["ema_down_13"]
    )

    df["recent_ema_down_26"] = recent_event(
        df["ema_down_26"]
    )

    df["recent_macd_up"] = recent_event(
        df["macd_up"]
    )

    df["recent_macd_down"] = recent_event(
        df["macd_down"]
    )

    return df.dropna(
        subset=[
            "ema_5",
            "ema_13",
            "ema_26",
            "macd",
            "macd_signal",
            "rsi",
        ]
    )


# ============================================================
# RSI RULES
# ============================================================

def long_rsi_ok(rsi, mode):

    if mode == "threshold":
        return rsi > 60

    if mode == "zone":
        return (
            rsi >= 60
            and rsi < 65
        )

    return False


def short_rsi_ok(rsi, mode):

    if mode == "threshold":
        return rsi < 40

    if mode == "zone":
        return (
            rsi > 30
            and rsi <= 40
        )

    return False


# ============================================================
# SIGNAL
# ============================================================

def get_signal(
    df,
    i,
    settings,
):

    row = df.iloc[i]

    long_setup = (
        bool(row["recent_ema_up_13"])
        and bool(row["recent_ema_up_26"])
        and bool(row["recent_macd_up"])
        and row["ema_5"] > row["ema_13"]
        and row["ema_13"] > row["ema_26"]
        and row["macd"] > row["macd_signal"]
        and long_rsi_ok(
            row["rsi"],
            settings["long_rsi_mode"],
        )
    )

    short_setup = (
        bool(row["recent_ema_down_13"])
        and bool(row["recent_ema_down_26"])
        and bool(row["recent_macd_down"])
        and row["ema_5"] < row["ema_13"]
        and row["ema_13"] < row["ema_26"]
        and row["macd"] < row["macd_signal"]
        and short_rsi_ok(
            row["rsi"],
            settings["short_rsi_mode"],
        )
    )

    return (
        bool(long_setup),
        bool(short_setup),
    )


# ============================================================
# TRADE SIMULATION
# ============================================================

def simulate_trade(
    df,
    signal_index,
    direction,
    variant,
):

    if signal_index + 1 >= len(df):
        return None

    entry_index = signal_index + 1

    entry_price = float(
        df.iloc[entry_index]["open"]
    )

    if entry_price <= 0:
        return None

    if direction == "LONG":

        stop = (
            entry_price
            * (1 - STOP_LOSS_PCT)
        )

        target = (
            entry_price
            * (1 + TARGET_PCT)
        )

    else:

        stop = (
            entry_price
            * (1 + STOP_LOSS_PCT)
        )

        target = (
            entry_price
            * (1 - TARGET_PCT)
        )

    last_index = min(
        entry_index + MAX_HOLDING_DAYS,
        len(df) - 1,
    )

    exit_index = None
    exit_price = None
    exit_reason = None

    for j in range(
        entry_index,
        last_index + 1,
    ):

        high = float(
            df.iloc[j]["high"]
        )

        low = float(
            df.iloc[j]["low"]
        )

        if direction == "LONG":

            stop_hit = low <= stop
            target_hit = high >= target

        else:

            stop_hit = high >= stop
            target_hit = low <= target

        if stop_hit:

            exit_index = j
            exit_price = stop
            exit_reason = "STOP_LOSS"
            break

        if target_hit:

            exit_index = j
            exit_price = target
            exit_reason = "TARGET"
            break

    if exit_index is None:

        exit_index = last_index

        exit_price = float(
            df.iloc[exit_index]["close"]
        )

        exit_reason = "TIME_EXIT"

    if direction == "LONG":

        gross = (
            (
                exit_price
                - entry_price
            )
            / entry_price
            * 100
        )

    else:

        gross = (
            (
                entry_price
                - exit_price
            )
            / entry_price
            * 100
        )

    net = (
        gross
        - TRANSACTION_COST_PCT
    )

    return {
        "variant": variant,
        "signal_date":
            df.index[signal_index],
        "entry_date":
            df.index[entry_index],
        "exit_date":
            df.index[exit_index],
        "direction": direction,
        "entry_price": entry_price,
        "exit_price": exit_price,
        "exit_reason": exit_reason,
        "holding_days":
            exit_index - entry_index + 1,
        "gross_pnl_pct": gross,
        "net_pnl_pct": net,
    }


# ============================================================
# METRICS
# ============================================================

def metrics(df):

    if df.empty:

        return {
            "trades": 0,
            "wins": 0,
            "losses": 0,
            "win_rate_pct": 0.0,
            "avg_net_pct": 0.0,
            "profit_factor": 0.0,
            "compounded_return_pct": 0.0,
            "max_drawdown_pct": 0.0,
        }

    returns = pd.to_numeric(
        df["net_pnl_pct"],
        errors="coerce",
    ).dropna()

    wins = returns[
        returns > 0
    ]

    losses = returns[
        returns <= 0
    ]

    gross_profit = wins.sum()

    gross_loss = abs(
        losses.sum()
    )

    if gross_loss > 0:
        pf = (
            gross_profit
            / gross_loss
        )
    else:
        pf = np.inf

    equity = (
        1 + returns / 100
    ).cumprod()

    compounded = (
        equity.iloc[-1] - 1
    ) * 100

    running_max = equity.cummax()

    drawdown = (
        equity / running_max - 1
    ) * 100

    return {
        "trades": len(returns),
        "wins": int(
            (returns > 0).sum()
        ),
        "losses": int(
            (returns <= 0).sum()
        ),
        "win_rate_pct":
            (returns > 0).mean() * 100,
        "avg_net_pct":
            returns.mean(),
        "profit_factor":
            pf,
        "compounded_return_pct":
            compounded,
        "max_drawdown_pct":
            drawdown.min(),
    }


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 90)
    print("DIRECTIONAL RSI STOCK ROBUSTNESS TEST")
    print("=" * 90)

    universe = pd.read_csv(
        UNIVERSE_FILE
    )

    universe = universe[
        universe[
            "instrument_type"
        ]
        .astype(str)
        .str.upper()
        .eq("STOCK")
    ]

    universe = universe[
        universe[
            "exchange"
        ]
        .astype(str)
        .str.upper()
        .isin(
            ["NSE", "BSE"]
        )
    ]

    universe = universe.drop_duplicates(
        "data_symbol"
    )

    print()
    print(
        f"Stocks selected : {len(universe)}"
    )

    all_trades = []

    usable = 0
    skipped = 0

    total = len(universe)

    # --------------------------------------------------------
    # PROCESS STOCKS
    # --------------------------------------------------------

    for count, (_, stock) in enumerate(
        universe.iterrows(),
        start=1,
    ):

        symbol = str(
            stock["symbol"]
        )

        exchange = str(
            stock["exchange"]
        ).upper()

        data_symbol = str(
            stock["data_symbol"]
        )

        print(
            f"\rProcessing "
            f"{count}/{total} "
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

            usable += 1

        except Exception:

            skipped += 1
            continue

        # ----------------------------------------------------
        # RUN BOTH VARIANTS
        # ----------------------------------------------------

        for variant, settings in VARIANTS.items():

            last_exit_index = -1

            for i in range(
                1,
                len(df) - 1,
            ):

                if i <= last_exit_index:
                    continue

                long_setup, short_setup = get_signal(
                    df,
                    i,
                    settings,
                )

                previous_long = False
                previous_short = False

                if i > 0:

                    (
                        previous_long,
                        previous_short,
                    ) = get_signal(
                        df,
                        i - 1,
                        settings,
                    )

                direction = None

                if (
                    long_setup
                    and not previous_long
                ):

                    direction = "LONG"

                elif (
                    short_setup
                    and not previous_short
                ):

                    direction = "SHORT"

                if direction is None:
                    continue

                trade = simulate_trade(
                    df,
                    i,
                    direction,
                    variant,
                )

                if trade is None:
                    continue

                trade["symbol"] = symbol
                trade["exchange"] = exchange
                trade["data_symbol"] = data_symbol

                all_trades.append(
                    trade
                )

                exit_date = pd.Timestamp(
                    trade["exit_date"]
                )

                positions = np.where(
                    df.index <= exit_date
                )[0]

                if len(positions):

                    last_exit_index = int(
                        positions[-1]
                    )

    print()

    trades = pd.DataFrame(
        all_trades
    )

    print()
    print("=" * 90)
    print("DATA COVERAGE")
    print("=" * 90)

    print(
        f"Selected : {total}"
    )

    print(
        f"Usable   : {usable}"
    )

    print(
        f"Skipped  : {skipped}"
    )

    if trades.empty:

        print()
        print("No trades generated.")
        return

    # ========================================================
    # STOCK-LEVEL RESULTS
    # ========================================================

    stock_rows = []

    for variant in VARIANTS:

        variant_trades = trades[
            trades["variant"]
            == variant
        ]

        for (
            exchange,
            symbol,
        ), stock_trades in variant_trades.groupby(
            ["exchange", "symbol"]
        ):

            result = metrics(
                stock_trades.sort_values(
                    "exit_date"
                )
            )

            stock_rows.append(
                {
                    "variant": variant,
                    "exchange": exchange,
                    "symbol": symbol,
                    **result,
                }
            )

    stock_results = pd.DataFrame(
        stock_rows
    )

    stock_results.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    # ========================================================
    # ROBUSTNESS SUMMARY
    # ========================================================

    summary_rows = []

    for variant in VARIANTS:

        subset = stock_results[
            stock_results["variant"]
            == variant
        ]

        active = subset[
            subset["trades"] > 0
        ]

        if active.empty:
            continue

        profitable = active[
            active["avg_net_pct"] > 0
        ]

        loss_making = active[
            active["avg_net_pct"] <= 0
        ]

        pf_valid = active[
            np.isfinite(
                active["profit_factor"]
            )
        ]

        summary_rows.append(
            {
                "variant": variant,
                "stocks_with_trades":
                    len(active),
                "profitable_stocks":
                    len(profitable),
                "losing_stocks":
                    len(loss_making),
                "profitable_stock_pct":
                    len(profitable)
                    / len(active)
                    * 100,
                "avg_stock_net_pct":
                    active[
                        "avg_net_pct"
                    ].mean(),
                "median_stock_net_pct":
                    active[
                        "avg_net_pct"
                    ].median(),
                "avg_stock_pf":
                    pf_valid[
                        "profit_factor"
                    ].mean(),
                "median_stock_pf":
                    pf_valid[
                        "profit_factor"
                    ].median(),
                "total_trades":
                    int(
                        active[
                            "trades"
                        ].sum()
                    ),
                "total_wins":
                    int(
                        active[
                            "wins"
                        ].sum()
                    ),
                "total_losses":
                    int(
                        active[
                            "losses"
                        ].sum()
                    ),
            }
        )

    summary = pd.DataFrame(
        summary_rows
    )

    summary.to_csv(
        SUMMARY_FILE,
        index=False,
    )

    # ========================================================
    # PRINT SUMMARY
    # ========================================================

    print()
    print("=" * 90)
    print("STOCK-LEVEL ROBUSTNESS")
    print("=" * 90)

    print(
        f"{'Variant':25s}"
        f"{'Stocks':>9s}"
        f"{'Profitable':>13s}"
        f"{'Prof %':>10s}"
        f"{'Avg Net':>12s}"
        f"{'Median':>12s}"
        f"{'Avg PF':>10s}"
        f"{'Median PF':>12s}"
    )

    print("-" * 90)

    for _, row in summary.iterrows():

        print(
            f"{row['variant']:25s}"
            f"{int(row['stocks_with_trades']):9d}"
            f"{int(row['profitable_stocks']):13d}"
            f"{row['profitable_stock_pct']:9.2f}%"
            f"{row['avg_stock_net_pct']:+11.2f}%"
            f"{row['median_stock_net_pct']:+11.2f}%"
            f"{row['avg_stock_pf']:10.2f}"
            f"{row['median_stock_pf']:12.2f}"
        )

    # ========================================================
    # TOP / BOTTOM STOCKS
    # ========================================================

    for variant in VARIANTS:

        subset = stock_results[
            stock_results["variant"]
            == variant
        ].copy()

        subset = subset[
            subset["trades"] >= 3
        ]

        if subset.empty:
            continue

        print()
        print("=" * 90)
        print(
            f"TOP STOCKS: {variant}"
        )
        print("=" * 90)

        top = subset.sort_values(
            "avg_net_pct",
            ascending=False,
        ).head(10)

        print(
            f"{'Symbol':18s}"
            f"{'Exch':>6s}"
            f"{'Trades':>9s}"
            f"{'Win %':>10s}"
            f"{'Avg Net':>12s}"
            f"{'PF':>9s}"
        )

        print("-" * 70)

        for _, row in top.iterrows():

            pf = row["profit_factor"]

            pf_text = (
                "INF"
                if np.isinf(pf)
                else f"{pf:.2f}"
            )

            print(
                f"{row['symbol'][:18]:18s}"
                f"{row['exchange']:>6s}"
                f"{int(row['trades']):9d}"
                f"{row['win_rate_pct']:9.2f}%"
                f"{row['avg_net_pct']:+11.2f}%"
                f"{pf_text:>9s}"
            )

        print()
        print(
            f"WORST STOCKS: {variant}"
        )

        bottom = subset.sort_values(
            "avg_net_pct",
            ascending=True,
        ).head(10)

        print(
            f"{'Symbol':18s}"
            f"{'Exch':>6s}"
            f"{'Trades':>9s}"
            f"{'Win %':>10s}"
            f"{'Avg Net':>12s}"
            f"{'PF':>9s}"
        )

        print("-" * 70)

        for _, row in bottom.iterrows():

            pf = row["profit_factor"]

            pf_text = (
                "INF"
                if np.isinf(pf)
                else f"{pf:.2f}"
            )

            print(
                f"{row['symbol'][:18]:18s}"
                f"{row['exchange']:>6s}"
                f"{int(row['trades']):9d}"
                f"{row['win_rate_pct']:9.2f}%"
                f"{row['avg_net_pct']:+11.2f}%"
                f"{pf_text:>9s}"
            )

    # ========================================================
    # FILES
    # ========================================================

    print()
    print("=" * 90)
    print("FILES SAVED")
    print("=" * 90)

    print(
        f"Stock results : {OUTPUT_FILE}"
    )

    print(
        f"Summary       : {SUMMARY_FILE}"
    )

    print()
    print("=" * 90)
    print("STOCK ROBUSTNESS TEST COMPLETE")
    print("=" * 90)


if __name__ == "__main__":
    main()