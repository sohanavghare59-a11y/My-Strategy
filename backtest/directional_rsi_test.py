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
REGIME_FILE = BASE_DIR / "output" / "nifty50_regime.csv"

SUMMARY_FILE = BASE_DIR / "output" / "directional_rsi_comparison.csv"
TRADES_FILE = BASE_DIR / "output" / "directional_rsi_trades.csv"


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

RSI_LONG_LEVEL = 60
RSI_SHORT_LEVEL = 40

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
        "regime_mode": "ALL",
    },

    "LONG_RSI_ZONE": {
        "long_rsi_mode": "zone",
        "short_rsi_mode": "threshold",
        "regime_mode": "ALL",
    },

    "SHORT_RSI_ZONE": {
        "long_rsi_mode": "threshold",
        "short_rsi_mode": "zone",
        "regime_mode": "ALL",
    },

    "BOTH_RSI_ZONE": {
        "long_rsi_mode": "zone",
        "short_rsi_mode": "zone",
        "regime_mode": "ALL",
    },

    "LONG_RSI_ZONE_NO_NEUTRAL": {
        "long_rsi_mode": "zone",
        "short_rsi_mode": "threshold",
        "regime_mode": "NO_NEUTRAL",
    },

    "LONG_RSI_ZONE_BEAR_ONLY": {
        "long_rsi_mode": "zone",
        "short_rsi_mode": "threshold",
        "regime_mode": "BEAR_ONLY",
    },
}


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
# DATA PREPARATION
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
            "Missing columns: "
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

    # --------------------------------------------------------
    # RSI
    # --------------------------------------------------------

    df["rsi"] = calculate_rsi(
        df["close"]
    )

    # --------------------------------------------------------
    # EMA CROSSOVERS
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

    # --------------------------------------------------------
    # MACD CROSSOVERS
    # --------------------------------------------------------

    df["macd_up"] = crossed_above(
        df["macd"],
        df["macd_signal"],
    )

    df["macd_down"] = crossed_below(
        df["macd"],
        df["macd_signal"],
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
# REGIME
# ============================================================

def load_regime():

    if not REGIME_FILE.exists():
        return {}

    regime = pd.read_csv(
        REGIME_FILE
    )

    regime["date"] = pd.to_datetime(
        regime["date"],
        errors="coerce",
    )

    regime = regime.dropna(
        subset=["date"]
    )

    return dict(
        zip(
            regime["date"].dt.normalize(),
            regime["regime"]
            .astype(str)
            .str.upper(),
        )
    )


# ============================================================
# RSI CONDITION
# ============================================================

def long_rsi_condition(
    rsi,
    mode,
):

    if mode == "threshold":
        return rsi > RSI_LONG_LEVEL

    if mode == "zone":
        return (
            rsi >= 60
            and rsi < 65
        )

    return False


def short_rsi_condition(
    rsi,
    mode,
):

    if mode == "threshold":
        return rsi < RSI_SHORT_LEVEL

    if mode == "zone":
        return (
            rsi > 30
            and rsi <= 40
        )

    return False


# ============================================================
# SETUP
# ============================================================

def raw_setup(
    df,
    i,
    settings,
    regime,
):

    row = df.iloc[i]

    ema_up_13 = recent_event(
        df["ema_up_13"]
    ).iloc[i]

    ema_up_26 = recent_event(
        df["ema_up_26"]
    ).iloc[i]

    ema_down_13 = recent_event(
        df["ema_down_13"]
    ).iloc[i]

    ema_down_26 = recent_event(
        df["ema_down_26"]
    ).iloc[i]

    macd_up = recent_event(
        df["macd_up"]
    ).iloc[i]

    macd_down = recent_event(
        df["macd_down"]
    ).iloc[i]

    long_setup = (
        ema_up_13
        and ema_up_26
        and macd_up
        and row["ema_5"] > row["ema_13"]
        and row["ema_13"] > row["ema_26"]
        and row["macd"] > row["macd_signal"]
        and long_rsi_condition(
            row["rsi"],
            settings["long_rsi_mode"],
        )
    )

    short_setup = (
        ema_down_13
        and ema_down_26
        and macd_down
        and row["ema_5"] < row["ema_13"]
        and row["ema_13"] < row["ema_26"]
        and row["macd"] < row["macd_signal"]
        and short_rsi_condition(
            row["rsi"],
            settings["short_rsi_mode"],
        )
    )

    current_regime = regime.get(
        df.index[i].normalize(),
        "UNKNOWN",
    )

    regime_mode = settings[
        "regime_mode"
    ]

    if regime_mode == "NO_NEUTRAL":

        if current_regime == "NEUTRAL":
            long_setup = False
            short_setup = False

    elif regime_mode == "BEAR_ONLY":

        if current_regime != "BEAR":
            long_setup = False
            short_setup = False

    return {
        "long": bool(long_setup),
        "short": bool(short_setup),
        "regime": current_regime,
    }


# ============================================================
# TRADE SIMULATION
# ============================================================

def simulate_trade(
    df,
    signal_index,
    direction,
    variant,
    regime_name,
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
        entry_index
        + MAX_HOLDING_DAYS,
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

        # Conservative assumption:
        # stop is assumed first if both
        # stop and target occur in one candle.

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
        "symbol": "",
        "exchange": "",
        "signal_date":
            df.index[
                signal_index
            ],
        "entry_date":
            df.index[
                entry_index
            ],
        "exit_date":
            df.index[
                exit_index
            ],
        "direction": direction,
        "entry_price": entry_price,
        "stop_loss": stop,
        "target": target,
        "exit_price": exit_price,
        "exit_reason": exit_reason,
        "holding_days":
            exit_index
            - entry_index
            + 1,
        "gross_pnl_pct": gross,
        "net_pnl_pct": net,
        "regime": regime_name,
        "rsi":
            float(
                df.iloc[
                    signal_index
                ]["rsi"]
            ),
    }


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
            "profit_factor": 0.0,
            "compounded_return_pct": 0.0,
            "max_drawdown_pct": 0.0,
        }

    work = df.sort_values(
        "exit_date"
    )

    net = pd.to_numeric(
        work["net_pnl_pct"],
        errors="coerce",
    ).dropna()

    wins = net[net > 0]
    losses = net[net <= 0]

    gross_profit = wins.sum()
    gross_loss = abs(
        losses.sum()
    )

    profit_factor = (
        gross_profit / gross_loss
        if gross_loss > 0
        else np.inf
    )

    equity = (
        1 + net / 100
    ).cumprod()

    compounded = (
        equity.iloc[-1] - 1
    ) * 100

    running_max = equity.cummax()

    drawdown = (
        equity / running_max - 1
    ) * 100

    return {
        "trades": len(net),
        "wins": int(
            (net > 0).sum()
        ),
        "losses": int(
            (net <= 0).sum()
        ),
        "win_rate_pct":
            (net > 0).mean() * 100,
        "avg_net_pct":
            net.mean(),
        "profit_factor":
            profit_factor,
        "compounded_return_pct":
            compounded,
        "max_drawdown_pct":
            drawdown.min(),
    }


# ============================================================
# UNIVERSE
# ============================================================

def load_universe():

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

    return universe


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 78)
    print("DIRECTIONAL RSI EXPERIMENT")
    print("=" * 78)

    universe = load_universe()
    regime = load_regime()

    print()
    print(
        f"Stocks selected : {len(universe)}"
    )

    all_trades = []

    data_cache = {}

    usable = 0
    skipped = 0

    total = len(universe)

    for count, (_, stock) in enumerate(
        universe.iterrows(),
        start=1,
    ):

        exchange = str(
            stock["exchange"]
        ).upper()

        symbol = str(
            stock["symbol"]
        )

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

            if data_symbol not in data_cache:

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

                data_cache[
                    data_symbol
                ] = prepare_data(
                    data
                )

            df = data_cache[
                data_symbol
            ]

            usable += 1

        except Exception:

            skipped += 1
            continue

        for variant, settings in VARIANTS.items():

            last_exit_index = -1

            for i in range(
                1,
                len(df) - 1,
            ):

                if i <= last_exit_index:
                    continue

                current = raw_setup(
                    df,
                    i,
                    settings,
                    regime,
                )

                direction = None

                if current["long"]:

                    previous = raw_setup(
                        df,
                        i - 1,
                        settings,
                        regime,
                    )

                    if not previous["long"]:
                        direction = "LONG"

                if (
                    direction is None
                    and current["short"]
                ):

                    previous = raw_setup(
                        df,
                        i - 1,
                        settings,
                        regime,
                    )

                    if not previous["short"]:
                        direction = "SHORT"

                if direction is None:
                    continue

                trade = simulate_trade(
                    df,
                    i,
                    direction,
                    variant,
                    current["regime"],
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

    print()
    print("=" * 78)
    print("DATA COVERAGE")
    print("=" * 78)

    print(
        f"Selected : {total}"
    )

    print(
        f"Usable   : {usable}"
    )

    print(
        f"Skipped  : {skipped}"
    )

    trades = pd.DataFrame(
        all_trades
    )

    if trades.empty:

        print()
        print("No trades generated.")
        return

    trades.to_csv(
        TRADES_FILE,
        index=False,
    )

    # ========================================================
    # SUMMARY
    # ========================================================

    summary_rows = []

    for variant in VARIANTS:

        subset = trades[
            trades["variant"]
            == variant
        ]

        metrics = calculate_metrics(
            subset
        )

        metrics["variant"] = variant

        summary_rows.append(
            metrics
        )

    summary = pd.DataFrame(
        summary_rows
    )

    summary = summary[
        [
            "variant",
            "trades",
            "wins",
            "losses",
            "win_rate_pct",
            "avg_net_pct",
            "profit_factor",
            "compounded_return_pct",
            "max_drawdown_pct",
        ]
    ]

    summary.to_csv(
        SUMMARY_FILE,
        index=False,
    )

    # ========================================================
    # RESULTS
    # ========================================================

    print()
    print("=" * 78)
    print("OVERALL COMPARISON")
    print("=" * 78)

    print(
        f"{'Variant':30s}"
        f"{'Trades':>8s}"
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
            f"{row['variant']:30s}"
            f"{int(row['trades']):8d}"
            f"{row['win_rate_pct']:9.2f}%"
            f"{row['avg_net_pct']:+11.2f}%"
            f"{pf_text:>8s}"
            f"{row['compounded_return_pct']:+11.2f}%"
            f"{row['max_drawdown_pct']:+11.2f}%"
        )

    # ========================================================
    # DIRECTION
    # ========================================================

    print()
    print("=" * 78)
    print("DIRECTION ANALYSIS")
    print("=" * 78)

    for variant in VARIANTS:

        for direction in [
            "LONG",
            "SHORT",
        ]:

            subset = trades[
                (
                    trades["variant"]
                    == variant
                )
                & (
                    trades["direction"]
                    == direction
                )
            ]

            if subset.empty:
                continue

            metrics = calculate_metrics(
                subset
            )

            print(
                f"{variant:30s} "
                f"{direction:5s} "
                f"trades={metrics['trades']:4d} "
                f"win={metrics['win_rate_pct']:6.2f}% "
                f"avg={metrics['avg_net_pct']:+6.2f}% "
                f"PF={metrics['profit_factor']:.2f}"
            )

    # ========================================================
    # REGIME
    # ========================================================

    print()
    print("=" * 78)
    print("REGIME ANALYSIS")
    print("=" * 78)

    for variant in VARIANTS:

        for regime_name in [
            "BULL",
            "BEAR",
            "NEUTRAL",
        ]:

            subset = trades[
                (
                    trades["variant"]
                    == variant
                )
                & (
                    trades["regime"]
                    == regime_name
                )
            ]

            if subset.empty:
                continue

            metrics = calculate_metrics(
                subset
            )

            print(
                f"{variant:30s} "
                f"{regime_name:8s} "
                f"trades={metrics['trades']:4d} "
                f"win={metrics['win_rate_pct']:6.2f}% "
                f"avg={metrics['avg_net_pct']:+6.2f}% "
                f"PF={metrics['profit_factor']:.2f}"
            )

    # ========================================================
    # TRADE COUNTS
    # ========================================================

    print()
    print("=" * 78)
    print("TRADE COUNT CHANGES")
    print("=" * 78)

    current_count = int(
        summary.loc[
            summary["variant"]
            == "CURRENT",
            "trades",
        ].iloc[0]
    )

    for _, row in summary.iterrows():

        change = (
            int(row["trades"])
            - current_count
        )

        print(
            f"{row['variant']:30s} "
            f"trades={int(row['trades']):4d} "
            f"vs current={change:+4d}"
        )

    # ========================================================
    # FILES
    # ========================================================

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

    print()
    print("=" * 78)
    print("DIRECTIONAL RSI EXPERIMENT COMPLETE")
    print("=" * 78)


if __name__ == "__main__":
    main()