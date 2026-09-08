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

SUMMARY_FILE = BASE_DIR / "output" / "directional_rsi_walk_forward.csv"


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
# WALK-FORWARD PERIODS
# ============================================================

PERIODS = [
    ("P1", "2024-10-24", "2025-04-11"),
    ("P2", "2025-04-12", "2025-09-28"),
    ("P3", "2025-09-29", "2026-03-17"),
    ("P4", "2026-03-18", "2026-09-03"),
]


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
    # CROSSOVERS
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

    # --------------------------------------------------------
    # PRECOMPUTE RECENT EVENTS
    # --------------------------------------------------------

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
# RSI CONDITIONS
# ============================================================

def long_rsi_condition(
    rsi,
    mode,
):

    if mode == "threshold":
        return rsi > 60

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
    regime,
):

    row = df.iloc[i]

    current_regime = regime.get(
        df.index[i].normalize(),
        "UNKNOWN",
    )

    long_setup = (
        bool(row["recent_ema_up_13"])
        and bool(row["recent_ema_up_26"])
        and bool(row["recent_macd_up"])
        and row["ema_5"] > row["ema_13"]
        and row["ema_13"] > row["ema_26"]
        and row["macd"] > row["macd_signal"]
        and long_rsi_condition(
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
        and short_rsi_condition(
            row["rsi"],
            settings["short_rsi_mode"],
        )
    )

    if settings["regime_mode"] == "NO_NEUTRAL":

        if current_regime == "NEUTRAL":
            long_setup = False
            short_setup = False

    elif settings["regime_mode"] == "BEAR_ONLY":

        if current_regime != "BEAR":
            long_setup = False
            short_setup = False

    return (
        bool(long_setup),
        bool(short_setup),
        current_regime,
    )


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
        "regime": regime_name,
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

    returns = pd.to_numeric(
        work["net_pnl_pct"],
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
        profit_factor = (
            gross_profit
            / gross_loss
        )
    else:
        profit_factor = np.inf

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

    print("=" * 90)
    print("DIRECTIONAL RSI WALK-FORWARD TEST")
    print("=" * 90)

    universe = load_universe()
    regime = load_regime()

    print()
    print(
        f"Stocks selected : {len(universe)}"
    )

    data_cache = {}

    all_trades = []

    usable = 0
    skipped = 0

    total = len(universe)

    # --------------------------------------------------------
    # LOAD DATA
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
            f"\rLoading "
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

            prepared = prepare_data(
                data
            )

            data_cache[
                data_symbol
            ] = (
                prepared,
                symbol,
                exchange,
            )

            usable += 1

        except Exception:

            skipped += 1

    print()

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

    # --------------------------------------------------------
    # GENERATE TRADES
    # --------------------------------------------------------

    for variant, settings in VARIANTS.items():

        for data_symbol, (
            df,
            symbol,
            exchange,
        ) in data_cache.items():

            last_exit_index = -1

            for i in range(
                1,
                len(df) - 1,
            ):

                if i <= last_exit_index:
                    continue

                long_setup, short_setup, current_regime = get_signal(
                    df,
                    i,
                    settings,
                    regime,
                )

                previous_long = False
                previous_short = False

                if i > 0:

                    (
                        previous_long,
                        previous_short,
                        _,
                    ) = get_signal(
                        df,
                        i - 1,
                        settings,
                        regime,
                    )

                direction = None

                # Fresh LONG
                if (
                    long_setup
                    and not previous_long
                ):
                    direction = "LONG"

                # Fresh SHORT
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
                    current_regime,
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

    trades = pd.DataFrame(
        all_trades
    )

    if trades.empty:

        print()
        print("No trades generated.")
        return

    # --------------------------------------------------------
    # WALK-FORWARD RESULTS
    # --------------------------------------------------------

    rows = []

    for period_name, start_date, end_date in PERIODS:

        start = pd.Timestamp(
            start_date
        )

        end = pd.Timestamp(
            end_date
        )

        period_trades = trades[
            (
                trades["signal_date"]
                >= start
            )
            & (
                trades["signal_date"]
                <= end
            )
        ]

        print()
        print("=" * 90)
        print(
            f"{period_name}: "
            f"{start_date} -> {end_date}"
        )
        print("=" * 90)

        print(
            f"{'Variant':30s}"
            f"{'Trades':>8s}"
            f"{'Win %':>10s}"
            f"{'Avg Net':>12s}"
            f"{'PF':>8s}"
            f"{'Comp.':>12s}"
            f"{'Max DD':>12s}"
        )

        print("-" * 90)

        for variant in VARIANTS:

            subset = period_trades[
                period_trades["variant"]
                == variant
            ]

            metrics = calculate_metrics(
                subset
            )

            pf = metrics[
                "profit_factor"
            ]

            pf_text = (
                "INF"
                if np.isinf(pf)
                else f"{pf:.2f}"
            )

            print(
                f"{variant:30s}"
                f"{metrics['trades']:8d}"
                f"{metrics['win_rate_pct']:9.2f}%"
                f"{metrics['avg_net_pct']:+11.2f}%"
                f"{pf_text:>8s}"
                f"{metrics['compounded_return_pct']:+11.2f}%"
                f"{metrics['max_drawdown_pct']:+11.2f}%"
            )

            row = {
                "period": period_name,
                "start_date": start_date,
                "end_date": end_date,
                "variant": variant,
                **metrics,
            }

            rows.append(row)

    # --------------------------------------------------------
    # STABILITY SUMMARY
    # --------------------------------------------------------

    results = pd.DataFrame(rows)

    results.to_csv(
        SUMMARY_FILE,
        index=False,
    )

    print()
    print("=" * 90)
    print("STABILITY SUMMARY")
    print("=" * 90)

    print(
        f"{'Variant':30s}"
        f"{'Periods':>9s}"
        f"{'Profitable':>12s}"
        f"{'PF > 1':>10s}"
        f"{'Avg PF':>10s}"
        f"{'Median PF':>12s}"
        f"{'Avg Net':>12s}"
        f"{'Median Net':>13s}"
    )

    print("-" * 90)

    for variant in VARIANTS:

        subset = results[
            results["variant"]
            == variant
        ]

        active = subset[
            subset["trades"] > 0
        ]

        if active.empty:
            continue

        pf_values = active[
            "profit_factor"
        ].replace(
            [np.inf, -np.inf],
            np.nan,
        ).dropna()

        profitable = int(
            (
                active[
                    "avg_net_pct"
                ]
                > 0
            ).sum()
        )

        pf_above_one = int(
            (
                active[
                    "profit_factor"
                ]
                > 1
            ).sum()
        )

        avg_pf = (
            pf_values.mean()
            if not pf_values.empty
            else 0
        )

        median_pf = (
            pf_values.median()
            if not pf_values.empty
            else 0
        )

        avg_net = (
            active[
                "avg_net_pct"
            ].mean()
        )

        median_net = (
            active[
                "avg_net_pct"
            ].median()
        )

        print(
            f"{variant:30s}"
            f"{len(active):9d}"
            f"{profitable:12d}"
            f"{pf_above_one:10d}"
            f"{avg_pf:10.2f}"
            f"{median_pf:12.2f}"
            f"{avg_net:+11.2f}%"
            f"{median_net:+12.2f}%"
        )

    print()
    print("=" * 90)
    print("FILES SAVED")
    print("=" * 90)

    print(
        f"Summary : {SUMMARY_FILE}"
    )

    print()
    print("=" * 90)
    print("DIRECTIONAL RSI WALK-FORWARD COMPLETE")
    print("=" * 90)


if __name__ == "__main__":
    main()