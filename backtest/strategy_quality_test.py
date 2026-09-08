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

REGIME_FILE = (
    BASE_DIR
    / "output"
    / "nifty50_regime.csv"
)

OUTPUT_DIR = BASE_DIR / "output"

SUMMARY_FILE = (
    OUTPUT_DIR
    / "strategy_quality_comparison.csv"
)

TRADES_FILE = (
    OUTPUT_DIR
    / "strategy_quality_trades.csv"
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
# EXPERIMENTS
# ============================================================

VARIANTS = {
    "BASE_CURRENT": {
        "use_rsi": True,
        "rsi_long_low": 60,
        "rsi_long_high": 100,
        "rsi_short_low": 0,
        "rsi_short_high": 40,
        "use_extension": False,
        "max_extension_pct": 999,
        "exclude_neutral": False,
    },

    "EMA_MACD": {
        "use_rsi": False,
        "rsi_long_low": 0,
        "rsi_long_high": 100,
        "rsi_short_low": 0,
        "rsi_short_high": 100,
        "use_extension": False,
        "max_extension_pct": 999,
        "exclude_neutral": False,
    },

    "RSI_ZONE": {
        "use_rsi": True,
        "rsi_long_low": 60,
        "rsi_long_high": 65,
        "rsi_short_low": 30,
        "rsi_short_high": 40,
        "use_extension": False,
        "max_extension_pct": 999,
        "exclude_neutral": False,
    },

    "EXT_3": {
        "use_rsi": True,
        "rsi_long_low": 60,
        "rsi_long_high": 100,
        "rsi_short_low": 0,
        "rsi_short_high": 40,
        "use_extension": True,
        "max_extension_pct": 3.0,
        "exclude_neutral": False,
    },

    "RSI_ZONE_EXT_3": {
        "use_rsi": True,
        "rsi_long_low": 60,
        "rsi_long_high": 65,
        "rsi_short_low": 30,
        "rsi_short_high": 40,
        "use_extension": True,
        "max_extension_pct": 3.0,
        "exclude_neutral": False,
    },

    "RSI_ZONE_NO_NEUTRAL": {
        "use_rsi": True,
        "rsi_long_low": 60,
        "rsi_long_high": 65,
        "rsi_short_low": 30,
        "rsi_short_high": 40,
        "use_extension": False,
        "max_extension_pct": 999,
        "exclude_neutral": True,
    },

    "RSI_ZONE_EXT_3_NO_NEUTRAL": {
        "use_rsi": True,
        "rsi_long_low": 60,
        "rsi_long_high": 65,
        "rsi_short_low": 30,
        "rsi_short_high": 40,
        "use_extension": True,
        "max_extension_pct": 3.0,
        "exclude_neutral": True,
    },
}


# ============================================================
# HELPERS
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

    df["rsi_up"] = (
        (df["rsi"] > 60)
        & (
            df["rsi"].shift(1)
            <= 60
        )
    ).fillna(False)

    df["rsi_down"] = (
        (df["rsi"] < 40)
        & (
            df["rsi"].shift(1)
            >= 40
        )
    ).fillna(False)

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
            regime["regime"].astype(str).str.upper(),
        )
    )


# ============================================================
# SIGNAL GENERATION
# ============================================================

def generate_signal(
    df,
    i,
    variant,
    regime,
):

    settings = VARIANTS[
        variant
    ]

    if i < 1:
        return None

    row = df.iloc[i]

    # --------------------------------------------------------
    # EMA events
    # --------------------------------------------------------

    ema_long = (
        recent_event(
            df["ema_up_13"]
        ).iloc[i]
        and recent_event(
            df["ema_up_26"]
        ).iloc[i]
        and row["ema_5"] > row["ema_13"]
        and row["ema_13"] > row["ema_26"]
    )

    ema_short = (
        recent_event(
            df["ema_down_13"]
        ).iloc[i]
        and recent_event(
            df["ema_down_26"]
        ).iloc[i]
        and row["ema_5"] < row["ema_13"]
        and row["ema_13"] < row["ema_26"]
    )

    # --------------------------------------------------------
    # MACD
    # --------------------------------------------------------

    macd_long = (
        recent_event(
            df["macd_up"]
        ).iloc[i]
        and row["macd"]
        > row["macd_signal"]
    )

    macd_short = (
        recent_event(
            df["macd_down"]
        ).iloc[i]
        and row["macd"]
        < row["macd_signal"]
    )

    # --------------------------------------------------------
    # RSI
    # --------------------------------------------------------

    rsi_long = True
    rsi_short = True

    if settings["use_rsi"]:

        rsi_long = (
            row["rsi"]
            >= settings["rsi_long_low"]
            and row["rsi"]
            < settings["rsi_long_high"]
        )

        rsi_short = (
            row["rsi"]
            > settings["rsi_short_low"]
            and row["rsi"]
            <= settings["rsi_short_high"]
        )

    # --------------------------------------------------------
    # Base setup
    # --------------------------------------------------------

    long_setup = (
        ema_long
        and macd_long
        and rsi_long
    )

    short_setup = (
        ema_short
        and macd_short
        and rsi_short
    )

    # --------------------------------------------------------
    # Extension
    # --------------------------------------------------------

    extension_pct = abs(
        (
            row["close"]
            - row["ema_5"]
        )
        / row["ema_5"]
        * 100
    )

    if settings["use_extension"]:

        if extension_pct > (
            settings[
                "max_extension_pct"
            ]
        ):
            long_setup = False
            short_setup = False

    # --------------------------------------------------------
    # Regime
    # --------------------------------------------------------

    signal_date = (
        df.index[i]
        .normalize()
    )

    current_regime = regime.get(
        signal_date,
        "UNKNOWN",
    )

    if settings["exclude_neutral"]:

        if current_regime == "NEUTRAL":

            long_setup = False
            short_setup = False

    # --------------------------------------------------------
    # Fresh signal
    # --------------------------------------------------------

    if i >= 1:

        previous = generate_raw_setup(
            df,
            i - 1,
            settings,
            regime,
        )

        previous_long = previous[
            "long"
        ]

        previous_short = previous[
            "short"
        ]

    else:

        previous_long = False
        previous_short = False

    if long_setup and not previous_long:

        return {
            "direction": "LONG",
            "regime": current_regime,
            "extension_pct": extension_pct,
            "rsi": float(row["rsi"]),
        }

    if short_setup and not previous_short:

        return {
            "direction": "SHORT",
            "regime": current_regime,
            "extension_pct": extension_pct,
            "rsi": float(row["rsi"]),
        }

    return None


# ============================================================
# RAW SETUP
# ============================================================

def generate_raw_setup(
    df,
    i,
    settings,
    regime,
):

    row = df.iloc[i]

    ema_long = (
        recent_event(
            df["ema_up_13"]
        ).iloc[i]
        and recent_event(
            df["ema_up_26"]
        ).iloc[i]
        and row["ema_5"] > row["ema_13"]
        and row["ema_13"] > row["ema_26"]
    )

    ema_short = (
        recent_event(
            df["ema_down_13"]
        ).iloc[i]
        and recent_event(
            df["ema_down_26"]
        ).iloc[i]
        and row["ema_5"] < row["ema_13"]
        and row["ema_13"] < row["ema_26"]
    )

    macd_long = (
        recent_event(
            df["macd_up"]
        ).iloc[i]
        and row["macd"]
        > row["macd_signal"]
    )

    macd_short = (
        recent_event(
            df["macd_down"]
        ).iloc[i]
        and row["macd"]
        < row["macd_signal"]
    )

    rsi_long = True
    rsi_short = True

    if settings["use_rsi"]:

        rsi_long = (
            row["rsi"]
            >= settings["rsi_long_low"]
            and row["rsi"]
            < settings["rsi_long_high"]
        )

        rsi_short = (
            row["rsi"]
            > settings["rsi_short_low"]
            and row["rsi"]
            <= settings["rsi_short_high"]
        )

    long_setup = (
        ema_long
        and macd_long
        and rsi_long
    )

    short_setup = (
        ema_short
        and macd_short
        and rsi_short
    )

    extension_pct = abs(
        (
            row["close"]
            - row["ema_5"]
        )
        / row["ema_5"]
        * 100
    )

    if settings["use_extension"]:

        if extension_pct > (
            settings[
                "max_extension_pct"
            ]
        ):
            long_setup = False
            short_setup = False

    current_regime = regime.get(
        df.index[i].normalize(),
        "UNKNOWN",
    )

    if (
        settings["exclude_neutral"]
        and current_regime == "NEUTRAL"
    ):
        long_setup = False
        short_setup = False

    return {
        "long": bool(long_setup),
        "short": bool(short_setup),
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
    extension_pct,
    rsi,
):

    if signal_index + 1 >= len(df):
        return None

    entry_index = signal_index + 1

    entry_date = df.index[
        entry_index
    ]

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
        # if both are hit in one candle,
        # assume stop occurs first.

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

    holding_days = (
        exit_index
        - entry_index
        + 1
    )

    return {
        "variant": variant,
        "direction": direction,
        "signal_date":
            df.index[
                signal_index
            ],
        "entry_date": entry_date,
        "exit_date":
            df.index[
                exit_index
            ],
        "entry_price":
            entry_price,
        "stop_loss": stop,
        "target": target,
        "exit_price":
            exit_price,
        "exit_reason":
            exit_reason,
        "holding_days":
            holding_days,
        "gross_pnl_pct":
            gross,
        "net_pnl_pct":
            net,
        "regime":
            regime_name,
        "rsi":
            rsi,
        "extension_pct":
            extension_pct,
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
            "win_rate_pct": 0,
            "avg_net_pct": 0,
            "profit_factor": 0,
            "compounded_return_pct": 0,
            "max_drawdown_pct": 0,
        }

    work = df.sort_values(
        "exit_date"
    ).copy()

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

    pf = (
        gross_profit / gross_loss
        if gross_loss > 0
        else np.inf
    )

    equity = (
        (1 + net / 100)
        .cumprod()
    )

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
        "win_rate_pct":
            (net > 0).mean() * 100,
        "avg_net_pct":
            net.mean(),
        "profit_factor":
            pf,
        "compounded_return_pct":
            compounded,
        "max_drawdown_pct":
            drawdown.min(),
    }


# ============================================================
# LOAD UNIVERSE
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
        .isin(["NSE", "BSE"])
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
    print("CONTROLLED STRATEGY QUALITY EXPERIMENT")
    print("=" * 78)

    universe = load_universe()
    regime = load_regime()

    print()
    print(
        f"Stocks selected : {len(universe)}"
    )

    all_trades = []

    data_cache = {}

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

        except Exception:

            skipped += 1

            continue

        # ----------------------------------------------------
        # Run every variant independently.
        # ----------------------------------------------------

        for variant in VARIANTS:

            trades_for_variant = []

            settings = VARIANTS[
                variant
            ]

            # Cache rolling event arrays.
            ema13_up_recent = recent_event(
                df["ema_up_13"]
            )

            ema26_up_recent = recent_event(
                df["ema_up_26"]
            )

            ema13_down_recent = recent_event(
                df["ema_down_13"]
            )

            ema26_down_recent = recent_event(
                df["ema_down_26"]
            )

            macd_up_recent = recent_event(
                df["macd_up"]
            )

            macd_down_recent = recent_event(
                df["macd_down"]
            )

            last_exit_index = -1

            for i in range(
                1,
                len(df) - 1,
            ):

                row = df.iloc[i]

                signal_date = (
                    df.index[i]
                    .normalize()
                )

                regime_name = regime.get(
                    signal_date,
                    "UNKNOWN",
                )

                # ------------------------------------------------
                # EMA
                # ------------------------------------------------

                ema_long = (
                    bool(
                        ema13_up_recent.iloc[i]
                    )
                    and bool(
                        ema26_up_recent.iloc[i]
                    )
                    and row["ema_5"]
                    > row["ema_13"]
                    and row["ema_13"]
                    > row["ema_26"]
                )

                ema_short = (
                    bool(
                        ema13_down_recent.iloc[i]
                    )
                    and bool(
                        ema26_down_recent.iloc[i]
                    )
                    and row["ema_5"]
                    < row["ema_13"]
                    and row["ema_13"]
                    < row["ema_26"]
                )

                # ------------------------------------------------
                # MACD
                # ------------------------------------------------

                macd_long = (
                    bool(
                        macd_up_recent.iloc[i]
                    )
                    and row["macd"]
                    > row["macd_signal"]
                )

                macd_short = (
                    bool(
                        macd_down_recent.iloc[i]
                    )
                    and row["macd"]
                    < row["macd_signal"]
                )

                # ------------------------------------------------
                # RSI
                # ------------------------------------------------

                if settings["use_rsi"]:

                    rsi_long = (
                        row["rsi"]
                        >= settings[
                            "rsi_long_low"
                        ]
                        and row["rsi"]
                        < settings[
                            "rsi_long_high"
                        ]
                    )

                    rsi_short = (
                        row["rsi"]
                        > settings[
                            "rsi_short_low"
                        ]
                        and row["rsi"]
                        <= settings[
                            "rsi_short_high"
                        ]
                    )

                else:

                    rsi_long = True
                    rsi_short = True

                long_setup = (
                    ema_long
                    and macd_long
                    and rsi_long
                )

                short_setup = (
                    ema_short
                    and macd_short
                    and rsi_short
                )

                # ------------------------------------------------
                # Extension
                # ------------------------------------------------

                extension_pct = abs(
                    (
                        row["close"]
                        - row["ema_5"]
                    )
                    / row["ema_5"]
                    * 100
                )

                if settings[
                    "use_extension"
                ]:

                    if (
                        extension_pct
                        > settings[
                            "max_extension_pct"
                        ]
                    ):

                        long_setup = False
                        short_setup = False

                # ------------------------------------------------
                # Regime
                # ------------------------------------------------

                if (
                    settings[
                        "exclude_neutral"
                    ]
                    and regime_name
                    == "NEUTRAL"
                ):

                    long_setup = False
                    short_setup = False

                # ------------------------------------------------
                # Freshness
                # ------------------------------------------------

                previous_long = False
                previous_short = False

                if i > 0:

                    previous = generate_raw_setup(
                        df,
                        i - 1,
                        settings,
                        regime,
                    )

                    previous_long = previous[
                        "long"
                    ]

                    previous_short = previous[
                        "short"
                    ]

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

                # ------------------------------------------------
                # No overlapping trades.
                # ------------------------------------------------

                if i <= last_exit_index:
                    continue

                trade = simulate_trade(
                    df,
                    i,
                    direction,
                    variant,
                    regime_name,
                    extension_pct,
                    float(row["rsi"]),
                )

                if trade is None:
                    continue

                trade["symbol"] = symbol
                trade["exchange"] = exchange
                trade["data_symbol"] = data_symbol

                trades_for_variant.append(
                    trade
                )

                exit_date = pd.Timestamp(
                    trade["exit_date"]
                )

                exit_positions = np.where(
                    df.index
                    <= exit_date
                )[0]

                if len(exit_positions):

                    last_exit_index = int(
                        exit_positions[-1]
                    )

            all_trades.extend(
                trades_for_variant
            )

    print()

    trades = pd.DataFrame(
        all_trades
    )

    if trades.empty:

        print(
            "No trades generated."
        )

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
    # PRINT
    # ========================================================

    print()
    print("=" * 78)
    print("EXPERIMENT RESULTS")
    print("=" * 78)

    print(
        f"{'Variant':25s}"
        f"{'Trades':>9s}"
        f"{'Win %':>10s}"
        f"{'Avg Net':>12s}"
        f"{'PF':>9s}"
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
            f"{row['variant']:25s}"
            f"{int(row['trades']):9d}"
            f"{row['win_rate_pct']:9.2f}%"
            f"{row['avg_net_pct']:+11.2f}%"
            f"{pf_text:>9s}"
            f"{row['compounded_return_pct']:+11.2f}%"
            f"{row['max_drawdown_pct']:+11.2f}%"
        )

    # ========================================================
    # DIRECTION ANALYSIS
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
                f"{variant:25s} "
                f"{direction:5s} "
                f"trades={metrics['trades']:4d} "
                f"win={metrics['win_rate_pct']:6.2f}% "
                f"avg={metrics['avg_net_pct']:+6.2f}% "
                f"PF={metrics['profit_factor']:.2f}"
            )

    # ========================================================
    # REGIME ANALYSIS
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
                f"{variant:25s} "
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

    base_count = int(
        summary.loc[
            summary["variant"]
            == "BASE_CURRENT",
            "trades",
        ].iloc[0]
    )

    for _, row in summary.iterrows():

        change = (
            int(row["trades"])
            - base_count
        )

        print(
            f"{row['variant']:25s} "
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
    print("CONTROLLED EXPERIMENT COMPLETE")
    print("=" * 78)


if __name__ == "__main__":
    main()