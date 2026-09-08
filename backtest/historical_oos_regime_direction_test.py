from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

INPUT_FILE = (
    BASE_DIR
    / "output"
    / "historical_oos_regime_trades.csv"
)

OUTPUT_FILE = (
    BASE_DIR
    / "output"
    / "historical_oos_regime_direction_comparison.csv"
)

OUTPUT_TRADES = (
    BASE_DIR
    / "output"
    / "historical_oos_regime_direction_trades.csv"
)


# ============================================================
# STRATEGY FILTERS
# ============================================================

# The underlying historical OOS test already used:
#
#   CURRENT
#   LONG_RSI_ZONE
#   EMA_MACD
#
# This experiment does NOT generate new signals.
#
# It takes the already-generated OOS trades and asks:
#
#   "What happens if we selectively allow/reject
#    trades based on NIFTY regime and direction?"
#
# This makes the experiment easy to audit and avoids changing
# the underlying signal-generation logic.


VARIANTS = [
    "CURRENT",
    "LONG_RSI_ZONE",
    "EMA_MACD",
]


# ============================================================
# METRICS
# ============================================================

def calculate_metrics(trades):

    if trades.empty:

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
        }

    returns = pd.to_numeric(
        trades["net_pnl_pct"],
        errors="coerce",
    ).dropna().to_numpy(
        dtype=float
    )

    if len(returns) == 0:

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
        }

    wins = returns > 0
    losses = returns <= 0

    gross_profit = returns[wins].sum()

    gross_loss = abs(
        returns[losses].sum()
    )

    if gross_loss > 0:

        profit_factor = (
            gross_profit
            / gross_loss
        )

    else:

        profit_factor = np.inf

    equity = np.cumprod(
        1.0
        + returns / 100.0
    )

    running_max = np.maximum.accumulate(
        equity
    )

    drawdown = (
        equity
        / running_max
        - 1.0
    )

    holding = pd.to_numeric(
        trades["holding_days"],
        errors="coerce",
    ).dropna()

    avg_holding = (
        float(holding.mean())
        if not holding.empty
        else 0.0
    )

    return {
        "trades":
            int(len(returns)),

        "wins":
            int(wins.sum()),

        "losses":
            int(losses.sum()),

        "win_rate":
            float(
                wins.mean()
                * 100.0
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
                    - 1.0
                )
                * 100.0
            ),

        "max_drawdown_pct":
            float(
                drawdown.min()
                * 100.0
            ),

        "avg_holding_days":
            avg_holding,
    }


# ============================================================
# FILTER DEFINITIONS
# ============================================================

def allow_all(
    regime,
    direction,
):

    return True


def no_neutral(
    regime,
    direction,
):

    return regime in {
        "BULL",
        "BEAR",
    }


def bear_only(
    regime,
    direction,
):

    return regime == "BEAR"


def bull_bear_short_only(
    regime,
    direction,
):

    if regime not in {
        "BULL",
        "BEAR",
    }:

        return False

    return direction == "SHORT"


def bull_short_bear_both(
    regime,
    direction,
):

    if regime == "BULL":

        return direction == "SHORT"

    if regime == "BEAR":

        return True

    return False


def bull_both_bear_short(
    regime,
    direction,
):

    if regime == "BULL":

        return True

    if regime == "BEAR":

        return direction == "SHORT"

    return False


def bull_short_bear_long(
    regime,
    direction,
):

    if regime == "BULL":

        return direction == "SHORT"

    if regime == "BEAR":

        return direction == "LONG"

    return False


def bull_long_bear_both(
    regime,
    direction,
):

    if regime == "BULL":

        return direction == "LONG"

    if regime == "BEAR":

        return True

    return False


FILTERS = {
    "ALL_TRADES":
        allow_all,

    "NO_NEUTRAL":
        no_neutral,

    "BEAR_ONLY":
        bear_only,

    "BULL_BEAR_SHORT_ONLY":
        bull_bear_short_only,

    "BULL_SHORT_BEAR_BOTH":
        bull_short_bear_both,

    "BULL_BOTH_BEAR_SHORT":
        bull_both_bear_short,

    "BULL_SHORT_BEAR_LONG":
        bull_short_bear_long,

    "BULL_LONG_BEAR_BOTH":
        bull_long_bear_both,
}


# ============================================================
# APPLY FILTER
# ============================================================

def apply_filter(
    trades,
    filter_name,
):

    filter_function = FILTERS[
        filter_name
    ]

    mask = trades.apply(
        lambda row:
            filter_function(
                str(
                    row["regime"]
                ),
                str(
                    row["direction"]
                ),
            ),
        axis=1,
    )

    return trades.loc[
        mask
    ].copy()


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 100)
    print(
        "HISTORICAL OOS REGIME × DIRECTION TEST"
    )
    print("=" * 100)

    if not INPUT_FILE.exists():

        print()
        print(
            "ERROR: Required input file not found:"
        )

        print(
            INPUT_FILE
        )

        print()
        print(
            "Run this first:"
        )

        print(
            "python -m backtest.historical_oos_regime_test"
        )

        return

    trades = pd.read_csv(
        INPUT_FILE
    )

    if trades.empty:

        print()
        print(
            "ERROR: Input trade file is empty."
        )

        return

    required_columns = [
        "variant",
        "regime",
        "direction",
        "signal_date",
        "net_pnl_pct",
        "holding_days",
    ]

    missing = [
        column
        for column in required_columns
        if column not in trades.columns
    ]

    if missing:

        print()
        print(
            "ERROR: Missing columns:"
        )

        print(
            missing
        )

        return

    trades["signal_date"] = pd.to_datetime(
        trades["signal_date"],
        errors="coerce",
    )

    trades["net_pnl_pct"] = pd.to_numeric(
        trades["net_pnl_pct"],
        errors="coerce",
    )

    trades["holding_days"] = pd.to_numeric(
        trades["holding_days"],
        errors="coerce",
    )

    trades = trades.dropna(
        subset=[
            "signal_date",
            "net_pnl_pct",
        ]
    ).copy()

    trades = trades.sort_values(
        [
            "variant",
            "signal_date",
        ]
    ).reset_index(
        drop=True
    )

    print()
    print(
        f"Input trades: {len(trades)}"
    )

    print()
    print(
        "Variants:"
    )

    for variant in VARIANTS:

        count = int(
            (
                trades["variant"]
                == variant
            ).sum()
        )

        print(
            f"  {variant:<20s}"
            f"{count:>6d} trades"
        )

    # ========================================================
    # TEST FILTERS
    # ========================================================

    all_results = []
    selected_trade_frames = []

    for variant in VARIANTS:

        variant_trades = trades[
            trades["variant"]
            == variant
        ].copy()

        print()
        print("=" * 100)
        print(
            f"{variant}"
        )
        print("=" * 100)

        print(
            f"{'Filter':<32s}"
            f"{'Trades':>9s}"
            f"{'Win %':>10s}"
            f"{'Avg Net':>12s}"
            f"{'PF':>10s}"
            f"{'Comp.':>12s}"
            f"{'Max DD':>12s}"
        )

        print(
            "-" * 100
        )

        for filter_name in FILTERS:

            selected = apply_filter(
                variant_trades,
                filter_name,
            )

            metrics = calculate_metrics(
                selected
            )

            row = {
                "variant":
                    variant,

                "filter":
                    filter_name,

                **metrics,
            }

            all_results.append(
                row
            )

            selected_copy = selected.copy()

            selected_copy[
                "filter"
            ] = filter_name

            selected_trade_frames.append(
                selected_copy
            )

            print(
                f"{filter_name:<32s}"
                f"{metrics['trades']:>9d}"
                f"{metrics['win_rate']:>9.2f}%"
                f"{metrics['avg_net_pct']:>+11.2f}%"
                f"{metrics['profit_factor']:>10.2f}"
                f"{metrics['compounded_return_pct']:>+11.2f}%"
                f"{metrics['max_drawdown_pct']:>+11.2f}%"
            )

    # ========================================================
    # SAVE RESULTS
    # ========================================================

    results_df = pd.DataFrame(
        all_results
    )

    results_df.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    selected_trades_df = pd.concat(
        selected_trade_frames,
        ignore_index=True,
    )

    selected_trades_df.to_csv(
        OUTPUT_TRADES,
        index=False,
    )

    # ========================================================
    # BEST FILTER BY AVERAGE NET
    # ========================================================

    print()
    print("=" * 100)
    print(
        "BEST FILTERS BY AVG NET"
    )
    print("=" * 100)

    for variant in VARIANTS:

        subset = results_df[
            results_df["variant"]
            == variant
        ].copy()

        subset = subset.sort_values(
            "avg_net_pct",
            ascending=False,
        )

        print()
        print(
            f"{variant}:"
        )

        for _, row in subset.head(3).iterrows():

            print(
                f"  {row['filter']:<32s}"
                f" Avg Net "
                f"{row['avg_net_pct']:+.2f}%"
                f" | PF "
                f"{row['profit_factor']:.2f}"
                f" | Trades "
                f"{int(row['trades'])}"
            )

    # ========================================================
    # BEST FILTER BY PROFIT FACTOR
    # ========================================================

    print()
    print("=" * 100)
    print(
        "BEST FILTERS BY PROFIT FACTOR"
    )
    print("=" * 100)

    for variant in VARIANTS:

        subset = results_df[
            results_df["variant"]
            == variant
        ].copy()

        subset = subset[
            subset["trades"]
            >= 30
        ]

        subset = subset.sort_values(
            "profit_factor",
            ascending=False,
        )

        print()
        print(
            f"{variant}:"
        )

        for _, row in subset.head(3).iterrows():

            print(
                f"  {row['filter']:<32s}"
                f" PF "
                f"{row['profit_factor']:.2f}"
                f" | Avg Net "
                f"{row['avg_net_pct']:+.2f}%"
                f" | Trades "
                f"{int(row['trades'])}"
            )

    # ========================================================
    # DIRECTIONAL EFFECT
    # ========================================================

    print()
    print("=" * 100)
    print(
        "BASELINE DIRECTION DISTRIBUTION"
    )
    print("=" * 100)

    for variant in VARIANTS:

        subset = trades[
            trades["variant"]
            == variant
        ]

        print()
        print(
            variant
        )

        for direction in [
            "LONG",
            "SHORT",
        ]:

            direction_trades = subset[
                subset["direction"]
                == direction
            ]

            metrics = calculate_metrics(
                direction_trades
            )

            print(
                f"  {direction:<8s}"
                f" Trades "
                f"{metrics['trades']:>4d}"
                f" | Win "
                f"{metrics['win_rate']:>6.2f}%"
                f" | Avg Net "
                f"{metrics['avg_net_pct']:>+6.2f}%"
                f" | PF "
                f"{metrics['profit_factor']:.2f}"
            )

    # ========================================================
    # REGIME DISTRIBUTION
    # ========================================================

    print()
    print("=" * 100)
    print(
        "BASELINE REGIME DISTRIBUTION"
    )
    print("=" * 100)

    for variant in VARIANTS:

        subset = trades[
            trades["variant"]
            == variant
        ]

        print()
        print(
            variant
        )

        for regime in [
            "BULL",
            "BEAR",
            "NEUTRAL",
        ]:

            regime_trades = subset[
                subset["regime"]
                == regime
            ]

            metrics = calculate_metrics(
                regime_trades
            )

            print(
                f"  {regime:<8s}"
                f" Trades "
                f"{metrics['trades']:>4d}"
                f" | Win "
                f"{metrics['win_rate']:>6.2f}%"
                f" | Avg Net "
                f"{metrics['avg_net_pct']:>+6.2f}%"
                f" | PF "
                f"{metrics['profit_factor']:.2f}"
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
        f"Comparison:"
        f" {OUTPUT_FILE}"
    )

    print(
        f"Filtered trades:"
        f" {OUTPUT_TRADES}"
    )

    print()
    print("=" * 100)
    print(
        "REGIME × DIRECTION TEST COMPLETE"
    )
    print("=" * 100)


if __name__ == "__main__":
    main()