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

OUTPUT_COMPARISON = (
    BASE_DIR
    / "output"
    / "historical_oos_direction_walk_forward.csv"
)

OUTPUT_TRADES = (
    BASE_DIR
    / "output"
    / "historical_oos_direction_walk_forward_trades.csv"
)


# ============================================================
# WALK-FORWARD PERIODS
# ============================================================

PERIODS = [
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
        }

    returns = pd.to_numeric(
        trades["net_pnl_pct"],
        errors="coerce",
    ).dropna().to_numpy(dtype=float)

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
        }

    wins = returns > 0
    losses = returns <= 0

    gross_profit = returns[wins].sum()
    gross_loss = abs(returns[losses].sum())

    if gross_loss > 0:
        profit_factor = (
            gross_profit / gross_loss
        )
    else:
        profit_factor = np.inf

    equity = np.cumprod(
        1.0 + returns / 100.0
    )

    running_max = np.maximum.accumulate(
        equity
    )

    drawdown = (
        equity / running_max - 1.0
    )

    return {
        "trades": int(len(returns)),
        "wins": int(wins.sum()),
        "losses": int(losses.sum()),
        "win_rate": float(wins.mean() * 100.0),
        "avg_net_pct": float(returns.mean()),
        "profit_factor": float(profit_factor),
        "compounded_return_pct": float(
            (equity[-1] - 1.0) * 100.0
        ),
        "max_drawdown_pct": float(
            drawdown.min() * 100.0
        ),
    }


# ============================================================
# DIRECTION FILTERS
# ============================================================

def all_trades(trades):
    return trades.copy()


def short_only(trades):
    return trades[
        trades["direction"] == "SHORT"
    ].copy()


def long_only(trades):
    return trades[
        trades["direction"] == "LONG"
    ].copy()


def no_neutral(trades):
    return trades[
        trades["regime"].isin(
            ["BULL", "BEAR"]
        )
    ].copy()


def short_no_neutral(trades):

    return trades[
        (trades["direction"] == "SHORT")
        & (
            trades["regime"].isin(
                ["BULL", "BEAR"]
            )
        )
    ].copy()


FILTERS = {
    "ALL": all_trades,
    "SHORT_ONLY": short_only,
    "LONG_ONLY": long_only,
    "NO_NEUTRAL": no_neutral,
    "SHORT_NO_NEUTRAL": short_no_neutral,
}


# ============================================================
# PERIOD ASSIGNMENT
# ============================================================

def assign_period(date):

    for name, start, end in PERIODS:

        start_date = pd.Timestamp(start)
        end_date = pd.Timestamp(end)

        if (
            date >= start_date
            and date <= end_date
        ):
            return name

    return None


# ============================================================
# SELECT DIRECTION USING PRIOR PERIOD
# ============================================================

def choose_direction_from_training(
    training_trades,
    variant,
):

    subset = training_trades[
        training_trades["variant"]
        == variant
    ].copy()

    long_trades = subset[
        subset["direction"] == "LONG"
    ]

    short_trades = subset[
        subset["direction"] == "SHORT"
    ]

    long_metrics = calculate_metrics(
        long_trades
    )

    short_metrics = calculate_metrics(
        short_trades
    )

    # Require enough trades before allowing
    # a direction to win the selection.
    #
    # This prevents one or two lucky trades
    # from determining the next period.

    minimum_training_trades = 20

    long_usable = (
        long_metrics["trades"]
        >= minimum_training_trades
    )

    short_usable = (
        short_metrics["trades"]
        >= minimum_training_trades
    )

    if (
        long_usable
        and short_usable
    ):

        # Primary criterion:
        # average net return.
        #
        # Secondary criterion:
        # profit factor.

        if (
            long_metrics["avg_net_pct"]
            > short_metrics["avg_net_pct"]
        ):

            return (
                "LONG",
                long_metrics,
                short_metrics,
            )

        if (
            short_metrics["avg_net_pct"]
            > long_metrics["avg_net_pct"]
        ):

            return (
                "SHORT",
                long_metrics,
                short_metrics,
            )

        if (
            long_metrics["profit_factor"]
            >= short_metrics["profit_factor"]
        ):

            return (
                "LONG",
                long_metrics,
                short_metrics,
            )

        return (
            "SHORT",
            long_metrics,
            short_metrics,
        )

    # If only one side has enough observations,
    # use that side.

    if long_usable:
        return (
            "LONG",
            long_metrics,
            short_metrics,
        )

    if short_usable:
        return (
            "SHORT",
            long_metrics,
            short_metrics,
        )

    # If neither side has enough observations,
    # remain neutral and take no trades.

    return (
        "NONE",
        long_metrics,
        short_metrics,
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 100)
    print(
        "TRUE WALK-FORWARD DIRECTION TEST"
    )
    print("=" * 100)

    print()
    print(
        "Important:"
    )
    print(
        "Direction selection uses ONLY the previous period."
    )
    print(
        "The selected direction is then applied to the next unseen period."
    )

    if not INPUT_FILE.exists():

        print()
        print(
            "ERROR: Input file not found:"
        )
        print(
            INPUT_FILE
        )

        print()
        print(
            "Run first:"
        )
        print(
            "python -m backtest.historical_oos_regime_test"
        )

        return

    trades = pd.read_csv(
        INPUT_FILE
    )

    required_columns = [
        "variant",
        "direction",
        "regime",
        "signal_date",
        "net_pnl_pct",
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

    trades = trades.dropna(
        subset=[
            "signal_date",
            "net_pnl_pct",
        ]
    ).copy()

    trades["period"] = trades[
        "signal_date"
    ].apply(
        assign_period
    )

    trades = trades[
        trades["period"].notna()
    ].copy()

    trades = trades.sort_values(
        "signal_date"
    ).reset_index(
        drop=True
    )

    variants = [
        "CURRENT",
        "EMA_MACD",
    ]

    # ========================================================
    # FIXED STRATEGIES
    # ========================================================

    print()
    print("=" * 100)
    print(
        "FIXED BASELINES"
    )
    print("=" * 100)

    fixed_rows = []

    for variant in variants:

        variant_trades = trades[
            trades["variant"]
            == variant
        ].copy()

        print()
        print(
            variant
        )

        for filter_name in [
            "ALL",
            "SHORT_ONLY",
            "NO_NEUTRAL",
            "SHORT_NO_NEUTRAL",
        ]:

            selected = FILTERS[
                filter_name
            ](
                variant_trades
            )

            metrics = calculate_metrics(
                selected
            )

            fixed_rows.append(
                {
                    "test_type":
                        "FIXED",

                    "variant":
                        variant,

                    "period":
                        "ALL",

                    "training_period":
                        "",

                    "selected_direction":
                        filter_name,

                    **metrics,
                }
            )

            print(
                f"  {filter_name:<20s}"
                f" Trades "
                f"{metrics['trades']:>4d}"
                f" | Win "
                f"{metrics['win_rate']:>6.2f}%"
                f" | Avg Net "
                f"{metrics['avg_net_pct']:>+6.2f}%"
                f" | PF "
                f"{metrics['profit_factor']:.2f}"
                f" | DD "
                f"{metrics['max_drawdown_pct']:>+7.2f}%"
            )

    # ========================================================
    # TRUE WALK-FORWARD
    # ========================================================

    wf_rows = []
    wf_trade_frames = []

    print()
    print("=" * 100)
    print(
        "TRUE WALK-FORWARD RESULTS"
    )
    print("=" * 100)

    for variant in variants:

        print()
        print(
            f"VARIANT: {variant}"
        )

        for index in range(
            len(PERIODS)
        ):

            period_name, start, end = (
                PERIODS[index]
            )

            test_trades = trades[
                (
                    trades["variant"]
                    == variant
                )
                & (
                    trades["period"]
                    == period_name
                )
            ].copy()

            # First period has no previous
            # training period.

            if index == 0:

                selected_direction = (
                    "ALL"
                )

                training_period = ""

                selected = test_trades.copy()

                training_note = (
                    "No prior period; "
                    "baseline only"
                )

            else:

                previous_period = (
                    PERIODS[index - 1][0]
                )

                training_trades = trades[
                    (
                        trades["variant"]
                        == variant
                    )
                    & (
                        trades["period"]
                        == previous_period
                    )
                ].copy()

                (
                    selected_direction,
                    long_metrics,
                    short_metrics,
                ) = choose_direction_from_training(
                    training_trades,
                    variant,
                )

                training_period = (
                    previous_period
                )

                if selected_direction == "SHORT":

                    selected = test_trades[
                        test_trades["direction"]
                        == "SHORT"
                    ].copy()

                elif selected_direction == "LONG":

                    selected = test_trades[
                        test_trades["direction"]
                        == "LONG"
                    ].copy()

                else:

                    selected = (
                        test_trades
                        .iloc[0:0]
                        .copy()
                    )

                training_note = (
                    f"Previous period "
                    f"{previous_period}: "
                    f"LONG avg "
                    f"{long_metrics['avg_net_pct']:+.2f}% "
                    f"PF "
                    f"{long_metrics['profit_factor']:.2f}; "
                    f"SHORT avg "
                    f"{short_metrics['avg_net_pct']:+.2f}% "
                    f"PF "
                    f"{short_metrics['profit_factor']:.2f}"
                )

            metrics = calculate_metrics(
                selected
            )

            wf_rows.append(
                {
                    "test_type":
                        "WALK_FORWARD",

                    "variant":
                        variant,

                    "period":
                        period_name,

                    "training_period":
                        training_period,

                    "selected_direction":
                        selected_direction,

                    "start_date":
                        start,

                    "end_date":
                        end,

                    "training_note":
                        training_note,

                    **metrics,
                }
            )

            selected_copy = selected.copy()

            selected_copy[
                "walk_forward_period"
            ] = period_name

            selected_copy[
                "training_period"
            ] = training_period

            selected_copy[
                "selected_direction"
            ] = selected_direction

            selected_copy[
                "variant_test"
            ] = variant

            wf_trade_frames.append(
                selected_copy
            )

            print()
            print(
                f"  {period_name}"
            )

            print(
                f"    Training: "
                f"{training_period or 'NONE'}"
            )

            print(
                f"    Selected: "
                f"{selected_direction}"
            )

            print(
                f"    Trades: "
                f"{metrics['trades']}"
            )

            print(
                f"    Win rate: "
                f"{metrics['win_rate']:.2f}%"
            )

            print(
                f"    Avg net: "
                f"{metrics['avg_net_pct']:+.2f}%"
            )

            print(
                f"    PF: "
                f"{metrics['profit_factor']:.2f}"
            )

            print(
                f"    Compound: "
                f"{metrics['compounded_return_pct']:+.2f}%"
            )

            print(
                f"    Max DD: "
                f"{metrics['max_drawdown_pct']:+.2f}%"
            )

    # ========================================================
    # WALK-FORWARD SUMMARY
    # ========================================================

    wf_df = pd.DataFrame(
        wf_rows
    )

    print()
    print("=" * 100)
    print(
        "WALK-FORWARD STABILITY"
    )
    print("=" * 100)

    for variant in variants:

        subset = wf_df[
            wf_df["variant"]
            == variant
        ].copy()

        # P1 is baseline because there was
        # no training period.

        test_subset = subset[
            subset["training_period"]
            != ""
        ].copy()

        profitable_periods = int(
            (
                test_subset["avg_net_pct"]
                > 0
            ).sum()
        )

        positive_pf_periods = int(
            (
                test_subset["profit_factor"]
                > 1.0
            ).sum()
        )

        print()
        print(
            variant
        )

        print(
            f"  Adaptive periods: "
            f"{len(test_subset)}"
        )

        print(
            f"  Profitable periods: "
            f"{profitable_periods}/"
            f"{len(test_subset)}"
        )

        print(
            f"  PF > 1 periods: "
            f"{positive_pf_periods}/"
            f"{len(test_subset)}"
        )

        if not test_subset.empty:

            print(
                f"  Average PF: "
                f"{test_subset['profit_factor'].mean():.2f}"
            )

            print(
                f"  Median PF: "
                f"{test_subset['profit_factor'].median():.2f}"
            )

            print(
                f"  Average net/trade: "
                f"{test_subset['avg_net_pct'].mean():+.2f}%"
            )

            print(
                f"  Median net/trade: "
                f"{test_subset['avg_net_pct'].median():+.2f}%"
            )

        print(
            "  Direction choices:"
        )

        print(
            test_subset[
                [
                    "period",
                    "training_period",
                    "selected_direction",
                ]
            ].to_string(
                index=False
            )
        )

    # ========================================================
    # COMBINED FIXED + WALK-FORWARD OUTPUT
    # ========================================================

    fixed_df = pd.DataFrame(
        fixed_rows
    )

    combined_df = pd.concat(
        [
            fixed_df,
            wf_df,
        ],
        ignore_index=True,
    )

    combined_df.to_csv(
        OUTPUT_COMPARISON,
        index=False,
    )

    if wf_trade_frames:

        wf_trades_df = pd.concat(
            wf_trade_frames,
            ignore_index=True,
        )

    else:

        wf_trades_df = pd.DataFrame()

    wf_trades_df.to_csv(
        OUTPUT_TRADES,
        index=False,
    )

    # ========================================================
    # FINAL INTERPRETATION
    # ========================================================

    print()
    print("=" * 100)
    print(
        "INTERPRETATION"
    )
    print("=" * 100)

    print()
    print(
        "SHORT_ONLY is NOT automatically accepted."
    )

    print(
        "It must outperform ALL/BASELINE across"
    )

    print(
        "multiple unseen walk-forward periods."
    )

    print()
    print(
        "If direction selection repeatedly chooses SHORT"
    )

    print(
        "using only prior-period information and remains"
    )

    print(
        "profitable in the following unseen period,"
    )

    print(
        "the short-side edge becomes much more credible."
    )

    print()
    print(
        "If the selected direction changes frequently"
    )

    print(
        "or fails in the next period, we reject the"
    )

    print(
        "adaptive direction filter."
    )

    print()
    print("=" * 100)
    print(
        "FILES SAVED"
    )
    print("=" * 100)

    print(
        OUTPUT_COMPARISON
    )

    print(
        OUTPUT_TRADES
    )

    print()
    print("=" * 100)
    print(
        "TRUE WALK-FORWARD TEST COMPLETE"
    )
    print("=" * 100)


if __name__ == "__main__":
    main()