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
    / "historical_oos_cost_sensitivity.csv"
)


# ============================================================
# TEST SETTINGS
# ============================================================

# The existing backtest trades already contain a base
# transaction-cost assumption.
#
# We reconstruct gross return from:
#
#     net = gross - base_cost
#
# and then test higher total costs.

BASE_COST_PCT = 0.15

COST_LEVELS = [
    0.00,
    0.05,
    0.10,
    0.15,
    0.20,
    0.25,
    0.30,
    0.40,
    0.50,
    0.60,
]


# ============================================================
# PERIODS
# ============================================================

PERIODS = [
    ("P1", "2025-09-04", "2025-12-31"),
    ("P2", "2026-01-01", "2026-03-31"),
    ("P3", "2026-04-01", "2026-06-30"),
    ("P4", "2026-07-01", "2026-09-05"),
]


# ============================================================
# METRICS
# ============================================================

def calculate_metrics(returns):

    returns = np.asarray(
        returns,
        dtype=float,
    )

    returns = returns[
        np.isfinite(returns)
    ]

    if len(returns) == 0:

        return {
            "trades": 0,
            "win_rate": 0.0,
            "avg_net_pct": 0.0,
            "profit_factor": 0.0,
            "compounded_return_pct": 0.0,
            "max_drawdown_pct": 0.0,
        }

    wins = returns > 0

    gross_profit = returns[wins].sum()

    gross_loss = abs(
        returns[~wins].sum()
    )

    if gross_loss > 0:
        profit_factor = (
            gross_profit
            / gross_loss
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
        equity / running_max
        - 1.0
    )

    return {
        "trades": int(len(returns)),
        "win_rate": float(
            wins.mean() * 100.0
        ),
        "avg_net_pct": float(
            returns.mean()
        ),
        "profit_factor": float(
            profit_factor
        ),
        "compounded_return_pct": float(
            (equity[-1] - 1.0)
            * 100.0
        ),
        "max_drawdown_pct": float(
            drawdown.min()
            * 100.0
        ),
    }


# ============================================================
# COST ADJUSTMENT
# ============================================================

def adjust_for_cost(
    trades,
    cost_pct,
):

    adjusted = trades.copy()

    # Existing net return was calculated using
    # BASE_COST_PCT.
    #
    # Recover approximate gross return first.

    adjusted[
        "gross_reconstructed_pct"
    ] = (
        adjusted["net_pnl_pct"]
        + BASE_COST_PCT
    )

    additional_cost = (
        cost_pct
        - BASE_COST_PCT
    )

    adjusted[
        "adjusted_net_pct"
    ] = (
        adjusted[
            "gross_reconstructed_pct"
        ]
        - cost_pct
    )

    return adjusted


# ============================================================
# FILTERS
# ============================================================

def apply_filter(
    trades,
    variant,
    filter_name,
):

    subset = trades[
        trades["variant"]
        == variant
    ].copy()

    if filter_name == "ALL":

        return subset

    if filter_name == "SHORT_ONLY":

        return subset[
            subset["direction"]
            == "SHORT"
        ].copy()

    if filter_name == "NO_NEUTRAL":

        return subset[
            subset["regime"].isin(
                ["BULL", "BEAR"]
            )
        ].copy()

    if filter_name == "SHORT_NO_NEUTRAL":

        return subset[
            (
                subset["direction"]
                == "SHORT"
            )
            & (
                subset["regime"].isin(
                    ["BULL", "BEAR"]
                )
            )
        ].copy()

    raise ValueError(
        f"Unknown filter: {filter_name}"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 100)
    print(
        "HISTORICAL OOS TRANSACTION-COST SENSITIVITY"
    )
    print("=" * 100)

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

    trades[
        "signal_date"
    ] = pd.to_datetime(
        trades["signal_date"],
        errors="coerce",
    )

    trades[
        "net_pnl_pct"
    ] = pd.to_numeric(
        trades["net_pnl_pct"],
        errors="coerce",
    )

    trades = trades.dropna(
        subset=[
            "signal_date",
            "net_pnl_pct",
        ]
    ).copy()

    trades = trades.sort_values(
        "signal_date"
    ).reset_index(
        drop=True
    )

    # ========================================================
    # TEST DEFINITIONS
    # ========================================================

    variants = [
        "CURRENT",
        "EMA_MACD",
    ]

    filters = [
        "ALL",
        "SHORT_ONLY",
        "NO_NEUTRAL",
        "SHORT_NO_NEUTRAL",
    ]

    results = []

    # ========================================================
    # OVERALL SENSITIVITY
    # ========================================================

    print()
    print("=" * 100)
    print(
        "OVERALL COST SENSITIVITY"
    )
    print("=" * 100)

    for variant in variants:

        for filter_name in filters:

            subset = apply_filter(
                trades,
                variant,
                filter_name,
            )

            print()
            print(
                f"{variant} | {filter_name}"
            )

            print(
                f"{'Cost':>8s}"
                f"{'Trades':>10s}"
                f"{'Win %':>10s}"
                f"{'Avg Net':>12s}"
                f"{'PF':>10s}"
                f"{'Compound':>14s}"
                f"{'Max DD':>12s}"
            )

            print(
                "-" * 76
            )

            for cost in COST_LEVELS:

                adjusted = adjust_for_cost(
                    subset,
                    cost,
                )

                metrics = calculate_metrics(
                    adjusted[
                        "adjusted_net_pct"
                    ]
                )

                results.append(
                    {
                        "scope":
                            "OVERALL",

                        "period":
                            "ALL",

                        "variant":
                            variant,

                        "filter":
                            filter_name,

                        "cost_pct":
                            cost,

                        **metrics,
                    }
                )

                print(
                    f"{cost:>7.2f}%"
                    f"{metrics['trades']:>10d}"
                    f"{metrics['win_rate']:>9.2f}%"
                    f"{metrics['avg_net_pct']:>+11.2f}%"
                    f"{metrics['profit_factor']:>9.2f}"
                    f"{metrics['compounded_return_pct']:>+13.2f}%"
                    f"{metrics['max_drawdown_pct']:>+11.2f}%"
                )

    # ========================================================
    # PERIOD-BY-PERIOD
    # ========================================================

    print()
    print("=" * 100)
    print(
        "PERIOD-BY-PERIOD COST SENSITIVITY"
    )
    print("=" * 100)

    for variant in variants:

        for filter_name in filters:

            subset = apply_filter(
                trades,
                variant,
                filter_name,
            )

            for period_name, start, end in PERIODS:

                period_trades = subset[
                    (
                        subset["signal_date"]
                        >= pd.Timestamp(start)
                    )
                    & (
                        subset["signal_date"]
                        <= pd.Timestamp(end)
                    )
                ].copy()

                for cost in COST_LEVELS:

                    adjusted = adjust_for_cost(
                        period_trades,
                        cost,
                    )

                    metrics = calculate_metrics(
                        adjusted[
                            "adjusted_net_pct"
                        ]
                    )

                    results.append(
                        {
                            "scope":
                                "PERIOD",

                            "period":
                                period_name,

                            "variant":
                                variant,

                            "filter":
                                filter_name,

                            "cost_pct":
                                cost,

                            **metrics,
                        }
                    )

    # ========================================================
    # BREAK-EVEN COST
    # ========================================================

    print()
    print("=" * 100)
    print(
        "BREAK-EVEN COST ANALYSIS"
    )
    print("=" * 100)

    break_even_rows = []

    for variant in variants:

        for filter_name in filters:

            subset = apply_filter(
                trades,
                variant,
                filter_name,
            )

            if subset.empty:
                continue

            # Average gross return is independent
            # of the assumed transaction cost.

            gross_return = (
                pd.to_numeric(
                    subset[
                        "net_pnl_pct"
                    ],
                    errors="coerce",
                )
                + BASE_COST_PCT
            )

            gross_return = gross_return[
                np.isfinite(
                    gross_return
                )
            ]

            if len(gross_return) == 0:
                continue

            avg_gross = float(
                gross_return.mean()
            )

            # Approximate break-even cost:
            # average gross return per trade.

            break_even = avg_gross

            break_even_rows.append(
                {
                    "variant":
                        variant,

                    "filter":
                        filter_name,

                    "trades":
                        len(subset),

                    "average_gross_pct":
                        avg_gross,

                    "approx_break_even_cost_pct":
                        break_even,
                }
            )

            print(
                f"{variant:<12s}"
                f"{filter_name:<22s}"
                f"Trades {len(subset):>4d}"
                f" | Avg gross "
                f"{avg_gross:+.2f}%"
                f" | Approx break-even cost "
                f"{break_even:.2f}%"
            )

    # ========================================================
    # ROBUSTNESS AT REALISTIC COSTS
    # ========================================================

    print()
    print("=" * 100)
    print(
        "ROBUSTNESS AT SELECTED COST LEVELS"
    )
    print("=" * 100)

    selected_costs = [
        0.15,
        0.25,
        0.35,
        0.50,
    ]

    for variant in variants:

        print()
        print(
            variant
        )

        for filter_name in filters:

            subset = apply_filter(
                trades,
                variant,
                filter_name,
            )

            print()
            print(
                f"  {filter_name}"
            )

            for cost in selected_costs:

                adjusted = adjust_for_cost(
                    subset,
                    cost,
                )

                metrics = calculate_metrics(
                    adjusted[
                        "adjusted_net_pct"
                    ]
                )

                print(
                    f"    Cost "
                    f"{cost:.2f}%"
                    f" | Avg "
                    f"{metrics['avg_net_pct']:+.2f}%"
                    f" | PF "
                    f"{metrics['profit_factor']:.2f}"
                    f" | Compound "
                    f"{metrics['compounded_return_pct']:+.2f}%"
                    f" | DD "
                    f"{metrics['max_drawdown_pct']:+.2f}%"
                )

    # ========================================================
    # SAVE
    # ========================================================

    result_df = pd.DataFrame(
        results
    )

    break_even_df = pd.DataFrame(
        break_even_rows
    )

    # Add break-even information to
    # a separate section in the same CSV
    # using a normal CSV-compatible table.

    result_df.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    break_even_file = (
        BASE_DIR
        / "output"
        / "historical_oos_break_even_cost.csv"
    )

    break_even_df.to_csv(
        break_even_file,
        index=False,
    )

    # ========================================================
    # FINAL GUIDANCE
    # ========================================================

    print()
    print("=" * 100)
    print(
        "HOW TO INTERPRET THIS TEST"
    )
    print("=" * 100)

    print()
    print(
        "The important comparison is not the 0.15% line alone."
    )

    print(
        "We want the strategy to remain profitable as"
    )

    print(
        "assumed trading costs increase."
    )

    print()
    print(
        "A robust candidate should ideally retain:"
    )

    print(
        "  - positive average net return"
    )

    print(
        "  - PF above 1.0"
    )

    print(
        "  - reasonable drawdown"
    )

    print(
        "  - positive performance across multiple periods"
    )

    print()
    print(
        "If the edge disappears around 0.20%-0.30%,"
    )

    print(
        "the strategy is probably too fragile for production."
    )

    print()
    print(
        "If SHORT_ONLY remains positive at materially higher"
    )

    print(
        "cost assumptions, the short-side edge becomes much"
    )

    print(
        "more credible."
    )

    print()
    print("=" * 100)
    print(
        "FILES SAVED"
    )
    print("=" * 100)

    print(
        OUTPUT_FILE
    )

    print(
        break_even_file
    )

    print()
    print("=" * 100)
    print(
        "COST SENSITIVITY TEST COMPLETE"
    )
    print("=" * 100)


if __name__ == "__main__":
    main()