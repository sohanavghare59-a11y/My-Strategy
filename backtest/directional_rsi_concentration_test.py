from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

TRADES_FILE = (
    BASE_DIR
    / "output"
    / "directional_rsi_trades.csv"
)

OUTPUT_FILE = (
    BASE_DIR
    / "output"
    / "directional_rsi_concentration_test.csv"
)


# ============================================================
# TEST SETTINGS
# ============================================================

VARIANTS = [
    "CURRENT",
    "LONG_RSI_ZONE",
]

TOP_N_VALUES = [
    0,
    1,
    3,
    5,
    10,
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
            "win_rate_pct": 0.0,
            "avg_net_pct": 0.0,
            "profit_factor": 0.0,
            "compounded_return_pct": 0.0,
            "max_drawdown_pct": 0.0,
            "gross_profit_pct": 0.0,
            "gross_loss_pct": 0.0,
        }

    returns = pd.to_numeric(
        trades["net_pnl_pct"],
        errors="coerce",
    ).dropna()

    if returns.empty:

        return {
            "trades": 0,
            "wins": 0,
            "losses": 0,
            "win_rate_pct": 0.0,
            "avg_net_pct": 0.0,
            "profit_factor": 0.0,
            "compounded_return_pct": 0.0,
            "max_drawdown_pct": 0.0,
            "gross_profit_pct": 0.0,
            "gross_loss_pct": 0.0,
        }

    wins = returns[
        returns > 0
    ]

    losses = returns[
        returns <= 0
    ]

    gross_profit = float(
        wins.sum()
    )

    gross_loss = float(
        abs(losses.sum())
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
        "trades": int(len(returns)),
        "wins": int(
            (returns > 0).sum()
        ),
        "losses": int(
            (returns <= 0).sum()
        ),
        "win_rate_pct": float(
            (returns > 0).mean()
            * 100
        ),
        "avg_net_pct": float(
            returns.mean()
        ),
        "profit_factor": float(
            profit_factor
        ),
        "compounded_return_pct": float(
            compounded
        ),
        "max_drawdown_pct": float(
            drawdown.min()
        ),
        "gross_profit_pct": gross_profit,
        "gross_loss_pct": gross_loss,
    }


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 95)
    print("DIRECTIONAL RSI CONCENTRATION TEST")
    print("=" * 95)

    # --------------------------------------------------------
    # LOAD TRADES
    # --------------------------------------------------------

    if not TRADES_FILE.exists():

        print()
        print(
            f"ERROR: Missing file:"
        )
        print(TRADES_FILE)
        return

    trades = pd.read_csv(
        TRADES_FILE
    )

    required_columns = [
        "variant",
        "symbol",
        "exchange",
        "signal_date",
        "exit_date",
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

        for column in missing:
            print(
                f"  - {column}"
            )

        return

    trades["signal_date"] = pd.to_datetime(
        trades["signal_date"],
        errors="coerce",
    )

    trades["exit_date"] = pd.to_datetime(
        trades["exit_date"],
        errors="coerce",
    )

    trades["net_pnl_pct"] = pd.to_numeric(
        trades["net_pnl_pct"],
        errors="coerce",
    )

    trades = trades.dropna(
        subset=[
            "variant",
            "symbol",
            "net_pnl_pct",
        ]
    )

    trades = trades[
        trades["variant"].isin(
            VARIANTS
        )
    ].copy()

    print()
    print(
        f"Trades loaded : {len(trades)}"
    )

    # ========================================================
    # STOCK CONTRIBUTION
    # ========================================================

    concentration_rows = []

    stock_summary_rows = []

    for variant in VARIANTS:

        variant_trades = trades[
            trades["variant"]
            == variant
        ].copy()

        if variant_trades.empty:
            continue

        # ----------------------------------------------------
        # STOCK-LEVEL TOTAL CONTRIBUTION
        # ----------------------------------------------------

        stock_summary = (
            variant_trades
            .groupby(
                [
                    "exchange",
                    "symbol",
                ],
                as_index=False,
            )
            .agg(
                trades=(
                    "net_pnl_pct",
                    "count",
                ),
                total_net_pct=(
                    "net_pnl_pct",
                    "sum",
                ),
                avg_net_pct=(
                    "net_pnl_pct",
                    "mean",
                ),
                wins=(
                    "net_pnl_pct",
                    lambda x:
                    int(
                        (x > 0).sum()
                    ),
                ),
                losses=(
                    "net_pnl_pct",
                    lambda x:
                    int(
                        (x <= 0).sum()
                    ),
                ),
            )
        )

        stock_summary["win_rate_pct"] = (
            stock_summary["wins"]
            / stock_summary["trades"]
            * 100
        )

        # Rank by total contribution.
        #
        # This answers:
        # "Which stocks are contributing the most
        #  to the strategy's total profit?"
        stock_summary = stock_summary.sort_values(
            "total_net_pct",
            ascending=False,
        ).reset_index(
            drop=True
        )

        stock_summary["rank"] = (
            stock_summary.index + 1
        )

        total_profit = (
            stock_summary[
                "total_net_pct"
            ].sum()
        )

        stock_summary["contribution_pct"] = (
            np.where(
                total_profit != 0,
                stock_summary[
                    "total_net_pct"
                ]
                / total_profit
                * 100,
                0.0,
            )
        )

        for _, row in stock_summary.iterrows():

            stock_summary_rows.append(
                {
                    "variant": variant,
                    "rank": int(
                        row["rank"]
                    ),
                    "exchange":
                        row["exchange"],
                    "symbol":
                        row["symbol"],
                    "trades":
                        int(row["trades"]),
                    "wins":
                        int(row["wins"]),
                    "losses":
                        int(row["losses"]),
                    "win_rate_pct":
                        row["win_rate_pct"],
                    "avg_net_pct":
                        row["avg_net_pct"],
                    "total_net_pct":
                        row["total_net_pct"],
                    "contribution_pct":
                        row["contribution_pct"],
                }
            )

        # ----------------------------------------------------
        # REMOVE TOP CONTRIBUTORS
        # ----------------------------------------------------

        for top_n in TOP_N_VALUES:

            if top_n == 0:

                excluded_stocks = set()

            else:

                top_stocks = stock_summary.head(
                    top_n
                )

                excluded_stocks = set(
                    zip(
                        top_stocks[
                            "exchange"
                        ],
                        top_stocks[
                            "symbol"
                        ],
                    )
                )

            if excluded_stocks:

                mask = [
                    (
                        exchange,
                        symbol,
                    )
                    not in excluded_stocks
                    for exchange, symbol
                    in zip(
                        variant_trades[
                            "exchange"
                        ],
                        variant_trades[
                            "symbol"
                        ],
                    )
                ]

                remaining = variant_trades.loc[
                    mask
                ].copy()

            else:

                remaining = (
                    variant_trades.copy()
                )

            result = calculate_metrics(
                remaining.sort_values(
                    "exit_date"
                )
            )

            concentration_rows.append(
                {
                    "variant": variant,
                    "removed_top_n_stocks":
                        top_n,
                    "remaining_stocks":
                        remaining[
                            [
                                "exchange",
                                "symbol",
                            ]
                        ]
                        .drop_duplicates()
                        .shape[0],
                    **result,
                }
            )

    # ========================================================
    # SAVE CONCENTRATION RESULTS
    # ========================================================

    concentration = pd.DataFrame(
        concentration_rows
    )

    concentration.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    # ========================================================
    # PRINT MAIN TABLE
    # ========================================================

    print()
    print("=" * 95)
    print("PERFORMANCE AFTER REMOVING TOP CONTRIBUTORS")
    print("=" * 95)

    print(
        f"{'Variant':22s}"
        f"{'Removed':>10s}"
        f"{'Trades':>10s}"
        f"{'Win %':>10s}"
        f"{'Avg Net':>12s}"
        f"{'PF':>10s}"
        f"{'Comp.':>12s}"
        f"{'Max DD':>12s}"
    )

    print("-" * 95)

    for _, row in concentration.iterrows():

        pf = row[
            "profit_factor"
        ]

        pf_text = (
            "INF"
            if np.isinf(pf)
            else f"{pf:.2f}"
        )

        print(
            f"{row['variant']:22s}"
            f"{int(row['removed_top_n_stocks']):10d}"
            f"{int(row['trades']):10d}"
            f"{row['win_rate_pct']:9.2f}%"
            f"{row['avg_net_pct']:+11.2f}%"
            f"{pf_text:>10s}"
            f"{row['compounded_return_pct']:+11.2f}%"
            f"{row['max_drawdown_pct']:+11.2f}%"
        )

    # ========================================================
    # TOP STOCK CONTRIBUTORS
    # ========================================================

    stock_results = pd.DataFrame(
        stock_summary_rows
    )

    for variant in VARIANTS:

        subset = stock_results[
            stock_results["variant"]
            == variant
        ].copy()

        if subset.empty:
            continue

        print()
        print("=" * 95)
        print(
            f"TOP 15 PROFIT CONTRIBUTORS: "
            f"{variant}"
        )
        print("=" * 95)

        top = subset.head(15)

        print(
            f"{'Rank':>5s}"
            f"{'Symbol':18s}"
            f"{'Exch':>6s}"
            f"{'Trades':>9s}"
            f"{'Avg Net':>12s}"
            f"{'Total Net':>13s}"
            f"{'Contribution':>15s}"
        )

        print("-" * 95)

        for _, row in top.iterrows():

            print(
                f"{int(row['rank']):5d}"
                f"{str(row['symbol'])[:18]:18s}"
                f"{str(row['exchange']):>6s}"
                f"{int(row['trades']):9d}"
                f"{row['avg_net_pct']:+11.2f}%"
                f"{row['total_net_pct']:+12.2f}%"
                f"{row['contribution_pct']:+14.2f}%"
            )

        # ----------------------------------------------------
        # CONCENTRATION CALCULATIONS
        # ----------------------------------------------------

        total_net = subset[
            "total_net_pct"
        ].sum()

        print()
        print(
            f"Total stock-level net contribution:"
            f" {total_net:+.2f}%"
        )

        for n in [1, 3, 5, 10]:

            top_n_total = subset.head(
                n
            )[
                "total_net_pct"
            ].sum()

            if total_net != 0:

                share = (
                    top_n_total
                    / total_net
                    * 100
                )

            else:

                share = 0.0

            print(
                f"Top {n:2d} stocks contribution:"
                f" {top_n_total:+.2f}%"
                f" ({share:+.2f}% of total)"
            )

        # ----------------------------------------------------
        # LOSING STOCKS
        # ----------------------------------------------------

        losing = subset[
            subset["total_net_pct"] < 0
        ]

        profitable = subset[
            subset["total_net_pct"] > 0
        ]

        print()
        print(
            f"Profitable stocks:"
            f" {len(profitable)}"
        )

        print(
            f"Losing stocks:"
            f" {len(losing)}"
        )

        if not losing.empty:

            loss_total = losing[
                "total_net_pct"
            ].sum()

            print(
                f"Total losing-stock contribution:"
                f" {loss_total:+.2f}%"
            )

    # ========================================================
    # DIRECT COMPARISON
    # ========================================================

    print()
    print("=" * 95)
    print("LONG_RSI_ZONE VS CURRENT")
    print("=" * 95)

    current = concentration[
        (
            concentration["variant"]
            == "CURRENT"
        )
        & (
            concentration[
                "removed_top_n_stocks"
            ]
            == 0
        )
    ]

    long_zone = concentration[
        (
            concentration["variant"]
            == "LONG_RSI_ZONE"
        )
        & (
            concentration[
                "removed_top_n_stocks"
            ]
            == 0
        )
    ]

    if (
        not current.empty
        and not long_zone.empty
    ):

        current_row = current.iloc[0]
        zone_row = long_zone.iloc[0]

        print()
        print(
            "FULL SAMPLE IMPROVEMENT"
        )

        print(
            f"Trade count:"
            f" {int(current_row['trades'])}"
            f" -> "
            f"{int(zone_row['trades'])}"
        )

        print(
            f"Win rate:"
            f" {current_row['win_rate_pct']:.2f}%"
            f" -> "
            f"{zone_row['win_rate_pct']:.2f}%"
        )

        print(
            f"Average net:"
            f" {current_row['avg_net_pct']:+.2f}%"
            f" -> "
            f"{zone_row['avg_net_pct']:+.2f}%"
        )

        print(
            f"Profit factor:"
            f" {current_row['profit_factor']:.2f}"
            f" -> "
            f"{zone_row['profit_factor']:.2f}"
        )

        print(
            f"Compounded:"
            f" {current_row['compounded_return_pct']:+.2f}%"
            f" -> "
            f"{zone_row['compounded_return_pct']:+.2f}%"
        )

        print(
            f"Max drawdown:"
            f" {current_row['max_drawdown_pct']:+.2f}%"
            f" -> "
            f"{zone_row['max_drawdown_pct']:+.2f}%"
        )

    # ========================================================
    # FILE
    # ========================================================

    print()
    print("=" * 95)
    print("FILES SAVED")
    print("=" * 95)

    print(
        f"Concentration:"
        f" {OUTPUT_FILE}"
    )

    print()
    print("=" * 95)
    print("CONCENTRATION TEST COMPLETE")
    print("=" * 95)


if __name__ == "__main__":
    main()