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
    / "directional_rsi_paired_company.csv"
)

SUMMARY_FILE = (
    BASE_DIR
    / "output"
    / "directional_rsi_paired_company_summary.csv"
)


# ============================================================
# VARIANTS
# ============================================================

CURRENT = "CURRENT"
LONG_ZONE = "LONG_RSI_ZONE"


# ============================================================
# METRICS
# ============================================================

def safe_mean(series):

    values = pd.to_numeric(
        series,
        errors="coerce",
    ).dropna()

    if values.empty:
        return 0.0

    return float(
        values.mean()
    )


def safe_median(series):

    values = pd.to_numeric(
        series,
        errors="coerce",
    ).dropna()

    if values.empty:
        return 0.0

    return float(
        values.median()
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 95)
    print("DIRECTIONAL RSI PAIRED COMPANY TEST")
    print("=" * 95)

    if not TRADES_FILE.exists():

        print()
        print(
            "ERROR: Missing trade file:"
        )
        print(TRADES_FILE)
        return

    trades = pd.read_csv(
        TRADES_FILE
    )

    required = [
        "variant",
        "symbol",
        "exchange",
        "net_pnl_pct",
    ]

    missing = [
        column
        for column in required
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
            [
                CURRENT,
                LONG_ZONE,
            ]
        )
    ].copy()

    print()
    print(
        f"Trades loaded : {len(trades)}"
    )

    # ========================================================
    # COMPANY-LEVEL AGGREGATION
    # ========================================================
    #
    # NSE and BSE listings with the same symbol are combined.
    #
    # This avoids treating:
    #
    #     ABC NSE
    #     ABC BSE
    #
    # as two completely independent companies.
    #
    # ========================================================

    company_rows = []

    for variant in [
        CURRENT,
        LONG_ZONE,
    ]:

        subset = trades[
            trades["variant"]
            == variant
        ].copy()

        grouped = (
            subset
            .groupby(
                "symbol",
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
                median_net_pct=(
                    "net_pnl_pct",
                    "median",
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

        grouped["win_rate_pct"] = (
            grouped["wins"]
            / grouped["trades"]
            * 100
        )

        grouped = grouped.rename(
            columns={
                "trades":
                    f"{variant}_trades",
                "total_net_pct":
                    f"{variant}_total_net_pct",
                "avg_net_pct":
                    f"{variant}_avg_net_pct",
                "median_net_pct":
                    f"{variant}_median_net_pct",
                "wins":
                    f"{variant}_wins",
                "losses":
                    f"{variant}_losses",
                "win_rate_pct":
                    f"{variant}_win_rate_pct",
            }
        )

        company_rows.append(
            grouped
        )

    current = company_rows[0]
    long_zone = company_rows[1]

    paired = pd.merge(
        current,
        long_zone,
        on="symbol",
        how="outer",
    )

    numeric_columns = [
        column
        for column in paired.columns
        if column != "symbol"
    ]

    for column in numeric_columns:

        paired[column] = pd.to_numeric(
            paired[column],
            errors="coerce",
        ).fillna(0.0)

    # ========================================================
    # PAIRED IMPROVEMENT
    # ========================================================

    paired[
        "avg_net_delta_pct"
    ] = (
        paired[
            "LONG_RSI_ZONE_avg_net_pct"
        ]
        - paired[
            "CURRENT_avg_net_pct"
        ]
    )

    paired[
        "total_net_delta_pct"
    ] = (
        paired[
            "LONG_RSI_ZONE_total_net_pct"
        ]
        - paired[
            "CURRENT_total_net_pct"
        ]
    )

    paired[
        "win_rate_delta_pct"
    ] = (
        paired[
            "LONG_RSI_ZONE_win_rate_pct"
        ]
        - paired[
            "CURRENT_win_rate_pct"
        ]
    )

    paired[
        "trade_count_delta"
    ] = (
        paired[
            "LONG_RSI_ZONE_trades"
        ]
        - paired[
            "CURRENT_trades"
        ]
    )

    # --------------------------------------------------------
    # Classification
    # --------------------------------------------------------

    def classify(value):

        if value > 0.05:
            return "IMPROVED"

        if value < -0.05:
            return "WORSENED"

        return "SIMILAR"

    paired["classification"] = (
        paired[
            "avg_net_delta_pct"
        ]
        .apply(classify)
    )

    paired = paired.sort_values(
        "avg_net_delta_pct",
        ascending=False,
    ).reset_index(
        drop=True
    )

    paired["rank"] = (
        paired.index + 1
    )

    # ========================================================
    # SAVE
    # ========================================================

    paired.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    # ========================================================
    # SUMMARY
    # ========================================================

    improved = paired[
        paired["classification"]
        == "IMPROVED"
    ]

    worsened = paired[
        paired["classification"]
        == "WORSENED"
    ]

    similar = paired[
        paired["classification"]
        == "SIMILAR"
    ]

    summary_rows = []

    summary_rows.append(
        {
            "metric":
                "companies_total",
            "value":
                len(paired),
        }
    )

    summary_rows.append(
        {
            "metric":
                "companies_improved",
            "value":
                len(improved),
        }
    )

    summary_rows.append(
        {
            "metric":
                "companies_worsened",
            "value":
                len(worsened),
        }
    )

    summary_rows.append(
        {
            "metric":
                "companies_similar",
            "value":
                len(similar),
        }
    )

    summary_rows.append(
        {
            "metric":
                "improved_pct",
            "value":
                (
                    len(improved)
                    / len(paired)
                    * 100
                )
                if len(paired)
                else 0,
        }
    )

    summary_rows.append(
        {
            "metric":
                "worsened_pct",
            "value":
                (
                    len(worsened)
                    / len(paired)
                    * 100
                )
                if len(paired)
                else 0,
        }
    )

    summary_rows.append(
        {
            "metric":
                "avg_company_delta_pct",
            "value":
                safe_mean(
                    paired[
                        "avg_net_delta_pct"
                    ]
                ),
        }
    )

    summary_rows.append(
        {
            "metric":
                "median_company_delta_pct",
            "value":
                safe_median(
                    paired[
                        "avg_net_delta_pct"
                    ]
                ),
        }
    )

    summary_rows.append(
        {
            "metric":
                "avg_win_rate_delta_pct",
            "value":
                safe_mean(
                    paired[
                        "win_rate_delta_pct"
                    ]
                ),
        }
    )

    summary_rows.append(
        {
            "metric":
                "avg_trade_count_delta",
            "value":
                safe_mean(
                    paired[
                        "trade_count_delta"
                    ]
                ),
        }
    )

    # ========================================================
    # REMOVE TOP IMPROVERS
    # ========================================================

    for n in [
        1,
        3,
        5,
        10,
    ]:

        remaining = paired.iloc[
            n:
        ]

        if remaining.empty:
            continue

        summary_rows.append(
            {
                "metric":
                    f"avg_delta_after_top_{n}_removed",
                "value":
                    safe_mean(
                        remaining[
                            "avg_net_delta_pct"
                        ]
                    ),
            }
        )

        summary_rows.append(
            {
                "metric":
                    f"median_delta_after_top_{n}_removed",
                "value":
                    safe_median(
                        remaining[
                            "avg_net_delta_pct"
                        ]
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
    # PRINT OVERALL
    # ========================================================

    print()
    print("=" * 95)
    print("COMPANY-LEVEL PAIRED ROBUSTNESS")
    print("=" * 95)

    print(
        f"Companies compared : "
        f"{len(paired)}"
    )

    print(
        f"Improved           : "
        f"{len(improved)} "
        f"({len(improved) / len(paired) * 100:.2f}%)"
    )

    print(
        f"Worsened           : "
        f"{len(worsened)} "
        f"({len(worsened) / len(paired) * 100:.2f}%)"
    )

    print(
        f"Similar            : "
        f"{len(similar)} "
        f"({len(similar) / len(paired) * 100:.2f}%)"
    )

    print()
    print(
        f"Average company delta : "
        f"{paired['avg_net_delta_pct'].mean():+.3f}%"
    )

    print(
        f"Median company delta  : "
        f"{paired['avg_net_delta_pct'].median():+.3f}%"
    )

    print(
        f"Average win-rate delta: "
        f"{paired['win_rate_delta_pct'].mean():+.2f} pp"
    )

    print(
        f"Average trade delta   : "
        f"{paired['trade_count_delta'].mean():+.2f}"
    )

    # ========================================================
    # TOP IMPROVERS
    # ========================================================

    print()
    print("=" * 95)
    print("TOP 15 COMPANIES IMPROVED")
    print("=" * 95)

    print(
        f"{'Rank':>5s}"
        f"{'Symbol':18s}"
        f"{'Current Avg':>14s}"
        f"{'Zone Avg':>12s}"
        f"{'Delta':>12s}"
        f"{'Trade Δ':>10s}"
    )

    print("-" * 95)

    for _, row in paired.head(15).iterrows():

        print(
            f"{int(row['rank']):5d}"
            f"{str(row['symbol'])[:18]:18s}"
            f"{row['CURRENT_avg_net_pct']:+13.2f}%"
            f"{row['LONG_RSI_ZONE_avg_net_pct']:+11.2f}%"
            f"{row['avg_net_delta_pct']:+11.2f}%"
            f"{int(row['trade_count_delta']):10d}"
        )

    # ========================================================
    # WORST DETERIORATIONS
    # ========================================================

    print()
    print("=" * 95)
    print("TOP 15 COMPANIES WORSENED")
    print("=" * 95)

    print(
        f"{'Symbol':18s}"
        f"{'Current Avg':>14s}"
        f"{'Zone Avg':>12s}"
        f"{'Delta':>12s}"
        f"{'Trade Δ':>10s}"
    )

    print("-" * 80)

    for _, row in paired.tail(15).sort_values(
        "avg_net_delta_pct"
    ).iterrows():

        print(
            f"{str(row['symbol'])[:18]:18s}"
            f"{row['CURRENT_avg_net_pct']:+13.2f}%"
            f"{row['LONG_RSI_ZONE_avg_net_pct']:+11.2f}%"
            f"{row['avg_net_delta_pct']:+11.2f}%"
            f"{int(row['trade_count_delta']):10d}"
        )

    # ========================================================
    # REMOVE TOP IMPROVERS
    # ========================================================

    print()
    print("=" * 95)
    print("ROBUSTNESS AFTER REMOVING TOP IMPROVERS")
    print("=" * 95)

    print(
        f"{'Removed':>10s}"
        f"{'Companies':>12s}"
        f"{'Avg Delta':>14s}"
        f"{'Median Delta':>16s}"
    )

    print("-" * 60)

    for n in [
        0,
        1,
        3,
        5,
        10,
    ]:

        remaining = paired.iloc[
            n:
        ]

        if remaining.empty:
            continue

        print(
            f"{n:10d}"
            f"{len(remaining):12d}"
            f"{remaining['avg_net_delta_pct'].mean():+13.3f}%"
            f"{remaining['avg_net_delta_pct'].median():+15.3f}%"
        )

    # ========================================================
    # TRADE COUNTS
    # ========================================================

    print()
    print("=" * 95)
    print("TRADE COUNT EFFECT")
    print("=" * 95)

    print(
        f"CURRENT total trades:"
        f" {int(paired['CURRENT_trades'].sum())}"
    )

    print(
        f"LONG_RSI_ZONE total trades:"
        f" {int(paired['LONG_RSI_ZONE_trades'].sum())}"
    )

    print(
        f"Trade reduction:"
        f" {int(paired['CURRENT_trades'].sum() - paired['LONG_RSI_ZONE_trades'].sum())}"
    )

    # ========================================================
    # FILES
    # ========================================================

    print()
    print("=" * 95)
    print("FILES SAVED")
    print("=" * 95)

    print(
        f"Company results : "
        f"{OUTPUT_FILE}"
    )

    print(
        f"Summary         : "
        f"{SUMMARY_FILE}"
    )

    print()
    print("=" * 95)
    print("PAIRED COMPANY TEST COMPLETE")
    print("=" * 95)


if __name__ == "__main__":
    main()