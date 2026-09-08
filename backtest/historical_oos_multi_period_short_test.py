from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

INPUT_FILE = (
    BASE_DIR
    / "output"
    / "historical_oos_regime_trades.csv"
)

PERIOD_OUTPUT = (
    BASE_DIR
    / "output"
    / "historical_oos_multi_period_short_periods.csv"
)

SUMMARY_OUTPUT = (
    BASE_DIR
    / "output"
    / "historical_oos_multi_period_short_summary.csv"
)

COST_PCT = 0.15


# ============================================================
# FIXED POLICIES
# ============================================================

POLICIES = [
    ("CURRENT", "ALL"),
    ("CURRENT", "SHORT_ONLY"),
    ("EMA_MACD", "ALL"),
    ("EMA_MACD", "SHORT_ONLY"),
]


# ============================================================
# CHRONOLOGICAL PERIODS
# ============================================================

# These are fixed BEFORE looking at individual results.
#
# The exact available history determines how many periods
# contain trades.
#
# We deliberately do NOT optimize the dates.

PERIODS = [
    ("P1", "2023-09-01", "2023-12-31"),
    ("P2", "2024-01-01", "2024-06-30"),
    ("P3", "2024-07-01", "2024-12-31"),
    ("P4", "2025-01-01", "2025-06-30"),
    ("P5", "2025-07-01", "2025-12-31"),
    ("P6", "2026-01-01", "2026-06-30"),
    ("P7", "2026-07-01", "2026-09-05"),
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
            "win_rate_pct": 0.0,
            "avg_net_pct": 0.0,
            "profit_factor": 0.0,
            "compound_pct": 0.0,
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
        "win_rate_pct": float(
            wins.mean() * 100.0
        ),
        "avg_net_pct": float(
            returns.mean()
        ),
        "profit_factor": float(
            profit_factor
        ),
        "compound_pct": float(
            (equity[-1] - 1.0)
            * 100.0
        ),
        "max_drawdown_pct": float(
            drawdown.min() * 100.0
        ),
    }


# ============================================================
# POLICY FILTER
# ============================================================

def apply_policy(
    trades,
    variant,
    policy,
):

    subset = trades[
        trades["variant"] == variant
    ].copy()

    if policy == "ALL":

        return subset

    if policy == "SHORT_ONLY":

        return subset[
            subset["direction"]
            == "SHORT"
        ].copy()

    raise ValueError(
        f"Unknown policy: {policy}"
    )


# ============================================================
# LOAD DATA
# ============================================================

def load_trades():

    if not INPUT_FILE.exists():

        raise FileNotFoundError(
            f"Input file not found: {INPUT_FILE}"
        )

    trades = pd.read_csv(
        INPUT_FILE
    )

    required = [
        "variant",
        "direction",
        "signal_date",
        "net_pnl_pct",
    ]

    missing = [
        column
        for column in required
        if column not in trades.columns
    ]

    if missing:

        raise ValueError(
            "Missing required columns: "
            + ", ".join(missing)
        )

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

    return trades


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 100)
    print(
        "MULTI-PERIOD FIXED SHORT-BIAS ROBUSTNESS TEST"
    )
    print("=" * 100)

    print()
    print(
        "Fixed transaction cost:"
    )

    print(
        f"  {COST_PCT:.2f}%"
    )

    print()
    print(
        "Policies:"
    )

    for variant, policy in POLICIES:

        print(
            f"  {variant:<12s} {policy}"
        )

    print()
    print(
        "IMPORTANT:"
    )

    print(
        "No period is used to optimize the policy."
    )

    print(
        "SHORT_ONLY is a fixed policy across all periods."
    )

    trades = load_trades()

    print()
    print(
        f"Total source trades: {len(trades)}"
    )

    print(
        f"Source date range: "
        f"{trades['signal_date'].min().date()} "
        f"-> "
        f"{trades['signal_date'].max().date()}"
    )

    period_rows = []

    # ========================================================
    # PERIOD-BY-PERIOD
    # ========================================================

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
        ].copy()

        print()
        print("=" * 100)
        print(
            f"{period_name}: "
            f"{start_date} -> {end_date}"
        )
        print("=" * 100)

        print(
            f"Source trades in period: "
            f"{len(period_trades)}"
        )

        for variant, policy in POLICIES:

            subset = apply_policy(
                period_trades,
                variant,
                policy,
            )

            metrics = calculate_metrics(
                subset["net_pnl_pct"]
            )

            row = {
                "period":
                    period_name,

                "start_date":
                    start_date,

                "end_date":
                    end_date,

                "variant":
                    variant,

                "policy":
                    policy,

                **metrics,
            }

            period_rows.append(
                row
            )

            print()
            print(
                f"{variant:<12s}"
                f"{policy:<15s}"
            )

            print(
                f"  Trades: "
                f"{metrics['trades']}"
            )

            print(
                f"  Win rate: "
                f"{metrics['win_rate_pct']:.2f}%"
            )

            print(
                f"  Avg net: "
                f"{metrics['avg_net_pct']:+.2f}%"
            )

            print(
                f"  PF: "
                f"{metrics['profit_factor']:.2f}"
            )

            print(
                f"  Compound: "
                f"{metrics['compound_pct']:+.2f}%"
            )

            print(
                f"  Max DD: "
                f"{metrics['max_drawdown_pct']:+.2f}%"
            )

    period_df = pd.DataFrame(
        period_rows
    )

    # ========================================================
    # COMBINED PERIOD STATISTICS
    # ========================================================

    summary_rows = []

    for variant, policy in POLICIES:

        subset = period_df[
            (
                period_df["variant"]
                == variant
            )
            & (
                period_df["policy"]
                == policy
            )
            & (
                period_df["trades"]
                > 0
            )
        ].copy()

        if subset.empty:
            continue

        all_trades = apply_policy(
            trades,
            variant,
            policy,
        )

        overall = calculate_metrics(
            all_trades["net_pnl_pct"]
        )

        profitable_periods = int(
            (
                subset["avg_net_pct"]
                > 0
            ).sum()
        )

        pf_positive_periods = int(
            (
                subset["profit_factor"]
                > 1.0
            ).sum()
        )

        positive_compound_periods = int(
            (
                subset["compound_pct"]
                > 0
            ).sum()
        )

        summary_rows.append(
            {
                "variant":
                    variant,

                "policy":
                    policy,

                "periods_with_trades":
                    len(subset),

                "profitable_periods":
                    profitable_periods,

                "pf_above_1_periods":
                    pf_positive_periods,

                "positive_compound_periods":
                    positive_compound_periods,

                "profitable_period_rate_pct":
                    (
                        profitable_periods
                        / len(subset)
                        * 100.0
                    ),

                "pf_above_1_rate_pct":
                    (
                        pf_positive_periods
                        / len(subset)
                        * 100.0
                    ),

                "overall_trades":
                    overall["trades"],

                "overall_win_rate_pct":
                    overall["win_rate_pct"],

                "overall_avg_net_pct":
                    overall["avg_net_pct"],

                "overall_profit_factor":
                    overall["profit_factor"],

                "overall_compound_pct":
                    overall["compound_pct"],

                "overall_max_drawdown_pct":
                    overall["max_drawdown_pct"],

                "mean_period_avg_net_pct":
                    subset[
                        "avg_net_pct"
                    ].mean(),

                "median_period_avg_net_pct":
                    subset[
                        "avg_net_pct"
                    ].median(),

                "mean_period_pf":
                    subset[
                        "profit_factor"
                    ].replace(
                        [np.inf, -np.inf],
                        np.nan,
                    ).mean(),

                "median_period_pf":
                    subset[
                        "profit_factor"
                    ].replace(
                        [np.inf, -np.inf],
                        np.nan,
                    ).median(),
            }
        )

    summary_df = pd.DataFrame(
        summary_rows
    )

    # ========================================================
    # PRINT SUMMARY
    # ========================================================

    print()
    print("=" * 100)
    print(
        "MULTI-PERIOD SUMMARY"
    )
    print("=" * 100)

    print()

    for _, row in summary_df.iterrows():

        print(
            f"{row['variant']:<12s}"
            f"{row['policy']:<15s}"
        )

        print(
            f"  Periods with trades: "
            f"{int(row['periods_with_trades'])}"
        )

        print(
            f"  Profitable periods: "
            f"{int(row['profitable_periods'])}"
            f"/"
            f"{int(row['periods_with_trades'])}"
        )

        print(
            f"  PF > 1 periods: "
            f"{int(row['pf_above_1_periods'])}"
            f"/"
            f"{int(row['periods_with_trades'])}"
        )

        print(
            f"  Overall trades: "
            f"{int(row['overall_trades'])}"
        )

        print(
            f"  Overall win rate: "
            f"{row['overall_win_rate_pct']:.2f}%"
        )

        print(
            f"  Overall avg net: "
            f"{row['overall_avg_net_pct']:+.2f}%"
        )

        print(
            f"  Overall PF: "
            f"{row['overall_profit_factor']:.2f}"
        )

        print(
            f"  Overall compound: "
            f"{row['overall_compound_pct']:+.2f}%"
        )

        print(
            f"  Overall max DD: "
            f"{row['overall_max_drawdown_pct']:+.2f}%"
        )

        print(
            f"  Mean period avg: "
            f"{row['mean_period_avg_net_pct']:+.2f}%"
        )

        print(
            f"  Median period avg: "
            f"{row['median_period_avg_net_pct']:+.2f}%"
        )

        print(
            f"  Mean period PF: "
            f"{row['mean_period_pf']:.2f}"
        )

        print(
            f"  Median period PF: "
            f"{row['median_period_pf']:.2f}"
        )

        print()

    # ========================================================
    # DIRECT SHORT-BIAS ADVANTAGE
    # ========================================================

    print("=" * 100)
    print(
        "SHORT-BIAS ADVANTAGE BY PERIOD"
    )
    print("=" * 100)

    advantage_rows = []

    for period_name in period_df[
        "period"
    ].unique():

        current_all = period_df[
            (
                period_df["period"]
                == period_name
            )
            & (
                period_df["variant"]
                == "CURRENT"
            )
            & (
                period_df["policy"]
                == "ALL"
            )
        ]

        current_short = period_df[
            (
                period_df["period"]
                == period_name
            )
            & (
                period_df["variant"]
                == "CURRENT"
            )
            & (
                period_df["policy"]
                == "SHORT_ONLY"
            )
        ]

        ema_all = period_df[
            (
                period_df["period"]
                == period_name
            )
            & (
                period_df["variant"]
                == "EMA_MACD"
            )
            & (
                period_df["policy"]
                == "ALL"
            )
        ]

        ema_short = period_df[
            (
                period_df["period"]
                == period_name
            )
            & (
                period_df["variant"]
                == "EMA_MACD"
            )
            & (
                period_df["policy"]
                == "SHORT_ONLY"
            )
        ]

        if (
            current_all.empty
            or current_short.empty
            or ema_all.empty
            or ema_short.empty
        ):
            continue

        ca = current_all.iloc[0]
        cs = current_short.iloc[0]
        ea = ema_all.iloc[0]
        es = ema_short.iloc[0]

        current_delta = (
            cs["avg_net_pct"]
            - ca["avg_net_pct"]
        )

        ema_delta = (
            es["avg_net_pct"]
            - ea["avg_net_pct"]
        )

        advantage_rows.append(
            {
                "period":
                    period_name,

                "current_all_avg":
                    ca["avg_net_pct"],

                "current_short_avg":
                    cs["avg_net_pct"],

                "current_short_advantage":
                    current_delta,

                "ema_macd_all_avg":
                    ea["avg_net_pct"],

                "ema_macd_short_avg":
                    es["avg_net_pct"],

                "ema_macd_short_advantage":
                    ema_delta,
            }
        )

        print()
        print(
            f"{period_name}"
        )

        print(
            f"  CURRENT short advantage: "
            f"{current_delta:+.2f}%"
        )

        print(
            f"  EMA+MACD short advantage: "
            f"{ema_delta:+.2f}%"
        )

    advantage_df = pd.DataFrame(
        advantage_rows
    )

    # ========================================================
    # SAVE
    # ========================================================

    period_df.to_csv(
        PERIOD_OUTPUT,
        index=False,
    )

    summary_df.to_csv(
        SUMMARY_OUTPUT,
        index=False,
    )

    advantage_file = (
        BASE_DIR
        / "output"
        / "historical_oos_multi_period_short_advantage.csv"
    )

    advantage_df.to_csv(
        advantage_file,
        index=False,
    )

    print()
    print("=" * 100)
    print(
        "FILES SAVED"
    )
    print("=" * 100)

    print()
    print(
        PERIOD_OUTPUT
    )

    print(
        SUMMARY_OUTPUT
    )

    print(
        advantage_file
    )

    print()
    print("=" * 100)
    print(
        "MULTI-PERIOD TEST COMPLETE"
    )
    print("=" * 100)


if __name__ == "__main__":
    main()