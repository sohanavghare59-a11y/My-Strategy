"""
Regime Component Test

Research-only analysis of the controlled RSI backtest.

Variants:
    CURRENT  = EMA + MACD + RSI
    EMA_MACD = EMA + MACD

Regime tests:
    ALL
    BULL_BEAR
    BULL_ONLY
    BEAR_ONLY
    NEUTRAL_ONLY

This script does not generate new trades.
It only analyzes the existing controlled RSI trades.
"""

from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

TRADES_FILE = BASE_DIR / "output" / "controlled_rsi_trades.csv"

OUTPUT_FILE = (
    BASE_DIR
    / "output"
    / "regime_component_comparison.csv"
)

DIRECTION_FILE = (
    BASE_DIR
    / "output"
    / "regime_component_direction.csv"
)

EXIT_FILE = (
    BASE_DIR
    / "output"
    / "regime_component_exit.csv"
)


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
            "avg_gross_pct": 0.0,
            "avg_winner_pct": 0.0,
            "avg_loser_pct": 0.0,
            "profit_factor": 0.0,
            "expectancy_pct": 0.0,
            "compounded_return_pct": 0.0,
            "max_drawdown_pct": 0.0,
            "avg_holding_days": 0.0,
            "target_exits": 0,
            "stop_exits": 0,
            "time_exits": 0,
        }

    net = pd.to_numeric(
        trades["net_pnl_pct"],
        errors="coerce",
    ).dropna()

    gross = pd.to_numeric(
        trades["gross_pnl_pct"],
        errors="coerce",
    ).dropna()

    holding = pd.to_numeric(
        trades["holding_days"],
        errors="coerce",
    ).dropna()

    winners = net[net > 0]
    losers = net[net <= 0]

    gross_profit = winners.sum()
    gross_loss = abs(losers.sum())

    if gross_loss > 0:
        profit_factor = gross_profit / gross_loss
    else:
        profit_factor = np.inf

    equity = (1 + net / 100).cumprod()

    compounded_return = (
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

        "win_rate_pct": (
            (net > 0).mean() * 100
        ),

        "avg_net_pct": net.mean(),

        "avg_gross_pct": gross.mean(),

        "avg_winner_pct": (
            winners.mean()
            if len(winners) > 0
            else 0.0
        ),

        "avg_loser_pct": (
            losers.mean()
            if len(losers) > 0
            else 0.0
        ),

        "profit_factor": profit_factor,

        "expectancy_pct": net.mean(),

        "compounded_return_pct": compounded_return,

        "max_drawdown_pct": drawdown.min(),

        "avg_holding_days": (
            holding.mean()
            if len(holding) > 0
            else 0.0
        ),

        "target_exits": int(
            (
                trades["exit_reason"]
                == "TARGET"
            ).sum()
        ),

        "stop_exits": int(
            (
                trades["exit_reason"]
                == "STOP_LOSS"
            ).sum()
        ),

        "time_exits": int(
            (
                trades["exit_reason"]
                == "TIME_EXIT"
            ).sum()
        ),
    }


# ============================================================
# LOAD TRADES
# ============================================================

def load_trades():
    if not TRADES_FILE.exists():
        raise FileNotFoundError(
            f"Missing controlled RSI trades file:\n"
            f"{TRADES_FILE}"
        )

    trades = pd.read_csv(
        TRADES_FILE
    )

    required_columns = [
        "variant",
        "direction",
        "regime",
        "net_pnl_pct",
        "gross_pnl_pct",
        "holding_days",
        "exit_reason",
    ]

    missing = [
        column
        for column in required_columns
        if column not in trades.columns
    ]

    if missing:
        raise ValueError(
            "Trades file is missing columns: "
            + ", ".join(missing)
        )

    trades["variant"] = (
        trades["variant"]
        .astype(str)
        .str.upper()
        .str.strip()
    )

    trades["direction"] = (
        trades["direction"]
        .astype(str)
        .str.upper()
        .str.strip()
    )

    trades["regime"] = (
        trades["regime"]
        .astype(str)
        .str.upper()
        .str.strip()
    )

    return trades


# ============================================================
# REGIME DEFINITIONS
# ============================================================

REGIME_TESTS = {
    "ALL": [
        "BULL",
        "BEAR",
        "NEUTRAL",
        "UNKNOWN",
    ],

    "BULL_BEAR": [
        "BULL",
        "BEAR",
    ],

    "BULL_ONLY": [
        "BULL",
    ],

    "BEAR_ONLY": [
        "BEAR",
    ],

    "NEUTRAL_ONLY": [
        "NEUTRAL",
    ],
}


# ============================================================
# MAIN
# ============================================================

def main():
    print("=" * 78)
    print("REGIME COMPONENT TEST")
    print("=" * 78)

    print()
    print("Using existing controlled RSI trades.")

    print()
    print(f"Trades file:")
    print(f"  {TRADES_FILE}")

    trades = load_trades()

    print()
    print(
        f"Total trades loaded : {len(trades)}"
    )

    # ========================================================
    # REGIME DISTRIBUTION
    # ========================================================

    print()
    print("=" * 78)
    print("INPUT REGIME DISTRIBUTION")
    print("=" * 78)

    regime_counts = (
        trades["regime"]
        .value_counts()
    )

    for regime, count in regime_counts.items():
        print(
            f"{regime:10s}: {int(count):4d}"
        )

    # ========================================================
    # MAIN REGIME TEST
    # ========================================================

    print()
    print("=" * 78)
    print("REGIME TEST RESULTS")
    print("=" * 78)

    print(
        f"{'Variant':12s}"
        f"{'Test':14s}"
        f"{'Trades':>8s}"
        f"{'Win %':>10s}"
        f"{'Avg Net':>11s}"
        f"{'PF':>8s}"
        f"{'Comp.':>12s}"
        f"{'Max DD':>12s}"
    )

    print("-" * 78)

    variants = [
        "CURRENT",
        "EMA_MACD",
    ]

    summary_rows = []

    for variant in variants:
        variant_trades = trades[
            trades["variant"] == variant
        ]

        for test_name, regimes in REGIME_TESTS.items():

            subset = variant_trades[
                variant_trades["regime"].isin(regimes)
            ]

            metrics = calculate_metrics(
                subset
            )

            metrics["variant"] = variant
            metrics["regime_test"] = test_name

            summary_rows.append(
                metrics
            )

            pf = metrics["profit_factor"]

            if np.isinf(pf):
                pf_text = "INF"
            else:
                pf_text = f"{pf:.2f}"

            print(
                f"{variant:12s}"
                f"{test_name:14s}"
                f"{metrics['trades']:8d}"
                f"{metrics['win_rate_pct']:9.2f}%"
                f"{metrics['avg_net_pct']:+10.2f}%"
                f"{pf_text:>8s}"
                f"{metrics['compounded_return_pct']:+11.2f}%"
                f"{metrics['max_drawdown_pct']:11.2f}%"
            )

    summary = pd.DataFrame(
        summary_rows
    )

    # ========================================================
    # DIRECTION ANALYSIS
    # ========================================================

    direction_rows = []

    for variant in variants:

        for test_name, regimes in REGIME_TESTS.items():

            base = trades[
                (
                    trades["variant"] == variant
                )
                & (
                    trades["regime"].isin(regimes)
                )
            ]

            for direction in [
                "LONG",
                "SHORT",
            ]:

                subset = base[
                    base["direction"] == direction
                ]

                if subset.empty:
                    continue

                metrics = calculate_metrics(
                    subset
                )

                metrics["variant"] = variant
                metrics["regime_test"] = test_name
                metrics["direction"] = direction

                direction_rows.append(
                    metrics
                )

    direction = pd.DataFrame(
        direction_rows
    )

    # ========================================================
    # EXIT ANALYSIS
    # ========================================================

    exit_rows = []

    for variant in variants:

        for test_name, regimes in REGIME_TESTS.items():

            subset = trades[
                (
                    trades["variant"] == variant
                )
                & (
                    trades["regime"].isin(regimes)
                )
            ]

            if subset.empty:
                continue

            for exit_reason in [
                "TARGET",
                "STOP_LOSS",
                "TIME_EXIT",
            ]:

                exit_subset = subset[
                    subset["exit_reason"]
                    == exit_reason
                ]

                if exit_subset.empty:
                    continue

                metrics = calculate_metrics(
                    exit_subset
                )

                metrics["variant"] = variant
                metrics["regime_test"] = test_name
                metrics["exit_reason"] = exit_reason

                exit_rows.append(
                    metrics
                )

    exit_summary = pd.DataFrame(
        exit_rows
    )

    # ========================================================
    # SAVE RESULTS
    # ========================================================

    summary.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    direction.to_csv(
        DIRECTION_FILE,
        index=False,
    )

    exit_summary.to_csv(
        EXIT_FILE,
        index=False,
    )

    # ========================================================
    # NEUTRAL REMOVAL IMPACT
    # ========================================================

    print()
    print("=" * 78)
    print("NEUTRAL REGIME REMOVAL IMPACT")
    print("=" * 78)

    for variant in variants:

        all_rows = summary[
            (
                summary["variant"] == variant
            )
            & (
                summary["regime_test"] == "ALL"
            )
        ]

        filtered_rows = summary[
            (
                summary["variant"] == variant
            )
            & (
                summary["regime_test"] == "BULL_BEAR"
            )
        ]

        if all_rows.empty or filtered_rows.empty:
            continue

        all_row = all_rows.iloc[0]
        filtered_row = filtered_rows.iloc[0]

        print()
        print(variant)

        print(
            f"Trades removed : "
            f"{int(all_row['trades'] - filtered_row['trades']):+d}"
        )

        print(
            f"Win rate change : "
            f"{filtered_row['win_rate_pct'] - all_row['win_rate_pct']:+.2f}%"
        )

        print(
            f"Avg net change : "
            f"{filtered_row['avg_net_pct'] - all_row['avg_net_pct']:+.2f}%"
        )

        print(
            f"PF change : "
            f"{filtered_row['profit_factor'] - all_row['profit_factor']:+.2f}"
        )

        print(
            f"Compounded change : "
            f"{filtered_row['compounded_return_pct'] - all_row['compounded_return_pct']:+.2f}%"
        )

        print(
            f"Max DD change : "
            f"{filtered_row['max_drawdown_pct'] - all_row['max_drawdown_pct']:+.2f}%"
        )

    # ========================================================
    # BEST TEST
    # ========================================================

    print()
    print("=" * 78)
    print("BEST REGIME TEST BY PROFIT FACTOR")
    print("=" * 78)

    for variant in variants:

        subset = summary[
            summary["variant"] == variant
        ].copy()

        # Ignore tiny samples.
        subset = subset[
            subset["trades"] >= 20
        ]

        if subset.empty:
            continue

        best = subset.sort_values(
            "profit_factor",
            ascending=False,
        ).iloc[0]

        print()
        print(variant)

        print(
            f"Best test    : "
            f"{best['regime_test']}"
        )

        print(
            f"Trades       : "
            f"{int(best['trades'])}"
        )

        print(
            f"Win rate     : "
            f"{best['win_rate_pct']:.2f}%"
        )

        print(
            f"Avg net      : "
            f"{best['avg_net_pct']:+.2f}%"
        )

        print(
            f"Profit factor: "
            f"{best['profit_factor']:.2f}"
        )

        print(
            f"Compounded   : "
            f"{best['compounded_return_pct']:+.2f}%"
        )

        print(
            f"Max DD       : "
            f"{best['max_drawdown_pct']:.2f}%"
        )

    # ========================================================
    # FILES
    # ========================================================

    print()
    print("=" * 78)
    print("FILES SAVED")
    print("=" * 78)

    print(
        f"Comparison : {OUTPUT_FILE}"
    )

    print(
        f"Direction  : {DIRECTION_FILE}"
    )

    print(
        f"Exit       : {EXIT_FILE}"
    )

    print()
    print("=" * 78)
    print("REGIME COMPONENT TEST COMPLETE")
    print("=" * 78)


if __name__ == "__main__":
    main()