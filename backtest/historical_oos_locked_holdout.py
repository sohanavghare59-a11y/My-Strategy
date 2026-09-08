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
    / "historical_oos_locked_holdout.csv"
)


# ============================================================
# LOCKED HOLDOUT DESIGN
# ============================================================

# P1-P3 are treated as the research period.
#
# P4 is the FINAL untouched holdout.
#
# We are NOT selecting a configuration using P4.

RESEARCH_START = "2025-09-04"
RESEARCH_END = "2026-06-30"

HOLDOUT_START = "2026-07-01"
HOLDOUT_END = "2026-09-05"


# ============================================================
# COST LEVELS
# ============================================================

COST_LEVELS = [
    0.15,
    0.20,
    0.25,
    0.30,
    0.35,
    0.40,
    0.50,
]


BASE_COST_PCT = 0.15


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
# FILTER
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
# COST ADJUSTMENT
# ============================================================

def adjust_cost(
    trades,
    cost_pct,
):

    result = trades.copy()

    result[
        "gross_reconstructed_pct"
    ] = (
        result["net_pnl_pct"]
        + BASE_COST_PCT
    )

    result[
        "adjusted_net_pct"
    ] = (
        result[
            "gross_reconstructed_pct"
        ]
        - cost_pct
    )

    return result


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 100)
    print(
        "LOCKED FINAL HOLDOUT TEST"
    )
    print("=" * 100)

    print()
    print(
        "Research period:"
    )

    print(
        f"  {RESEARCH_START} -> {RESEARCH_END}"
    )

    print()
    print(
        "FINAL HOLDOUT:"
    )

    print(
        f"  {HOLDOUT_START} -> {HOLDOUT_END}"
    )

    print()
    print(
        "IMPORTANT:"
    )

    print(
        "P4 is evaluated separately and is NOT used"
    )

    print(
        "to select the strategy."
    )

    if not INPUT_FILE.exists():

        print()
        print(
            "ERROR: Input file not found:"
        )

        print(
            INPUT_FILE
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
    # PERIOD SPLIT
    # ========================================================

    research = trades[
        (
            trades["signal_date"]
            >= pd.Timestamp(
                RESEARCH_START
            )
        )
        & (
            trades["signal_date"]
            <= pd.Timestamp(
                RESEARCH_END
            )
        )
    ].copy()

    holdout = trades[
        (
            trades["signal_date"]
            >= pd.Timestamp(
                HOLDOUT_START
            )
        )
        & (
            trades["signal_date"]
            <= pd.Timestamp(
                HOLDOUT_END
            )
        )
    ].copy()

    print()
    print(
        f"Research trades available: {len(research)}"
    )

    print(
        f"Holdout trades available: {len(holdout)}"
    )

    # ========================================================
    # CANDIDATES
    # ========================================================

    candidates = [
        (
            "CURRENT",
            "ALL",
        ),
        (
            "CURRENT",
            "SHORT_ONLY",
        ),
        (
            "CURRENT",
            "SHORT_NO_NEUTRAL",
        ),
        (
            "EMA_MACD",
            "ALL",
        ),
        (
            "EMA_MACD",
            "SHORT_ONLY",
        ),
        (
            "EMA_MACD",
            "SHORT_NO_NEUTRAL",
        ),
    ]

    results = []

    # ========================================================
    # RESEARCH PERIOD
    # ========================================================

    print()
    print("=" * 100)
    print(
        "RESEARCH PERIOD: P1-P3"
    )
    print("=" * 100)

    research_summary = []

    for variant, filter_name in candidates:

        subset = apply_filter(
            research,
            variant,
            filter_name,
        )

        metrics = calculate_metrics(
            subset["net_pnl_pct"]
        )

        research_summary.append(
            {
                "variant":
                    variant,

                "filter":
                    filter_name,

                "period":
                    "RESEARCH_P1_P3",

                **metrics,
            }
        )

        print()
        print(
            f"{variant:<12s}"
            f"{filter_name:<22s}"
        )

        print(
            f"  Trades: "
            f"{metrics['trades']}"
        )

        print(
            f"  Win rate: "
            f"{metrics['win_rate']:.2f}%"
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
            f"{metrics['compounded_return_pct']:+.2f}%"
        )

        print(
            f"  Max DD: "
            f"{metrics['max_drawdown_pct']:+.2f}%"
        )

    # ========================================================
    # FINAL HOLDOUT
    # ========================================================

    print()
    print("=" * 100)
    print(
        "FINAL HOLDOUT: P4"
    )
    print("=" * 100)

    holdout_summary = []

    for variant, filter_name in candidates:

        subset = apply_filter(
            holdout,
            variant,
            filter_name,
        )

        metrics = calculate_metrics(
            subset["net_pnl_pct"]
        )

        holdout_summary.append(
            {
                "variant":
                    variant,

                "filter":
                    filter_name,

                "period":
                    "FINAL_HOLDOUT_P4",

                **metrics,
            }
        )

        print()
        print(
            f"{variant:<12s}"
            f"{filter_name:<22s}"
        )

        print(
            f"  Trades: "
            f"{metrics['trades']}"
        )

        print(
            f"  Win rate: "
            f"{metrics['win_rate']:.2f}%"
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
            f"{metrics['compounded_return_pct']:+.2f}%"
        )

        print(
            f"  Max DD: "
            f"{metrics['max_drawdown_pct']:+.2f}%"
        )

    # ========================================================
    # HOLDOUT COST SENSITIVITY
    # ========================================================

    print()
    print("=" * 100)
    print(
        "FINAL HOLDOUT COST SENSITIVITY"
    )
    print("=" * 100)

    for variant, filter_name in candidates:

        subset = apply_filter(
            holdout,
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

            adjusted = adjust_cost(
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
                    "variant":
                        variant,

                    "filter":
                        filter_name,

                    "period":
                        "FINAL_HOLDOUT_P4",

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
    # RESEARCH -> HOLDOUT COMPARISON
    # ========================================================

    comparison_rows = []

    research_df = pd.DataFrame(
        research_summary
    )

    holdout_df = pd.DataFrame(
        holdout_summary
    )

    for variant, filter_name in candidates:

        r = research_df[
            (
                research_df["variant"]
                == variant
            )
            & (
                research_df["filter"]
                == filter_name
            )
        ]

        h = holdout_df[
            (
                holdout_df["variant"]
                == variant
            )
            & (
                holdout_df["filter"]
                == filter_name
            )
        ]

        if r.empty or h.empty:
            continue

        r = r.iloc[0]
        h = h.iloc[0]

        comparison_rows.append(
            {
                "variant":
                    variant,

                "filter":
                    filter_name,

                "research_trades":
                    r["trades"],

                "research_win_rate":
                    r["win_rate"],

                "research_avg_net_pct":
                    r["avg_net_pct"],

                "research_profit_factor":
                    r["profit_factor"],

                "research_compounded_pct":
                    r["compounded_return_pct"],

                "holdout_trades":
                    h["trades"],

                "holdout_win_rate":
                    h["win_rate"],

                "holdout_avg_net_pct":
                    h["avg_net_pct"],

                "holdout_profit_factor":
                    h["profit_factor"],

                "holdout_compounded_pct":
                    h["compounded_return_pct"],

                "holdout_max_drawdown_pct":
                    h["max_drawdown_pct"],

                "avg_net_change":
                    h["avg_net_pct"]
                    - r["avg_net_pct"],

                "pf_change":
                    h["profit_factor"]
                    - r["profit_factor"],
            }
        )

    comparison_df = pd.DataFrame(
        comparison_rows
    )

    # ========================================================
    # SAVE
    # ========================================================

    output_frames = []

    for row in research_summary:

        output_frames.append(
            {
                "section":
                    "RESEARCH",

                **row,

                "cost_pct":
                    BASE_COST_PCT,
            }
        )

    for row in holdout_summary:

        output_frames.append(
            {
                "section":
                    "HOLDOUT",

                **row,

                "cost_pct":
                    BASE_COST_PCT,
            }
        )

    cost_df = pd.DataFrame(
        results
    )

    summary_file = (
        BASE_DIR
        / "output"
        / "historical_oos_locked_holdout_summary.csv"
    )

    comparison_file = (
        BASE_DIR
        / "output"
        / "historical_oos_locked_holdout_comparison.csv"
    )

    cost_df.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    pd.DataFrame(
        output_frames
    ).to_csv(
        summary_file,
        index=False,
    )

    comparison_df.to_csv(
        comparison_file,
        index=False,
    )

    # ========================================================
    # FINAL VERDICT FRAMEWORK
    # ========================================================

    print()
    print("=" * 100)
    print(
        "FINAL HOLDOUT INTERPRETATION"
    )
    print("=" * 100)

    print()
    print(
        "The strongest candidate is NOT chosen from P4."
    )

    print(
        "We judge whether a configuration that looked"
    )

    print(
        "reasonable during P1-P3 continued to work"
    )

    print(
        "during the untouched P4 period."
    )

    print()
    print(
        "Important:"
    )

    print(
        "A single successful P4 does not prove the strategy."
    )

    print(
        "A failed P4 does not automatically disprove it."
    )

    print(
        "We are looking for consistency and degradation"
    )

    print(
        "that remains economically acceptable."
    )

    print()
    print("=" * 100)
    print(
        "FILES SAVED"
    )
    print("=" * 100)

    print()
    print(
        OUTPUT_FILE
    )

    print(
        summary_file
    )

    print(
        comparison_file
    )

    print()
    print("=" * 100)
    print(
        "LOCKED HOLDOUT TEST COMPLETE"
    )
    print("=" * 100)


if __name__ == "__main__":
    main()