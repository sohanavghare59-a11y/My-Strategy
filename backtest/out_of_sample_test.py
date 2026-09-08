"""
Out-of-Sample Strategy Validation
=================================

Research-only chronological validation.

Purpose
-------
Test whether the regime insight discovered in the historical
research survives on unseen future data.

Variants:

1. CURRENT
   EMA + MACD + RSI

2. CURRENT_REGIME
   EMA + MACD + RSI
   BULL + BEAR regimes only
   NEUTRAL excluded

3. EMA_MACD
   EMA + MACD

4. EMA_MACD_REGIME
   EMA + MACD
   BULL + BEAR regimes only
   NEUTRAL excluded

Important
---------
This script does NOT modify the production strategy.

It performs a chronological split:

    TRAIN = first 70%
    TEST  = final 30%

The regime itself is calculated using NIFTY 50 data and is
available only from information known at the signal date.

The purpose is validation, not parameter optimization.
"""

from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

TRADES_FILE = (
    BASE_DIR
    / "output"
    / "controlled_rsi_trades.csv"
)

REGIME_FILE = (
    BASE_DIR
    / "output"
    / "nifty50_regime.csv"
)

OUTPUT_FILE = (
    BASE_DIR
    / "output"
    / "out_of_sample_comparison.csv"
)

DIRECTION_FILE = (
    BASE_DIR
    / "output"
    / "out_of_sample_direction.csv"
)

REGIME_BREAKDOWN_FILE = (
    BASE_DIR
    / "output"
    / "out_of_sample_regime.csv"
)

SPLIT_RATIO = 0.70


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

    gross_loss = abs(
        losers.sum()
    )

    if gross_loss > 0:

        profit_factor = (
            gross_profit
            / gross_loss
        )

    else:

        profit_factor = np.inf

    equity = (
        1 + net / 100
    ).cumprod()

    compounded = (
        equity.iloc[-1]
        - 1
    ) * 100

    running_max = equity.cummax()

    drawdown = (
        equity
        / running_max
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

        "win_rate_pct": (
            (net > 0).mean()
            * 100
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

        "compounded_return_pct": (
            compounded
        ),

        "max_drawdown_pct": (
            drawdown.min()
        ),

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
            f"Missing trades file:\n"
            f"{TRADES_FILE}"
        )

    trades = pd.read_csv(
        TRADES_FILE
    )

    required = [
        "variant",
        "direction",
        "regime",
        "signal_date",
        "net_pnl_pct",
        "gross_pnl_pct",
        "holding_days",
        "exit_reason",
    ]

    missing = [
        column
        for column in required
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

    trades["signal_date"] = pd.to_datetime(
        trades["signal_date"],
        errors="coerce",
    )

    trades = trades.dropna(
        subset=["signal_date"]
    )

    trades = trades.sort_values(
        "signal_date"
    ).reset_index(
        drop=True
    )

    return trades


# ============================================================
# LOAD REGIME
# ============================================================

def load_regime():

    if not REGIME_FILE.exists():

        raise FileNotFoundError(
            f"Missing NIFTY regime file:\n"
            f"{REGIME_FILE}"
        )

    regime = pd.read_csv(
        REGIME_FILE
    )

    date_column = None

    for candidate in [
        "date",
        "Date",
        "signal_date",
    ]:

        if candidate in regime.columns:

            date_column = candidate
            break

    if date_column is None:

        raise ValueError(
            "Could not find date column in "
            f"{REGIME_FILE}"
        )

    regime_column = None

    for candidate in [
        "regime",
        "Regime",
    ]:

        if candidate in regime.columns:

            regime_column = candidate
            break

    if regime_column is None:

        raise ValueError(
            "Could not find regime column in "
            f"{REGIME_FILE}"
        )

    regime = regime[
        [date_column, regime_column]
    ].copy()

    regime.columns = [
        "signal_date",
        "regime",
    ]

    regime["signal_date"] = pd.to_datetime(
        regime["signal_date"],
        errors="coerce",
    )

    regime["regime"] = (
        regime["regime"]
        .astype(str)
        .str.upper()
        .str.strip()
    )

    regime = regime.dropna(
        subset=["signal_date"]
    )

    regime = regime.drop_duplicates(
        subset=["signal_date"],
        keep="last",
    )

    return regime


# ============================================================
# MAP REGIME
# ============================================================

def map_regime(
    trades,
    regime,
):

    result = trades.drop(
        columns=["regime"],
        errors="ignore",
    ).merge(
        regime,
        on="signal_date",
        how="left",
    )

    result["regime"] = (
        result["regime"]
        .fillna("UNKNOWN")
        .astype(str)
        .str.upper()
        .str.strip()
    )

    return result


# ============================================================
# CHRONOLOGICAL SPLIT
# ============================================================

def chronological_split(trades):

    unique_dates = (
        trades["signal_date"]
        .dropna()
        .drop_duplicates()
        .sort_values()
        .reset_index(
            drop=True
        )
    )

    if len(unique_dates) < 10:

        raise ValueError(
            "Not enough unique signal dates "
            "for chronological split."
        )

    split_index = int(
        len(unique_dates)
        * SPLIT_RATIO
    )

    if split_index <= 0:

        split_index = 1

    if split_index >= len(unique_dates):

        split_index = (
            len(unique_dates) - 1
        )

    train_end = (
        unique_dates.iloc[
            split_index - 1
        ]
    )

    test_start = (
        unique_dates.iloc[
            split_index
        ]
    )

    train = trades[
        trades["signal_date"]
        <= train_end
    ].copy()

    test = trades[
        trades["signal_date"]
        >= test_start
    ].copy()

    return (
        train,
        test,
        train_end,
        test_start,
    )


# ============================================================
# VARIANT DEFINITIONS
# ============================================================

VARIANTS = {
    "CURRENT": {
        "source_variant": "CURRENT",
        "regime_filter": False,
    },

    "CURRENT_REGIME": {
        "source_variant": "CURRENT",
        "regime_filter": True,
    },

    "EMA_MACD": {
        "source_variant": "EMA_MACD",
        "regime_filter": False,
    },

    "EMA_MACD_REGIME": {
        "source_variant": "EMA_MACD",
        "regime_filter": True,
    },
}


# ============================================================
# FILTER VARIANT
# ============================================================

def get_variant_trades(
    trades,
    variant_name,
):

    settings = VARIANTS[
        variant_name
    ]

    result = trades[
        trades["variant"]
        == settings["source_variant"]
    ].copy()

    if settings["regime_filter"]:

        result = result[
            result["regime"].isin(
                [
                    "BULL",
                    "BEAR",
                ]
            )
        ].copy()

    return result


# ============================================================
# PRINT METRICS
# ============================================================

def print_metric_row(
    label,
    metrics,
):

    pf = metrics[
        "profit_factor"
    ]

    if np.isinf(pf):

        pf_text = "INF"

    else:

        pf_text = f"{pf:.2f}"

    print(
        f"{label:22s}"
        f"{metrics['trades']:8d}"
        f"{metrics['win_rate_pct']:9.2f}%"
        f"{metrics['avg_net_pct']:+10.2f}%"
        f"{pf_text:>8s}"
        f"{metrics['compounded_return_pct']:+11.2f}%"
        f"{metrics['max_drawdown_pct']:11.2f}%"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 86)
    print("OUT-OF-SAMPLE STRATEGY VALIDATION")
    print("=" * 86)

    print()
    print(
        "This is a chronological research test."
    )

    print(
        "No production strategy is modified."
    )

    # ========================================================
    # LOAD
    # ========================================================

    trades = load_trades()

    regime = load_regime()

    print()
    print(
        f"Trades loaded : {len(trades)}"
    )

    print(
        f"Regime rows   : {len(regime)}"
    )

    # ========================================================
    # REGIME MAPPING
    # ========================================================

    mapped = map_regime(
        trades,
        regime,
    )

    mapped_count = (
        mapped["regime"]
        != "UNKNOWN"
    ).sum()

    print()
    print(
        "REGIME MAPPING"
    )

    print(
        f"Mapped   : {mapped_count}/{len(mapped)}"
    )

    print(
        f"Unknown  : "
        f"{len(mapped) - mapped_count}"
    )

    # ========================================================
    # SPLIT
    # ========================================================

    train_all, test_all, train_end, test_start = (
        chronological_split(mapped)
    )

    print()
    print("=" * 86)
    print("CHRONOLOGICAL SPLIT")
    print("=" * 86)

    print(
        f"Train end  : "
        f"{train_end.date()}"
    )

    print(
        f"Test start : "
        f"{test_start.date()}"
    )

    print(
        f"Train trades: "
        f"{len(train_all)}"
    )

    print(
        f"Test trades : "
        f"{len(test_all)}"
    )

    # ========================================================
    # REGIME DISTRIBUTION
    # ========================================================

    print()
    print("=" * 86)
    print("TRAIN / TEST REGIME DISTRIBUTION")
    print("=" * 86)

    for name, data in [
        ("TRAIN", train_all),
        ("TEST", test_all),
    ]:

        print()
        print(name)

        counts = (
            data["regime"]
            .value_counts()
        )

        for regime_name, count in (
            counts.items()
        ):

            print(
                f"  {regime_name:10s}: "
                f"{int(count):4d}"
            )

    # ========================================================
    # RESULTS
    # ========================================================

    summary_rows = []

    print()
    print("=" * 86)
    print("TRAINING RESULTS")
    print("=" * 86)

    print(
        f"{'Variant':22s}"
        f"{'Trades':>8s}"
        f"{'Win %':>10s}"
        f"{'Avg Net':>11s}"
        f"{'PF':>8s}"
        f"{'Comp.':>12s}"
        f"{'Max DD':>12s}"
    )

    print("-" * 86)

    for variant_name in VARIANTS:

        variant_train = get_variant_trades(
            train_all,
            variant_name,
        )

        metrics = calculate_metrics(
            variant_train
        )

        print_metric_row(
            variant_name,
            metrics,
        )

        row = metrics.copy()

        row["variant"] = variant_name
        row["period"] = "TRAIN"

        summary_rows.append(
            row
        )

    print()
    print("=" * 86)
    print("OUT-OF-SAMPLE TEST RESULTS")
    print("=" * 86)

    print(
        f"{'Variant':22s}"
        f"{'Trades':>8s}"
        f"{'Win %':>10s}"
        f"{'Avg Net':>11s}"
        f"{'PF':>8s}"
        f"{'Comp.':>12s}"
        f"{'Max DD':>12s}"
    )

    print("-" * 86)

    for variant_name in VARIANTS:

        variant_test = get_variant_trades(
            test_all,
            variant_name,
        )

        metrics = calculate_metrics(
            variant_test
        )

        print_metric_row(
            variant_name,
            metrics,
        )

        row = metrics.copy()

        row["variant"] = variant_name
        row["period"] = "TEST"

        summary_rows.append(
            row
        )

    summary = pd.DataFrame(
        summary_rows
    )

    # ========================================================
    # TEST REGIME BREAKDOWN
    # ========================================================

    regime_rows = []

    for variant_name in VARIANTS:

        variant_test = get_variant_trades(
            test_all,
            variant_name,
        )

        for regime_name in [
            "BULL",
            "BEAR",
            "NEUTRAL",
            "UNKNOWN",
        ]:

            subset = variant_test[
                variant_test["regime"]
                == regime_name
            ]

            if subset.empty:
                continue

            metrics = calculate_metrics(
                subset
            )

            metrics["variant"] = (
                variant_name
            )

            metrics["regime"] = (
                regime_name
            )

            regime_rows.append(
                metrics
            )

    regime_summary = pd.DataFrame(
        regime_rows
    )

    # ========================================================
    # TEST DIRECTION
    # ========================================================

    direction_rows = []

    for variant_name in VARIANTS:

        variant_test = get_variant_trades(
            test_all,
            variant_name,
        )

        for direction in [
            "LONG",
            "SHORT",
        ]:

            subset = variant_test[
                variant_test["direction"]
                == direction
            ]

            if subset.empty:
                continue

            metrics = calculate_metrics(
                subset
            )

            metrics["variant"] = (
                variant_name
            )

            metrics["direction"] = (
                direction
            )

            direction_rows.append(
                metrics
            )

    direction_summary = pd.DataFrame(
        direction_rows
    )

    # ========================================================
    # REGIME FILTER IMPACT
    # ========================================================

    print()
    print("=" * 86)
    print("OUT-OF-SAMPLE REGIME FILTER IMPACT")
    print("=" * 86)

    for base_variant, filtered_variant in [
        (
            "CURRENT",
            "CURRENT_REGIME",
        ),
        (
            "EMA_MACD",
            "EMA_MACD_REGIME",
        ),
    ]:

        base = summary[
            (
                summary["variant"]
                == base_variant
            )
            & (
                summary["period"]
                == "TEST"
            )
        ]

        filtered = summary[
            (
                summary["variant"]
                == filtered_variant
            )
            & (
                summary["period"]
                == "TEST"
            )
        ]

        if base.empty or filtered.empty:
            continue

        base = base.iloc[0]
        filtered = filtered.iloc[0]

        print()
        print(
            f"{base_variant}"
        )

        print(
            f"Trades: "
            f"{int(base['trades'])}"
            f" -> "
            f"{int(filtered['trades'])}"
        )

        print(
            f"Win rate change: "
            f"{filtered['win_rate_pct'] - base['win_rate_pct']:+.2f}%"
        )

        print(
            f"Avg net change: "
            f"{filtered['avg_net_pct'] - base['avg_net_pct']:+.2f}%"
        )

        print(
            f"PF change: "
            f"{filtered['profit_factor'] - base['profit_factor']:+.2f}"
        )

        print(
            f"Compounded change: "
            f"{filtered['compounded_return_pct'] - base['compounded_return_pct']:+.2f}%"
        )

        print(
            f"Max DD change: "
            f"{filtered['max_drawdown_pct'] - base['max_drawdown_pct']:+.2f}%"
        )

    # ========================================================
    # TRAIN VS TEST STABILITY
    # ========================================================

    print()
    print("=" * 86)
    print("TRAIN VS TEST STABILITY")
    print("=" * 86)

    for variant_name in VARIANTS:

        train_row = summary[
            (
                summary["variant"]
                == variant_name
            )
            & (
                summary["period"]
                == "TRAIN"
            )
        ]

        test_row = summary[
            (
                summary["variant"]
                == variant_name
            )
            & (
                summary["period"]
                == "TEST"
            )
        ]

        if train_row.empty or test_row.empty:
            continue

        train_row = train_row.iloc[0]
        test_row = test_row.iloc[0]

        print()
        print(
            variant_name
        )

        print(
            f"Train PF : "
            f"{train_row['profit_factor']:.2f}"
        )

        print(
            f"Test PF  : "
            f"{test_row['profit_factor']:.2f}"
        )

        print(
            f"Train avg net : "
            f"{train_row['avg_net_pct']:+.2f}%"
        )

        print(
            f"Test avg net  : "
            f"{test_row['avg_net_pct']:+.2f}%"
        )

        print(
            f"Train win rate: "
            f"{train_row['win_rate_pct']:.2f}%"
        )

        print(
            f"Test win rate : "
            f"{test_row['win_rate_pct']:.2f}%"
        )

    # ========================================================
    # SAVE
    # ========================================================

    summary.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    direction_summary.to_csv(
        DIRECTION_FILE,
        index=False,
    )

    regime_summary.to_csv(
        REGIME_BREAKDOWN_FILE,
        index=False,
    )

    print()
    print("=" * 86)
    print("FILES SAVED")
    print("=" * 86)

    print(
        f"Comparison:"
    )

    print(
        OUTPUT_FILE
    )

    print()
    print(
        f"Direction:"
    )

    print(
        DIRECTION_FILE
    )

    print()
    print(
        f"Regime:"
    )

    print(
        REGIME_BREAKDOWN_FILE
    )

    print()
    print("=" * 86)
    print("OUT-OF-SAMPLE TEST COMPLETE")
    print("=" * 86)


if __name__ == "__main__":
    main()