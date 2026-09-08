"""
Signal Diagnostics for the Daily Swing Trading Agent.

Purpose
-------
Research-only analysis of the existing baseline backtest.

This module does NOT change the trading strategy or backtest results.

It analyzes:
    - LONG vs SHORT
    - regime
    - holding period
    - exit behavior
    - return distribution
    - signal quality where available

Primary input:
    output/backtest_trades.csv

Optional regime input:
    output/nifty50_regime.csv

Outputs:
    output/signal_diagnostics_overall.csv
    output/signal_diagnostics_direction.csv
    output/signal_diagnostics_regime.csv
    output/signal_diagnostics_exit.csv
    output/signal_diagnostics_monthly.csv
"""


from __future__ import annotations

from pathlib import Path
import pandas as pd
import numpy as np


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

BASELINE_FILE = BASE_DIR / "output" / "backtest_trades.csv"
REGIME_FILE = BASE_DIR / "output" / "nifty50_regime.csv"

OUTPUT_OVERALL = BASE_DIR / "output" / "signal_diagnostics_overall.csv"
OUTPUT_DIRECTION = BASE_DIR / "output" / "signal_diagnostics_direction.csv"
OUTPUT_REGIME = BASE_DIR / "output" / "signal_diagnostics_regime.csv"
OUTPUT_EXIT = BASE_DIR / "output" / "signal_diagnostics_exit.csv"
OUTPUT_MONTHLY = BASE_DIR / "output" / "signal_diagnostics_monthly.csv"


# ============================================================
# REQUIRED COLUMNS
# ============================================================

REQUIRED_COLUMNS = [
    "signal_date",
    "entry_date",
    "exit_date",
    "direction",
    "entry_price",
    "stop_loss",
    "target",
    "exit_price",
    "raw_exit_price",
    "exit_reason",
    "holding_days",
    "gross_pnl_pct",
    "transaction_cost_pct",
    "net_pnl_pct",
    "symbol",
    "exchange",
]


# ============================================================
# HELPERS
# ============================================================

def safe_float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return np.nan


def calculate_metrics(df: pd.DataFrame) -> dict:
    if df.empty:
        return {
            "trades": 0,
            "wins": 0,
            "losses": 0,
            "win_rate_pct": 0.0,
            "avg_net_pct": 0.0,
            "median_net_pct": 0.0,
            "avg_gross_pct": 0.0,
            "avg_winner_pct": 0.0,
            "avg_loser_pct": 0.0,
            "profit_factor": 0.0,
            "expectancy_pct": 0.0,
            "compounded_return_pct": 0.0,
            "max_drawdown_pct": 0.0,
            "avg_holding_days": 0.0,
            "median_holding_days": 0.0,
            "target_exits": 0,
            "stop_exits": 0,
            "time_exits": 0,
        }

    net = pd.to_numeric(df["net_pnl_pct"], errors="coerce").dropna()
    gross = pd.to_numeric(df["gross_pnl_pct"], errors="coerce").dropna()
    holding = pd.to_numeric(df["holding_days"], errors="coerce").dropna()

    wins = net[net > 0]
    losses = net[net <= 0]

    gross_profit = wins.sum()
    gross_loss = abs(losses.sum())

    if gross_loss > 0:
        profit_factor = gross_profit / gross_loss
    else:
        profit_factor = np.inf if gross_profit > 0 else 0.0

    equity = (1.0 + net / 100.0).cumprod()
    compounded = (equity.iloc[-1] - 1.0) * 100.0

    running_max = equity.cummax()
    drawdown = (equity / running_max - 1.0) * 100.0
    max_drawdown = drawdown.min()

    return {
        "trades": len(net),
        "wins": int((net > 0).sum()),
        "losses": int((net <= 0).sum()),
        "win_rate_pct": (net > 0).mean() * 100.0,
        "avg_net_pct": net.mean(),
        "median_net_pct": net.median(),
        "avg_gross_pct": gross.mean() if not gross.empty else 0.0,
        "avg_winner_pct": wins.mean() if not wins.empty else 0.0,
        "avg_loser_pct": losses.mean() if not losses.empty else 0.0,
        "profit_factor": profit_factor,
        "expectancy_pct": net.mean(),
        "compounded_return_pct": compounded,
        "max_drawdown_pct": max_drawdown,
        "avg_holding_days": holding.mean() if not holding.empty else 0.0,
        "median_holding_days": holding.median() if not holding.empty else 0.0,
        "target_exits": int(
            (df["exit_reason"].astype(str).str.upper() == "TARGET").sum()
        ),
        "stop_exits": int(
            (df["exit_reason"].astype(str).str.upper() == "STOP_LOSS").sum()
        ),
        "time_exits": int(
            (df["exit_reason"].astype(str).str.upper() == "TIME_EXIT").sum()
        ),
    }


def print_metrics(title: str, metrics: dict):
    print()
    print("=" * 78)
    print(title)
    print("=" * 78)

    for key, value in metrics.items():
        if isinstance(value, float):
            print(f"{key:25s}: {value:.4f}")
        else:
            print(f"{key:25s}: {value}")


def load_baseline() -> pd.DataFrame:
    if not BASELINE_FILE.exists():
        raise FileNotFoundError(
            f"Baseline file not found: {BASELINE_FILE}"
        )

    df = pd.read_csv(BASELINE_FILE)

    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]

    if missing:
        raise ValueError(
            "Baseline file is missing required columns:\n"
            + "\n".join(f"  - {c}" for c in missing)
        )

    df["signal_date"] = pd.to_datetime(
        df["signal_date"],
        errors="coerce",
    )

    df["entry_date"] = pd.to_datetime(
        df["entry_date"],
        errors="coerce",
    )

    df["exit_date"] = pd.to_datetime(
        df["exit_date"],
        errors="coerce",
    )

    for column in [
        "entry_price",
        "stop_loss",
        "target",
        "exit_price",
        "raw_exit_price",
        "holding_days",
        "gross_pnl_pct",
        "transaction_cost_pct",
        "net_pnl_pct",
    ]:
        df[column] = pd.to_numeric(
            df[column],
            errors="coerce",
        )

    df["direction"] = (
        df["direction"]
        .astype(str)
        .str.upper()
        .str.strip()
    )

    df["exit_reason"] = (
        df["exit_reason"]
        .astype(str)
        .str.upper()
        .str.strip()
    )

    return df


def attach_regime(df: pd.DataFrame) -> pd.DataFrame:
    if not REGIME_FILE.exists():
        print()
        print("WARNING: NIFTY regime file not found.")
        print(f"Expected: {REGIME_FILE}")
        df["regime"] = "UNKNOWN"
        return df

    regime = pd.read_csv(REGIME_FILE)

    required = ["date", "regime"]

    missing = [c for c in required if c not in regime.columns]

    if missing:
        print()
        print("WARNING: Regime file is missing:")
        for column in missing:
            print(f"  - {column}")

        df["regime"] = "UNKNOWN"
        return df

    regime["date"] = pd.to_datetime(
        regime["date"],
        errors="coerce",
    )

    regime["regime"] = (
        regime["regime"]
        .astype(str)
        .str.upper()
        .str.strip()
    )

    regime = regime[
        ["date", "regime"]
    ].dropna(subset=["date"])

    regime = regime.drop_duplicates(
        subset=["date"],
        keep="last",
    )

    df = df.copy()

    df["signal_date_only"] = (
        df["signal_date"]
        .dt.normalize()
    )

    regime["date_only"] = (
        regime["date"]
        .dt.normalize()
    )

    mapping = regime.set_index(
        "date_only"
    )["regime"]

    df["regime"] = (
        df["signal_date_only"]
        .map(mapping)
        .fillna("UNKNOWN")
    )

    return df


# ============================================================
# DIAGNOSTIC 1 — OVERALL
# ============================================================

def overall_diagnostics(df: pd.DataFrame) -> pd.DataFrame:
    metrics = calculate_metrics(df)

    result = pd.DataFrame(
        [metrics]
    )

    return result


# ============================================================
# DIAGNOSTIC 2 — DIRECTION
# ============================================================

def direction_diagnostics(df: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for direction in ["LONG", "SHORT"]:
        subset = df[
            df["direction"] == direction
        ]

        metrics = calculate_metrics(subset)
        metrics["direction"] = direction

        rows.append(metrics)

    return pd.DataFrame(rows)


# ============================================================
# DIAGNOSTIC 3 — REGIME
# ============================================================

def regime_diagnostics(df: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for regime in [
        "BULL",
        "BEAR",
        "NEUTRAL",
        "UNKNOWN",
    ]:
        subset = df[
            df["regime"] == regime
        ]

        if subset.empty:
            continue

        metrics = calculate_metrics(subset)
        metrics["regime"] = regime

        rows.append(metrics)

    return pd.DataFrame(rows)


# ============================================================
# DIAGNOSTIC 4 — EXIT BEHAVIOR
# ============================================================

def exit_diagnostics(df: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for exit_reason in [
        "TARGET",
        "STOP_LOSS",
        "TIME_EXIT",
    ]:
        subset = df[
            df["exit_reason"] == exit_reason
        ]

        if subset.empty:
            continue

        net = pd.to_numeric(
            subset["net_pnl_pct"],
            errors="coerce",
        ).dropna()

        rows.append(
            {
                "exit_reason": exit_reason,
                "trades": len(subset),
                "avg_net_pct": net.mean(),
                "median_net_pct": net.median(),
                "min_net_pct": net.min(),
                "max_net_pct": net.max(),
                "avg_holding_days": pd.to_numeric(
                    subset["holding_days"],
                    errors="coerce",
                ).mean(),
            }
        )

    return pd.DataFrame(rows)


# ============================================================
# DIAGNOSTIC 5 — MONTHLY
# ============================================================

def monthly_diagnostics(df: pd.DataFrame) -> pd.DataFrame:
    temp = df.copy()

    temp["month"] = (
        temp["signal_date"]
        .dt.to_period("M")
        .astype(str)
    )

    rows = []

    for month, subset in temp.groupby(
        "month",
        sort=True,
    ):
        metrics = calculate_metrics(subset)
        metrics["month"] = month
        rows.append(metrics)

    return pd.DataFrame(rows)


# ============================================================
# CONCENTRATION / STOCK DIAGNOSTIC
# ============================================================

def stock_diagnostics(df: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for symbol, subset in df.groupby(
        "symbol",
        sort=True,
    ):
        metrics = calculate_metrics(subset)

        metrics["symbol"] = symbol

        if "exchange" in subset.columns:
            metrics["exchange"] = (
                subset["exchange"]
                .astype(str)
                .iloc[0]
            )

        rows.append(metrics)

    result = pd.DataFrame(rows)

    if result.empty:
        return result

    return result.sort_values(
        by="avg_net_pct",
        ascending=False,
    )


# ============================================================
# PRINT IMPORTANT FINDINGS
# ============================================================

def print_direction_table(df: pd.DataFrame):
    print()
    print("=" * 78)
    print("DIRECTION ANALYSIS")
    print("=" * 78)

    for direction in ["LONG", "SHORT"]:
        subset = df[
            df["direction"] == direction
        ]

        if subset.empty:
            continue

        metrics = calculate_metrics(subset)

        print(
            f"{direction:8s} "
            f"trades={metrics['trades']:4d} "
            f"win_rate={metrics['win_rate_pct']:6.2f}% "
            f"avg_net={metrics['avg_net_pct']:+.2f}% "
            f"PF={metrics['profit_factor']:.2f}"
        )


def print_regime_table(df: pd.DataFrame):
    print()
    print("=" * 78)
    print("REGIME ANALYSIS")
    print("=" * 78)

    for regime in [
        "BULL",
        "BEAR",
        "NEUTRAL",
        "UNKNOWN",
    ]:
        subset = df[
            df["regime"] == regime
        ]

        if subset.empty:
            continue

        metrics = calculate_metrics(subset)

        print(
            f"{regime:8s} "
            f"trades={metrics['trades']:4d} "
            f"win_rate={metrics['win_rate_pct']:6.2f}% "
            f"avg_net={metrics['avg_net_pct']:+.2f}% "
            f"PF={metrics['profit_factor']:.2f}"
        )


def print_exit_table(df: pd.DataFrame):
    print()
    print("=" * 78)
    print("EXIT ANALYSIS")
    print("=" * 78)

    for exit_reason in [
        "TARGET",
        "STOP_LOSS",
        "TIME_EXIT",
    ]:
        subset = df[
            df["exit_reason"] == exit_reason
        ]

        if subset.empty:
            continue

        net = pd.to_numeric(
            subset["net_pnl_pct"],
            errors="coerce",
        ).dropna()

        holding = pd.to_numeric(
            subset["holding_days"],
            errors="coerce",
        ).dropna()

        print(
            f"{exit_reason:12s} "
            f"trades={len(subset):4d} "
            f"avg_net={net.mean():+.2f}% "
            f"avg_hold={holding.mean():.2f}"
        )


# ============================================================
# MAIN
# ============================================================

def main():
    print("=" * 78)
    print("SIGNAL DIAGNOSTICS")
    print("=" * 78)

    print()
    print(f"Baseline file : {BASELINE_FILE}")
    print(f"Regime file   : {REGIME_FILE}")

    df = load_baseline()

    print()
    print(f"Trades loaded : {len(df)}")

    df = attach_regime(df)

    mapped = (
        df["regime"]
        .isin(["BULL", "BEAR", "NEUTRAL"])
        .sum()
    )

    print(
        f"Regime mapped : {mapped}/{len(df)}"
    )

    # --------------------------------------------------------
    # Overall
    # --------------------------------------------------------

    overall = overall_diagnostics(df)

    print_metrics(
        "OVERALL BASELINE",
        calculate_metrics(df),
    )

    # --------------------------------------------------------
    # Direction
    # --------------------------------------------------------

    print_direction_table(df)

    direction = direction_diagnostics(df)

    # --------------------------------------------------------
    # Regime
    # --------------------------------------------------------

    print_regime_table(df)

    regime = regime_diagnostics(df)

    # --------------------------------------------------------
    # Exit
    # --------------------------------------------------------

    print_exit_table(df)

    exit_stats = exit_diagnostics(df)

    # --------------------------------------------------------
    # Monthly
    # --------------------------------------------------------

    monthly = monthly_diagnostics(df)

    # --------------------------------------------------------
    # Stock-level
    # --------------------------------------------------------

    stock = stock_diagnostics(df)

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    overall.to_csv(
        OUTPUT_OVERALL,
        index=False,
    )

    direction.to_csv(
        OUTPUT_DIRECTION,
        index=False,
    )

    regime.to_csv(
        OUTPUT_REGIME,
        index=False,
    )

    exit_stats.to_csv(
        OUTPUT_EXIT,
        index=False,
    )

    monthly.to_csv(
        OUTPUT_MONTHLY,
        index=False,
    )

    # --------------------------------------------------------
    # Stock ranking output
    # --------------------------------------------------------

    stock_output = (
        BASE_DIR
        / "output"
        / "signal_diagnostics_stocks.csv"
    )

    stock.to_csv(
        stock_output,
        index=False,
    )

    # --------------------------------------------------------
    # Final summary
    # --------------------------------------------------------

    print()
    print("=" * 78)
    print("DIAGNOSTIC FILES SAVED")
    print("=" * 78)

    print(
        f"Overall       : {OUTPUT_OVERALL}"
    )

    print(
        f"Direction     : {OUTPUT_DIRECTION}"
    )

    print(
        f"Regime        : {OUTPUT_REGIME}"
    )

    print(
        f"Exit          : {OUTPUT_EXIT}"
    )

    print(
        f"Monthly       : {OUTPUT_MONTHLY}"
    )

    print(
        f"Stock         : {stock_output}"
    )

    print()
    print("=" * 78)
    print("SIGNAL DIAGNOSTICS COMPLETE")
    print("=" * 78)


if __name__ == "__main__":
    main()