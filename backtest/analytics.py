"""
Backtest Analytics
==================

Compares the original unfiltered baseline backtest against the
regime-filtered backtest.

The original baseline trades do not contain a regime column, so
this module reconstructs the NIFTY 50 regime using each trade's
signal_date.

This keeps the comparison consistent with the regime-filtered
backtest methodology.
"""

from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[1]
OUTPUT_DIR = BASE_DIR / "output"

BASELINE_FILE = OUTPUT_DIR / "backtest_trades.csv"
FILTERED_FILE = OUTPUT_DIR / "backtest_regime_filtered_trades.csv"
REGIME_FILE = OUTPUT_DIR / "nifty50_regime.csv"

COMPARISON_FILE = OUTPUT_DIR / "backtest_regime_comparison.csv"
DIAGNOSTICS_FILE = OUTPUT_DIR / "backtest_regime_diagnostics.csv"
BULL_BEAR_FILE = OUTPUT_DIR / "backtest_bull_bear_comparison.csv"


# ============================================================
# DISPLAY
# ============================================================

LINE = "=" * 78
SUBLINE = "-" * 78


# ============================================================
# LOAD CSV
# ============================================================

def load_csv(path: Path, name: str) -> pd.DataFrame:

    if not path.exists():
        raise FileNotFoundError(
            f"{name} file not found:\n{path}"
        )

    df = pd.read_csv(path)

    if df.empty:
        raise ValueError(
            f"{name} file is empty:\n{path}"
        )

    return df


# ============================================================
# FIND COLUMN
# ============================================================

def find_column(df: pd.DataFrame, candidates):

    normalized = {
        str(column).strip().lower(): column
        for column in df.columns
    }

    for candidate in candidates:

        key = candidate.strip().lower()

        if key in normalized:
            return normalized[key]

    return None


# ============================================================
# NORMALIZE REGIME
# ============================================================

def normalize_regime(value):

    if pd.isna(value):
        return "UNKNOWN"

    text = str(value).strip().upper()

    if text in {"BULL", "BULLISH"}:
        return "BULL"

    if text in {"BEAR", "BEARISH"}:
        return "BEAR"

    if text in {
        "NEUTRAL",
        "SIDEWAYS",
        "FLAT",
    }:
        return "NEUTRAL"

    return "UNKNOWN"


# ============================================================
# ATTACH REGIME TO BASELINE TRADES
# ============================================================

def attach_baseline_regime(
    baseline: pd.DataFrame,
    regime_df: pd.DataFrame,
) -> pd.DataFrame:

    baseline = baseline.copy()
    regime_df = regime_df.copy()

    signal_col = find_column(
        baseline,
        [
            "signal_date",
            "date",
        ],
    )

    if signal_col is None:
        raise ValueError(
            "Baseline backtest does not contain signal_date/date."
        )

    regime_date_col = find_column(
        regime_df,
        [
            "date",
        ],
    )

    if regime_date_col is None:
        raise ValueError(
            "NIFTY regime file does not contain a date column."
        )

    regime_col = find_column(
        regime_df,
        [
            "regime",
            "market_regime",
            "regime_label",
            "market_regime_label",
        ],
    )

    if regime_col is None:
        raise ValueError(
            "NIFTY regime file does not contain a regime column."
        )

    baseline["_regime_match_date"] = pd.to_datetime(
        baseline[signal_col],
        errors="coerce",
    ).dt.normalize()

    regime_df["_regime_match_date"] = pd.to_datetime(
        regime_df[regime_date_col],
        errors="coerce",
    ).dt.normalize()

    regime_map = (
        regime_df[
            [
                "_regime_match_date",
                regime_col,
            ]
        ]
        .dropna(
            subset=["_regime_match_date"]
        )
        .drop_duplicates(
            subset=["_regime_match_date"],
            keep="last",
        )
        .copy()
    )

    regime_map["mapped_regime"] = (
        regime_map[regime_col]
        .apply(normalize_regime)
    )

    regime_map = regime_map[
        [
            "_regime_match_date",
            "mapped_regime",
        ]
    ]

    baseline = baseline.merge(
        regime_map,
        how="left",
        on="_regime_match_date",
    )

    baseline["regime"] = (
        baseline["mapped_regime"]
        .fillna("UNKNOWN")
    )

    baseline.drop(
        columns=[
            "_regime_match_date",
            "mapped_regime",
        ],
        inplace=True,
        errors="ignore",
    )

    return baseline


# ============================================================
# NORMALIZE FILTERED REGIME
# ============================================================

def normalize_filtered_regime(
    filtered: pd.DataFrame,
) -> pd.DataFrame:

    filtered = filtered.copy()

    regime_col = find_column(
        filtered,
        [
            "market_regime",
            "regime",
            "market_regime_label",
            "regime_label",
        ],
    )

    if regime_col is None:
        raise ValueError(
            "Filtered backtest does not contain a regime column."
        )

    filtered["regime"] = (
        filtered[regime_col]
        .apply(normalize_regime)
    )

    return filtered


# ============================================================
# RETURN COLUMNS
# ============================================================

def get_return_column(df: pd.DataFrame):

    candidates = [
        "net_pnl_pct",
        "pnl_pct",
        "net_return_pct",
        "return_pct",
        "return",
    ]

    column = find_column(
        df,
        candidates,
    )

    if column is None:
        raise ValueError(
            "Could not find a net return column."
        )

    return column


def get_gross_return_column(
    df: pd.DataFrame,
):

    candidates = [
        "gross_pnl_pct",
        "gross_return_pct",
        "gross_return",
    ]

    return find_column(
        df,
        candidates,
    )


# ============================================================
# METRICS
# ============================================================

def calculate_metrics(
    df: pd.DataFrame,
) -> dict:

    if df.empty:

        return {
            "trades": 0,
            "wins": 0,
            "losses": 0,
            "win_rate": 0.0,
            "avg_net": 0.0,
            "avg_gross": 0.0,
            "average_winner": 0.0,
            "average_loser": 0.0,
            "profit_factor": 0.0,
            "expectancy": 0.0,
            "compounded": 0.0,
            "max_drawdown": 0.0,
            "avg_holding": 0.0,
            "median_holding": 0.0,
            "target_exits": 0,
            "stop_exits": 0,
            "time_exits": 0,
        }

    df = df.copy()

    return_col = get_return_column(df)

    returns = pd.to_numeric(
        df[return_col],
        errors="coerce",
    ).dropna()

    if returns.empty:

        return {
            "trades": 0,
            "wins": 0,
            "losses": 0,
            "win_rate": 0.0,
            "avg_net": 0.0,
            "avg_gross": 0.0,
            "average_winner": 0.0,
            "average_loser": 0.0,
            "profit_factor": 0.0,
            "expectancy": 0.0,
            "compounded": 0.0,
            "max_drawdown": 0.0,
            "avg_holding": 0.0,
            "median_holding": 0.0,
            "target_exits": 0,
            "stop_exits": 0,
            "time_exits": 0,
        }

    wins = returns[
        returns > 0
    ]

    losses = returns[
        returns < 0
    ]

    gross_profit = wins.sum()

    gross_loss = abs(
        losses.sum()
    )

    if gross_loss > 0:

        profit_factor = (
            gross_profit / gross_loss
        )

    elif gross_profit > 0:

        profit_factor = np.inf

    else:

        profit_factor = 0.0

    compounded = (
        np.prod(
            1.0 + returns / 100.0
        ) - 1.0
    ) * 100.0

    equity = (
        1.0 + returns / 100.0
    ).cumprod()

    running_peak = equity.cummax()

    drawdown = (
        equity / running_peak - 1.0
    ) * 100.0

    max_drawdown = drawdown.min()

    # Gross return

    gross_col = get_gross_return_column(
        df
    )

    if gross_col:

        gross_returns = pd.to_numeric(
            df[gross_col],
            errors="coerce",
        ).dropna()

        if not gross_returns.empty:

            avg_gross = (
                gross_returns.mean()
            )

        else:

            avg_gross = 0.0

    else:

        avg_gross = 0.0

    # Holding period

    holding_col = find_column(
        df,
        [
            "holding_days",
            "holding_sessions",
            "holding_period",
        ],
    )

    if holding_col:

        holding = pd.to_numeric(
            df[holding_col],
            errors="coerce",
        ).dropna()

        if not holding.empty:

            avg_holding = (
                holding.mean()
            )

            median_holding = (
                holding.median()
            )

        else:

            avg_holding = 0.0
            median_holding = 0.0

    else:

        avg_holding = 0.0
        median_holding = 0.0

    # Exit reason

    exit_col = find_column(
        df,
        [
            "exit_reason",
            "exit_type",
        ],
    )

    if exit_col:

        exits = (
            df[exit_col]
            .astype(str)
            .str.upper()
        )

        target_exits = int(
            exits.str.contains(
                "TARGET",
                na=False,
            ).sum()
        )

        stop_exits = int(
            exits.str.contains(
                "STOP",
                na=False,
            ).sum()
        )

        time_exits = int(
            exits.str.contains(
                "TIME",
                na=False,
            ).sum()
        )

    else:

        target_exits = 0
        stop_exits = 0
        time_exits = 0

    return {
        "trades": int(len(returns)),
        "wins": int(
            (returns > 0).sum()
        ),
        "losses": int(
            (returns < 0).sum()
        ),
        "win_rate": (
            (returns > 0).mean()
            * 100.0
        ),
        "avg_net": returns.mean(),
        "avg_gross": avg_gross,
        "average_winner": (
            wins.mean()
            if not wins.empty
            else 0.0
        ),
        "average_loser": (
            losses.mean()
            if not losses.empty
            else 0.0
        ),
        "profit_factor": profit_factor,
        "expectancy": returns.mean(),
        "compounded": compounded,
        "max_drawdown": max_drawdown,
        "avg_holding": avg_holding,
        "median_holding": median_holding,
        "target_exits": target_exits,
        "stop_exits": stop_exits,
        "time_exits": time_exits,
    }


# ============================================================
# PRINT GENERAL COMPARISON
# ============================================================

def print_comparison(
    title,
    baseline_metrics,
    filtered_metrics,
):

    print()
    print(LINE)
    print(title)
    print(LINE)

    print(
        f"{'Metric':35s}"
        f"{'BASELINE':>20s}"
        f"{'FILTERED':>20s}"
    )

    print(SUBLINE)

    rows = [
        ("Trades", "trades", 0),
        ("Win rate %", "win_rate", 2),
        ("Avg net return %", "avg_net", 2),
        ("Avg gross return %", "avg_gross", 2),
        ("Average winner %", "average_winner", 2),
        ("Average loser %", "average_loser", 2),
        ("Profit factor", "profit_factor", 2),
        ("Expectancy %", "expectancy", 2),
        ("Compounded return %", "compounded", 2),
        ("Max drawdown %", "max_drawdown", 2),
        ("Avg holding days", "avg_holding", 2),
        ("Median holding days", "median_holding", 2),
        ("Target exits", "target_exits", 0),
        ("Stop exits", "stop_exits", 0),
        ("Time exits", "time_exits", 0),
    ]

    for label, key, decimals in rows:

        b = baseline_metrics[key]
        f = filtered_metrics[key]

        if key == "profit_factor":

            b_text = (
                "INF"
                if np.isinf(b)
                else f"{b:.2f}"
            )

            f_text = (
                "INF"
                if np.isinf(f)
                else f"{f:.2f}"
            )

        elif decimals == 0:

            b_text = f"{b:.0f}"
            f_text = f"{f:.0f}"

        else:

            b_text = f"{b:.{decimals}f}"
            f_text = f"{f:.{decimals}f}"

        print(
            f"{label:35s}"
            f"{b_text:>20s}"
            f"{f_text:>20s}"
        )


# ============================================================
# REGIME BREAKDOWN
# ============================================================

def regime_breakdown(
    df: pd.DataFrame,
):

    results = []

    for regime in [
        "BULL",
        "BEAR",
        "NEUTRAL",
        "UNKNOWN",
    ]:

        subset = df[
            df["regime"] == regime
        ]

        metrics = calculate_metrics(
            subset
        )

        results.append({
            "regime": regime,
            **metrics,
        })

    return pd.DataFrame(
        results
    )


def print_regime_breakdown(
    title,
    df: pd.DataFrame,
):

    print()
    print(LINE)
    print(title)
    print(LINE)

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

        metrics = calculate_metrics(
            subset
        )

        pf = metrics[
            "profit_factor"
        ]

        if np.isinf(pf):

            pf_text = "INF"

        else:

            pf_text = f"{pf:.2f}"

        print(
            f"{regime:8s}"
            f" trades={metrics['trades']:<5d}"
            f" win_rate={metrics['win_rate']:.2f}%"
            f" avg_net={metrics['avg_net']:.2f}%"
            f" PF={pf_text}"
        )


# ============================================================
# BULL / BEAR COMPARISON
# ============================================================

def build_bull_bear_comparison(
    baseline: pd.DataFrame,
    filtered: pd.DataFrame,
):

    rows = []

    for regime in [
        "BULL",
        "BEAR",
    ]:

        baseline_subset = baseline[
            baseline["regime"] == regime
        ]

        filtered_subset = filtered[
            filtered["regime"] == regime
        ]

        b = calculate_metrics(
            baseline_subset
        )

        f = calculate_metrics(
            filtered_subset
        )

        trade_change = (
            f["trades"]
            - b["trades"]
        )

        if b["trades"] > 0:

            trade_reduction = (
                (
                    b["trades"]
                    - f["trades"]
                )
                / b["trades"]
                * 100.0
            )

        else:

            trade_reduction = np.nan

        rows.append({
            "regime": regime,

            "baseline_trades":
                b["trades"],

            "filtered_trades":
                f["trades"],

            "baseline_win_rate":
                b["win_rate"],

            "filtered_win_rate":
                f["win_rate"],

            "baseline_avg_net":
                b["avg_net"],

            "filtered_avg_net":
                f["avg_net"],

            "baseline_profit_factor":
                b["profit_factor"],

            "filtered_profit_factor":
                f["profit_factor"],

            "baseline_compounded":
                b["compounded"],

            "filtered_compounded":
                f["compounded"],

            "baseline_max_drawdown":
                b["max_drawdown"],

            "filtered_max_drawdown":
                f["max_drawdown"],

            "baseline_avg_holding":
                b["avg_holding"],

            "filtered_avg_holding":
                f["avg_holding"],

            "baseline_target_exits":
                b["target_exits"],

            "filtered_target_exits":
                f["target_exits"],

            "baseline_stop_exits":
                b["stop_exits"],

            "filtered_stop_exits":
                f["stop_exits"],

            "trade_change":
                trade_change,

            "trade_reduction_pct":
                trade_reduction,

            "win_rate_change":
                f["win_rate"]
                - b["win_rate"],

            "avg_net_change":
                f["avg_net"]
                - b["avg_net"],

            "profit_factor_change":
                f["profit_factor"]
                - b["profit_factor"],
        })

    return pd.DataFrame(
        rows
    )


def print_bull_bear_comparison(
    comparison: pd.DataFrame,
):

    for regime in [
        "BULL",
        "BEAR",
    ]:

        row = comparison[
            comparison["regime"] == regime
        ]

        if row.empty:
            continue

        row = row.iloc[0]

        print()
        print(LINE)
        print(f"{regime} MARKET")
        print(LINE)

        print(
            f"{'Metric':35s}"
            f"{'BASELINE':>20s}"
            f"{'FILTERED':>20s}"
        )

        print(SUBLINE)

        metrics = [
            (
                "Trades",
                "baseline_trades",
                "filtered_trades",
                0,
            ),
            (
                "Win rate %",
                "baseline_win_rate",
                "filtered_win_rate",
                2,
            ),
            (
                "Avg net return %",
                "baseline_avg_net",
                "filtered_avg_net",
                2,
            ),
            (
                "Profit factor",
                "baseline_profit_factor",
                "filtered_profit_factor",
                2,
            ),
            (
                "Compounded return %",
                "baseline_compounded",
                "filtered_compounded",
                2,
            ),
            (
                "Max drawdown %",
                "baseline_max_drawdown",
                "filtered_max_drawdown",
                2,
            ),
            (
                "Avg holding days",
                "baseline_avg_holding",
                "filtered_avg_holding",
                2,
            ),
            (
                "Target exits",
                "baseline_target_exits",
                "filtered_target_exits",
                0,
            ),
            (
                "Stop exits",
                "baseline_stop_exits",
                "filtered_stop_exits",
                0,
            ),
        ]

        for (
            label,
            baseline_key,
            filtered_key,
            decimals,
        ) in metrics:

            b = row[baseline_key]
            f = row[filtered_key]

            if "profit_factor" in baseline_key:

                b_text = (
                    "INF"
                    if np.isinf(b)
                    else f"{b:.2f}"
                )

                f_text = (
                    "INF"
                    if np.isinf(f)
                    else f"{f:.2f}"
                )

            elif decimals == 0:

                b_text = f"{b:.0f}"
                f_text = f"{f:.0f}"

            else:

                b_text = f"{b:.{decimals}f}"
                f_text = f"{f:.{decimals}f}"

            print(
                f"{label:35s}"
                f"{b_text:>20s}"
                f"{f_text:>20s}"
            )

        print()
        print("FILTER IMPACT")
        print(SUBLINE)

        print(
            f"Trades removed       : "
            f"{row['trade_change']:.0f}"
        )

        if pd.isna(
            row["trade_reduction_pct"]
        ):

            print(
                "Trade reduction      : N/A"
            )

        else:

            print(
                f"Trade reduction      : "
                f"{row['trade_reduction_pct']:.2f}%"
            )

        print(
            f"Win-rate change       : "
            f"{row['win_rate_change']:+.2f}%"
        )

        print(
            f"Avg net change        : "
            f"{row['avg_net_change']:+.2f}%"
        )

        print(
            f"Profit-factor change  : "
            f"{row['profit_factor_change']:+.2f}"
        )


# ============================================================
# MAIN
# ============================================================

def main():

    print(LINE)
    print("REGIME-FILTERED BACKTEST ANALYTICS")
    print(LINE)

    # --------------------------------------------------------
    # LOAD
    # --------------------------------------------------------

    baseline = load_csv(
        BASELINE_FILE,
        "Baseline backtest",
    )

    filtered = load_csv(
        FILTERED_FILE,
        "Regime-filtered backtest",
    )

    regime_df = load_csv(
        REGIME_FILE,
        "NIFTY regime",
    )

    print()
    print(
        f"Baseline trades loaded : "
        f"{len(baseline)}"
    )

    print(
        f"Regime trades loaded   : "
        f"{len(filtered)}"
    )

    print(
        f"NIFTY regime rows      : "
        f"{len(regime_df)}"
    )

    # --------------------------------------------------------
    # MAP BASELINE REGIME
    # --------------------------------------------------------

    baseline = attach_baseline_regime(
        baseline,
        regime_df,
    )

    filtered = normalize_filtered_regime(
        filtered
    )

    # --------------------------------------------------------
    # REGIME MAPPING CHECK
    # --------------------------------------------------------

    baseline_known = baseline[
        "regime"
    ].isin(
        [
            "BULL",
            "BEAR",
            "NEUTRAL",
        ]
    ).sum()

    filtered_known = filtered[
        "regime"
    ].isin(
        [
            "BULL",
            "BEAR",
            "NEUTRAL",
        ]
    ).sum()

    print()
    print(LINE)
    print("REGIME MAPPING CHECK")
    print(LINE)

    print(
        f"Baseline trades mapped : "
        f"{baseline_known}/{len(baseline)}"
    )

    print(
        f"Filtered trades mapped : "
        f"{filtered_known}/{len(filtered)}"
    )

    # --------------------------------------------------------
    # COUNTS
    # --------------------------------------------------------

    print()
    print("BASELINE REGIME COUNTS")

    for regime in [
        "BULL",
        "BEAR",
        "NEUTRAL",
        "UNKNOWN",
    ]:

        count = int(
            (
                baseline["regime"]
                == regime
            ).sum()
        )

        print(
            f"{regime:8s}: {count}"
        )

    print()
    print("FILTERED REGIME COUNTS")

    for regime in [
        "BULL",
        "BEAR",
        "NEUTRAL",
        "UNKNOWN",
    ]:

        count = int(
            (
                filtered["regime"]
                == regime
            ).sum()
        )

        print(
            f"{regime:8s}: {count}"
        )

    # --------------------------------------------------------
    # OVERALL
    # --------------------------------------------------------

    baseline_metrics = calculate_metrics(
        baseline
    )

    filtered_metrics = calculate_metrics(
        filtered
    )

    print_comparison(
        "BASELINE VS REGIME-FILTERED",
        baseline_metrics,
        filtered_metrics,
    )

    # --------------------------------------------------------
    # FILTER IMPACT
    # --------------------------------------------------------

    print()
    print(LINE)
    print("REGIME FILTER IMPACT")
    print(LINE)

    trades_removed = (
        len(baseline)
        - len(filtered)
    )

    trade_reduction = (
        trades_removed
        / len(baseline)
        * 100.0
    )

    win_rate_change = (
        filtered_metrics["win_rate"]
        - baseline_metrics["win_rate"]
    )

    avg_net_change = (
        filtered_metrics["avg_net"]
        - baseline_metrics["avg_net"]
    )

    pf_change = (
        filtered_metrics["profit_factor"]
        - baseline_metrics["profit_factor"]
    )

    compounded_change = (
        filtered_metrics["compounded"]
        - baseline_metrics["compounded"]
    )

    drawdown_change = (
        filtered_metrics["max_drawdown"]
        - baseline_metrics["max_drawdown"]
    )

    print(
        f"Trades removed         : "
        f"{trades_removed}"
    )

    print(
        f"Trade reduction        : "
        f"{trade_reduction:.2f}%"
    )

    print(
        f"Win-rate change        : "
        f"{win_rate_change:+.2f}%"
    )

    print(
        f"Avg net return change  : "
        f"{avg_net_change:+.2f}%"
    )

    print(
        f"Profit factor change   : "
        f"{pf_change:+.2f}"
    )

    print(
        f"Compounded change      : "
        f"{compounded_change:+.2f}%"
    )

    print(
        f"Max drawdown change    : "
        f"{drawdown_change:+.2f}%"
    )

    # --------------------------------------------------------
    # BREAKDOWNS
    # --------------------------------------------------------

    print_regime_breakdown(
        "BASELINE REGIME BREAKDOWN",
        baseline,
    )

    print_regime_breakdown(
        "FILTERED REGIME BREAKDOWN",
        filtered,
    )

    # --------------------------------------------------------
    # BULL / BEAR
    # --------------------------------------------------------

    comparison = build_bull_bear_comparison(
        baseline,
        filtered,
    )

    print_bull_bear_comparison(
        comparison
    )

    # --------------------------------------------------------
    # FILTERED DIRECTION
    # --------------------------------------------------------

    print()
    print(LINE)
    print("FILTERED REGIME + DIRECTION")
    print(LINE)

    direction_col = find_column(
        filtered,
        ["direction"],
    )

    if direction_col:

        temp = filtered.copy()

        temp["direction_normalized"] = (
            temp[direction_col]
            .astype(str)
            .str.upper()
        )

        for regime in [
            "BULL",
            "BEAR",
        ]:

            for direction in [
                "LONG",
                "SHORT",
            ]:

                subset = temp[
                    (
                        temp["regime"]
                        == regime
                    )
                    &
                    (
                        temp[
                            "direction_normalized"
                        ]
                        == direction
                    )
                ]

                if subset.empty:
                    continue

                metrics = calculate_metrics(
                    subset
                )

                pf = metrics[
                    "profit_factor"
                ]

                if np.isinf(pf):

                    pf_text = "INF"

                else:

                    pf_text = f"{pf:.2f}"

                print(
                    f"{regime:8s}"
                    f"{direction:8s}"
                    f"trades={metrics['trades']:<5d}"
                    f"win_rate={metrics['win_rate']:.2f}%"
                    f"avg_net={metrics['avg_net']:.2f}%"
                    f"PF={pf_text}"
                )

    # --------------------------------------------------------
    # SAVE OVERALL COMPARISON
    # --------------------------------------------------------

    overall_comparison = pd.DataFrame([
        {
            "dataset": "BASELINE",
            **baseline_metrics,
        },
        {
            "dataset": "REGIME_FILTERED",
            **filtered_metrics,
        },
    ])

    overall_comparison.to_csv(
        COMPARISON_FILE,
        index=False,
    )

    # --------------------------------------------------------
    # SAVE REGIME DIAGNOSTICS
    # --------------------------------------------------------

    baseline_breakdown = regime_breakdown(
        baseline
    )

    baseline_breakdown.insert(
        0,
        "dataset",
        "BASELINE",
    )

    filtered_breakdown = regime_breakdown(
        filtered
    )

    filtered_breakdown.insert(
        0,
        "dataset",
        "REGIME_FILTERED",
    )

    diagnostics = pd.concat(
        [
            baseline_breakdown,
            filtered_breakdown,
        ],
        ignore_index=True,
    )

    diagnostics.to_csv(
        DIAGNOSTICS_FILE,
        index=False,
    )

    # --------------------------------------------------------
    # SAVE BULL / BEAR COMPARISON
    # --------------------------------------------------------

    comparison.to_csv(
        BULL_BEAR_FILE,
        index=False,
    )

    # --------------------------------------------------------
    # DONE
    # --------------------------------------------------------

    print()
    print(LINE)
    print("FILES SAVED")
    print(LINE)

    print(
        f"Overall comparison   : "
        f"{COMPARISON_FILE}"
    )

    print(
        f"Regime diagnostics   : "
        f"{DIAGNOSTICS_FILE}"
    )

    print(
        f"Bull/Bear comparison : "
        f"{BULL_BEAR_FILE}"
    )

    print()
    print(LINE)
    print("REGIME ANALYTICS COMPLETE")
    print(LINE)


if __name__ == "__main__":
    main()