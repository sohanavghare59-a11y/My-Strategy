"""
Walk-Forward Stability Test
===========================

Research-only validation.

Tests four fixed strategy variants across chronological periods:

1. CURRENT
   EMA + MACD + RSI

2. CURRENT_REGIME
   EMA + MACD + RSI
   BULL + BEAR only

3. EMA_MACD
   EMA + MACD

4. EMA_MACD_REGIME
   EMA + MACD
   BULL + BEAR only

The purpose is to determine whether the observed edge is
stable across different historical periods rather than being
concentrated in one favorable period.

No production configuration is changed.
"""

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
    / "walk_forward_comparison.csv"
)

DIRECTION_FILE = (
    BASE_DIR
    / "output"
    / "walk_forward_direction.csv"
)

REGIME_FILE_OUTPUT = (
    BASE_DIR
    / "output"
    / "walk_forward_regime.csv"
)


# ============================================================
# VARIANTS
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
# LOAD TRADES
# ============================================================

def load_trades():

    if not TRADES_FILE.exists():

        raise FileNotFoundError(
            f"Missing file:\n{TRADES_FILE}"
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
            "Missing required columns: "
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

    trades["net_pnl_pct"] = pd.to_numeric(
        trades["net_pnl_pct"],
        errors="coerce",
    )

    trades["gross_pnl_pct"] = pd.to_numeric(
        trades["gross_pnl_pct"],
        errors="coerce",
    )

    trades["holding_days"] = pd.to_numeric(
        trades["holding_days"],
        errors="coerce",
    )

    trades = trades.dropna(
        subset=[
            "signal_date",
            "net_pnl_pct",
        ]
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
            f"Missing file:\n{REGIME_FILE}"
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
            "No date column found in "
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
            "No regime column found in "
            f"{REGIME_FILE}"
        )

    regime = regime[
        [
            date_column,
            regime_column,
        ]
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

    original_regime = trades[
        [
            "signal_date",
            "regime",
        ]
    ].copy()

    mapped = trades.drop(
        columns=["regime"]
    ).merge(
        regime,
        on="signal_date",
        how="left",
    )

    mapped["regime"] = (
        mapped["regime"]
        .fillna("UNKNOWN")
        .astype(str)
        .str.upper()
        .str.strip()
    )

    return mapped


# ============================================================
# METRICS
# ============================================================

def calculate_metrics(
    trades,
):

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

    winners = net[
        net > 0
    ]

    losers = net[
        net <= 0
    ]

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
        1
        + net / 100
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
        "trades": int(len(net)),

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

        "avg_gross_pct": (
            gross.mean()
            if len(gross) > 0
            else 0.0
        ),

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

        "profit_factor": (
            profit_factor
        ),

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
# VARIANT FILTER
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
# CREATE WALK-FORWARD PERIODS
# ============================================================

def create_periods(
    trades,
):

    dates = (
        trades["signal_date"]
        .drop_duplicates()
        .sort_values()
        .reset_index(
            drop=True
        )
    )

    if len(dates) < 20:

        raise ValueError(
            "Not enough signal dates for "
            "walk-forward testing."
        )

    minimum_date = dates.iloc[0]
    maximum_date = dates.iloc[-1]

    total_days = (
        maximum_date
        - minimum_date
    ).days

    # Four chronological calendar periods.
    boundaries = []

    for fraction in [
        0.25,
        0.50,
        0.75,
        1.00,
    ]:

        target = (
            minimum_date
            + pd.Timedelta(
                days=int(
                    total_days
                    * fraction
                )
            )
        )

        boundaries.append(
            target
        )

    periods = []

    start = minimum_date

    for index, end in enumerate(
        boundaries,
        start=1,
    ):

        if index == 1:

            label = "PERIOD_1"

        elif index == 2:

            label = "PERIOD_2"

        elif index == 3:

            label = "PERIOD_3"

        else:

            label = "PERIOD_4"

        periods.append(
            {
                "period": label,
                "start": start,
                "end": end,
            }
        )

        start = (
            end
            + pd.Timedelta(
                days=1
            )
        )

    return periods


# ============================================================
# PRINT METRICS
# ============================================================

def print_metrics_row(
    variant,
    period,
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
        f"{variant:22s}"
        f"{period:12s}"
        f"{metrics['trades']:7d}"
        f"{metrics['win_rate_pct']:9.2f}%"
        f"{metrics['avg_net_pct']:+10.2f}%"
        f"{pf_text:>7s}"
        f"{metrics['compounded_return_pct']:+11.2f}%"
        f"{metrics['max_drawdown_pct']:11.2f}%"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 96)
    print("WALK-FORWARD STABILITY TEST")
    print("=" * 96)

    print()
    print(
        "Research only."
    )

    print(
        "Production configuration will NOT be changed."
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
    # MAP REGIME
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
        f"Mapped  : "
        f"{mapped_count}/{len(mapped)}"
    )

    print(
        f"Unknown : "
        f"{len(mapped) - mapped_count}"
    )

    # ========================================================
    # PERIODS
    # ========================================================

    periods = create_periods(
        mapped
    )

    print()
    print("=" * 96)
    print("WALK-FORWARD PERIODS")
    print("=" * 96)

    for item in periods:

        period_data = mapped[
            (
                mapped["signal_date"]
                >= item["start"]
            )
            & (
                mapped["signal_date"]
                <= item["end"]
            )
        ]

        print(
            f"{item['period']:12s}"
            f"{item['start'].date()}"
            f" -> "
            f"{item['end'].date()}"
            f" | trades={len(period_data)}"
        )

    # ========================================================
    # PERIOD RESULTS
    # ========================================================

    summary_rows = []

    print()
    print("=" * 96)
    print("PERIOD-BY-PERIOD RESULTS")
    print("=" * 96)

    print(
        f"{'Variant':22s}"
        f"{'Period':12s}"
        f"{'Trades':7s}"
        f"{'Win %':10s}"
        f"{'Avg Net':11s}"
        f"{'PF':8s}"
        f"{'Comp.':12s}"
        f"{'Max DD':12s}"
    )

    print("-" * 96)

    for item in periods:

        period_data = mapped[
            (
                mapped["signal_date"]
                >= item["start"]
            )
            & (
                mapped["signal_date"]
                <= item["end"]
            )
        ]

        for variant_name in VARIANTS:

            variant_data = get_variant_trades(
                period_data,
                variant_name,
            )

            metrics = calculate_metrics(
                variant_data
            )

            print_metrics_row(
                variant_name,
                item["period"],
                metrics,
            )

            row = metrics.copy()

            row["variant"] = (
                variant_name
            )

            row["period"] = (
                item["period"]
            )

            row["period_start"] = (
                item["start"].date()
            )

            row["period_end"] = (
                item["end"].date()
            )

            summary_rows.append(
                row
            )

    summary = pd.DataFrame(
        summary_rows
    )

    # ========================================================
    # DIRECTION STABILITY
    # ========================================================

    direction_rows = []

    for item in periods:

        period_data = mapped[
            (
                mapped["signal_date"]
                >= item["start"]
            )
            & (
                mapped["signal_date"]
                <= item["end"]
            )
        ]

        for variant_name in VARIANTS:

            variant_data = get_variant_trades(
                period_data,
                variant_name,
            )

            for direction in [
                "LONG",
                "SHORT",
            ]:

                subset = variant_data[
                    variant_data[
                        "direction"
                    ]
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

                metrics["period"] = (
                    item["period"]
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
    # REGIME STABILITY
    # ========================================================

    regime_rows = []

    for item in periods:

        period_data = mapped[
            (
                mapped["signal_date"]
                >= item["start"]
            )
            & (
                mapped["signal_date"]
                <= item["end"]
            )
        ]

        for variant_name in VARIANTS:

            variant_data = get_variant_trades(
                period_data,
                variant_name,
            )

            for regime_name in [
                "BULL",
                "BEAR",
                "NEUTRAL",
                "UNKNOWN",
            ]:

                subset = variant_data[
                    variant_data[
                        "regime"
                    ]
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

                metrics["period"] = (
                    item["period"]
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
    # STABILITY SUMMARY
    # ========================================================

    print()
    print("=" * 96)
    print("STABILITY SUMMARY")
    print("=" * 96)

    for variant_name in VARIANTS:

        data = summary[
            summary["variant"]
            == variant_name
        ].copy()

        valid = data[
            data["trades"] > 0
        ]

        profitable_periods = (
            valid[
                valid["avg_net_pct"] > 0
            ]
        )

        pf_above_one = (
            valid[
                valid["profit_factor"] > 1
            ]
        )

        print()
        print(
            variant_name
        )

        print(
            f"Periods with trades : "
            f"{len(valid)}"
        )

        print(
            f"Profitable periods  : "
            f"{len(profitable_periods)}"
        )

        print(
            f"PF > 1 periods     : "
            f"{len(pf_above_one)}"
        )

        if len(valid) > 0:

            print(
                f"Average period PF  : "
                f"{valid['profit_factor'].mean():.2f}"
            )

            print(
                f"Median period PF   : "
                f"{valid['profit_factor'].median():.2f}"
            )

            print(
                f"Average period net : "
                f"{valid['avg_net_pct'].mean():+.2f}%"
            )

            print(
                f"Median period net  : "
                f"{valid['avg_net_pct'].median():+.2f}%"
            )

    # ========================================================
    # REGIME FILTER ADVANTAGE
    # ========================================================

    print()
    print("=" * 96)
    print("REGIME FILTER ADVANTAGE BY PERIOD")
    print("=" * 96)

    for period_name in [
        item["period"]
        for item in periods
    ]:

        print()
        print(
            period_name
        )

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
                    == period_name
                )
            ]

            filtered = summary[
                (
                    summary["variant"]
                    == filtered_variant
                )
                & (
                    summary["period"]
                    == period_name
                )
            ]

            if base.empty or filtered.empty:
                continue

            base = base.iloc[0]
            filtered = filtered.iloc[0]

            print(
                f"{base_variant:12s}"
                f" | trades "
                f"{int(base['trades'])}"
                f" -> "
                f"{int(filtered['trades'])}"
                f" | PF "
                f"{base['profit_factor']:.2f}"
                f" -> "
                f"{filtered['profit_factor']:.2f}"
                f" | Avg Net "
                f"{base['avg_net_pct']:+.2f}%"
                f" -> "
                f"{filtered['avg_net_pct']:+.2f}%"
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
        REGIME_FILE_OUTPUT,
        index=False,
    )

    print()
    print("=" * 96)
    print("FILES SAVED")
    print("=" * 96)

    print(
        OUTPUT_FILE
    )

    print(
        DIRECTION_FILE
    )

    print(
        REGIME_FILE_OUTPUT
    )

    print()
    print("=" * 96)
    print("WALK-FORWARD TEST COMPLETE")
    print("=" * 96)


if __name__ == "__main__":
    main()