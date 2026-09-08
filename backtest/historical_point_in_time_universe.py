"""
Historical Point-in-Time Universe Builder
=========================================

Builds a historical stock-selection universe using only information
available in each stock's first half of historical data.

This is a RESEARCH utility.

It does NOT modify:
    - production scanner
    - production strategy
    - existing production outputs

Method
------
1. Read stocks with sufficient historical coverage.
2. Load each stock's explicit historical daily cache.
3. Require at least MIN_HISTORY rows.
4. Split each stock's history into two halves.
5. Calculate average volume using ONLY the first half.
6. Rank NSE and BSE stocks separately.
7. Select TOP_N_PER_EXCHANGE stocks from each exchange.
8. Save the selected historical universe.

The second half is reserved for OOS testing.

This deliberately avoids using current average volume to select
historical stocks.
"""

from __future__ import annotations

import argparse
import os

import pandas as pd

from data.market_data import get_historical_data


# ============================================================
# CONFIGURATION
# ============================================================

COVERAGE_FILE = "output/historical_coverage_universe.csv"

OUTPUT_FILE = "output/historical_point_in_time_universe.csv"

START_DATE = "2021-01-01"
END_DATE = "2026-09-06"

INTERVAL = "1d"

MIN_HISTORY = 400

TOP_N_PER_EXCHANGE = 50

CACHE_HOURS = 168


# ============================================================
# HELPERS
# ============================================================

def load_coverage() -> pd.DataFrame:
    """
    Load the historical coverage results.

    Only PASS stocks are eligible.
    """

    if not os.path.exists(COVERAGE_FILE):

        raise FileNotFoundError(
            f"Coverage file not found: {COVERAGE_FILE}"
        )

    data = pd.read_csv(
        COVERAGE_FILE
    )

    required = [
        "exchange",
        "symbol",
        "name",
        "data_symbol",
        "avg_volume",
        "historical_status",
        "rows",
    ]

    missing = [
        column
        for column in required
        if column not in data.columns
    ]

    if missing:

        raise ValueError(
            "Missing required columns: "
            + ", ".join(missing)
        )

    data = data[
        data["historical_status"]
        .astype(str)
        .str.upper()
        .eq("PASS")
    ].copy()

    data["rows"] = pd.to_numeric(
        data["rows"],
        errors="coerce",
    )

    data = data[
        data["rows"] >= MIN_HISTORY
    ].copy()

    data["exchange"] = (
        data["exchange"]
        .astype(str)
        .str.upper()
        .str.strip()
    )

    data["data_symbol"] = (
        data["data_symbol"]
        .astype(str)
        .str.strip()
    )

    data = data[
        data["data_symbol"].ne("")
        & data["data_symbol"].ne("nan")
    ].copy()

    data = data.drop_duplicates(
        subset=["data_symbol"],
        keep="first",
    ).reset_index(drop=True)

    return data


def calculate_first_half_average_volume(
    data: pd.DataFrame,
) -> tuple[float, int, str, str]:
    """
    Calculate average volume using only the first half
    of the available historical data.

    Returns:
        average_volume,
        first_half_rows,
        first_half_start,
        first_half_end
    """

    if data.empty:

        return 0.0, 0, "", ""

    data = data.sort_index().copy()

    midpoint = len(data) // 2

    first_half = data.iloc[
        :midpoint
    ].copy()

    if first_half.empty:

        return 0.0, 0, "", ""

    volume = pd.to_numeric(
        first_half["Volume"],
        errors="coerce",
    )

    volume = volume[
        volume > 0
    ]

    if volume.empty:

        average_volume = 0.0

    else:

        average_volume = float(
            volume.mean()
        )

    first_date = (
        first_half.index[0]
        .date()
        .isoformat()
    )

    last_date = (
        first_half.index[-1]
        .date()
        .isoformat()
    )

    return (
        average_volume,
        len(first_half),
        first_date,
        last_date,
    )


def process_stock(
    row: pd.Series,
) -> dict:
    """
    Load one stock and calculate its first-half
    historical liquidity.
    """

    exchange = str(
        row["exchange"]
    ).strip()

    symbol = str(
        row["symbol"]
    ).strip()

    name = str(
        row["name"]
    ).strip()

    data_symbol = str(
        row["data_symbol"]
    ).strip()

    result = {
        "exchange": exchange,
        "symbol": symbol,
        "name": name,
        "data_symbol": data_symbol,
        "full_history_rows": 0,
        "full_history_start": "",
        "full_history_end": "",
        "selection_history_rows": 0,
        "selection_history_start": "",
        "selection_history_end": "",
        "selection_avg_volume": 0.0,
        "selection_status": "ERROR",
        "error": "",
    }

    try:

        history = get_historical_data(
            data_symbol,
            start_date=START_DATE,
            end_date=END_DATE,
            interval=INTERVAL,
            cache_hours=CACHE_HOURS,
            force_refresh=False,
            minimum_rows=MIN_HISTORY,
        )

        if history.empty:

            result["selection_status"] = (
                "NO_DATA"
            )

            result["error"] = (
                "Historical data returned empty."
            )

            return result

        history = history.sort_index()

        result["full_history_rows"] = len(
            history
        )

        result["full_history_start"] = (
            history.index[0]
            .date()
            .isoformat()
        )

        result["full_history_end"] = (
            history.index[-1]
            .date()
            .isoformat()
        )

        if len(history) < MIN_HISTORY:

            result["selection_status"] = (
                "INSUFFICIENT_HISTORY"
            )

            result["error"] = (
                f"Only {len(history)} rows; "
                f"minimum is {MIN_HISTORY}."
            )

            return result

        (
            average_volume,
            first_half_rows,
            first_half_start,
            first_half_end,
        ) = calculate_first_half_average_volume(
            history
        )

        result[
            "selection_avg_volume"
        ] = average_volume

        result[
            "selection_history_rows"
        ] = first_half_rows

        result[
            "selection_history_start"
        ] = first_half_start

        result[
            "selection_history_end"
        ] = first_half_end

        if average_volume <= 0:

            result["selection_status"] = (
                "NO_VOLUME"
            )

            result["error"] = (
                "No positive volume observations "
                "in first-half history."
            )

            return result

        result["selection_status"] = (
            "PASS"
        )

        return result

    except Exception as exc:

        result["selection_status"] = (
            "ERROR"
        )

        result["error"] = (
            f"{type(exc).__name__}: {exc}"
        )

        return result


# ============================================================
# UNIVERSE BUILD
# ============================================================

def build_universe(
    top_n: int,
) -> pd.DataFrame:
    """
    Build the historical point-in-time universe.
    """

    coverage = load_coverage()

    print("=" * 70)
    print("HISTORICAL POINT-IN-TIME UNIVERSE BUILDER")
    print("=" * 70)

    print(
        f"Eligible coverage stocks: "
        f"{len(coverage)}"
    )

    print(
        f"Minimum history: "
        f"{MIN_HISTORY} rows"
    )

    print(
        f"Selection method: "
        f"first-half average volume"
    )

    print(
        f"Selection: "
        f"top {top_n} NSE + top {top_n} BSE"
    )

    print("=" * 70)

    results = []

    total = len(
        coverage
    )

    for position, (_, row) in enumerate(
        coverage.iterrows(),
        start=1,
    ):

        exchange = str(
            row["exchange"]
        ).strip()

        symbol = str(
            row["symbol"]
        ).strip()

        print(
            f"[{position}/{total}] "
            f"{exchange} {symbol}"
        )

        result = process_stock(
            row
        )

        results.append(
            result
        )

    research = pd.DataFrame(
        results
    )

    passed = research[
        research["selection_status"]
        .eq("PASS")
    ].copy()

    if passed.empty:

        raise RuntimeError(
            "No stocks passed historical "
            "selection requirements."
        )

    passed = passed.sort_values(
        by=[
            "exchange",
            "selection_avg_volume",
        ],
        ascending=[
            True,
            False,
        ],
    ).copy()

    selected_parts = []

    for exchange in [
        "NSE",
        "BSE",
    ]:

        exchange_data = passed[
            passed["exchange"]
            .eq(exchange)
        ].copy()

        selected = exchange_data.head(
            top_n
        ).copy()

        if not selected.empty:

            selected[
                "selection_rank"
            ] = range(
                1,
                len(selected) + 1,
            )

            selected_parts.append(
                selected
            )

    if not selected_parts:

        raise RuntimeError(
            "No exchange produced a selected universe."
        )

    selected = pd.concat(
        selected_parts,
        ignore_index=True,
    )

    selected[
        "selection_method"
    ] = "FIRST_HALF_AVG_VOLUME"

    selected[
        "selection_top_n"
    ] = top_n

    selected[
        "oos_start_date"
    ] = selected[
        "selection_history_end"
    ]

    # Keep research diagnostics as separate columns.
    selected = selected[
        [
            "exchange",
            "symbol",
            "name",
            "data_symbol",
            "full_history_rows",
            "full_history_start",
            "full_history_end",
            "selection_history_rows",
            "selection_history_start",
            "selection_history_end",
            "selection_avg_volume",
            "selection_rank",
            "selection_method",
            "selection_top_n",
            "oos_start_date",
        ]
    ].copy()

    return selected


# ============================================================
# SUMMARY
# ============================================================

def print_summary(
    selected: pd.DataFrame,
) -> None:
    """
    Print selected-universe summary.
    """

    print()
    print("=" * 70)
    print("HISTORICAL POINT-IN-TIME UNIVERSE SUMMARY")
    print("=" * 70)

    print(
        f"Total selected: "
        f"{len(selected)}"
    )

    for exchange in [
        "NSE",
        "BSE",
    ]:

        exchange_data = selected[
            selected["exchange"]
            .eq(exchange)
        ]

        print(
            f"{exchange}: "
            f"{len(exchange_data)}"
        )

    print()

    if not selected.empty:

        print(
            "Selection-history date ranges:"
        )

        for exchange in [
            "NSE",
            "BSE",
        ]:

            exchange_data = selected[
                selected["exchange"]
                .eq(exchange)
            ]

            if exchange_data.empty:
                continue

            print(
                f"  {exchange}: "
                f"{exchange_data['selection_history_start'].min()} "
                f"-> "
                f"{exchange_data['selection_history_end'].max()}"
            )

        print()

        print(
            "Top selected stocks by historical "
            "average volume:"
        )

        display_columns = [
            "exchange",
            "symbol",
            "selection_avg_volume",
            "selection_rank",
        ]

        print(
            selected[
                display_columns
            ]
            .head(10)
            .to_string(index=False)
        )

    print()
    print(
        f"Saved to: {OUTPUT_FILE}"
    )

    print("=" * 70)


# ============================================================
# MAIN
# ============================================================

def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments.
    """

    parser = argparse.ArgumentParser(
        description=(
            "Build a historical point-in-time "
            "stock universe."
        )
    )

    parser.add_argument(
        "--top-n",
        type=int,
        default=TOP_N_PER_EXCHANGE,
        help=(
            "Number of stocks per exchange. "
            "Default: 50."
        ),
    )

    return parser.parse_args()


def main() -> None:

    args = parse_args()

    if args.top_n <= 0:

        raise ValueError(
            "--top-n must be greater than zero."
        )

    selected = build_universe(
        top_n=args.top_n
    )

    os.makedirs(
        os.path.dirname(OUTPUT_FILE),
        exist_ok=True,
    )

    selected.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    print_summary(
        selected
    )


if __name__ == "__main__":

    main()