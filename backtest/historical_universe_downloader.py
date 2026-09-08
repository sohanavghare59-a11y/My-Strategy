"""
Historical Universe Downloader
==============================

Downloads explicit daily historical data for the current liquid
universe using the long-range market-data cache.

This is a RESEARCH utility.

It does NOT modify:
    - production scanner
    - production strategy
    - production configuration
    - existing backtests

Default historical range:
    2021-01-01 -> 2026-09-06

The downloader is resume-safe:
    existing valid historical caches are reused.
"""

from __future__ import annotations

import argparse
import os
import time
from datetime import datetime

import pandas as pd

from data.market_data import get_historical_data


# ============================================================
# CONFIGURATION
# ============================================================

INPUT_FILE = "output/liquid_universe.csv"

OUTPUT_FILE = "output/historical_coverage_universe.csv"

START_DATE = "2021-01-01"
END_DATE = "2026-09-06"

INTERVAL = "1d"

MIN_HISTORY = 400

DEFAULT_LIMIT = 20

DELAY_SECONDS = 0.25

CACHE_HOURS = 168


# ============================================================
# HELPERS
# ============================================================

def ensure_output_directory() -> None:
    """
    Ensure output directory exists.
    """

    os.makedirs(
        os.path.dirname(OUTPUT_FILE),
        exist_ok=True,
    )


def load_liquid_universe() -> pd.DataFrame:
    """
    Load the current liquid universe.

    Only PASS stocks are used.

    Index instruments are excluded because this downloader
    is intended to construct a historical stock universe.
    """

    if not os.path.exists(INPUT_FILE):

        raise FileNotFoundError(
            f"Input file not found: {INPUT_FILE}"
        )

    data = pd.read_csv(
        INPUT_FILE
    )

    required = [
        "exchange",
        "instrument_type",
        "symbol",
        "name",
        "data_symbol",
        "latest_close",
        "avg_volume",
        "liquidity_status",
    ]

    missing = [
        column
        for column in required
        if column not in data.columns
    ]

    if missing:

        raise ValueError(
            "Missing required columns in liquid universe: "
            + ", ".join(missing)
        )

    data = data[
        data["liquidity_status"]
        .astype(str)
        .str.upper()
        .eq("PASS")
    ].copy()

    data = data[
        data["instrument_type"]
        .astype(str)
        .str.upper()
        .eq("STOCK")
    ].copy()

    data["avg_volume"] = pd.to_numeric(
        data["avg_volume"],
        errors="coerce",
    )

    data = data[
        data["avg_volume"] > 0
    ].copy()

    data["data_symbol"] = (
        data["data_symbol"]
        .astype(str)
        .str.strip()
    )

    data = data[
        data["data_symbol"].ne("")
        & data["data_symbol"].ne("nan")
    ].copy()

    # One Yahoo symbol should represent one research series.
    data = data.drop_duplicates(
        subset=["data_symbol"],
        keep="first",
    ).copy()

    # Rank by current liquidity so the first test batch is useful.
    data = data.sort_values(
        by="avg_volume",
        ascending=False,
    ).reset_index(drop=True)

    return data


def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments.
    """

    parser = argparse.ArgumentParser(
        description=(
            "Download long historical daily data for "
            "the liquid stock universe."
        )
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=DEFAULT_LIMIT,
        help=(
            "Maximum number of stocks to process. "
            "Use 20 for the first test. "
            "Use 0 for all stocks."
        ),
    )

    parser.add_argument(
        "--start",
        default=START_DATE,
        help="Historical start date YYYY-MM-DD.",
    )

    parser.add_argument(
        "--end",
        default=END_DATE,
        help="Historical end date YYYY-MM-DD.",
    )

    parser.add_argument(
        "--delay",
        type=float,
        default=DELAY_SECONDS,
        help="Delay between Yahoo requests in seconds.",
    )

    parser.add_argument(
        "--force-refresh",
        action="store_true",
        help="Ignore existing historical caches.",
    )

    return parser.parse_args()


# ============================================================
# MAIN DOWNLOAD
# ============================================================

def run_downloader(
    limit: int,
    start_date: str,
    end_date: str,
    delay: float,
    force_refresh: bool,
) -> pd.DataFrame:
    """
    Download historical data for the selected universe.
    """

    ensure_output_directory()

    universe = load_liquid_universe()

    total_candidates = len(
        universe
    )

    if limit > 0:

        selected = universe.head(
            limit
        ).copy()

    else:

        selected = universe.copy()

    print("=" * 70)
    print("HISTORICAL UNIVERSE DOWNLOADER")
    print("=" * 70)

    print(
        f"Input universe: {total_candidates} PASS stocks"
    )

    print(
        f"Selected for this run: {len(selected)} stocks"
    )

    print(
        f"Historical range: {start_date} -> {end_date}"
    )

    print(
        f"Minimum history: {MIN_HISTORY} rows"
    )

    print(
        f"Delay: {delay:.2f} seconds"
    )

    print("=" * 70)

    results = []

    total = len(
        selected
    )

    for position, (_, row) in enumerate(
        selected.iterrows(),
        start=1,
    ):

        exchange = str(
            row["exchange"]
        ).strip()

        symbol = str(
            row["symbol"]
        ).strip()

        data_symbol = str(
            row["data_symbol"]
        ).strip()

        name = str(
            row["name"]
        ).strip()

        avg_volume = row[
            "avg_volume"
        ]

        print()
        print(
            f"[{position}/{total}] "
            f"{exchange} {symbol} "
            f"-> {data_symbol}"
        )

        result = {
            "exchange": exchange,
            "symbol": symbol,
            "name": name,
            "data_symbol": data_symbol,
            "avg_volume": avg_volume,
            "start_date_requested": start_date,
            "end_date_requested": end_date,
            "rows": 0,
            "first_date": "",
            "last_date": "",
            "historical_status": "ERROR",
            "error": "",
            "downloaded_at": datetime.now().isoformat(
                timespec="seconds"
            ),
        }

        try:

            data = get_historical_data(
                data_symbol,
                start_date=start_date,
                end_date=end_date,
                interval=INTERVAL,
                cache_hours=CACHE_HOURS,
                force_refresh=force_refresh,
                minimum_rows=MIN_HISTORY,
            )

            if data.empty:

                result["historical_status"] = (
                    "NO_DATA"
                )

                result["error"] = (
                    "Historical data returned empty."
                )

                print(
                    "  STATUS: NO_DATA"
                )

            else:

                result["rows"] = len(
                    data
                )

                result["first_date"] = (
                    data.index[0].date().isoformat()
                )

                result["last_date"] = (
                    data.index[-1].date().isoformat()
                )

                if len(data) >= MIN_HISTORY:

                    result[
                        "historical_status"
                    ] = "PASS"

                else:

                    result[
                        "historical_status"
                    ] = "INSUFFICIENT_HISTORY"

                print(
                    f"  Rows: {result['rows']}"
                )

                print(
                    f"  Range: "
                    f"{result['first_date']} "
                    f"-> "
                    f"{result['last_date']}"
                )

                print(
                    f"  STATUS: "
                    f"{result['historical_status']}"
                )

        except Exception as exc:

            result["historical_status"] = (
                "ERROR"
            )

            result["error"] = (
                f"{type(exc).__name__}: {exc}"
            )

            print(
                f"  ERROR: {result['error']}"
            )

        results.append(
            result
        )

        # Small pause between requests.
        if (
            position < total
            and delay > 0
        ):

            time.sleep(
                delay
            )

    result_data = pd.DataFrame(
        results
    )

    result_data.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    return result_data


# ============================================================
# SUMMARY
# ============================================================

def print_summary(
    results: pd.DataFrame,
) -> None:
    """
    Print a concise download summary.
    """

    print()
    print("=" * 70)
    print("HISTORICAL DOWNLOAD SUMMARY")
    print("=" * 70)

    if results.empty:

        print(
            "No stocks were processed."
        )

        return

    status_counts = (
        results[
            "historical_status"
        ]
        .value_counts()
    )

    print(
        f"Processed: {len(results)}"
    )

    for status, count in status_counts.items():

        print(
            f"{status}: {count}"
        )

    passed = results[
        results[
            "historical_status"
        ].eq("PASS")
    ]

    if not passed.empty:

        rows = pd.to_numeric(
            passed["rows"],
            errors="coerce",
        )

        print()
        print(
            f"PASS minimum rows: {MIN_HISTORY}"
        )

        print(
            f"PASS median rows: "
            f"{rows.median():.0f}"
        )

        print(
            f"PASS minimum rows observed: "
            f"{rows.min():.0f}"
        )

        print(
            f"PASS maximum rows observed: "
            f"{rows.max():.0f}"
        )

    print()
    print(
        f"Results saved to: {OUTPUT_FILE}"
    )

    print("=" * 70)


# ============================================================
# ENTRY POINT
# ============================================================

def main() -> None:

    args = parse_args()

    if args.limit < 0:

        raise ValueError(
            "--limit must be >= 0"
        )

    results = run_downloader(
        limit=args.limit,
        start_date=args.start,
        end_date=args.end,
        delay=args.delay,
        force_refresh=args.force_refresh,
    )

    print_summary(
        results
    )


if __name__ == "__main__":

    main()