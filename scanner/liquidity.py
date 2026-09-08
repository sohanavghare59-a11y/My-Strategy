"""
Daily liquidity filter for the swing trading agent.

Stocks:
    - Minimum latest close: configurable
    - Minimum 20-day average volume: configurable

Indices:
    - Always retained

The filter uses the master universe and Yahoo Finance daily data.
"""

from pathlib import Path

import pandas as pd

from config.settings import (
    VOLUME_AVG_PERIOD,
)
from data.market_data import get_daily_data


PROJECT_ROOT = Path.cwd()
OUTPUT_DIR = PROJECT_ROOT / "output"

MASTER_FILE = OUTPUT_DIR / "master_universe.csv"
LIQUIDITY_FILE = OUTPUT_DIR / "liquid_universe.csv"

MIN_PRICE = 50.0
MIN_AVG_VOLUME = 100_000

LOOKBACK_PERIOD = "3mo"


def check_stock_liquidity(data):
    """
    Check whether a stock passes the liquidity rules.
    """

    if data is None or data.empty:
        return False, {
            "latest_close": None,
            "avg_volume": None,
            "liquid": False,
            "reason": "NO_DATA",
        }

    required = ["Close", "Volume"]

    for column in required:
        if column not in data.columns:
            return False, {
                "latest_close": None,
                "avg_volume": None,
                "liquid": False,
                "reason": f"MISSING_{column}",
            }

    data = data.copy()

    data["Close"] = pd.to_numeric(
        data["Close"],
        errors="coerce",
    )

    data["Volume"] = pd.to_numeric(
        data["Volume"],
        errors="coerce",
    )

    data = data.dropna(
        subset=["Close", "Volume"]
    )

    if data.empty:
        return False, {
            "latest_close": None,
            "avg_volume": None,
            "liquid": False,
            "reason": "NO_VALID_DATA",
        }

    latest_close = float(data["Close"].iloc[-1])

    avg_volume = float(
        data["Volume"]
        .tail(VOLUME_AVG_PERIOD)
        .mean()
    )

    if latest_close < MIN_PRICE:
        return False, {
            "latest_close": latest_close,
            "avg_volume": avg_volume,
            "liquid": False,
            "reason": "PRICE_TOO_LOW",
        }

    if avg_volume < MIN_AVG_VOLUME:
        return False, {
            "latest_close": latest_close,
            "avg_volume": avg_volume,
            "liquid": False,
            "reason": "VOLUME_TOO_LOW",
        }

    return True, {
        "latest_close": latest_close,
        "avg_volume": avg_volume,
        "liquid": True,
        "reason": "PASS",
    }


def run_liquidity_filter():
    """
    Run the liquidity filter across the master universe.
    """

    print("=" * 60)
    print("LIQUIDITY FILTER")
    print("=" * 60)

    if not MASTER_FILE.exists():
        raise FileNotFoundError(
            f"Missing master universe: {MASTER_FILE}"
        )

    universe = pd.read_csv(
        MASTER_FILE,
        dtype=str,
    )

    print(f"Master instruments: {len(universe)}")

    results = []

    for index, row in universe.iterrows():

        exchange = str(row["exchange"])
        instrument_type = str(row["instrument_type"])
        symbol = str(row["symbol"])
        data_symbol = str(row["data_symbol"])

        print(
            f"[{index + 1}/{len(universe)}] "
            f"{exchange} {instrument_type} "
            f"{symbol}"
        )

        if instrument_type == "INDEX":
            result = row.to_dict()

            result["latest_close"] = None
            result["avg_volume"] = None
            result["liquidity_status"] = "INDEX"
            result["liquidity_reason"] = "INDEX_RETAINED"

            results.append(result)
            continue

        try:
            data = get_daily_data(
                data_symbol,
                period=LOOKBACK_PERIOD,
            )

            passed, details = check_stock_liquidity(
                data
            )

            result = row.to_dict()

            result["latest_close"] = details[
                "latest_close"
            ]

            result["avg_volume"] = details[
                "avg_volume"
            ]

            result["liquidity_status"] = (
                "PASS" if passed else "FAIL"
            )

            result["liquidity_reason"] = details[
                "reason"
            ]

            results.append(result)

        except Exception as exc:

            result = row.to_dict()

            result["latest_close"] = None
            result["avg_volume"] = None
            result["liquidity_status"] = "FAIL"
            result["liquidity_reason"] = (
                f"ERROR: {exc}"
            )

            results.append(result)

    result_df = pd.DataFrame(results)

    result_df.to_csv(
        LIQUIDITY_FILE,
        index=False,
    )

    print()
    print("=" * 60)
    print("LIQUIDITY FILTER COMPLETE")
    print("=" * 60)

    print(
        f"Total instruments: {len(result_df)}"
    )

    print()
    print(
        result_df[
            "liquidity_status"
        ].value_counts()
    )

    print()
    print(
        f"Saved to: {LIQUIDITY_FILE}"
    )

    return result_df


if __name__ == "__main__":
    run_liquidity_filter()