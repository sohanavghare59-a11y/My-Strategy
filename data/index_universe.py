"""
NSE/BSE Index Universe
======================

Builds the index universe used by the swing-trading agent.

Only indices with a verified Yahoo Finance daily-data symbol
are included.

NSE:
    NIFTY 50
    NIFTY BANK
    NIFTY NEXT 50
    NIFTY 100
    NIFTY 200
    NIFTY 500
    NIFTY MIDCAP 150
    NIFTY INDIA VIX

BSE:
    SENSEX
    BSE 100
    BSE 200
    BSE 500

The file can be used independently:

    python -m data.index_universe
"""

from pathlib import Path
from typing import List

import pandas as pd


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_ROOT / "output"

OUTPUT_FILE = OUTPUT_DIR / "index_universe.csv"


# ============================================================
# INDEX DEFINITIONS
# ============================================================

# ------------------------------------------------------------
# NSE INDICES
# ------------------------------------------------------------

NSE_INDICES = [
    {
        "exchange": "NSE",
        "instrument_type": "INDEX",
        "symbol": "NIFTY 50",
        "name": "NIFTY 50",
        "data_symbol": "^NSEI",
    },
    {
        "exchange": "NSE",
        "instrument_type": "INDEX",
        "symbol": "NIFTY BANK",
        "name": "NIFTY Bank",
        "data_symbol": "^NSEBANK",
    },
    {
        "exchange": "NSE",
        "instrument_type": "INDEX",
        "symbol": "NIFTY NEXT 50",
        "name": "NIFTY Next 50",
        "data_symbol": "^NSMIDCP",
    },
    {
        "exchange": "NSE",
        "instrument_type": "INDEX",
        "symbol": "NIFTY 100",
        "name": "NIFTY 100",
        "data_symbol": "^CNX100",
    },
    {
        "exchange": "NSE",
        "instrument_type": "INDEX",
        "symbol": "NIFTY 200",
        "name": "NIFTY 200",
        "data_symbol": "^CNX200",
    },
    {
        "exchange": "NSE",
        "instrument_type": "INDEX",
        "symbol": "NIFTY 500",
        "name": "NIFTY 500",
        "data_symbol": "^CRSLDX",
    },
    {
        "exchange": "NSE",
        "instrument_type": "INDEX",
        "symbol": "NIFTY MIDCAP 150",
        "name": "NIFTY Midcap 150",
        "data_symbol": "^NSEMDCP50",
    },
    {
        "exchange": "NSE",
        "instrument_type": "INDEX",
        "symbol": "NIFTY INDIA VIX",
        "name": "India VIX",
        "data_symbol": "^INDIAVIX",
    },
]


# ------------------------------------------------------------
# BSE INDICES
# ------------------------------------------------------------

# Verified against the current yfinance environment.
#
# The previous symbols:
#     ^BSE100
#     ^BSE200
#     ^BSE500
#
# returned no data.
#
# The working Yahoo symbols are:
#     BSE-100.BO
#     BSE-200.BO
#     BSE-500.BO
#
# SENSEX remains:
#     ^BSESN
#
# BSE MidCap and BSE SmallCap are intentionally excluded for now
# because the candidate Yahoo symbols tested did not provide a
# sufficiently usable historical daily series.

BSE_INDICES = [
    {
        "exchange": "BSE",
        "instrument_type": "INDEX",
        "symbol": "SENSEX",
        "name": "BSE SENSEX",
        "data_symbol": "^BSESN",
    },
    {
        "exchange": "BSE",
        "instrument_type": "INDEX",
        "symbol": "BSE 100",
        "name": "BSE 100",
        "data_symbol": "BSE-100.BO",
    },
    {
        "exchange": "BSE",
        "instrument_type": "INDEX",
        "symbol": "BSE 200",
        "name": "BSE 200",
        "data_symbol": "BSE-200.BO",
    },
    {
        "exchange": "BSE",
        "instrument_type": "INDEX",
        "symbol": "BSE 500",
        "name": "BSE 500",
        "data_symbol": "BSE-500.BO",
    },
]


# ============================================================
# BUILD INDEX UNIVERSE
# ============================================================

def build_index_universe() -> pd.DataFrame:
    """
    Build the combined NSE + BSE index universe.
    """

    rows: List[dict] = []

    rows.extend(NSE_INDICES)
    rows.extend(BSE_INDICES)

    df = pd.DataFrame(rows)

    required_columns = [
        "exchange",
        "instrument_type",
        "symbol",
        "name",
        "data_symbol",
    ]

    df = df[required_columns].copy()

    # Clean strings.
    for column in required_columns:
        df[column] = (
            df[column]
            .astype(str)
            .str.strip()
        )

    # Remove duplicate index definitions.
    df = (
        df.drop_duplicates(
            subset=[
                "exchange",
                "instrument_type",
                "symbol",
            ]
        )
        .reset_index(drop=True)
    )

    return df


# ============================================================
# VALIDATION
# ============================================================

def validate_index_universe(
    df: pd.DataFrame,
) -> bool:
    """
    Validate the index universe.
    """

    required_columns = [
        "exchange",
        "instrument_type",
        "symbol",
        "name",
        "data_symbol",
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}"
        )

    if df.empty:
        raise ValueError(
            "Index universe is empty."
        )

    if df["symbol"].isna().any():
        raise ValueError(
            "Index symbol contains null values."
        )

    if df["data_symbol"].isna().any():
        raise ValueError(
            "Index data_symbol contains null values."
        )

    if (
        df["symbol"]
        .astype(str)
        .str.strip()
        .eq("")
        .any()
    ):
        raise ValueError(
            "Index symbol contains empty values."
        )

    if (
        df["data_symbol"]
        .astype(str)
        .str.strip()
        .eq("")
        .any()
    ):
        raise ValueError(
            "Index data_symbol contains empty values."
        )

    invalid_exchange = ~df["exchange"].isin(
        ["NSE", "BSE"]
    )

    if invalid_exchange.any():
        raise ValueError(
            "Invalid exchange values found."
        )

    invalid_type = (
        df["instrument_type"] != "INDEX"
    )

    if invalid_type.any():
        raise ValueError(
            "Invalid instrument_type found."
        )

    return True


# ============================================================
# SAVE
# ============================================================

def save_index_universe(
    df: pd.DataFrame,
    output_file: Path = OUTPUT_FILE,
) -> Path:
    """
    Save the index universe to CSV.
    """

    validate_index_universe(df)

    output_file.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    df.to_csv(
        output_file,
        index=False,
    )

    return output_file


# ============================================================
# LOAD
# ============================================================

def load_index_universe(
    input_file: Path = OUTPUT_FILE,
) -> pd.DataFrame:
    """
    Load the previously saved index universe.
    """

    if not input_file.exists():
        raise FileNotFoundError(
            f"Index universe file not found: {input_file}"
        )

    df = pd.read_csv(input_file)

    validate_index_universe(df)

    return df


# ============================================================
# TEST
# ============================================================

def run_test() -> None:
    """
    Run index-universe validation test.
    """

    print("=" * 70)
    print("NSE/BSE INDEX UNIVERSE TEST")
    print("=" * 70)

    print()
    print("Building index universe...")

    df = build_index_universe()

    print()
    print("Validating index universe...")

    validate_index_universe(df)

    print("Validation: PASSED")

    print()
    print("=" * 70)
    print("INDEX UNIVERSE SUMMARY")
    print("=" * 70)

    print(f"Total indices: {len(df)}")

    print(
        f"NSE indices:   "
        f"{(df['exchange'] == 'NSE').sum()}"
    )

    print(
        f"BSE indices:   "
        f"{(df['exchange'] == 'BSE').sum()}"
    )

    print()
    print("=" * 70)
    print("INDEX LIST")
    print("=" * 70)

    print(
        df.to_string(index=False)
    )

    print()
    print("=" * 70)
    print("NULL VALUE CHECK")
    print("=" * 70)

    print(
        df.isna().sum().to_string()
    )

    print()
    print("=" * 70)
    print("SAVING INDEX UNIVERSE")
    print("=" * 70)

    output_path = save_index_universe(df)

    print(f"Saved to: {output_path}")

    print()
    print("=" * 70)
    print("SAVED FILE CHECK")
    print("=" * 70)

    print(
        f"File exists: "
        f"{output_path.exists()}"
    )

    saved_df = pd.read_csv(
        output_path
    )

    print(
        f"Saved rows: "
        f"{len(saved_df)}"
    )

    print(
        f"Saved columns: "
        f"{list(saved_df.columns)}"
    )

    print()
    print("=" * 70)
    print("INDEX UNIVERSE TEST COMPLETED")
    print("=" * 70)


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    run_test()