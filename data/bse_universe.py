"""
Clean BSE stock universe builder.

Source
------
Dhan public instrument master.

For BSE cash-market equity stocks we use:

    SEM_EXM_EXCH_ID == BSE
    SEM_SEGMENT == E
    SEM_EXCH_INSTRUMENT_TYPE == ES

The important distinction is that:

    SEM_INSTRUMENT_NAME == EQUITY

is too broad and includes debt, ETFs, bonds and other
security types.

The Dhan exchange instrument type:

    ES

is used for ordinary equity shares.

Output
------
output/bse_raw_dhan_master.csv
output/bse_universe.csv
"""

from __future__ import annotations

from pathlib import Path
from typing import List

import pandas as pd


# ---------------------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------------------

DHAN_MASTER_URL = (
    "https://images.dhan.co/api-data/api-scrip-master.csv"
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]

OUTPUT_DIR = PROJECT_ROOT / "output"

RAW_OUTPUT_FILE = OUTPUT_DIR / "bse_raw_dhan_master.csv"
BSE_OUTPUT_FILE = OUTPUT_DIR / "bse_universe.csv"


# ---------------------------------------------------------------------
# HELPERS
# ---------------------------------------------------------------------

def clean_text(value) -> str:
    """Convert a value to a clean string."""
    if pd.isna(value):
        return ""

    return str(value).strip()


def find_column(
    df: pd.DataFrame,
    candidates: List[str],
) -> str | None:
    """
    Return the first matching column from a list of candidates.

    Matching is case-insensitive and ignores surrounding spaces.
    """
    normalized = {
        str(column).strip().upper(): column
        for column in df.columns
    }

    for candidate in candidates:
        key = candidate.strip().upper()

        if key in normalized:
            return normalized[key]

    return None


def normalize_yahoo_symbol(symbol: str) -> str:
    """
    Convert a BSE trading symbol into a Yahoo Finance symbol.

    Example:

        20MICRONS -> 20MICRONS.BO
    """
    symbol = clean_text(symbol)

    if not symbol:
        return ""

    symbol = symbol.upper()

    if symbol.endswith(".BO"):
        return symbol

    return f"{symbol}.BO"


# ---------------------------------------------------------------------
# DOWNLOAD
# ---------------------------------------------------------------------

def download_dhan_master() -> pd.DataFrame:
    """Download the Dhan public instrument master."""
    print("Downloading Dhan instrument master...")

    df = pd.read_csv(
        DHAN_MASTER_URL,
        dtype=str,
        low_memory=False,
    )

    df.columns = [
        str(column).strip()
        for column in df.columns
    ]

    print(f"Downloaded rows: {len(df)}")
    print(f"Downloaded columns: {len(df.columns)}")

    return df


# ---------------------------------------------------------------------
# BUILD BSE UNIVERSE
# ---------------------------------------------------------------------

def build_bse_universe(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build a clean BSE stock universe.

    Main filter:

        Exchange       = BSE
        Segment        = E
        Instrument type = ES
    """

    required_columns = [
        "SEM_EXM_EXCH_ID",
        "SEM_SEGMENT",
        "SEM_EXCH_INSTRUMENT_TYPE",
    ]

    missing = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing:
        raise ValueError(
            "Required Dhan columns are missing: "
            + ", ".join(missing)
        )

    # -------------------------------------------------------------
    # Normalize source columns
    # -------------------------------------------------------------

    for column in required_columns:
        df[column] = (
            df[column]
            .fillna("")
            .astype(str)
            .str.strip()
            .str.upper()
        )

    # -------------------------------------------------------------
    # BSE CASH SEGMENT
    # -------------------------------------------------------------

    bse = df[
        df["SEM_EXM_EXCH_ID"].eq("BSE")
        & df["SEM_SEGMENT"].eq("E")
    ].copy()

    print()
    print("=" * 70)
    print("BSE SOURCE FILTER")
    print("=" * 70)

    print(
        f"After BSE + segment E filter : {len(bse)}"
    )

    # -------------------------------------------------------------
    # EXACT EQUITY SHARE FILTER
    # -------------------------------------------------------------

    bse = bse[
        bse["SEM_EXCH_INSTRUMENT_TYPE"].eq("ES")
    ].copy()

    print(
        f"After ES equity filter       : {len(bse)}"
    )

    if bse.empty:
        raise ValueError(
            "No BSE ES equity records were found."
        )

    # -------------------------------------------------------------
    # LOCATE IMPORTANT COLUMNS
    # -------------------------------------------------------------

    trading_symbol_col = find_column(
        bse,
        [
            "SEM_TRADING_SYMBOL",
        ],
    )

    custom_symbol_col = find_column(
        bse,
        [
            "SEM_CUSTOM_SYMBOL",
        ],
    )

    security_id_col = find_column(
        bse,
        [
            "SEM_SMST_SECURITY_ID",
        ],
    )

    series_col = find_column(
        bse,
        [
            "SEM_SERIES",
        ],
    )

    instrument_name_col = find_column(
        bse,
        [
            "SEM_CUSTOM_SYMBOL",
            "SEM_TRADING_SYMBOL",
        ],
    )

    if trading_symbol_col is None:
        raise ValueError(
            "SEM_TRADING_SYMBOL column not found."
        )

    if security_id_col is None:
        raise ValueError(
            "SEM_SMST_SECURITY_ID column not found."
        )

    # -------------------------------------------------------------
    # CREATE STANDARDIZED UNIVERSE
    # -------------------------------------------------------------

    result = pd.DataFrame(index=bse.index)

    result["exchange"] = "BSE"

    result["instrument_type"] = "STOCK"

    result["symbol"] = (
        bse[trading_symbol_col]
        .fillna("")
        .astype(str)
        .str.strip()
        .str.upper()
    )

    # Prefer custom company name when available.
    if custom_symbol_col is not None:
        custom_name = (
            bse[custom_symbol_col]
            .fillna("")
            .astype(str)
            .str.strip()
        )
    else:
        custom_name = pd.Series(
            "",
            index=bse.index,
            dtype="object",
        )

    trading_name = (
        bse[trading_symbol_col]
        .fillna("")
        .astype(str)
        .str.strip()
    )

    result["name"] = custom_name.where(
        custom_name.ne(""),
        trading_name,
    )

    # Dhan master currently does not expose a reliable ISIN field
    # in this source, so leave it empty rather than inventing one.
    result["isin"] = ""

    result["security_id"] = (
        bse[security_id_col]
        .fillna("")
        .astype(str)
        .str.strip()
    )

    if series_col is not None:
        result["series"] = (
            bse[series_col]
            .fillna("")
            .astype(str)
            .str.strip()
            .str.upper()
        )
    else:
        result["series"] = ""

    result["data_symbol"] = (
        result["symbol"]
        .map(normalize_yahoo_symbol)
    )

    # -------------------------------------------------------------
    # REMOVE INVALID SYMBOLS
    # -------------------------------------------------------------

    before_symbol_filter = len(result)

    result = result[
        result["symbol"].ne("")
        & result["data_symbol"].ne("")
    ].copy()

    removed_empty_symbols = (
        before_symbol_filter - len(result)
    )

    print(
        f"Removed empty trading symbols : "
        f"{removed_empty_symbols}"
    )

    # -------------------------------------------------------------
    # REMOVE DUPLICATE TRADING SYMBOLS
    # -------------------------------------------------------------

    before_dedup = len(result)

    result = result.drop_duplicates(
        subset=["symbol"],
        keep="first",
    ).copy()

    removed_duplicates = (
        before_dedup - len(result)
    )

    print(
        f"Removed duplicate symbols      : "
        f"{removed_duplicates}"
    )

    # -------------------------------------------------------------
    # SORT
    # -------------------------------------------------------------

    result = result.sort_values(
        by=["symbol"],
        kind="stable",
    ).reset_index(drop=True)

    return result


# ---------------------------------------------------------------------
# VALIDATION
# ---------------------------------------------------------------------

def validate_bse_universe(
    df: pd.DataFrame,
) -> None:
    """Validate the standardized BSE universe."""

    required_columns = [
        "exchange",
        "instrument_type",
        "symbol",
        "name",
        "isin",
        "security_id",
        "series",
        "data_symbol",
    ]

    missing = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing:
        raise ValueError(
            "Missing output columns: "
            + ", ".join(missing)
        )

    if df.empty:
        raise ValueError(
            "BSE universe is empty."
        )

    if not df["symbol"].is_unique:
        raise ValueError(
            "BSE symbols are not unique."
        )

    if not df["security_id"].is_unique:
        raise ValueError(
            "BSE security IDs are not unique."
        )

    if not df["exchange"].eq("BSE").all():
        raise ValueError(
            "Universe contains non-BSE records."
        )

    if not df["instrument_type"].eq("STOCK").all():
        raise ValueError(
            "Universe contains non-STOCK records."
        )

    for column in [
        "symbol",
        "name",
        "security_id",
        "data_symbol",
    ]:
        if df[column].fillna("").astype(str).str.strip().eq("").any():
            raise ValueError(
                f"Column '{column}' contains empty values."
            )


# ---------------------------------------------------------------------
# SUMMARY
# ---------------------------------------------------------------------

def print_summary(
    df: pd.DataFrame,
) -> None:
    """Print BSE universe summary."""

    print()
    print("=" * 70)
    print("CLEAN BSE UNIVERSE SUMMARY")
    print("=" * 70)

    print(
        f"Total BSE stock records: {len(df)}"
    )

    print(
        f"Unique symbols:          "
        f"{df['symbol'].nunique()}"
    )

    print(
        f"Unique security IDs:     "
        f"{df['security_id'].nunique()}"
    )

    print(
        f"Unique data symbols:     "
        f"{df['data_symbol'].nunique()}"
    )

    print()
    print("=" * 70)
    print("EXCHANGE / INSTRUMENT")
    print("=" * 70)

    print(
        df[
            ["exchange", "instrument_type"]
        ]
        .drop_duplicates()
        .to_string(index=False)
    )

    print()
    print("=" * 70)
    print("SERIES COUNTS")
    print("=" * 70)

    print(
        df["series"]
        .replace("", "EMPTY")
        .value_counts()
        .to_string()
    )

    print()
    print("=" * 70)
    print("FIRST 50 STOCKS")
    print("=" * 70)

    display_columns = [
        "exchange",
        "instrument_type",
        "symbol",
        "name",
        "isin",
        "security_id",
        "series",
        "data_symbol",
    ]

    print(
        df[display_columns]
        .head(50)
        .to_string(index=False)
    )

    print()
    print("=" * 70)
    print("EMPTY VALUE CHECK")
    print("=" * 70)

    for column in display_columns:
        empty_count = (
            df[column]
            .fillna("")
            .astype(str)
            .str.strip()
            .eq("")
            .sum()
        )

        print(
            f"{column:<17}: {empty_count}"
        )


# ---------------------------------------------------------------------
# TEST
# ---------------------------------------------------------------------

def run_test() -> pd.DataFrame:
    """Run complete BSE universe builder test."""

    print("=" * 70)
    print("CLEAN BSE STOCK UNIVERSE BUILDER")
    print("=" * 70)

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # -------------------------------------------------------------
    # Download raw master
    # -------------------------------------------------------------

    raw_df = download_dhan_master()

    raw_df.to_csv(
        RAW_OUTPUT_FILE,
        index=False,
    )

    print(
        f"Raw source saved to: {RAW_OUTPUT_FILE}"
    )

    # -------------------------------------------------------------
    # Build clean universe
    # -------------------------------------------------------------

    bse_df = build_bse_universe(
        raw_df
    )

    # -------------------------------------------------------------
    # Validate
    # -------------------------------------------------------------

    print()
    print("Validating cleaned BSE universe...")

    validate_bse_universe(
        bse_df
    )

    print("Validation: PASSED")

    # -------------------------------------------------------------
    # Summary
    # -------------------------------------------------------------

    print_summary(
        bse_df
    )

    # -------------------------------------------------------------
    # Save
    # -------------------------------------------------------------

    bse_df.to_csv(
        BSE_OUTPUT_FILE,
        index=False,
    )

    print()
    print(
        f"Saved BSE universe to: {BSE_OUTPUT_FILE}"
    )

    print()
    print("=" * 70)
    print("BSE UNIVERSE BUILD COMPLETED")
    print("=" * 70)

    return bse_df


# ---------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------

if __name__ == "__main__":
    run_test()