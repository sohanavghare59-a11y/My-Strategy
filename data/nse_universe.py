"""
NSE Stock Universe
==================

Downloads the NSE equity security list and converts it
into a standardized DataFrame for the swing trading agent.

Primary NSE source:
    NSE Equity List CSV

Output columns:
    exchange
    instrument_type
    symbol
    name
    isin
    series
    data_symbol

Yahoo Finance mapping:
    NSE symbol -> SYMBOL.NS
"""

from __future__ import annotations

import os
from io import StringIO
from urllib.request import Request, urlopen

import pandas as pd


# ======================================================================
# NSE DATA SOURCE
# ======================================================================

NSE_EQUITY_URL = (
    "https://nsearchives.nseindia.com/"
    "content/equities/EQUITY_L.csv"
)


# ======================================================================
# HTTP SETTINGS
# ======================================================================

USER_AGENT = (
    "Mozilla/5.0 "
    "(Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 "
    "(KHTML, like Gecko) "
    "Chrome/139.0 Safari/537.36"
)


# ======================================================================
# DOWNLOAD NSE FILE
# ======================================================================

def download_nse_equity_csv() -> pd.DataFrame:
    """
    Download the NSE equity security list.

    Returns
    -------
    pandas.DataFrame
        Raw NSE equity data.
    """

    request = Request(
        NSE_EQUITY_URL,
        headers={
            "User-Agent": USER_AGENT,
            "Accept": (
                "text/csv,"
                "application/csv,"
                "application/octet-stream,"
                "*/*"
            ),
        },
    )

    try:
        with urlopen(
            request,
            timeout=30,
        ) as response:

            content = response.read()

    except Exception as exc:
        raise RuntimeError(
            "Failed to download NSE equity universe: "
            f"{exc}"
        ) from exc

    if not content:
        raise ValueError(
            "NSE returned an empty equity file."
        )

    try:
        text = content.decode(
            "utf-8-sig",
            errors="replace",
        )

        df = pd.read_csv(
            StringIO(text)
        )

    except Exception as exc:
        raise RuntimeError(
            "Failed to parse NSE equity CSV: "
            f"{exc}"
        ) from exc

    if df.empty:
        raise ValueError(
            "NSE equity universe is empty."
        )

    return df


# ======================================================================
# NORMALIZE NSE COLUMN NAMES
# ======================================================================

def normalize_nse_columns(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Normalize NSE column names.

    Example:

        "SYMBOL" -> "SYMBOL"
        "NAME OF COMPANY" -> "NAME_OF_COMPANY"
    """

    result = df.copy()

    result.columns = [
        str(column)
        .strip()
        .upper()
        .replace(" ", "_")
        for column in result.columns
    ]

    return result


# ======================================================================
# STANDARDIZE NSE UNIVERSE
# ======================================================================

def standardize_nse_universe(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Convert raw NSE data into the standard universe format.

    Standard output columns:

        exchange
        instrument_type
        symbol
        name
        isin
        series
        data_symbol
    """

    if df is None or df.empty:
        raise ValueError(
            "Cannot standardize an empty NSE universe."
        )

    result = normalize_nse_columns(df)

    # ------------------------------------------------------------------
    # Required column
    # ------------------------------------------------------------------

    if "SYMBOL" not in result.columns:
        raise ValueError(
            "NSE file is missing the SYMBOL column. "
            f"Available columns: {list(result.columns)}"
        )

    # ------------------------------------------------------------------
    # Find company-name column
    # ------------------------------------------------------------------

    name_column = None

    for candidate in (
        "NAME_OF_COMPANY",
        "NAME",
        "COMPANY_NAME",
    ):
        if candidate in result.columns:
            name_column = candidate
            break

    # ------------------------------------------------------------------
    # Find ISIN column
    # ------------------------------------------------------------------

    isin_column = None

    for candidate in (
        "ISIN_NUMBER",
        "ISIN",
    ):
        if candidate in result.columns:
            isin_column = candidate
            break

    # ------------------------------------------------------------------
    # Find series column
    # ------------------------------------------------------------------

    series_column = None

    for candidate in (
        "SERIES",
        "SERIES_NAME",
    ):
        if candidate in result.columns:
            series_column = candidate
            break

    # ------------------------------------------------------------------
    # IMPORTANT:
    #
    # Reset the source index before creating the standardized
    # DataFrame. This prevents pandas index-alignment problems.
    # ------------------------------------------------------------------

    result = result.reset_index(
        drop=True
    )

    # ------------------------------------------------------------------
    # Create standardized DataFrame
    # ------------------------------------------------------------------

    standardized = pd.DataFrame(
        index=result.index
    )

    # These are created with a matching index.
    # This prevents NaN values in exchange/instrument_type.
    standardized["exchange"] = pd.Series(
        "NSE",
        index=result.index,
        dtype="string",
    )

    standardized["instrument_type"] = pd.Series(
        "STOCK",
        index=result.index,
        dtype="string",
    )

    # ------------------------------------------------------------------
    # Symbol
    # ------------------------------------------------------------------

    standardized["symbol"] = (
        result["SYMBOL"]
        .fillna("")
        .astype(str)
        .str.strip()
        .str.upper()
    )

    # ------------------------------------------------------------------
    # Company name
    # ------------------------------------------------------------------

    if name_column is not None:

        standardized["name"] = (
            result[name_column]
            .fillna("")
            .astype(str)
            .str.strip()
        )

    else:

        standardized["name"] = ""

    # ------------------------------------------------------------------
    # ISIN
    # ------------------------------------------------------------------

    if isin_column is not None:

        standardized["isin"] = (
            result[isin_column]
            .fillna("")
            .astype(str)
            .str.strip()
            .str.upper()
        )

    else:

        standardized["isin"] = ""

    # ------------------------------------------------------------------
    # Series
    # ------------------------------------------------------------------

    if series_column is not None:

        standardized["series"] = (
            result[series_column]
            .fillna("")
            .astype(str)
            .str.strip()
            .str.upper()
        )

    else:

        standardized["series"] = ""

    # ------------------------------------------------------------------
    # Yahoo Finance symbol
    # ------------------------------------------------------------------

    standardized["data_symbol"] = (
        standardized["symbol"]
        + ".NS"
    )

    # ------------------------------------------------------------------
    # Remove invalid symbols
    # ------------------------------------------------------------------

    standardized = standardized[
        standardized["symbol"].notna()
    ]

    standardized = standardized[
        standardized["symbol"].str.len() > 0
    ]

    standardized = standardized[
        standardized["symbol"] != "NAN"
    ]

    # ------------------------------------------------------------------
    # Keep valid NSE equity series
    #
    # EQ = normal equity
    # BE/BZ = trade-to-trade equity series
    # SM/ST = SME-related equity series
    # ------------------------------------------------------------------

    if "series" in standardized.columns:

        series_values = (
            standardized["series"]
            .fillna("")
            .astype(str)
            .str.upper()
            .str.strip()
        )

        if series_values.ne("").any():

            standardized = standardized[
                series_values.isin(
                    [
                        "EQ",
                        "BE",
                        "BZ",
                        "SM",
                        "ST",
                    ]
                )
            ]

    # ------------------------------------------------------------------
    # Remove duplicate symbols
    # ------------------------------------------------------------------

    standardized = (
        standardized
        .drop_duplicates(
            subset=[
                "exchange",
                "instrument_type",
                "symbol",
            ],
            keep="first",
        )
        .reset_index(drop=True)
    )

    # ------------------------------------------------------------------
    # Final cleanup
    # ------------------------------------------------------------------

    standardized["exchange"] = (
        standardized["exchange"]
        .fillna("NSE")
        .astype(str)
        .str.upper()
        .str.strip()
    )

    standardized["instrument_type"] = (
        standardized["instrument_type"]
        .fillna("STOCK")
        .astype(str)
        .str.upper()
        .str.strip()
    )

    # ------------------------------------------------------------------
    # Sort alphabetically
    # ------------------------------------------------------------------

    standardized = (
        standardized
        .sort_values(
            by="symbol"
        )
        .reset_index(drop=True)
    )

    # ------------------------------------------------------------------
    # Ensure exact column order
    # ------------------------------------------------------------------

    standardized = standardized[
        [
            "exchange",
            "instrument_type",
            "symbol",
            "name",
            "isin",
            "series",
            "data_symbol",
        ]
    ]

    return standardized


# ======================================================================
# PUBLIC FUNCTION
# ======================================================================

def get_nse_stock_universe() -> pd.DataFrame:
    """
    Download and return the standardized NSE stock universe.
    """

    raw = download_nse_equity_csv()

    return standardize_nse_universe(
        raw
    )


# ======================================================================
# SAVE NSE UNIVERSE
# ======================================================================

def save_nse_universe(
    df: pd.DataFrame,
    output_path: str = "output/nse_universe.csv",
) -> str:
    """
    Save the standardized NSE universe to CSV.

    Returns
    -------
    str
        Output file path.
    """

    standardized = standardize_nse_universe(
        df
    )

    output_directory = os.path.dirname(
        output_path
    )

    if output_directory:
        os.makedirs(
            output_directory,
            exist_ok=True,
        )

    standardized.to_csv(
        output_path,
        index=False,
    )

    return output_path


# ======================================================================
# VALIDATION
# ======================================================================

def validate_nse_universe(
    df: pd.DataFrame,
) -> None:
    """
    Validate the standardized NSE universe.
    """

    if df is None:
        raise ValueError(
            "NSE universe is None."
        )

    if df.empty:
        raise ValueError(
            "NSE universe is empty."
        )

    required_columns = [
        "exchange",
        "instrument_type",
        "symbol",
        "name",
        "isin",
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
            "NSE universe is missing columns: "
            f"{missing}"
        )

    # ------------------------------------------------------------------
    # Check exchange values
    # ------------------------------------------------------------------

    if not (
        df["exchange"]
        .eq("NSE")
        .all()
    ):
        raise ValueError(
            "NSE universe contains invalid exchange values."
        )

    # ------------------------------------------------------------------
    # Check instrument type
    # ------------------------------------------------------------------

    if not (
        df["instrument_type"]
        .eq("STOCK")
        .all()
    ):
        raise ValueError(
            "NSE universe contains invalid instrument types."
        )

    # ------------------------------------------------------------------
    # Check symbols
    # ------------------------------------------------------------------

    if (
        df["symbol"]
        .isna()
        .any()
    ):
        raise ValueError(
            "NSE universe contains missing symbols."
        )

    if (
        df["data_symbol"]
        .isna()
        .any()
    ):
        raise ValueError(
            "NSE universe contains missing data symbols."
        )

    # ------------------------------------------------------------------
    # Check Yahoo suffix
    # ------------------------------------------------------------------

    invalid_yahoo_symbols = df[
        ~df["data_symbol"]
        .str.endswith(".NS")
    ]

    if not invalid_yahoo_symbols.empty:
        raise ValueError(
            "Some NSE symbols do not have the .NS Yahoo suffix."
        )


# ======================================================================
# TEST
# ======================================================================

def _run_test() -> None:
    """
    Test NSE universe download, standardization,
    validation and saving.
    """

    print("=" * 70)
    print("NSE STOCK UNIVERSE TEST")
    print("=" * 70)

    print()
    print("Downloading NSE equity universe...")

    try:

        universe = get_nse_stock_universe()

    except Exception as exc:

        print()
        print(
            "ERROR:",
            exc,
        )

        return

    # ------------------------------------------------------------------
    # Validate
    # ------------------------------------------------------------------

    print()
    print("Validating NSE universe...")

    try:

        validate_nse_universe(
            universe
        )

        print(
            "Validation: PASSED"
        )

    except Exception as exc:

        print(
            "Validation: FAILED"
        )

        print(
            "ERROR:",
            exc,
        )

        return

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------

    print()
    print("=" * 70)
    print("NSE UNIVERSE SUMMARY")
    print("=" * 70)

    print(
        "Total NSE stocks:",
        len(universe),
    )

    print(
        "Unique symbols:",
        universe["symbol"].nunique(),
    )

    print(
        "Unique ISINs:",
        universe["isin"].replace(
            "",
            pd.NA,
        ).nunique(),
    )

    # ------------------------------------------------------------------
    # Exchange check
    # ------------------------------------------------------------------

    print()
    print("=" * 70)
    print("EXCHANGE / INSTRUMENT CHECK")
    print("=" * 70)

    print(
        universe[
            [
                "exchange",
                "instrument_type",
            ]
        ]
        .drop_duplicates()
        .to_string(
            index=False
        )
    )

    # ------------------------------------------------------------------
    # Series counts
    # ------------------------------------------------------------------

    print()
    print("=" * 70)
    print("SERIES COUNTS")
    print("=" * 70)

    print(
        universe["series"]
        .value_counts()
        .to_string()
    )

    # ------------------------------------------------------------------
    # First 20
    # ------------------------------------------------------------------

    print()
    print("=" * 70)
    print("FIRST 20 STOCKS")
    print("=" * 70)

    print(
        universe.head(20).to_string(
            index=False
        )
    )

    # ------------------------------------------------------------------
    # Last 10
    # ------------------------------------------------------------------

    print()
    print("=" * 70)
    print("LAST 10 STOCKS")
    print("=" * 70)

    print(
        universe.tail(10).to_string(
            index=False
        )
    )

    # ------------------------------------------------------------------
    # Yahoo symbols
    # ------------------------------------------------------------------

    print()
    print("=" * 70)
    print("YAHOO FINANCE SYMBOL EXAMPLES")
    print("=" * 70)

    print(
        universe[
            [
                "symbol",
                "name",
                "data_symbol",
            ]
        ]
        .head(10)
        .to_string(
            index=False
        )
    )

    # ------------------------------------------------------------------
    # Check for NaN
    # ------------------------------------------------------------------

    print()
    print("=" * 70)
    print("NULL VALUE CHECK")
    print("=" * 70)

    null_counts = (
        universe.isna()
        .sum()
    )

    print(
        null_counts.to_string()
    )

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------

    print()
    print("=" * 70)
    print("SAVING NSE UNIVERSE")
    print("=" * 70)

    try:

        output_path = save_nse_universe(
            universe,
            "output/nse_universe.csv",
        )

        print(
            "Saved to:",
            output_path,
        )

    except Exception as exc:

        print(
            "WARNING: Could not save universe:",
            exc,
        )

    # ------------------------------------------------------------------
    # Final status
    # ------------------------------------------------------------------

    print()
    print("=" * 70)
    print("NSE STOCK UNIVERSE TEST COMPLETED")
    print("=" * 70)


# ======================================================================
# ENTRY POINT
# ======================================================================

if __name__ == "__main__":
    _run_test()