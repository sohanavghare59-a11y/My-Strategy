"""
BSE Yahoo Finance Full Universe Validator
==========================================

Validates BSE stocks against Yahoo Finance.

Input:
    output/bse_universe.csv

Outputs:
    output/bse_validated_universe.csv
    output/bse_invalid_universe.csv

The validator:
    1. Tries data_symbol.BO
    2. Tries symbol.BO
    3. Tries security_id.BO
    4. Requires at least MIN_ROWS daily candles
    5. Saves progress continuously
    6. Reuses previously validated results
"""

from __future__ import annotations

import time
from pathlib import Path

import pandas as pd
import yfinance as yf


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path.cwd()

INPUT_FILE = (
    PROJECT_ROOT
    / "output"
    / "bse_universe.csv"
)

VALID_OUTPUT_FILE = (
    PROJECT_ROOT
    / "output"
    / "bse_validated_universe.csv"
)

INVALID_OUTPUT_FILE = (
    PROJECT_ROOT
    / "output"
    / "bse_invalid_universe.csv"
)


# ============================================================
# FULL VALIDATION
# ============================================================

TEST_MODE = False

TEST_LIMIT = 100


# ============================================================
# YAHOO SETTINGS
# ============================================================

PERIOD = "1y"

INTERVAL = "1d"

MIN_ROWS = 30

DELAY = 0.20


# ============================================================
# PROGRESS SETTINGS
# ============================================================

SAVE_EVERY = 25


# ============================================================
# STRING HELPERS
# ============================================================

def clean_string(value) -> str:
    """
    Convert a value into a clean string.
    """

    if pd.isna(value):
        return ""

    return str(value).strip()


def normalize_yahoo_symbol(
    symbol: str,
) -> str:
    """
    Convert a BSE symbol into Yahoo format.

    Examples:

        TCS
        -> TCS.BO

        TCS.BO
        -> TCS.BO

        500570
        -> 500570.BO
    """

    symbol = clean_string(symbol)

    if not symbol:
        return ""

    symbol = symbol.upper()

    if symbol.endswith(".BO"):
        return symbol

    return f"{symbol}.BO"


# ============================================================
# CANDIDATE GENERATION
# ============================================================

def add_candidate(
    candidates: list[str],
    value,
) -> None:
    """
    Add a Yahoo candidate without duplicates.
    """

    value = clean_string(value)

    if not value:
        return

    candidate = normalize_yahoo_symbol(
        value
    )

    if (
        candidate
        and candidate not in candidates
    ):
        candidates.append(candidate)


def build_candidates(
    row: pd.Series,
) -> list[str]:
    """
    Build Yahoo Finance candidates.

    Priority:

        1. data_symbol
        2. symbol
        3. security_id
    """

    candidates: list[str] = []

    add_candidate(
        candidates,
        row.get(
            "data_symbol",
            "",
        ),
    )

    add_candidate(
        candidates,
        row.get(
            "symbol",
            "",
        ),
    )

    add_candidate(
        candidates,
        row.get(
            "security_id",
            "",
        ),
    )

    return candidates


# ============================================================
# DOWNLOAD DATA
# ============================================================

def download_candidate(
    yahoo_symbol: str,
) -> tuple[
    bool,
    pd.DataFrame | None,
    str,
]:
    """
    Download daily Yahoo Finance data.
    """

    try:

        data = yf.download(
            yahoo_symbol,
            period=PERIOD,
            interval=INTERVAL,
            auto_adjust=False,
            progress=False,
            threads=False,
        )

        if (
            data is None
            or data.empty
        ):
            return (
                False,
                None,
                "Yahoo returned no data",
            )

        # ----------------------------------------------------
        # Flatten MultiIndex columns
        # ----------------------------------------------------

        if isinstance(
            data.columns,
            pd.MultiIndex,
        ):

            data.columns = [
                column[0]
                if isinstance(
                    column,
                    tuple,
                )
                else column
                for column in data.columns
            ]

        # ----------------------------------------------------
        # Required columns
        # ----------------------------------------------------

        required_columns = [
            "Open",
            "High",
            "Low",
            "Close",
            "Volume",
        ]

        missing_columns = [
            column
            for column in required_columns
            if column not in data.columns
        ]

        if missing_columns:

            return (
                False,
                None,
                (
                    "Missing columns: "
                    f"{missing_columns}"
                ),
            )

        # ----------------------------------------------------
        # Remove rows without OHLC
        # ----------------------------------------------------

        data = data.dropna(
            subset=[
                "Open",
                "High",
                "Low",
                "Close",
            ]
        )

        # ----------------------------------------------------
        # Minimum history
        # ----------------------------------------------------

        if len(data) < MIN_ROWS:

            return (
                False,
                None,
                (
                    f"Only {len(data)} rows; "
                    f"minimum is {MIN_ROWS}"
                ),
            )

        return (
            True,
            data,
            "",
        )

    except Exception as exc:

        return (
            False,
            None,
            (
                f"{type(exc).__name__}: "
                f"{exc}"
            ),
        )


# ============================================================
# VALIDATE ONE STOCK
# ============================================================

def validate_row(
    row: pd.Series,
) -> tuple[
    bool,
    str,
    str,
    int,
    str,
    str,
]:
    """
    Validate one BSE stock.

    Returns:

        valid
        yahoo_symbol
        validation_method
        data_rows
        last_date
        error
    """

    candidates = build_candidates(
        row
    )

    if not candidates:

        return (
            False,
            "",
            "",
            0,
            "",
            "No Yahoo candidates",
        )

    errors: list[str] = []

    for index, candidate in enumerate(
        candidates
    ):

        (
            valid,
            data,
            error,
        ) = download_candidate(
            candidate
        )

        if (
            valid
            and data is not None
        ):

            if index == 0:
                method = "data_symbol"

            elif index == 1:
                method = "symbol"

            else:
                method = "security_id"

            try:

                last_date = str(
                    pd.to_datetime(
                        data.index[-1]
                    ).date()
                )

            except Exception:

                last_date = str(
                    data.index[-1]
                )

            return (
                True,
                candidate,
                method,
                len(data),
                last_date,
                "",
            )

        errors.append(
            f"{candidate}: {error}"
        )

        time.sleep(
            DELAY
        )

    return (
        False,
        "",
        "",
        0,
        "",
        " | ".join(errors),
    )


# ============================================================
# LOAD PREVIOUS PROGRESS
# ============================================================

def load_previous_results():
    """
    Load previously saved validation results.

    This allows the validator to resume.
    """

    previous_valid = {}

    previous_invalid = {}

    if VALID_OUTPUT_FILE.exists():

        try:

            valid_df = pd.read_csv(
                VALID_OUTPUT_FILE,
                dtype=str,
            )

            valid_df = valid_df.fillna("")

            for _, row in valid_df.iterrows():

                key = clean_string(
                    row.get(
                        "security_id",
                        "",
                    )
                )

                if key:
                    previous_valid[
                        key
                    ] = row.to_dict()

        except Exception as exc:

            print(
                "Warning: could not load "
                f"previous valid results: {exc}"
            )

    if INVALID_OUTPUT_FILE.exists():

        try:

            invalid_df = pd.read_csv(
                INVALID_OUTPUT_FILE,
                dtype=str,
            )

            invalid_df = invalid_df.fillna("")

            for _, row in invalid_df.iterrows():

                key = clean_string(
                    row.get(
                        "security_id",
                        "",
                    )
                )

                if key:
                    previous_invalid[
                        key
                    ] = row.to_dict()

        except Exception as exc:

            print(
                "Warning: could not load "
                f"previous invalid results: {exc}"
            )

    return (
        previous_valid,
        previous_invalid,
    )


# ============================================================
# SAVE RESULTS
# ============================================================

def save_results(
    valid_rows: list[dict],
    invalid_rows: list[dict],
) -> None:
    """
    Save current validation results.
    """

    valid_df = pd.DataFrame(
        valid_rows
    )

    invalid_df = pd.DataFrame(
        invalid_rows
    )

    VALID_OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    valid_df.to_csv(
        VALID_OUTPUT_FILE,
        index=False,
    )

    invalid_df.to_csv(
        INVALID_OUTPUT_FILE,
        index=False,
    )


# ============================================================
# MAIN
# ============================================================

def main() -> None:

    print("=" * 70)
    print(
        "BSE YAHOO FINANCE FULL UNIVERSE VALIDATOR"
    )
    print("=" * 70)

    print()

    # --------------------------------------------------------
    # Check input
    # --------------------------------------------------------

    if not INPUT_FILE.exists():

        raise FileNotFoundError(
            "BSE universe file not found:\n"
            f"{INPUT_FILE}"
        )

    # --------------------------------------------------------
    # Load universe
    # --------------------------------------------------------

    universe = pd.read_csv(
        INPUT_FILE,
        dtype=str,
    )

    universe = universe.fillna("")

    print(
        f"Total BSE records: {len(universe)}"
    )

    # --------------------------------------------------------
    # Test mode
    # --------------------------------------------------------

    if TEST_MODE:

        universe = (
            universe
            .head(TEST_LIMIT)
            .copy()
        )

        print(
            f"TEST MODE: {len(universe)} records"
        )

    else:

        print(
            "FULL MODE: validating all records"
        )

    print()

    # --------------------------------------------------------
    # Load previous results
    # --------------------------------------------------------

    (
        previous_valid,
        previous_invalid,
    ) = load_previous_results()

    previous_count = (
        len(previous_valid)
        + len(previous_invalid)
    )

    print(
        f"Previous saved results: "
        f"{previous_count}"
    )

    print()

    # --------------------------------------------------------
    # Result containers
    # --------------------------------------------------------

    valid_rows: list[dict] = []

    invalid_rows: list[dict] = []

    skipped_count = 0

    new_valid_count = 0

    new_invalid_count = 0

    total = len(universe)

    # --------------------------------------------------------
    # Validate
    # --------------------------------------------------------

    for position, (_, row) in enumerate(
        universe.iterrows(),
        start=1,
    ):

        security_id = clean_string(
            row.get(
                "security_id",
                "",
            )
        )

        symbol = clean_string(
            row.get(
                "symbol",
                "",
            )
        )

        # ----------------------------------------------------
        # Resume previous result
        # ----------------------------------------------------

        if security_id in previous_valid:

            valid_rows.append(
                previous_valid[
                    security_id
                ]
            )

            skipped_count += 1

            print(
                f"[{position:>4}/{total}] "
                f"{symbol:<20} "
                "SKIP -> already VALID"
            )

            continue

        if security_id in previous_invalid:

            invalid_rows.append(
                previous_invalid[
                    security_id
                ]
            )

            skipped_count += 1

            print(
                f"[{position:>4}/{total}] "
                f"{symbol:<20} "
                "SKIP -> already INVALID"
            )

            continue

        # ----------------------------------------------------
        # Build candidates
        # ----------------------------------------------------

        candidates = build_candidates(
            row
        )

        print(
            f"[{position:>4}/{total}] "
            f"{symbol:<20} "
            f"ID={security_id:<10}"
        )

        print(
            f"       Candidates: "
            f"{candidates}"
        )

        # ----------------------------------------------------
        # Validate
        # ----------------------------------------------------

        (
            valid,
            yahoo_symbol,
            validation_method,
            data_rows,
            last_date,
            error,
        ) = validate_row(
            row
        )

        result = row.to_dict()

        result[
            "yahoo_symbol"
        ] = yahoo_symbol

        result[
            "validation_method"
        ] = validation_method

        result[
            "data_rows"
        ] = data_rows

        result[
            "last_date"
        ] = last_date

        result[
            "validation_error"
        ] = error

        # ----------------------------------------------------
        # Valid
        # ----------------------------------------------------

        if valid:

            result[
                "data_symbol"
            ] = yahoo_symbol

            valid_rows.append(
                result
            )

            new_valid_count += 1

            print(
                f"       VALID -> "
                f"{yahoo_symbol} "
                f"[{validation_method}] "
                f"rows={data_rows} "
                f"last={last_date}"
            )

        # ----------------------------------------------------
        # Invalid
        # ----------------------------------------------------

        else:

            invalid_rows.append(
                result
            )

            new_invalid_count += 1

            print(
                f"       INVALID"
            )

            print(
                f"       {error}"
            )

        print()

        # ----------------------------------------------------
        # Save periodically
        # ----------------------------------------------------

        processed_new = (
            new_valid_count
            + new_invalid_count
        )

        if (
            processed_new > 0
            and processed_new % SAVE_EVERY == 0
        ):

            save_results(
                valid_rows,
                invalid_rows,
            )

            print(
                "       >>> PROGRESS SAVED <<<"
            )

            print()

        # ----------------------------------------------------
        # Delay
        # ----------------------------------------------------

        time.sleep(
            DELAY
        )

    # ========================================================
    # FINAL SAVE
    # ========================================================

    save_results(
        valid_rows,
        invalid_rows,
    )

    # ========================================================
    # SUMMARY
    # ========================================================

    valid_count = len(
        valid_rows
    )

    invalid_count = len(
        invalid_rows
    )

    processed_count = (
        valid_count
        + invalid_count
    )

    if processed_count > 0:

        success_rate = (
            valid_count
            / processed_count
            * 100
        )

    else:

        success_rate = 0.0

    print()

    print("=" * 70)
    print(
        "VALIDATION SUMMARY"
    )
    print("=" * 70)

    print(
        f"Universe records: "
        f"{total}"
    )

    print(
        f"Valid:            "
        f"{valid_count}"
    )

    print(
        f"Invalid:          "
        f"{invalid_count}"
    )

    print(
        f"Skipped/resumed:  "
        f"{skipped_count}"
    )

    print(
        f"New valid:        "
        f"{new_valid_count}"
    )

    print(
        f"New invalid:      "
        f"{new_invalid_count}"
    )

    print(
        f"Success rate:     "
        f"{success_rate:.2f}%"
    )

    # ========================================================
    # METHODS
    # ========================================================

    print()

    print(
        "VALIDATION METHODS"
    )

    print("-" * 70)

    if valid_count > 0:

        method_series = pd.Series(
            [
                row.get(
                    "validation_method",
                    "",
                )
                for row in valid_rows
            ]
        )

        method_counts = (
            method_series
            .value_counts()
        )

        for method, count in (
            method_counts.items()
        ):

            print(
                f"{method:<20} "
                f"{count}"
            )

    else:

        print(
            "No valid records."
        )

    # ========================================================
    # OUTPUTS
    # ========================================================

    print()

    print(
        "OUTPUT FILES"
    )

    print("-" * 70)

    print(
        f"Valid:   {VALID_OUTPUT_FILE}"
    )

    print(
        f"Invalid: {INVALID_OUTPUT_FILE}"
    )

    print()

    print("=" * 70)
    print(
        "BSE VALIDATION COMPLETE"
    )
    print("=" * 70)


if __name__ == "__main__":
    main()