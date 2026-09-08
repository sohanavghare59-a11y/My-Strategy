"""
Daily market data utilities.

Provides:
    - Yahoo Finance daily data
    - OHLCV normalization
    - Separate cache files for different periods
    - Explicit historical date-range downloads
    - Cache validation
    - Latest daily row helpers

Important:
    Backtesting requests such as period="2y" must never reuse
    a shorter cache such as period="3mo".

Historical backtesting can also request explicit dates, for example:

    start_date="2021-01-01"
    end_date="2026-09-06"

Date-range caches are kept separate from normal period caches.
"""

from __future__ import annotations

import os
import re
import time
from datetime import datetime, timedelta
from typing import Optional

import pandas as pd
import yfinance as yf


# ============================================================
# CONFIGURATION
# ============================================================

CACHE_DIRECTORY = os.path.join(
    "data",
    "cache",
)

DEFAULT_PERIOD = "2y"
DEFAULT_INTERVAL = "1d"

DEFAULT_CACHE_HOURS = 4

# Minimum expected rows for common Yahoo periods.
# These are sanity checks only.
EXPECTED_MIN_ROWS = {
    "1mo": 15,
    "3mo": 45,
    "6mo": 90,
    "1y": 180,
    "2y": 400,
    "5y": 900,
}

# Explicit historical ranges used by long-history research.
HISTORICAL_DEFAULT_START = "2021-01-01"
HISTORICAL_DEFAULT_END = "2026-09-06"

# Minimum rows for an explicit historical range.
# This is intentionally conservative because different stocks
# have different listing dates.
HISTORICAL_MIN_ROWS = 400


# ============================================================
# DIRECTORY
# ============================================================

def ensure_cache_directory() -> None:
    """
    Create cache directory if it does not exist.
    """

    os.makedirs(
        CACHE_DIRECTORY,
        exist_ok=True,
    )


# ============================================================
# SYMBOL NORMALIZATION
# ============================================================

def normalize_symbol(symbol: str) -> str:
    """
    Normalize a Yahoo Finance symbol.
    """

    if symbol is None:
        raise ValueError(
            "Symbol cannot be None."
        )

    symbol = str(symbol).strip().upper()

    if not symbol:
        raise ValueError(
            "Symbol cannot be empty."
        )

    return symbol


# ============================================================
# PERIOD / DATE NORMALIZATION
# ============================================================

def normalize_date_string(
    value: str,
    name: str = "date",
) -> str:
    """
    Validate and normalize a date string.

    Accepted input:
        YYYY-MM-DD

    Returns:
        YYYY-MM-DD
    """

    if value is None:
        raise ValueError(
            f"{name} cannot be None."
        )

    text = str(value).strip()

    try:
        parsed = datetime.strptime(
            text,
            "%Y-%m-%d",
        )

    except ValueError as exc:

        raise ValueError(
            f"{name} must use YYYY-MM-DD format. "
            f"Received {value!r}."
        ) from exc

    return parsed.strftime(
        "%Y-%m-%d"
    )


def validate_date_range(
    start_date: str,
    end_date: str,
) -> tuple[str, str]:
    """
    Validate an explicit historical date range.

    Yahoo Finance uses an exclusive end date.
    """

    start = normalize_date_string(
        start_date,
        "start_date",
    )

    end = normalize_date_string(
        end_date,
        "end_date",
    )

    start_dt = datetime.strptime(
        start,
        "%Y-%m-%d",
    )

    end_dt = datetime.strptime(
        end,
        "%Y-%m-%d",
    )

    if end_dt <= start_dt:

        raise ValueError(
            "end_date must be later than start_date. "
            f"Received start_date={start}, "
            f"end_date={end}."
        )

    return start, end


# ============================================================
# CACHE FILENAME HELPERS
# ============================================================

def _safe_cache_component(
    value: str,
) -> str:
    """
    Make a value safe for use inside a filename.
    """

    value = str(value).strip()

    value = (
        value
        .replace("/", "_")
        .replace("\\", "_")
        .replace(":", "_")
        .replace("*", "_")
        .replace("?", "_")
        .replace('"', "_")
        .replace("<", "_")
        .replace(">", "_")
        .replace("|", "_")
    )

    value = re.sub(
        r"[^A-Za-z0-9_.=-]+",
        "_",
        value,
    )

    return value


def symbol_cache_filename(
    symbol: str,
    period: str = DEFAULT_PERIOD,
    interval: str = DEFAULT_INTERVAL,
) -> str:
    """
    Create a unique period-based cache filename.

    Example:

        RELIANCE.NS__2y__1d.csv
    """

    symbol = normalize_symbol(
        symbol
    )

    safe_symbol = _safe_cache_component(
        symbol
    )

    period_safe = _safe_cache_component(
        str(period).strip().lower()
    )

    interval_safe = _safe_cache_component(
        str(interval).strip().lower()
    )

    return (
        f"{safe_symbol}"
        f"__{period_safe}"
        f"__{interval_safe}.csv"
    )


def historical_cache_filename(
    symbol: str,
    start_date: str,
    end_date: str,
    interval: str = DEFAULT_INTERVAL,
) -> str:
    """
    Create a unique explicit-date historical cache filename.

    Example:

        RELIANCE.NS__from_2021-01-01__to_2026-09-06__1d.csv
    """

    symbol = normalize_symbol(
        symbol
    )

    start_date, end_date = validate_date_range(
        start_date,
        end_date,
    )

    safe_symbol = _safe_cache_component(
        symbol
    )

    interval_safe = _safe_cache_component(
        str(interval).strip().lower()
    )

    return (
        f"{safe_symbol}"
        f"__from_{start_date}"
        f"__to_{end_date}"
        f"__{interval_safe}.csv"
    )


# ============================================================
# CACHE PATHS
# ============================================================

def get_cache_path(
    symbol: str,
    period: str = DEFAULT_PERIOD,
    interval: str = DEFAULT_INTERVAL,
) -> str:
    """
    Return complete period-based cache path.
    """

    ensure_cache_directory()

    return os.path.join(
        CACHE_DIRECTORY,
        symbol_cache_filename(
            symbol,
            period,
            interval,
        ),
    )


def get_historical_cache_path(
    symbol: str,
    start_date: str,
    end_date: str,
    interval: str = DEFAULT_INTERVAL,
) -> str:
    """
    Return complete explicit-date cache path.
    """

    ensure_cache_directory()

    return os.path.join(
        CACHE_DIRECTORY,
        historical_cache_filename(
            symbol,
            start_date,
            end_date,
            interval,
        ),
    )


# ============================================================
# CACHE VALIDITY
# ============================================================

def is_cache_valid(
    path: str,
    cache_hours: int = DEFAULT_CACHE_HOURS,
) -> bool:
    """
    Return True if cache exists and is newer than cache_hours.
    """

    if not os.path.exists(path):
        return False

    try:

        modified_time = datetime.fromtimestamp(
            os.path.getmtime(path)
        )

        age = (
            datetime.now()
            - modified_time
        )

        return age < timedelta(
            hours=cache_hours
        )

    except OSError:

        return False


# ============================================================
# OHLCV NORMALIZATION
# ============================================================

def _flatten_columns(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Flatten Yahoo Finance MultiIndex columns.
    """

    if not isinstance(
        df.columns,
        pd.MultiIndex,
    ):
        return df

    flattened = []

    for column in df.columns:

        parts = [
            str(part)
            for part in column
            if str(part).strip()
        ]

        flattened.append(
            parts[0]
            if parts
            else ""
        )

    result = df.copy()

    result.columns = flattened

    return result


def _normalize_ohlcv(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Normalize Yahoo Finance data into:

        Open
        High
        Low
        Close
        Volume

    with a DatetimeIndex.
    """

    if df is None:
        return pd.DataFrame()

    if df.empty:
        return pd.DataFrame()

    data = df.copy()

    data = _flatten_columns(
        data
    )

    required = [
        "Open",
        "High",
        "Low",
        "Close",
        "Volume",
    ]

    # --------------------------------------------------------
    # Case-insensitive column matching.
    # --------------------------------------------------------

    column_map = {}

    for column in data.columns:

        normalized = str(
            column
        ).strip().lower()

        column_map[normalized] = column

    rename_map = {}

    for required_column in required:

        key = required_column.lower()

        if key in column_map:

            rename_map[
                column_map[key]
            ] = required_column

    data = data.rename(
        columns=rename_map
    )

    missing = [
        column
        for column in required
        if column not in data.columns
    ]

    if missing:

        raise ValueError(
            "Missing OHLCV columns: "
            + ", ".join(missing)
        )

    data = data[
        required
    ].copy()

    # --------------------------------------------------------
    # Numeric conversion.
    # --------------------------------------------------------

    for column in required:

        data[column] = pd.to_numeric(
            data[column],
            errors="coerce",
        )

    data = data.dropna(
        subset=required
    )

    # --------------------------------------------------------
    # Datetime index.
    # --------------------------------------------------------

    if not isinstance(
        data.index,
        pd.DatetimeIndex,
    ):

        data.index = pd.to_datetime(
            data.index,
            errors="coerce",
        )

    data = data[
        ~data.index.isna()
    ].copy()

    # Remove timezone for consistency.
    try:

        if data.index.tz is not None:

            data.index = (
                data.index
                .tz_localize(None)
            )

    except Exception:

        pass

    # --------------------------------------------------------
    # Remove duplicate dates.
    # --------------------------------------------------------

    data = data[
        ~data.index.duplicated(
            keep="last"
        )
    ].copy()

    data = data.sort_index()

    return data


# ============================================================
# DOWNLOAD — PERIOD BASED
# ============================================================

def download_daily_data(
    symbol: str,
    period: str = DEFAULT_PERIOD,
    interval: str = DEFAULT_INTERVAL,
    retries: int = 2,
    delay: float = 1.0,
) -> pd.DataFrame:
    """
    Download daily OHLCV data from Yahoo Finance using
    a Yahoo period such as 2y or 5y.
    """

    symbol = normalize_symbol(
        symbol
    )

    if interval != "1d":

        raise ValueError(
            "This market-data module is "
            "configured for daily data only. "
            f"Received interval={interval!r}"
        )

    last_error = None

    for attempt in range(
        retries + 1
    ):

        try:

            ticker = yf.Ticker(
                symbol
            )

            data = ticker.history(
                period=period,
                interval=interval,
                auto_adjust=False,
                actions=False,
            )

            data = _normalize_ohlcv(
                data
            )

            if data.empty:

                raise ValueError(
                    f"No daily data returned "
                    f"for {symbol}."
                )

            return data

        except Exception as exc:

            last_error = exc

            if attempt < retries:

                time.sleep(
                    delay
                )

    raise RuntimeError(
        f"Failed to download "
        f"{symbol}: {last_error}"
    )


# ============================================================
# DOWNLOAD — EXPLICIT HISTORICAL RANGE
# ============================================================

def download_historical_data(
    symbol: str,
    start_date: str,
    end_date: str,
    interval: str = DEFAULT_INTERVAL,
    retries: int = 2,
    delay: float = 1.0,
) -> pd.DataFrame:
    """
    Download daily OHLCV data for an explicit date range.

    Parameters
    ----------
    symbol:
        Yahoo Finance symbol.

    start_date:
        Inclusive start date in YYYY-MM-DD format.

    end_date:
        Exclusive end date in YYYY-MM-DD format.

    interval:
        Currently only 1d is supported.

    Returns
    -------
    pandas.DataFrame
        Normalized daily OHLCV data.
    """

    symbol = normalize_symbol(
        symbol
    )

    start_date, end_date = validate_date_range(
        start_date,
        end_date,
    )

    if interval != "1d":

        raise ValueError(
            "This market-data module is "
            "configured for daily data only. "
            f"Received interval={interval!r}"
        )

    last_error = None

    for attempt in range(
        retries + 1
    ):

        try:

            ticker = yf.Ticker(
                symbol
            )

            data = ticker.history(
                start=start_date,
                end=end_date,
                interval=interval,
                auto_adjust=False,
                actions=False,
            )

            data = _normalize_ohlcv(
                data
            )

            if data.empty:

                raise ValueError(
                    f"No historical daily data returned "
                    f"for {symbol} between "
                    f"{start_date} and {end_date}."
                )

            return data

        except Exception as exc:

            last_error = exc

            if attempt < retries:

                time.sleep(
                    delay
                )

    raise RuntimeError(
        f"Failed historical download for "
        f"{symbol} "
        f"({start_date} -> {end_date}): "
        f"{last_error}"
    )


# ============================================================
# SAVE CACHE — PERIOD
# ============================================================

def save_cache(
    symbol: str,
    data: pd.DataFrame,
    period: str = DEFAULT_PERIOD,
    interval: str = DEFAULT_INTERVAL,
) -> str:
    """
    Save data to a period-specific cache.
    """

    symbol = normalize_symbol(
        symbol
    )

    normalized = _normalize_ohlcv(
        data
    )

    if normalized.empty:

        raise ValueError(
            f"Cannot cache empty data "
            f"for {symbol}."
        )

    path = get_cache_path(
        symbol,
        period,
        interval,
    )

    normalized.to_csv(
        path,
        index=True,
    )

    return path


# ============================================================
# SAVE CACHE — HISTORICAL RANGE
# ============================================================

def save_historical_cache(
    symbol: str,
    data: pd.DataFrame,
    start_date: str,
    end_date: str,
    interval: str = DEFAULT_INTERVAL,
) -> str:
    """
    Save explicit-date historical data to a dedicated cache.
    """

    symbol = normalize_symbol(
        symbol
    )

    start_date, end_date = validate_date_range(
        start_date,
        end_date,
    )

    normalized = _normalize_ohlcv(
        data
    )

    if normalized.empty:

        raise ValueError(
            f"Cannot cache empty historical data "
            f"for {symbol}."
        )

    path = get_historical_cache_path(
        symbol,
        start_date,
        end_date,
        interval,
    )

    normalized.to_csv(
        path,
        index=True,
    )

    return path


# ============================================================
# LOAD CACHE — PERIOD
# ============================================================

def load_cache(
    symbol: str,
    period: str = DEFAULT_PERIOD,
    interval: str = DEFAULT_INTERVAL,
) -> pd.DataFrame:
    """
    Load period-specific cache.
    """

    symbol = normalize_symbol(
        symbol
    )

    path = get_cache_path(
        symbol,
        period,
        interval,
    )

    if not os.path.exists(path):

        return pd.DataFrame()

    try:

        data = pd.read_csv(
            path,
            index_col=0,
            parse_dates=True,
        )

        return _normalize_ohlcv(
            data
        )

    except Exception:

        return pd.DataFrame()


# ============================================================
# LOAD CACHE — HISTORICAL RANGE
# ============================================================

def load_historical_cache(
    symbol: str,
    start_date: str,
    end_date: str,
    interval: str = DEFAULT_INTERVAL,
) -> pd.DataFrame:
    """
    Load explicit-date historical cache.
    """

    symbol = normalize_symbol(
        symbol
    )

    start_date, end_date = validate_date_range(
        start_date,
        end_date,
    )

    path = get_historical_cache_path(
        symbol,
        start_date,
        end_date,
        interval,
    )

    if not os.path.exists(path):

        return pd.DataFrame()

    try:

        data = pd.read_csv(
            path,
            index_col=0,
            parse_dates=True,
        )

        return _normalize_ohlcv(
            data
        )

    except Exception:

        return pd.DataFrame()


# ============================================================
# CACHE ROW SANITY CHECK
# ============================================================

def _cache_has_enough_rows(
    data: pd.DataFrame,
    period: str,
) -> bool:
    """
    Prevent an unexpectedly short period cache from being used.
    """

    if data is None or data.empty:

        return False

    minimum = EXPECTED_MIN_ROWS.get(
        str(period).lower()
    )

    if minimum is None:

        return True

    return len(data) >= minimum


def _historical_cache_has_enough_rows(
    data: pd.DataFrame,
    minimum_rows: int = HISTORICAL_MIN_ROWS,
) -> bool:
    """
    Validate an explicit historical cache.
    """

    if data is None or data.empty:

        return False

    return len(data) >= minimum_rows


# ============================================================
# MAIN DATA ACCESS — PERIOD BASED
# ============================================================

def get_daily_data(
    symbol: str,
    period: str = DEFAULT_PERIOD,
    interval: str = DEFAULT_INTERVAL,
    cache_hours: int = DEFAULT_CACHE_HOURS,
    force_refresh: bool = False,
) -> pd.DataFrame:
    """
    Get daily OHLCV data using a Yahoo period.

    Cache behavior
    --------------
    A cache is specific to:

        symbol + period + interval

    This prevents a 3-month liquidity cache from being
    accidentally returned when the caller requests 2 years.

    If the cache is unexpectedly short for the requested
    period, it is ignored and a fresh Yahoo download is made.
    """

    symbol = normalize_symbol(
        symbol
    )

    path = get_cache_path(
        symbol,
        period,
        interval,
    )

    # --------------------------------------------------------
    # Try cache.
    # --------------------------------------------------------

    if not force_refresh:

        if is_cache_valid(
            path,
            cache_hours,
        ):

            cached = load_cache(
                symbol,
                period,
                interval,
            )

            if _cache_has_enough_rows(
                cached,
                period,
            ):

                return cached

    # --------------------------------------------------------
    # Download fresh data.
    # --------------------------------------------------------

    data = download_daily_data(
        symbol,
        period=period,
        interval=interval,
    )

    # --------------------------------------------------------
    # Validate downloaded length.
    #
    # Do not reject it if Yahoo itself returned less than
    # our rough expected amount, but print a warning.
    # --------------------------------------------------------

    minimum = EXPECTED_MIN_ROWS.get(
        str(period).lower()
    )

    if (
        minimum is not None
        and len(data) < minimum
    ):

        print(
            f"Warning: Yahoo returned only "
            f"{len(data)} rows for {symbol} "
            f"with period={period}."
        )

    # --------------------------------------------------------
    # Save period-specific cache.
    # --------------------------------------------------------

    save_cache(
        symbol,
        data,
        period,
        interval,
    )

    return data


# ============================================================
# MAIN DATA ACCESS — EXPLICIT HISTORICAL RANGE
# ============================================================

def get_historical_data(
    symbol: str,
    start_date: str,
    end_date: str,
    interval: str = DEFAULT_INTERVAL,
    cache_hours: int = 168,
    force_refresh: bool = False,
    minimum_rows: int = HISTORICAL_MIN_ROWS,
) -> pd.DataFrame:
    """
    Get daily OHLCV data for an explicit historical date range.

    Example
    -------

        data = get_historical_data(
            "RELIANCE.NS",
            "2021-01-01",
            "2026-09-06",
        )

    Why this exists
    ---------------

    Long historical backtests should not depend on a rolling
    Yahoo period such as "2y".

    Explicit date ranges make the research window clear and
    produce dedicated cache files.
    """

    symbol = normalize_symbol(
        symbol
    )

    start_date, end_date = validate_date_range(
        start_date,
        end_date,
    )

    path = get_historical_cache_path(
        symbol,
        start_date,
        end_date,
        interval,
    )

    # --------------------------------------------------------
    # Try cache.
    # --------------------------------------------------------

    if not force_refresh:

        if is_cache_valid(
            path,
            cache_hours,
        ):

            cached = load_historical_cache(
                symbol,
                start_date,
                end_date,
                interval,
            )

            if _historical_cache_has_enough_rows(
                cached,
                minimum_rows,
            ):

                return cached

    # --------------------------------------------------------
    # Download.
    # --------------------------------------------------------

    data = download_historical_data(
        symbol,
        start_date,
        end_date,
        interval=interval,
    )

    # --------------------------------------------------------
    # Sanity warning.
    # --------------------------------------------------------

    if len(data) < minimum_rows:

        print(
            f"Warning: historical data for {symbol} "
            f"contains only {len(data)} rows. "
            f"Minimum expected for research is "
            f"{minimum_rows}."
        )

    # --------------------------------------------------------
    # Save dedicated historical cache.
    # --------------------------------------------------------

    save_historical_cache(
        symbol,
        data,
        start_date,
        end_date,
        interval,
    )

    return data


# ============================================================
# LATEST ROW
# ============================================================

def get_latest_daily_row(
    symbol: str,
    period: str = DEFAULT_PERIOD,
    cache_hours: int = DEFAULT_CACHE_HOURS,
) -> Optional[pd.Series]:
    """
    Return the latest daily OHLCV row.
    """

    data = get_daily_data(
        symbol,
        period=period,
        interval="1d",
        cache_hours=cache_hours,
    )

    if data.empty:

        return None

    return data.iloc[-1]


# ============================================================
# TEST
# ============================================================

def _test() -> None:

    print("=" * 70)
    print("MARKET DATA MODULE TEST")
    print("=" * 70)

    symbol = "RELIANCE.NS"

    print(
        f"\nTesting normal 2y data: {symbol}"
    )

    data_2y = get_daily_data(
        symbol,
        period="2y",
        interval="1d",
        force_refresh=False,
    )

    print(
        f"2y rows: {len(data_2y)}"
    )

    if not data_2y.empty:

        print(
            f"2y first date: "
            f"{data_2y.index[0].date()}"
        )

        print(
            f"2y last date: "
            f"{data_2y.index[-1].date()}"
        )

    print(
        "\nTesting explicit historical range..."
    )

    historical = get_historical_data(
        symbol,
        start_date=HISTORICAL_DEFAULT_START,
        end_date=HISTORICAL_DEFAULT_END,
        interval="1d",
        force_refresh=False,
    )

    print(
        f"Historical rows: {len(historical)}"
    )

    if not historical.empty:

        print(
            f"Historical first date: "
            f"{historical.index[0].date()}"
        )

        print(
            f"Historical last date: "
            f"{historical.index[-1].date()}"
        )

    print(
        "\nHistorical cache path:"
    )

    print(
        get_historical_cache_path(
            symbol,
            HISTORICAL_DEFAULT_START,
            HISTORICAL_DEFAULT_END,
            "1d",
        )
    )

    print(
        "\nMarket data test complete."
    )


if __name__ == "__main__":
    _test()