"""
Daily Swing Trading Signal Scanner.

Scans the liquid NSE/BSE stock universe plus all configured indices
using the daily timeframe strategy.
"""

from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

from config.settings import (
    MAX_WORKERS,
    MASTER_UNIVERSE_FILE,
    MIN_RISK_REWARD,
    SIGNAL_OUTPUT_FILE,
    STOP_LOSS_PCT,
    TARGET_1_PCT,
    TIMEFRAME_PERIOD,
)
from indicators import calculate_indicators, get_latest_indicators
from strategy import get_latest_signal
from data.market_data import get_daily_data


PROJECT_ROOT = Path(__file__).resolve().parent.parent
MASTER_UNIVERSE_PATH = PROJECT_ROOT / MASTER_UNIVERSE_FILE
SIGNAL_OUTPUT_PATH = PROJECT_ROOT / SIGNAL_OUTPUT_FILE

MIN_DATA_ROWS = 60


def calculate_risk_levels(entry_price, direction):
    """Calculate stop loss, target, risk and reward."""

    entry_price = float(entry_price)

    if direction == "LONG":
        stop_loss = entry_price * (1 - STOP_LOSS_PCT)
        target = entry_price * (1 + TARGET_1_PCT)

    elif direction == "SHORT":
        stop_loss = entry_price * (1 + STOP_LOSS_PCT)
        target = entry_price * (1 - TARGET_1_PCT)

    else:
        raise ValueError(f"Unknown direction: {direction}")

    risk = abs(entry_price - stop_loss)
    reward = abs(target - entry_price)

    risk_reward = reward / risk if risk != 0 else 0.0

    return {
        "entry": entry_price,
        "stop_loss": stop_loss,
        "target": target,
        "risk": risk,
        "reward": reward,
        "risk_reward": risk_reward,
    }


def make_base_result(row):
    """Create standard output structure."""

    return {
        "exchange": row.get("exchange", ""),
        "instrument_type": row.get("instrument_type", ""),
        "symbol": row.get("symbol", ""),
        "name": row.get("name", ""),
        "isin": row.get("isin", ""),
        "security_id": row.get("security_id", ""),
        "series": row.get("series", ""),
        "data_symbol": row.get("data_symbol", ""),
        "scan_status": "",
        "signal": "",
        "signal_date": "",
        "entry": "",
        "stop_loss": "",
        "target": "",
        "risk": "",
        "reward": "",
        "risk_reward": "",
        "rsi": "",
        "ema_5": "",
        "ema_13": "",
        "ema_26": "",
        "macd": "",
        "macd_signal": "",
        "volume_ratio": "",
        "scan_error": "",
    }


def scan_instrument(row):
    """Scan one instrument."""

    result = make_base_result(row)

    data_symbol = str(row.get("data_symbol", "")).strip()

    try:
        if not data_symbol:
            result["scan_status"] = "ERROR"
            result["scan_error"] = "EMPTY_DATA_SYMBOL"
            return result

        # ---------------------------------------------------------
        # Download daily data
        # ---------------------------------------------------------

        df = get_daily_data(
            data_symbol,
            period=TIMEFRAME_PERIOD,
        )

        if df is None or df.empty:
            result["scan_status"] = "ERROR"
            result["scan_error"] = (
                f"No market data returned for {data_symbol}"
            )
            return result

        # ---------------------------------------------------------
        # Minimum history
        # ---------------------------------------------------------

        row_count = len(df)

        if row_count < MIN_DATA_ROWS:
            result["scan_status"] = "INSUFFICIENT_DATA"
            result["scan_error"] = (
                f"INSUFFICIENT_DATA_ROWS_{row_count}"
            )
            return result

        # ---------------------------------------------------------
        # Calculate indicators
        # ---------------------------------------------------------

        indicators = calculate_indicators(df)

        if indicators is None or indicators.empty:
            result["scan_status"] = "ERROR"
            result["scan_error"] = "INDICATOR_CALCULATION_FAILED"
            return result

        # ---------------------------------------------------------
        # Get latest indicators
        # ---------------------------------------------------------

        latest = get_latest_indicators(indicators)

        if latest is None:
            result["scan_status"] = "ERROR"
            result["scan_error"] = "LATEST_INDICATORS_UNAVAILABLE"
            return result

        # ---------------------------------------------------------
        # Get latest strategy signal
        # ---------------------------------------------------------

        signal_info = get_latest_signal(indicators)

        if signal_info is None:
            signal_info = {}

        signal = signal_info.get("signal", "")
        direction = signal_info.get("direction", signal)

        # ---------------------------------------------------------
        # IMPORTANT:
        # get_latest_indicators() uses lowercase keys.
        # ---------------------------------------------------------

        result["rsi"] = latest.get("rsi", "")
        result["ema_5"] = latest.get("ema_5", "")
        result["ema_13"] = latest.get("ema_13", "")
        result["ema_26"] = latest.get("ema_26", "")
        result["macd"] = latest.get("macd", "")
        result["macd_signal"] = latest.get("macd_signal", "")
        result["volume_ratio"] = latest.get("volume_ratio", "")

        # ---------------------------------------------------------
        # No signal
        # ---------------------------------------------------------

        if not signal or signal in ("NONE", "NO_SIGNAL"):
            result["scan_status"] = "NO_SIGNAL"
            result["signal"] = "NO_SIGNAL"
            return result

        # ---------------------------------------------------------
        # Signal found
        # ---------------------------------------------------------

        result["scan_status"] = "SIGNAL"
        result["signal"] = direction

        result["signal_date"] = str(
            signal_info.get("date", "")
        )

        # ---------------------------------------------------------
        # Entry = latest daily close
        # ---------------------------------------------------------

        entry_price = latest.get("close")

        if entry_price is None or pd.isna(entry_price):
            result["scan_status"] = "ERROR"
            result["scan_error"] = "ENTRY_PRICE_UNAVAILABLE"
            return result

        # ---------------------------------------------------------
        # Risk management
        # ---------------------------------------------------------

        risk = calculate_risk_levels(
            entry_price,
            direction,
        )

        result["entry"] = risk["entry"]
        result["stop_loss"] = risk["stop_loss"]
        result["target"] = risk["target"]
        result["risk"] = risk["risk"]
        result["reward"] = risk["reward"]
        result["risk_reward"] = risk["risk_reward"]

        return result

    except Exception as exc:
        result["scan_status"] = "ERROR"
        result["scan_error"] = str(exc)
        return result


def load_scan_universe():
    """
    Load the master universe.

    Include:
        - PASS stocks from liquid_universe.csv
        - all indices
    """

    if not MASTER_UNIVERSE_PATH.exists():
        raise FileNotFoundError(
            f"Master universe not found: {MASTER_UNIVERSE_PATH}"
        )

    master_df = pd.read_csv(
        MASTER_UNIVERSE_PATH
    )

    required_columns = [
        "exchange",
        "instrument_type",
        "symbol",
        "name",
        "data_symbol",
    ]

    missing = [
        column
        for column in required_columns
        if column not in master_df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing master universe columns: {missing}"
        )

    liquid_path = (
        PROJECT_ROOT
        / "output"
        / "liquid_universe.csv"
    )

    if not liquid_path.exists():
        raise FileNotFoundError(
            f"Liquid universe not found: {liquid_path}"
        )

    liquid_df = pd.read_csv(
        liquid_path
    )

    if "liquidity_status" not in liquid_df.columns:
        raise ValueError(
            "liquid_universe.csv is missing liquidity_status"
        )

    liquid_symbols = set(
        liquid_df.loc[
            liquid_df["liquidity_status"].eq("PASS"),
            "data_symbol",
        ]
        .dropna()
        .astype(str)
        .str.strip()
    )

    instrument_type = (
        master_df["instrument_type"]
        .fillna("")
        .astype(str)
        .str.upper()
        .str.strip()
    )

    data_symbols = (
        master_df["data_symbol"]
        .fillna("")
        .astype(str)
        .str.strip()
    )

    stock_mask = (
        instrument_type.eq("STOCK")
        & data_symbols.isin(liquid_symbols)
    )

    index_mask = instrument_type.eq("INDEX")

    scan_df = master_df.loc[
        stock_mask | index_mask
    ].copy()

    scan_df = scan_df.drop_duplicates(
        subset=[
            "exchange",
            "instrument_type",
            "data_symbol",
        ]
    )

    return scan_df.reset_index(drop=True)


def save_results(results):
    """Save results to CSV."""

    SIGNAL_OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    result_df = pd.DataFrame(
        results
    )

    result_df.to_csv(
        SIGNAL_OUTPUT_PATH,
        index=False,
    )

    return result_df


def run_scanner():
    """Run complete daily scanner."""

    print("=" * 60)
    print("DAILY SWING TRADING SIGNAL SCANNER")
    print("=" * 60)

    print()
    print("Loading liquid universe...")

    universe = load_scan_universe()

    print(
        f"Total instruments to scan: {len(universe)}"
    )

    print()
    print("Starting daily scan...")
    print(f"Workers: {MAX_WORKERS}")
    print(f"Minimum daily rows: {MIN_DATA_ROWS}")
    print(f"Period: {TIMEFRAME_PERIOD}")

    results = []

    rows = universe.to_dict(
        "records"
    )

    with ThreadPoolExecutor(
        max_workers=MAX_WORKERS
    ) as executor:

        futures = {
            executor.submit(
                scan_instrument,
                row,
            ): row
            for row in rows
        }

        completed = 0

        for future in as_completed(
            futures
        ):
            row = futures[future]

            try:
                result = future.result()

            except Exception as exc:
                result = make_base_result(row)
                result["scan_status"] = "ERROR"
                result["scan_error"] = str(exc)

            results.append(result)

            completed += 1

            if (
                completed % 100 == 0
                or completed == len(rows)
            ):
                print(
                    f"Progress: "
                    f"{completed}/{len(rows)}"
                )

    result_df = save_results(
        results
    )

    print()
    print("=" * 60)
    print("DAILY SIGNAL SCANNER COMPLETE")
    print("=" * 60)

    print(
        f"Total instruments: "
        f"{len(result_df)}"
    )

    print()
    print("SCAN STATUS:")

    if not result_df.empty:
        print(
            result_df[
                "scan_status"
            ]
            .value_counts()
            .to_string()
        )

    signals = result_df[
        result_df["scan_status"].eq(
            "SIGNAL"
        )
    ].copy()

    print()
    print(
        f"TOTAL SIGNALS: {len(signals)}"
    )

    if not signals.empty:

        display_columns = [
            "exchange",
            "symbol",
            "signal",
            "signal_date",
            "entry",
            "stop_loss",
            "target",
            "risk_reward",
            "rsi",
        ]

        print()

        print(
            signals[
                display_columns
            ]
            .sort_values(
                [
                    "signal",
                    "exchange",
                    "symbol",
                ]
            )
            .to_string(index=False)
        )

    insufficient = result_df[
        result_df["scan_status"].eq(
            "INSUFFICIENT_DATA"
        )
    ]

    errors = result_df[
        result_df["scan_status"].eq(
            "ERROR"
        )
    ]

    print()
    print(
        f"INSUFFICIENT DATA: "
        f"{len(insufficient)}"
    )

    print(
        f"TRUE ERRORS: {len(errors)}"
    )

    print()
    print(
        f"Saved to: "
        f"{SIGNAL_OUTPUT_PATH}"
    )

    return result_df


if __name__ == "__main__":
    run_scanner()