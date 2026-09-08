"""
Master universe builder for the swing trading agent.
Combines NSE stocks, validated BSE stocks, and indices.
"""

from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path.cwd()
OUTPUT_DIR = PROJECT_ROOT / "output"

NSE_FILE = OUTPUT_DIR / "nse_universe.csv"
BSE_FILE = OUTPUT_DIR / "bse_validated_universe.csv"
INDEX_FILE = OUTPUT_DIR / "index_universe.csv"
MASTER_FILE = OUTPUT_DIR / "master_universe.csv"

MASTER_COLUMNS = [
    "exchange",
    "instrument_type",
    "symbol",
    "name",
    "isin",
    "security_id",
    "series",
    "data_symbol",
]


def load_nse():
    print("Loading NSE universe...")

    if not NSE_FILE.exists():
        raise FileNotFoundError(f"Missing file: {NSE_FILE}")

    df = pd.read_csv(NSE_FILE, dtype=str)

    required = [
        "exchange",
        "instrument_type",
        "symbol",
        "name",
        "isin",
        "series",
        "data_symbol",
    ]

    for column in required:
        if column not in df.columns:
            raise ValueError(f"NSE missing column: {column}")

    df = df[required].copy()
    df["security_id"] = ""

    return df[MASTER_COLUMNS]


def load_bse():
    print("Loading VALIDATED BSE universe...")

    if not BSE_FILE.exists():
        raise FileNotFoundError(f"Missing file: {BSE_FILE}")

    df = pd.read_csv(BSE_FILE, dtype=str)

    required = [
        "exchange",
        "instrument_type",
        "symbol",
        "name",
        "isin",
        "security_id",
        "series",
        "data_symbol",
    ]

    for column in required:
        if column not in df.columns:
            raise ValueError(f"BSE missing column: {column}")

    df = df[required].copy()

    df["exchange"] = "BSE"
    df["instrument_type"] = "STOCK"

    return df[MASTER_COLUMNS]


def load_indices():
    print("Loading index universe...")

    if not INDEX_FILE.exists():
        raise FileNotFoundError(f"Missing file: {INDEX_FILE}")

    df = pd.read_csv(INDEX_FILE, dtype=str)

    required = [
        "exchange",
        "instrument_type",
        "symbol",
        "name",
        "data_symbol",
    ]

    for column in required:
        if column not in df.columns:
            raise ValueError(f"Index missing column: {column}")

    df = df[required].copy()

    df["isin"] = ""
    df["security_id"] = ""
    df["series"] = ""

    return df[MASTER_COLUMNS]


def clean_data(df):
    for column in MASTER_COLUMNS:
        df[column] = df[column].fillna("").astype(str).str.strip()

    df = df.drop_duplicates(
        subset=["exchange", "instrument_type", "data_symbol"]
    ).reset_index(drop=True)

    return df


def build_master_universe():
    print("=" * 60)
    print("BUILDING MASTER UNIVERSE")
    print("=" * 60)

    nse = load_nse()
    print(f"NSE stocks: {len(nse)}")

    bse = load_bse()
    print(f"Validated BSE stocks: {len(bse)}")

    indices = load_indices()
    print(f"Indices: {len(indices)}")

    master = pd.concat(
        [nse, bse, indices],
        ignore_index=True,
    )

    master = clean_data(master)

    MASTER_FILE.parent.mkdir(parents=True, exist_ok=True)
    master.to_csv(MASTER_FILE, index=False)

    print()
    print("MASTER UNIVERSE CREATED")
    print(f"Total rows: {len(master)}")
    print()
    print("Breakdown:")
    print(
        master.groupby(
            ["exchange", "instrument_type"]
        ).size()
    )
    print()
    print(f"Saved to: {MASTER_FILE}")

    return master


if __name__ == "__main__":
    build_master_universe()
