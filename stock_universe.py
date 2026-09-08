"""
Stock Universe module — fetches the NIFTY stock list and all major
NSE/BSE indices dynamically.

No hardcoded stock lists — everything is fetched at runtime so the agent
always has the current universe.
"""

import yfinance as yf
import pandas as pd
import json
import os
from datetime import datetime, timedelta

CACHE_DIR = os.path.join(os.path.dirname(__file__), "data", "cache")
os.makedirs(CACHE_DIR, exist_ok=True)

# ─── Major NSE Indices ───────────────────────────────────────────
NSE_INDICES = [
    "^NSEI",        # NIFTY 50
    "^NSEBANK",     # NIFTY Bank
    "^CNXFIN",      # NIFTY Financial Services (FinNIFTY)
]

# ─── Major BSE Indices ───────────────────────────────────────────
BSE_INDICES = [
    "^BSESN",       # SENSEX
]

ALL_INDICES = NSE_INDICES + BSE_INDICES


def fetch_nifty500_tickers():
    """
    Fetch the NSE stock list. Uses NIFTY_LIST from config (default: nifty200).
    Tries multiple sources:
      1. NSE India official CSV (most reliable)
      2. yfinance constituent list
      3. Wikipedia table (needs lxml installed)
      4. Curated fallback list (50 stocks)

    Returns a list of ticker symbols (without .NS suffix).
    """
    import requests
    from config import NIFTY_LIST

    tickers = []

    # ─── Source 1: NSE India official CSV ────────────────────────
    try:
        url = f"https://archives.nseindia.com/content/indices/{NIFTY_LIST}"
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
        }
        r = requests.get(url, headers=headers, timeout=10)
        if r.status_code == 200 and len(r.content) > 100:
            df = pd.read_csv(pd.io.common.StringIO(r.text))
            if "Symbol" in df.columns:
                tickers = df["Symbol"].str.strip().tolist()
                print(f"  ✓ Fetched {len(tickers)} {NIFTY_LIST.replace('ind_','').replace('list.csv','').upper()} stocks from NSE India")
                return tickers
    except Exception as e:
        print(f"  [!] NSE CSV fetch failed: {e}")

    # ─── Source 2: yfinance constituent list ─────────────────────
    try:
        nifty = yf.Ticker("^NSEI")
        constituents = nifty.info.get("components", [])
        if constituents and len(constituents) > 10:
            tickers = [c["symbol"].replace(".NS", "") for c in constituents]
            print(f"  ✓ Fetched {len(tickers)} stocks from yfinance constituents")
            return tickers
    except Exception:
        pass

    # ─── Source 3: Wikipedia ──────────────────────────────────────
    try:
        tables = pd.read_html("https://en.wikipedia.org/wiki/NIFTY_200")
        for table in tables:
            if "Symbol" in table.columns:
                tickers = table["Symbol"].str.strip().tolist()
                print(f"  ✓ Fetched {len(tickers)} stocks from Wikipedia")
                return tickers
    except Exception:
        pass

    # ─── Source 4: Fallback watchlist ────────────────────────────
    from config import FALLBACK_WATCHLIST
    print(f"  ⚠ Using fallback watchlist ({len(FALLBACK_WATCHLIST)} stocks)")
    return FALLBACK_WATCHLIST


def fetch_index_tickers():
    """Return the list of index tickers for yfinance."""
    return ALL_INDICES


def get_index_name(ticker):
    """Map a yfinance index ticker to a human-readable name."""
    names = {
        "^NSEI": "NIFTY 50",
        "^NSEBANK": "Bank Nifty",
        "^CNXFIN": "FinNIFTY",
        "^BSESN": "SENSEX",
    }
    return names.get(ticker, ticker)


if __name__ == "__main__":
    print("=" * 60)
    print("STOCK UNIVERSE TEST")
    print("=" * 60)

    print("\nFetching stock tickers...")
    stocks = fetch_nifty500_tickers()
    print(f"  Stocks: {len(stocks)}")
    print(f"  First 5: {stocks[:5]}")

    print("\nIndex tickers:")
    indices = fetch_index_tickers()
    for idx in indices:
        print(f"  {idx} → {get_index_name(idx)}")

    print(f"\nTotal: {len(stocks)} stocks + {len(indices)} indices")
    print("=" * 60)
