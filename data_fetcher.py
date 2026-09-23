"""
Data fetcher v4 — chunked batch downloads (rate-limit resistant).

PROBLEM (v3): 500 stocks x (history + ticker.info) = 1000+ individual
requests fired at Yahoo Finance in one parallel burst. Yahoo blocks the
client after roughly the first 20 requests, so only the alphabetically
first stocks (starting with A) ever came back — the scan "finished" with
~20 stocks.

FIX (v4): downloads OHLCV in batches of 40 symbols per request via
yf.download(), with NO per-stock .info calls, a 4-hour disk cache, and a
polite pause between chunks. A 500-stock scan now needs only ~15 requests
total instead of 1000+, and also runs several times faster.

Public API (unchanged):
    fetch_stocks_and_indices(stock_tickers, index_tickers, progress_callback)
    fetch_news(ticker_symbol, company_name, days)
    get_market_regime(index_data)
"""

import yfinance as yf
import feedparser
import pandas as pd
from datetime import datetime, timedelta
import time
import json
import os

from config import (
    TIMEFRAME_INTERVAL, TIMEFRAME_PERIOD,
    EMA_FAST, EMA_MID, EMA_SLOW,
    MACD_FAST, MACD_SLOW, MACD_SIGNAL,
    RSI_PERIOD, RSI_BULL_THRESHOLD, RSI_BEAR_THRESHOLD,
    VOLUME_AVG_PERIOD, VOLUME_SPIKE_MULT,
    ADX_PERIOD, ATR_PERIOD, REGIME_EMA_PERIOD,
)
from indicators import (
    ema_ribbon, ribbon_state, macd, macd_state,
    rsi as calc_rsi, rsi_state, volume_state,
    support_resistance, candle_signal,
    adx as calc_adx, adx_state,
)

CACHE_DIR = os.path.join(os.path.dirname(__file__), "data", "cache")
os.makedirs(CACHE_DIR, exist_ok=True)

# How many symbols to request per batch (Yahoo allows ~100 per request;
# 40 keeps responses small and reliable, especially on 512 MB servers)
CHUNK_SIZE = 40

# Polite pause between batches so Yahoo does not block us
INTER_CHUNK_DELAY = 1.5  # seconds


def _cache_key(symbol):
    today = datetime.now().strftime("%Y%m%d")
    safe = symbol.replace("^", "IDX_")
    return f"{safe}_{today}"


def _load_cache(key, max_hours=4):
    path = os.path.join(CACHE_DIR, f"{key}.json")
    if os.path.exists(path):
        mtime = datetime.fromtimestamp(os.path.getmtime(path))
        if datetime.now() - mtime < timedelta(hours=max_hours):
            with open(path, "r") as f:
                return json.load(f)
    return None


def _save_cache(key, data):
    path = os.path.join(CACHE_DIR, f"{key}.json")
    with open(path, "w") as f:
        json.dump(data, f, default=str)


def _calc_atr(hist, period=14):
    """Average True Range — daily volatility measure."""
    high = hist["High"]
    low = hist["Low"]
    close = hist["Close"]

    tr = pd.concat([
        high - low,
        (high - close.shift(1)).abs(),
        (low - close.shift(1)).abs(),
    ], axis=1).max(axis=1)

    atr_series = tr.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    return atr_series


def _calc_above_ema200(close, period=REGIME_EMA_PERIOD):
    """True if latest close is above its long-term EMA, False if below, None if not enough data."""
    ema_val = float(close.ewm(span=period, adjust=False, min_periods=100).mean().iloc[-1])
    latest = float(close.iloc[-1])
    if ema_val != ema_val or latest != latest:
        return None
    return bool(latest > ema_val)


def _build_symbol_data(symbol, hist, is_index=False):
    """
    Build the full indicator dict for one symbol from its OHLCV DataFrame.
    Pure computation — no network calls (that is the whole point of v4).
    """
    try:
        close = hist["Close"]
        volume = hist["Volume"]
        if len(close) < 2:
            return None

        latest_close = float(close.iloc[-1])
        prev_close = float(close.iloc[-2]) if len(close) >= 2 else latest_close
        price_change_pct = ((latest_close - prev_close) / prev_close) * 100

        # ── Indicators ──
        ribbon = ema_ribbon(close, EMA_FAST, EMA_MID, EMA_SLOW)
        rb_state = ribbon_state(ribbon, close, lookback=3)

        macd_data = macd(close, MACD_FAST, MACD_SLOW, MACD_SIGNAL)
        mc_state = macd_state(macd_data, lookback=3)

        rsi_series = calc_rsi(close, RSI_PERIOD)
        rs_state = rsi_state(rsi_series, RSI_BULL_THRESHOLD, RSI_BEAR_THRESHOLD, lookback=3)

        # ADX trend strength
        adx_series = calc_adx(hist, period=ADX_PERIOD)
        ad_state = adx_state(adx_series, period=ADX_PERIOD)

        # ATR volatility (for adaptive stop loss)
        atr_series = _calc_atr(hist, period=ATR_PERIOD)
        atr_v = float(atr_series.iloc[-1])
        if atr_v != atr_v:  # NaN fallback
            atr_v = latest_close * 0.02

        # Long-term trend: above/below EMA 200
        above_ema200 = _calc_above_ema200(close)

        vol_state = volume_state(volume, VOLUME_AVG_PERIOD, VOLUME_SPIKE_MULT)
        sr = support_resistance(hist, lookback=20)
        candle = candle_signal(hist)

        return {
            "ticker": symbol,
            "is_index": is_index,
            "name": symbol,
            "sector": "Index" if is_index else "",
            "industry": "",
            "latest_close": round(latest_close, 2),
            "price_change_pct": round(price_change_pct, 2),
            "prev_close": round(prev_close, 2),
            "ema": rb_state,
            "macd": {
                "macd": macd_data["latest"]["macd"],
                "signal": macd_data["latest"]["signal"],
                "histogram": macd_data["latest"]["histogram"],
                "signal_state": mc_state["signal"],
                "bullish_crossover": mc_state["bullish_crossover"],
                "bearish_crossover": mc_state["bearish_crossover"],
                "histogram_rising": mc_state["histogram_rising"],
            },
            "rsi": rs_state,
            "adx": ad_state,
            "atr": round(atr_v, 2),
            "above_ema200": above_ema200,
            "volume": vol_state,
            "support_resistance": sr,
            "candle": candle,
            "pe_ratio": None,
            "roe": None,
            "market_cap": None,
            "bars": len(close),
            "fetched_at": datetime.now().isoformat(),
        }
    except Exception as e:
        print(f"    [!] Error building indicators for {symbol}: {e}")
        return None


def _download_batch(yf_symbols):
    """
    Download daily OHLCV for a batch of Yahoo symbols in one call.
    Returns {yf_symbol: DataFrame} for the symbols that came back.
    """
    try:
        raw = yf.download(
            yf_symbols,
            period=TIMEFRAME_PERIOD,
            interval=TIMEFRAME_INTERVAL,
            group_by="ticker",
            auto_adjust=True,
            actions=False,
            progress=False,
            threads=False,  # sequential — 40 parallel threads crash on small servers
        )
    except Exception as e:
        print(f"  [!] Batch download failed ({len(yf_symbols)} symbols): {e}")
        return {}

    out = {}
    if raw is None or raw.empty:
        return out

    # Single symbol may come back without a MultiIndex — normalise
    if not isinstance(raw.columns, pd.MultiIndex):
        sym = yf_symbols[0]
        df = raw.dropna(subset=["Close"])
        if not df.empty and df["Close"].notna().any():
            out[sym] = df
        return out

    available = set(raw.columns.get_level_values(0))
    for sym in yf_symbols:
        if sym not in available:
            continue
        try:
            df = raw[sym].dropna(subset=["Close"])
            if not df.empty and df["Close"].notna().any():
                out[sym] = df
        except Exception:
            continue
    return out


def fetch_symbol_data(symbol, is_index=False):
    """Fetch a single symbol (uses the same batch machinery). Cached for 4 hours."""
    key = _cache_key(symbol)
    cached = _load_cache(key)
    if cached:
        return cached

    yf_symbol = symbol if is_index else f"{symbol}.NS"
    frames = _download_batch([yf_symbol])
    df = frames.get(yf_symbol)
    if df is None or df.empty:
        return None

    data = _build_symbol_data(symbol, df, is_index=is_index)
    if data is None:
        return None
    _save_cache(key, data)
    return data


def fetch_stocks_and_indices(stock_tickers, index_tickers,
                             progress_callback=None):
    """
    Fetch both stocks and indices using CHUNKED batch downloads.

    For each symbol: serve from the 4-hour disk cache if possible, otherwise
    download in batches of CHUNK_SIZE symbols per request. Symbols that come
    back empty get ONE retry after a short pause (Yahoo hiccups are often
    transient). Truly dead tickers are skipped with a warning.

    Returns (stock_data, stock_news, index_data). News is fetched separately
    by the app for just the top-N stocks.
    """
    stock_data = {}
    index_data = {}
    failed = []

    def _process(symbols, is_index, store):
        # 1. Serve whatever we can from cache
        todo = []
        for s in symbols:
            cached = _load_cache(_cache_key(s))
            if cached:
                store[s] = cached
            else:
                todo.append(s)

        # 2. Download the rest in chunks
        done = len(symbols) - len(todo)
        for i in range(0, len(todo), CHUNK_SIZE):
            chunk = todo[i:i + CHUNK_SIZE]
            yf_syms = [s if is_index else f"{s}.NS" for s in chunk]

            frames = _download_batch(yf_syms)

            # One retry pass for anything missing in this chunk
            missing = [y for y in yf_syms if y not in frames]
            if missing:
                time.sleep(4)
                frames.update(_download_batch(missing))

            for s in chunk:
                y = s if is_index else f"{s}.NS"
                df = frames.get(y)
                if df is not None and not df.empty:
                    data = _build_symbol_data(s, df, is_index=is_index)
                    if data:
                        _save_cache(_cache_key(s), data)
                        store[s] = data
                    else:
                        failed.append(s)
                else:
                    failed.append(s)

                done += 1
                if progress_callback:
                    progress_callback(done, len(symbols), s)

            time.sleep(INTER_CHUNK_DELAY)

    _process(stock_tickers, False, stock_data)
    _process(index_tickers, True, index_data)

    if failed:
        print(f"\n  ⚠ {len(failed)} symbol(s) failed: "
              f"{', '.join(failed[:10])}{'...' if len(failed) > 10 else ''}\n")

    return stock_data, {}, index_data


def fetch_news(ticker_symbol, company_name=None, days=3):
    query = company_name or ticker_symbol
    query = query.replace(" ", "+")
    url = (f"https://news.google.com/rss/search?q={query}+stock"
           f"+site:moneycontrol.com+OR+site:livemint.com+OR+site:economictimes.com"
           f"&hl=en-IN&gl=IN&ceid=IN:en")

    try:
        feed = feedparser.parse(url)
        cutoff = datetime.now() - timedelta(days=days)
        articles = []

        for entry in feed.entries[:10]:
            published_str = entry.get("published", "")
            try:
                published = datetime.strptime(
                    published_str, "%a, %d %b %Y %H:%M:%S %Z"
                )
            except (ValueError, TypeError):
                published = datetime.now()

            if published >= cutoff:
                articles.append({
                    "title": entry.get("title", ""),
                    "link": entry.get("link", ""),
                    "published": published.strftime("%Y-%m-%d %H:%M"),
                    "source": entry.get("source", {}).get("title", "Unknown"),
                })

        return articles

    except Exception:
        return []


def get_market_regime(index_data):
    """
    Derive the market regime from NIFTY 50 data.
    Returns "bullish" if NIFTY is above its 200 EMA,
    "bearish" if below, None if NIFTY data is unavailable.
    """
    nifty = index_data.get("^NSEI") if index_data else None
    if not nifty:
        return None
    if nifty.get("above_ema200") is True:
        return "bullish"
    if nifty.get("above_ema200") is False:
        return "bearish"
    return None
