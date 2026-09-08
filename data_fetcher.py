"""
Data fetcher v3 — parallel fetching for NIFTY + indices.
Single thread pool for stocks and indices together.
"""

import yfinance as yf
import feedparser
import pandas as pd
from datetime import datetime, timedelta
import time
import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

from config import (
    TIMEFRAME_INTERVAL, TIMEFRAME_PERIOD, MAX_WORKERS, FETCH_DELAY,
    EMA_FAST, EMA_MID, EMA_SLOW,
    MACD_FAST, MACD_SLOW, MACD_SIGNAL,
    RSI_PERIOD, RSI_BULL_THRESHOLD, RSI_BEAR_THRESHOLD,
    VOLUME_AVG_PERIOD, VOLUME_SPIKE_MULT,
)
from indicators import (
    ema_ribbon, ribbon_state, macd, macd_state,
    rsi as calc_rsi, rsi_state, volume_state,
    support_resistance, candle_signal,
)

CACHE_DIR = os.path.join(os.path.dirname(__file__), "data", "cache")
os.makedirs(CACHE_DIR, exist_ok=True)


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


def fetch_symbol_data(symbol, is_index=False):
    """Fetch daily OHLCV + indicators for a stock or index."""
    key = _cache_key(symbol)
    cached = _load_cache(key)
    if cached:
        return cached

    yf_symbol = symbol if is_index else f"{symbol}.NS"

    try:
        ticker = yf.Ticker(yf_symbol)
        hist = ticker.history(period=TIMEFRAME_PERIOD, interval=TIMEFRAME_INTERVAL)
        if hist.empty:
            return None

        close = hist["Close"]
        volume = hist["Volume"]
        latest_close = float(close.iloc[-1])
        latest_volume = float(volume.iloc[-1])

        prev_close = float(close.iloc[-2]) if len(close) >= 2 else latest_close
        price_change_pct = ((latest_close - prev_close) / prev_close) * 100

        # ── Indicators ──
        ribbon = ema_ribbon(close, EMA_FAST, EMA_MID, EMA_SLOW)
        rb_state = ribbon_state(ribbon, close, lookback=3)

        macd_data = macd(close, MACD_FAST, MACD_SLOW, MACD_SIGNAL)
        mc_state = macd_state(macd_data, lookback=3)

        rsi_series = calc_rsi(close, RSI_PERIOD)
        rs_state = rsi_state(rsi_series, RSI_BULL_THRESHOLD, RSI_BEAR_THRESHOLD, lookback=3)

        vol_state = volume_state(volume, VOLUME_AVG_PERIOD, VOLUME_SPIKE_MULT)
        sr = support_resistance(hist, lookback=20)
        candle = candle_signal(hist)

        # Fundamentals (only for stocks)
        info = {}
        if not is_index:
            try:
                info = ticker.info or {}
            except Exception:
                pass

        data = {
            "ticker": symbol,
            "is_index": is_index,
            "name": info.get("longName") or info.get("shortName", symbol),
            "sector": info.get("sector", "Index" if is_index else "Unknown"),
            "industry": info.get("industry", ""),
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
            "volume": vol_state,
            "support_resistance": sr,
            "candle": candle,
            "pe_ratio": info.get("trailingPE") if not is_index else None,
            "roe": info.get("returnOnEquity") if not is_index else None,
            "market_cap": info.get("marketCap") if not is_index else None,
            "bars": len(close),
            "fetched_at": datetime.now().isoformat(),
        }

        _save_cache(key, data)
        return data

    except Exception as e:
        print(f"    [!] Error fetching {symbol}: {e}")
        return None


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


def _fetch_one(ticker, is_index, with_news):
    """Worker function for parallel fetching."""
    data = fetch_symbol_data(ticker, is_index=is_index)
    if data and with_news and not is_index:
        news = fetch_news(ticker, company_name=data.get("name"), days=3)
        time.sleep(FETCH_DELAY)
        return ticker, data, news
    elif data:
        return ticker, data, []
    return ticker, None, []


def fetch_all_parallel(tickers, is_index=False, with_news=True,
                       progress_callback=None, max_workers=None):
    """Fetch data for multiple tickers in parallel."""
    if max_workers is None:
        max_workers = MAX_WORKERS

    stock_data = {}
    news_data = {}
    failed = []
    completed = 0
    total = len(tickers)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(_fetch_one, t, is_index, with_news): t
            for t in tickers
        }

        for future in as_completed(futures):
            ticker, data, news = future.result()
            completed += 1

            if progress_callback:
                progress_callback(completed, total, ticker)

            if data:
                stock_data[ticker] = data
                if news:
                    news_data[ticker] = news
            else:
                failed.append(ticker)

    if failed:
        print(f"\n  ⚠ {len(failed)} symbol(s) failed: {', '.join(failed[:10])}{'...' if len(failed) > 10 else ''}\n")

    return stock_data, news_data


def fetch_stocks_and_indices(stock_tickers, index_tickers,
                             progress_callback=None):
    """Fetch both stocks and indices in a SINGLE thread pool."""
    max_workers = MAX_WORKERS
    stock_data = {}
    stock_news = {}
    index_data = {}
    failed = []
    completed = 0
    total = len(stock_tickers) + len(index_tickers)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {}

        # Submit stocks
        for t in stock_tickers:
            futures[executor.submit(_fetch_one, t, False, False)] = ("stock", t)

        # Submit indices
        for t in index_tickers:
            futures[executor.submit(_fetch_one, t, True, False)] = ("index", t)

        for future in as_completed(futures):
            kind, ticker = futures[future]
            result = future.result()
            data = result[1]
            news = result[2]
            completed += 1

            if progress_callback and kind == "stock":
                progress_callback(completed, total, ticker)

            if data:
                if kind == "stock":
                    stock_data[ticker] = data
                    if news:
                        stock_news[ticker] = news
                else:
                    index_data[ticker] = data
            else:
                failed.append(ticker)

    if failed:
        print(f"\n  ⚠ {len(failed)} symbol(s) failed: {', '.join(failed[:10])}{'...' if len(failed) > 10 else ''}\n")

    return stock_data, stock_news, index_data
