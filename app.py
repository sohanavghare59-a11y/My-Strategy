"""
Flask web dashboard v3 — serves swing-trading picks for NIFTY 500 + indices.
Run: python app.py  then open http://localhost:5000
"""

from flask import Flask, render_template, jsonify
import threading
import time
from datetime import datetime

from config import TOP_N_STOCKS, TOP_N_INDICES
from stock_universe import fetch_nifty500_tickers, fetch_index_tickers, get_index_name
from data_fetcher import fetch_stocks_and_indices, fetch_news
from screener import screen_all, screen_indices
from agent import enrich_with_summary, market_overview

app = Flask(__name__)

_state = {
    "results": [],
    "index_results": [],
    "overview": "",
    "last_updated": None,
    "status": "idle",
    "progress": "",
    "total_stocks": 0,
    "total_indices": 0,
}


def run_research():
    """Background research: fetch NIFTY + indices → screen → score → enrich."""
    _state["status"] = "running"
    _state["progress"] = "Loading stock universe..."

    try:
        # 1. Get stock list + index list
        stock_tickers = fetch_nifty500_tickers()
        index_tickers = fetch_index_tickers()
        _state["total_stocks"] = len(stock_tickers)
        _state["total_indices"] = len(index_tickers)

        print(f"  ✓ {len(stock_tickers)} stocks + {len(index_tickers)} indices to fetch")

        _state["progress"] = f"Fetching {len(stock_tickers)} stocks + {len(index_tickers)} indices..."

        def progress(done, total, ticker):
            msg = f"Fetched {done}/{total} — {ticker}"
            _state["progress"] = msg
            if done % 25 == 0 or done == total:
                print(f"  [{done}/{total}] {ticker}")

        # 2. Fetch all data in parallel
        stock_data, stock_news, index_data = fetch_stocks_and_indices(
            stock_tickers, index_tickers, progress_callback=progress
        )

        print(f"\n  ✓ Fetched {len(stock_data)} stocks + {len(index_data)} indices")

        # 3. Screen stocks
        _state["progress"] = "Screening stocks..."
        print("  Screening stocks...")
        results = screen_all(stock_data, stock_news)

        # 3.5 Fetch news ONLY for top N stocks
        _state["progress"] = f"Fetching news for top {TOP_N_STOCKS} stocks..."
        print(f"  Fetching news for top {TOP_N_STOCKS} stocks...")
        for r in results[:TOP_N_STOCKS]:
            try:
                news = fetch_news(r["ticker"], company_name=r["name"], days=3)
                r["news"] = news[:5]
                from screener_logic import score_news
                ns, ns_signals = score_news(news)
                r["news_score"] = round(ns, 1)
                r["signals"]["news"] = ns_signals
                time.sleep(0.2)
            except Exception:
                r["news"] = []
                r["news_score"] = 50.0

        # 4. Screen indices
        _state["progress"] = "Screening indices..."
        print("  Screening indices...")
        index_results = screen_indices(index_data)

        # 5. Enrich top N stocks with summaries
        _state["progress"] = "Writing summaries..."
        print("  Writing summaries...")
        for r in results[:TOP_N_STOCKS]:
            try:
                enrich_with_summary(r)
            except Exception as e:
                print(f"  [!] Summary failed for {r['ticker']}: {e}")

        # 6. Market overview
        _state["progress"] = "Generating market overview..."
        print("  Generating market overview...")
        try:
            overview = market_overview(results[:TOP_N_STOCKS])
        except Exception:
            overview = ""

        _state["results"] = results[:TOP_N_STOCKS]
        _state["index_results"] = index_results[:TOP_N_INDICES]
        _state["overview"] = overview
        _state["last_updated"] = datetime.now().strftime("%d %b %Y, %I:%M %p")
        _state["status"] = "done"
        _state["progress"] = ""

        print(f"\n  ✓ DONE! {len(results)} stocks screened, top {TOP_N_STOCKS} ready.")
        print(f"  Dashboard is live at http://localhost:5000\n")

    except Exception as e:
        import traceback
        _state["status"] = "error"
        _state["progress"] = str(e)
        print(f"\n  ❌ ERROR: {e}")
        traceback.print_exc()
        print()


@app.route("/")
def dashboard():
    try:
        return render_template(
            "dashboard.html",
            results=_state["results"],
            index_results=_state["index_results"],
            overview=_state["overview"],
            last_updated=_state["last_updated"],
            status=_state["status"],
            progress=_state["progress"],
            top_n=TOP_N_STOCKS,
            total_stocks=_state["total_stocks"],
            total_indices=_state["total_indices"],
        )
    except Exception as e:
        import traceback
        traceback.print_exc()
        return f"""<html><body style="background:#0d1117;color:#c9d1d9;font-family:monospace;padding:40px;">
        <h2 style="color:#f85149;">Dashboard render error</h2>
        <pre style="color:#8b949e;white-space:pre-wrap;">{e}</pre>
        <p style="color:#d29922;">The data may still be loading. Wait a few seconds and refresh.</p>
        <p>Status: {_state['status']}</p>
        <p>Progress: {_state['progress']}</p>
        <p>Results: {len(_state['results'])} stocks, {len(_state['index_results'])} indices</p>
        </body></html>"""


@app.route("/api/results")
def api_results():
    return jsonify({
        "results": _state["results"],
        "index_results": _state["index_results"],
        "overview": _state["overview"],
        "last_updated": _state["last_updated"],
        "status": _state["status"],
        "progress": _state["progress"],
    })


@app.route("/refresh")
def refresh():
    if _state["status"] != "running":
        thread = threading.Thread(target=run_research, daemon=True)
        thread.start()
    return jsonify({"status": "refreshing", "progress": _state["progress"]})


if __name__ == "__main__":
    print("=" * 60)
    print("  Swing Trading Research Agent v3")
    print("  NIFTY + All NSE/BSE Indices | Daily | Parallel")
    print("=" * 60)
    print("\n  Starting dashboard NOW — data will load in background...\n")

    thread = threading.Thread(target=run_research, daemon=True)
    thread.start()

    print("  → Open http://localhost:5000 in your browser NOW")
    print("  → You'll see 'Screening...' while data loads")
    print("  → Page auto-refreshes every 8 seconds\n")

    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)
