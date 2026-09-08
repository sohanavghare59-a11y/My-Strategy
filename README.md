# 📈 Swing Trading Research Agent v3

Screens **NIFTY 500 stocks + all major NSE/BSE indices** for swing-trading setups using EMA 5/13/26, MACD, RSI 60-40, volume, and support/resistance — on the daily timeframe, with **parallel fetching** for speed.

## What's New in v3

- **NIFTY 500 stocks** — dynamically fetched at runtime (not hardcoded)
- **All major NSE + BSE indices** — 23 NSE indices + 20 BSE indices = 43 total
- **Parallel fetching** — 8 threads simultaneously (NIFTY 500 in ~5 min instead of ~30 min)
- **Index dashboard strip** — see all indices at a glance with trend + recommendation
- **Top 20 stock picks** — increased from 10
- **Graceful error handling** — failed stocks are skipped silently

## Strategy

| Indicator | Settings | Purpose |
|---|---|---|
| EMA Ribbon | 5 / 13 / 26 | Trend direction + crossover signals |
| MACD | 12 / 26 / 9 | Momentum confirmation |
| RSI | 14, 60-40 mode | Trend strength + buy zone detection |
| Volume | 20-bar avg, 1.3x | Institutional confirmation |
| Support/Resistance | 20-bar swings | Entry/exit levels |
| Candle Pattern | Latest bar | Bullish/bearish confirmation |

### Trade Setup
- **Entry**: Current close
- **Stop Loss**: Smart (min of 2.5% below entry, 1% below EMA-26, 0.5% below support)
- **Target 1**: +5% / **Target 2**: +10%
- **Risk:Reward** must be ≥ 1:2 to qualify as BUY

## Quick Start

```bash
pip install -r requirements.txt
python app.py
# → http://localhost:5000
```

First run takes ~5-10 minutes (fetching 500 stocks + 43 indices in parallel). Cached for 4 hours.

## Project Structure

```
swingagent_v3/
├── app.py                # Flask dashboard
├── config.py             # All settings
├── stock_universe.py     # Dynamic NIFTY 500 + index list fetcher
├── indicators.py         # EMA, MACD, RSI, volume, S/R, candles
├── data_fetcher.py       # Parallel yfinance fetcher
├── screener_logic.py     # Shared scoring functions
├── screener.py           # Stock + index screening
├── agent.py              # LLM summaries
├── requirements.txt
├── templates/
│   └── dashboard.html
└── data/
    └── cache/
```

## Indices Tracked

**NSE (23):** NIFTY 50, Bank Nifty, IT, Auto, Pharma, FMCG, Metal, Energy, Media, Realty, PSU Bank, MNC, Infra, PSU, Financial Services, Commodities, Consumption, Midcap 100, Smallcap 100, 200, 500, 100, 500 TRI

**BSE (20):** SENSEX, Bankex, Midcap, Smallcap, Teck, Consumer Goods, Healthcare, PSU, Metal, Power, Auto, Finance, Consumer Disc., FMCG, IT, Industrials, Energy, Infrastructure, Telecom, Utilities

## API

```bash
curl http://localhost:5000/api/results    # JSON (stocks + indices)
curl http://localhost:5000/refresh       # Trigger refresh
```

## Limitations

- **Not investment advice.** Research tool only.
- yfinance rate limits may slow down with 500+ stocks.
- Always paper-trade before using real capital.
