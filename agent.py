"""
Agent module — LLM-powered swing-trading summaries with risk/reward context.
Falls back to rule-based summaries if no OPENAI_API_KEY is set.
"""

from config import OPENAI_API_KEY, LLM_MODEL, USE_LLM_SUMMARY


def _rule_based_summary(result):
    """Generate a swing-trading summary from the screening signals."""
    ticker = result["ticker"]
    name = result["name"]
    signal = result["signal"]
    price = result["latest_close"]
    change = result["price_change_pct"]
    direction = result["trade_direction"]
    trend = result["trend"]
    trade = result["trade"]

    summary = (
        f"{name} ({ticker}) is trading at \u20b9{price} ({change:+.2f}%) "
        f"with a {signal} signal. "
        f"Trend: {trend}, direction: {direction}. "
    )

    # signals is a list of strings from screener_logic
    sig_list = result.get("signals", [])
    if sig_list:
        summary += "Conditions: " + "; ".join(sig_list[:4]) + ". "

    # Trade levels + risk/reward
    if trade and direction in ("LONG", "SHORT"):
        summary += (
            f"Entry \u20b9{trade['entry']}, SL \u20b9{trade['stop_loss']} "
            f"({trade['risk_pct']}% risk), "
            f"T1 \u20b9{trade['target1']} ({trade['reward1_pct']}%), "
            f"T2 \u20b9{trade['target2']} ({trade['reward2_pct']}%). "
            f"Risk:Reward = 1:{trade['rr1']} (T1), 1:{trade['rr2']} (T2)."
        )

    return summary


def _llm_summary(result):
    """Use OpenAI to generate a swing-trading summary."""
    try:
        from openai import OpenAI
        client = OpenAI(api_key=OPENAI_API_KEY)

        ind = result["indicators"]
        trade = result["trade"]
        news = result["news"]
        news_text = "\n".join([f"- {n['title']} ({n['source']})" for n in news[:5]])

        prompt = f"""You are a swing-trading research analyst. Write a 3-4 sentence summary explaining why {result['name']} ({result['ticker']}) is a swing trade pick today.

Stock data:
- Price: \u20b9{result['latest_close']} ({result['price_change_pct']:+.2f}% today)
- Signal: {result['signal']}
- Trend: {result['trend']}
- Trade direction: {result['trade_direction']}

Indicators:
- EMA 5/13/26: state={ind['ema']['state']}, fast={ind['ema']['ema_fast']}, mid={ind['ema']['ema_mid']}, slow={ind['ema']['ema_slow']}
- MACD: {ind['macd']['signal']}, bullish_crossover={ind['macd']['bullish_crossover']}, histogram_rising={ind['macd']['histogram_rising']}
- RSI: {ind['rsi']['value']} ({ind['rsi']['zone']}), crossed_above_60={ind['rsi']['crossed_above_60']}
- Volume: {ind['volume']['ratio']}x avg, confirmed={ind['volume']['confirmed']}, trend={ind['volume']['trend']}
- Support: \u20b9{ind['support_resistance']['support']} ({ind['support_resistance']['support_pct']}% below)
- Resistance: \u20b9{ind['support_resistance']['resistance']} ({ind['support_resistance']['resistance_pct']}% above)
- Candle: {ind['candle']['pattern']}

Trade setup:
- Entry: \u20b9{trade['entry']}
- Stop Loss: \u20b9{trade['stop_loss']} ({trade['risk_pct']}% risk)
- Target 1: \u20b9{trade['target1']} ({trade['reward1_pct']}% reward, R:R = 1:{trade['rr1']})
- Target 2: \u20b9{trade['target2']} ({trade['reward2_pct']}% reward, R:R = 1:{trade['rr2']})

Recent news:
{news_text or 'No significant news'}

Write a clear, factual summary for a retail swing trader. Mention the risk/reward ratio and key risks. Do not give financial advice."""

        response = client.chat.completions.create(
            model=LLM_MODEL,
            messages=[
                {"role": "system", "content": "You are a swing-trading research analyst assistant."},
                {"role": "user", "content": prompt},
            ],
            max_tokens=250,
            temperature=0.3,
        )
        return response.choices[0].message.content.strip()

    except Exception as e:
        print(f"  [!] LLM summary failed for {result['ticker']}: {e}")
        return _rule_based_summary(result)


def enrich_with_summary(result):
    """Add a qualitative summary to a screening result."""
    if USE_LLM_SUMMARY:
        result["summary"] = _llm_summary(result)
    else:
        result["summary"] = _rule_based_summary(result)
    return result


# ─── Market Overview ─────────────────────────────────────────────

def market_overview(results):
    """Generate a brief market overview from all screened stocks."""
    if USE_LLM_SUMMARY:
        try:
            from openai import OpenAI
            client = OpenAI(api_key=OPENAI_API_KEY)

            top_picks = results[:5]
            summaries = "\n".join([
                f"- {r['ticker']} ({r['sector']}): {r['signal']}, "
                f"trend={r['trend']}, "
                f"R:R=1:{r['trade']['rr1'] if r['trade'] else 'N/A'}"
                for r in top_picks
            ])

            prompt = f"""Write a 3-4 sentence market overview for today's NSE swing-trading session based on these top picks:

{summaries}

What sectors are strong? What's the overall market mood for swing traders? Be factual and concise."""

            response = client.chat.completions.create(
                model=LLM_MODEL,
                messages=[
                    {"role": "system", "content": "You are a swing-trading market analyst."},
                    {"role": "user", "content": prompt},
                ],
                max_tokens=150,
                temperature=0.3,
            )
            return response.choices[0].message.content.strip()
        except Exception:
            pass

    # Rule-based overview
    bull_count = sum(1 for r in results if r["trend"] == "bullish")
    bear_count = sum(1 for r in results if r["trend"] == "bearish")
    buy_count = sum(1 for r in results if "BUY" in r["signal"])

    sectors = {}
    for r in results[:10]:
        sec = r["sector"]
        sectors[sec] = sectors.get(sec, 0) + 1

    if sectors:
        top_sectors = sorted(sectors.items(), key=lambda x: x[1], reverse=True)
        sector_names = [s[0] for s in top_sectors[:3]]

        mood = "bullish" if bull_count > bear_count else "bearish" if bear_count > bull_count else "mixed"

        top = results[0]
        top_rr = top['trade']['rr1'] if top.get('trade') else 'N/A'

        return (
            f"Today's swing screening identified {len(results)} stocks "
            f"({bull_count} bullish, {bear_count} bearish, mood: {mood}). "
            f"{buy_count} stocks rated BUY. "
            f"Strongest sectors: {', '.join(sector_names)}. "
            f"Top pick: {top['name']} ({top['ticker']}) "
            f"with a {top['signal']} signal "
            f"and R:R of 1:{top_rr}."
        )
    return "No stocks met the screening criteria today."
