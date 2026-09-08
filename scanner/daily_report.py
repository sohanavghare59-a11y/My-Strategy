"""
Daily Swing Trading Report Generator
====================================

Reads:
    output/ranked_daily_signals.csv

Creates:
    output/daily_report.txt

The report is based only on signals already generated
by the deterministic strategy and ranking engine.
"""

from __future__ import annotations

from datetime import datetime

import pandas as pd


INPUT_FILE = "output/ranked_daily_signals.csv"
OUTPUT_FILE = "output/daily_report.txt"


def money(value) -> str:
    """Format a number as Indian Rupees."""
    try:
        return f"₹{float(value):,.2f}"
    except (TypeError, ValueError):
        return "N/A"


def number(value, decimals: int = 2) -> str:
    """Format a numeric value."""
    try:
        return f"{float(value):,.{decimals}f}"
    except (TypeError, ValueError):
        return "N/A"


def build_signal_line(row: pd.Series, position: int) -> list[str]:
    """Build report lines for one signal."""

    symbol = str(row.get("symbol", "N/A"))
    exchange = str(row.get("exchange", "N/A"))
    signal = str(row.get("signal", "N/A")).upper()
    quality = str(row.get("quality", "N/A"))
    score = number(row.get("quality_score"))

    entry = money(row.get("entry"))
    stop = money(row.get("stop_loss"))
    target = money(row.get("target"))

    rsi = number(row.get("rsi"))
    volume_ratio = number(row.get("volume_ratio"))

    risk_reward = number(row.get("risk_reward"))

    signal_date = str(row.get("signal_date", "N/A"))

    return [
        f"{position}. {symbol} ({exchange})",
        f"   Direction : {signal}",
        f"   Quality   : {quality} ({score}/100)",
        f"   Date      : {signal_date}",
        f"   Entry     : {entry}",
        f"   Stop Loss : {stop}",
        f"   Target    : {target}",
        f"   Risk/Reward: {risk_reward}:1",
        f"   RSI       : {rsi}",
        f"   Volume    : {volume_ratio}x average",
        "",
    ]


def generate_report(df: pd.DataFrame) -> str:
    """Generate complete daily trading report."""

    signals = df[
        df["scan_status"].astype(str).str.upper() == "SIGNAL"
    ].copy()

    if signals.empty:
        today = datetime.now().strftime("%d-%b-%Y")

        return (
            "=" * 70
            + "\n"
            + "NSE/BSE DAILY SWING TRADING REPORT\n"
            + "=" * 70
            + "\n\n"
            + f"Report Date: {today}\n\n"
            + "NO VALID SWING SIGNALS TODAY.\n"
        )

    signals["quality_score"] = pd.to_numeric(
        signals["quality_score"],
        errors="coerce",
    )

    signals = signals.sort_values(
        "quality_score",
        ascending=False,
    )

    longs = signals[
        signals["signal"].astype(str).str.upper() == "LONG"
    ].copy()

    shorts = signals[
        signals["signal"].astype(str).str.upper() == "SHORT"
    ].copy()

    strong_count = (
        signals["quality"].astype(str).str.upper() == "STRONG"
    ).sum()

    good_count = (
        signals["quality"].astype(str).str.upper() == "GOOD"
    ).sum()

    moderate_count = (
        signals["quality"].astype(str).str.upper() == "MODERATE"
    ).sum()

    report_date = datetime.now().strftime("%d-%b-%Y")

    lines: list[str] = []

    lines.append("=" * 70)
    lines.append("NSE/BSE DAILY SWING TRADING REPORT")
    lines.append("=" * 70)
    lines.append("")
    lines.append(f"Report Date: {report_date}")
    lines.append("Strategy: EMA 5/13/26 + MACD + RSI")
    lines.append("Timeframe: Daily")
    lines.append("")

    lines.append("-" * 70)
    lines.append("SIGNAL SUMMARY")
    lines.append("-" * 70)
    lines.append(f"Total Signals : {len(signals)}")
    lines.append(f"LONG Signals  : {len(longs)}")
    lines.append(f"SHORT Signals : {len(shorts)}")
    lines.append(f"STRONG        : {strong_count}")
    lines.append(f"GOOD          : {good_count}")
    lines.append(f"MODERATE      : {moderate_count}")
    lines.append("")

    # ==============================================================
    # TOP LONG SIGNALS
    # ==============================================================

    lines.append("=" * 70)
    lines.append("TOP LONG OPPORTUNITIES")
    lines.append("=" * 70)
    lines.append("")

    if longs.empty:
        lines.append("No LONG signals.")
        lines.append("")

    else:
        for position, (_, row) in enumerate(
            longs.head(5).iterrows(),
            start=1,
        ):
            lines.extend(
                build_signal_line(
                    row,
                    position,
                )
            )

    # ==============================================================
    # TOP SHORT SIGNALS
    # ==============================================================

    lines.append("=" * 70)
    lines.append("TOP SHORT OPPORTUNITIES")
    lines.append("=" * 70)
    lines.append("")

    if shorts.empty:
        lines.append("No SHORT signals.")
        lines.append("")

    else:
        for position, (_, row) in enumerate(
            shorts.head(5).iterrows(),
            start=1,
        ):
            lines.extend(
                build_signal_line(
                    row,
                    position,
                )
            )

    # ==============================================================
    # ALL SIGNALS
    # ==============================================================

    lines.append("=" * 70)
    lines.append("ALL VALID SIGNALS")
    lines.append("=" * 70)
    lines.append("")

    for position, (_, row) in enumerate(
        signals.iterrows(),
        start=1,
    ):
        symbol = str(row.get("symbol", "N/A"))
        signal = str(row.get("signal", "N/A")).upper()
        quality = str(row.get("quality", "N/A"))
        score = number(row.get("quality_score"))

        lines.append(
            f"{position:02d}. "
            f"{symbol:<15} "
            f"{signal:<6} "
            f"{quality:<9} "
            f"{score}/100"
        )

    lines.append("")

    # ==============================================================
    # RISK RULES
    # ==============================================================

    lines.append("=" * 70)
    lines.append("RISK MANAGEMENT")
    lines.append("=" * 70)
    lines.append("")
    lines.append("Stop Loss      : 2.5%")
    lines.append("Target         : 5.0%")
    lines.append("Minimum R:R    : 2.0:1")
    lines.append("")
    lines.append(
        "IMPORTANT: Position size should be determined from "
        "account risk, not from conviction or signal score."
    )
    lines.append("")

    # ==============================================================
    # DISCLAIMER
    # ==============================================================

    lines.append("=" * 70)
    lines.append("SYSTEM NOTE")
    lines.append("=" * 70)
    lines.append("")
    lines.append(
        "This report is generated from the deterministic trading "
        "strategy and ranking engine."
    )
    lines.append(
        "It is a research/decision-support output and is not a "
        "guarantee of future returns."
    )
    lines.append("")

    return "\n".join(lines)


def main() -> None:
    """Generate and save the daily report."""

    print("=" * 70)
    print("DAILY SWING TRADING REPORT GENERATOR")
    print("=" * 70)

    print()
    print(f"Loading: {INPUT_FILE}")

    df = pd.read_csv(INPUT_FILE)

    print(f"Rows loaded: {len(df)}")

    report = generate_report(df)

    with open(
        OUTPUT_FILE,
        "w",
        encoding="utf-8",
    ) as file:
        file.write(report)

    print()
    print(report)

    print()
    print("=" * 70)
    print("REPORT GENERATION COMPLETE")
    print("=" * 70)
    print()
    print(f"Saved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()