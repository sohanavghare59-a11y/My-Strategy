"""
Trade Quality Analysis
======================

Research-only diagnostic.

Analyzes the existing controlled RSI experiment and compares:

    CURRENT
        EMA + MACD + RSI

    EMA_MACD
        EMA + MACD

The analysis examines:

- RSI
- MACD histogram
- EMA separation
- Volume ratio
- Direction
- Market regime
- Exit reason
- Holding period
- Winner vs loser behavior
- Indicator buckets

No production strategy settings are modified.
No market data is downloaded.
"""

from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

INPUT_FILE = (
    BASE_DIR
    / "output"
    / "controlled_rsi_trades.csv"
)

OUTPUT_OVERALL = (
    BASE_DIR
    / "output"
    / "trade_quality_overall.csv"
)

OUTPUT_BUCKETS = (
    BASE_DIR
    / "output"
    / "trade_quality_buckets.csv"
)

OUTPUT_DIRECTION = (
    BASE_DIR
    / "output"
    / "trade_quality_direction.csv"
)

OUTPUT_REGIME = (
    BASE_DIR
    / "output"
    / "trade_quality_regime.csv"
)

OUTPUT_EXIT = (
    BASE_DIR
    / "output"
    / "trade_quality_exit.csv"
)


# ============================================================
# HELPERS
# ============================================================

def find_column(df, candidates):

    lower_map = {
        str(column).lower().strip(): column
        for column in df.columns
    }

    for candidate in candidates:

        key = candidate.lower().strip()

        if key in lower_map:
            return lower_map[key]

    return None


def print_separator():

    print("-" * 100)


def safe_mean(series):

    series = pd.to_numeric(
        series,
        errors="coerce",
    ).dropna()

    if series.empty:
        return 0.0

    return float(series.mean())


def calculate_metrics(df):

    if df.empty:

        return {
            "trades": 0,
            "win_rate_pct": 0.0,
            "avg_net_pct": 0.0,
            "profit_factor": 0.0,
            "avg_winner_pct": 0.0,
            "avg_loser_pct": 0.0,
            "avg_holding_days": 0.0,
        }

    net = pd.to_numeric(
        df["net_pnl_pct"],
        errors="coerce",
    ).dropna()

    if net.empty:

        return {
            "trades": 0,
            "win_rate_pct": 0.0,
            "avg_net_pct": 0.0,
            "profit_factor": 0.0,
            "avg_winner_pct": 0.0,
            "avg_loser_pct": 0.0,
            "avg_holding_days": 0.0,
        }

    winners = net[
        net > 0
    ]

    losers = net[
        net <= 0
    ]

    gross_profit = winners.sum()

    gross_loss = abs(
        losers.sum()
    )

    if gross_loss > 0:

        pf = (
            gross_profit
            / gross_loss
        )

    else:

        pf = np.inf

    holding = pd.to_numeric(
        df["holding_days"],
        errors="coerce",
    )

    return {
        "trades": int(len(net)),

        "win_rate_pct": (
            (net > 0).mean()
            * 100
        ),

        "avg_net_pct": float(
            net.mean()
        ),

        "profit_factor": float(
            pf
        ),

        "avg_winner_pct": (
            float(winners.mean())
            if not winners.empty
            else 0.0
        ),

        "avg_loser_pct": (
            float(losers.mean())
            if not losers.empty
            else 0.0
        ),

        "avg_holding_days": (
            float(holding.mean())
            if holding.notna().any()
            else 0.0
        ),
    }


def make_bucket(
    series,
    labels,
):

    numeric = pd.to_numeric(
        series,
        errors="coerce",
    )

    try:

        result = pd.qcut(
            numeric,
            q=4,
            labels=labels,
            duplicates="drop",
        )

    except Exception:

        result = pd.Series(
            [np.nan] * len(series),
            index=series.index,
        )

    return result


# ============================================================
# LOAD DATA
# ============================================================

def load_data():

    if not INPUT_FILE.exists():

        raise FileNotFoundError(
            f"Missing input file:\n{INPUT_FILE}"
        )

    df = pd.read_csv(
        INPUT_FILE
    )

    required = [
        "variant",
        "direction",
        "regime",
        "net_pnl_pct",
        "holding_days",
    ]

    missing = [
        column
        for column in required
        if column not in df.columns
    ]

    if missing:

        raise ValueError(
            "Missing required columns: "
            + ", ".join(missing)
        )

    df["variant"] = (
        df["variant"]
        .astype(str)
        .str.upper()
        .str.strip()
    )

    df["direction"] = (
        df["direction"]
        .astype(str)
        .str.upper()
        .str.strip()
    )

    df["regime"] = (
        df["regime"]
        .astype(str)
        .str.upper()
        .str.strip()
    )

    return df


# ============================================================
# DETECT INDICATOR COLUMNS
# ============================================================

def detect_columns(df):

    columns = {}

    columns["rsi"] = find_column(
        df,
        [
            "rsi",
            "rsi_value",
            "entry_rsi",
            "signal_rsi",
        ],
    )

    columns["macd_histogram"] = find_column(
        df,
        [
            "macd_histogram",
            "macd_hist",
            "macd_diff",
            "histogram",
            "entry_macd_histogram",
        ],
    )

    columns["ema_5"] = find_column(
        df,
        [
            "ema_5",
            "ema5",
            "entry_ema_5",
        ],
    )

    columns["ema_13"] = find_column(
        df,
        [
            "ema_13",
            "ema13",
            "entry_ema_13",
        ],
    )

    columns["ema_26"] = find_column(
        df,
        [
            "ema_26",
            "ema26",
            "entry_ema_26",
        ],
    )

    columns["volume_ratio"] = find_column(
        df,
        [
            "volume_ratio",
            "volume_ratio_entry",
            "entry_volume_ratio",
        ],
    )

    columns["macd"] = find_column(
        df,
        [
            "macd",
            "entry_macd",
        ],
    )

    columns["macd_signal"] = find_column(
        df,
        [
            "macd_signal",
            "macd_signal_line",
            "entry_macd_signal",
        ],
    )

    columns["exit_reason"] = find_column(
        df,
        [
            "exit_reason",
        ],
    )

    return columns


# ============================================================
# OVERALL
# ============================================================

def analyze_overall(df):

    rows = []

    for variant in [
        "CURRENT",
        "EMA_MACD",
    ]:

        subset = df[
            df["variant"]
            == variant
        ]

        metrics = calculate_metrics(
            subset
        )

        metrics["variant"] = variant

        rows.append(
            metrics
        )

    result = pd.DataFrame(
        rows
    )

    return result


# ============================================================
# GROUP ANALYSIS
# ============================================================

def analyze_group(
    df,
    group_column,
    output_name,
):

    rows = []

    groups = [
        value
        for value in
        df[group_column]
        .dropna()
        .unique()
    ]

    for variant in [
        "CURRENT",
        "EMA_MACD",
    ]:

        variant_df = df[
            df["variant"]
            == variant
        ]

        for group in sorted(
            groups,
            key=str,
        ):

            subset = variant_df[
                variant_df[
                    group_column
                ]
                == group
            ]

            if subset.empty:
                continue

            metrics = calculate_metrics(
                subset
            )

            metrics["variant"] = variant

            metrics[
                group_column
            ] = group

            rows.append(
                metrics
            )

    return pd.DataFrame(
        rows
    )


# ============================================================
# BUCKET ANALYSIS
# ============================================================

def analyze_buckets(
    df,
    column_name,
    metric_name,
):

    if column_name is None:

        return pd.DataFrame()

    numeric = pd.to_numeric(
        df[column_name],
        errors="coerce",
    )

    valid = df[
        numeric.notna()
    ].copy()

    if len(valid) < 20:

        return pd.DataFrame()

    valid[
        f"{metric_name}_bucket"
    ] = make_bucket(
        valid[column_name],
        [
            "Q1_LOWEST",
            "Q2",
            "Q3",
            "Q4_HIGHEST",
        ],
    )

    rows = []

    bucket_column = (
        f"{metric_name}_bucket"
    )

    for variant in [
        "CURRENT",
        "EMA_MACD",
    ]:

        variant_df = valid[
            valid["variant"]
            == variant
        ]

        for bucket in [
            "Q1_LOWEST",
            "Q2",
            "Q3",
            "Q4_HIGHEST",
        ]:

            subset = variant_df[
                variant_df[
                    bucket_column
                ]
                == bucket
            ]

            if subset.empty:
                continue

            metrics = calculate_metrics(
                subset
            )

            metrics["variant"] = variant

            metrics["factor"] = (
                metric_name
            )

            metrics["bucket"] = (
                bucket
            )

            metrics["raw_mean"] = (
                safe_mean(
                    subset[
                        column_name
                    ]
                )
            )

            rows.append(
                metrics
            )

    return pd.DataFrame(
        rows
    )


# ============================================================
# EMA SEPARATION
# ============================================================

def add_ema_separation(
    df,
    columns,
):

    ema5 = columns["ema_5"]
    ema13 = columns["ema_13"]
    ema26 = columns["ema_26"]

    if not all(
        [
            ema5,
            ema13,
            ema26,
        ]
    ):

        return df

    e5 = pd.to_numeric(
        df[ema5],
        errors="coerce",
    )

    e13 = pd.to_numeric(
        df[ema13],
        errors="coerce",
    )

    e26 = pd.to_numeric(
        df[ema26],
        errors="coerce",
    )

    denominator = e26.abs()

    denominator = denominator.replace(
        0,
        np.nan,
    )

    df["ema_separation_pct"] = (
        (e5 - e26)
        / denominator
        * 100
    )

    return df


# ============================================================
# MACD HISTOGRAM
# ============================================================

def add_macd_histogram(
    df,
    columns,
):

    if columns[
        "macd_histogram"
    ] is not None:

        return df

    macd = columns["macd"]
    signal = columns["macd_signal"]

    if not macd or not signal:

        return df

    m = pd.to_numeric(
        df[macd],
        errors="coerce",
    )

    s = pd.to_numeric(
        df[signal],
        errors="coerce",
    )

    df["macd_histogram_calc"] = (
        m - s
    )

    columns[
        "macd_histogram"
    ] = "macd_histogram_calc"

    return df


# ============================================================
# PRINT GROUP TABLE
# ============================================================

def print_group_table(
    title,
    result,
    group_column,
):

    print()
    print("=" * 100)
    print(title)
    print("=" * 100)

    if result.empty:

        print(
            "No data available."
        )

        return

    for variant in [
        "CURRENT",
        "EMA_MACD",
    ]:

        subset = result[
            result["variant"]
            == variant
        ]

        print()
        print(
            variant
        )

        print(
            f"{group_column:15s}"
            f"{'Trades':>9s}"
            f"{'Win %':>10s}"
            f"{'Avg Net':>11s}"
            f"{'PF':>8s}"
            f"{'Avg Hold':>11s}"
        )

        print("-" * 70)

        for _, row in subset.iterrows():

            pf = row[
                "profit_factor"
            ]

            pf_text = (
                "INF"
                if np.isinf(pf)
                else f"{pf:.2f}"
            )

            print(
                f"{str(row[group_column]):15s}"
                f"{int(row['trades']):9d}"
                f"{row['win_rate_pct']:9.2f}%"
                f"{row['avg_net_pct']:+10.2f}%"
                f"{pf_text:>8s}"
                f"{row['avg_holding_days']:10.2f}"
            )


# ============================================================
# EXIT ANALYSIS
# ============================================================

def analyze_exit(
    df,
    columns,
):

    exit_column = columns[
        "exit_reason"
    ]

    if exit_column is None:

        return pd.DataFrame()

    rows = []

    for variant in [
        "CURRENT",
        "EMA_MACD",
    ]:

        variant_df = df[
            df["variant"]
            == variant
        ]

        for exit_reason in sorted(
            variant_df[
                exit_column
            ]
            .dropna()
            .astype(str)
            .unique()
        ):

            subset = variant_df[
                variant_df[
                    exit_column
                ]
                .astype(str)
                == exit_reason
            ]

            metrics = calculate_metrics(
                subset
            )

            metrics["variant"] = (
                variant
            )

            metrics["exit_reason"] = (
                exit_reason
            )

            rows.append(
                metrics
            )

    return pd.DataFrame(
        rows
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 100)
    print("TRADE QUALITY ANALYSIS")
    print("=" * 100)

    print()
    print(
        "Research only."
    )

    print(
        "No production strategy settings will be changed."
    )

    # ========================================================
    # LOAD
    # ========================================================

    df = load_data()

    print()
    print(
        f"Trades loaded : {len(df)}"
    )

    print(
        f"Columns found : {len(df.columns)}"
    )

    # ========================================================
    # SHOW COLUMNS
    # ========================================================

    print()
    print("=" * 100)
    print("INPUT COLUMNS")
    print("=" * 100)

    for column in df.columns:

        print(
            f"  {column}"
        )

    # ========================================================
    # DETECT
    # ========================================================

    columns = detect_columns(
        df
    )

    df = add_ema_separation(
        df,
        columns,
    )

    df = add_macd_histogram(
        df,
        columns,
    )

    print()
    print("=" * 100)
    print("DETECTED INDICATORS")
    print("=" * 100)

    for key, value in columns.items():

        print(
            f"{key:20s}: "
            f"{value}"
        )

    if "ema_separation_pct" in df.columns:

        columns[
            "ema_separation"
        ] = "ema_separation_pct"

    else:

        columns[
            "ema_separation"
        ] = None

    # ========================================================
    # OVERALL
    # ========================================================

    overall = analyze_overall(
        df
    )

    print()
    print("=" * 100)
    print("OVERALL TRADE QUALITY")
    print("=" * 100)

    print(
        f"{'Variant':18s}"
        f"{'Trades':>9s}"
        f"{'Win %':>10s}"
        f"{'Avg Net':>11s}"
        f"{'PF':>8s}"
        f"{'Winner':>11s}"
        f"{'Loser':>11s}"
        f"{'Avg Hold':>11s}"
    )

    print("-" * 90)

    for _, row in overall.iterrows():

        pf = row[
            "profit_factor"
        ]

        pf_text = (
            "INF"
            if np.isinf(pf)
            else f"{pf:.2f}"
        )

        print(
            f"{row['variant']:18s}"
            f"{int(row['trades']):9d}"
            f"{row['win_rate_pct']:9.2f}%"
            f"{row['avg_net_pct']:+10.2f}%"
            f"{pf_text:>8s}"
            f"{row['avg_winner_pct']:+10.2f}%"
            f"{row['avg_loser_pct']:+10.2f}%"
            f"{row['avg_holding_days']:10.2f}"
        )

    # ========================================================
    # DIRECTION
    # ========================================================

    direction = analyze_group(
        df,
        "direction",
        "direction",
    )

    print_group_table(
        "DIRECTION ANALYSIS",
        direction,
        "direction",
    )

    # ========================================================
    # REGIME
    # ========================================================

    regime = analyze_group(
        df,
        "regime",
        "regime",
    )

    print_group_table(
        "MARKET REGIME ANALYSIS",
        regime,
        "regime",
    )

    # ========================================================
    # EXIT
    # ========================================================

    exit_result = analyze_exit(
        df,
        columns,
    )

    print()
    print("=" * 100)
    print("EXIT ANALYSIS")
    print("=" * 100)

    if exit_result.empty:

        print(
            "Exit reason column not available."
        )

    else:

        print(
            f"{'Variant':18s}"
            f"{'Exit':18s}"
            f"{'Trades':>9s}"
            f"{'Win %':>10s}"
            f"{'Avg Net':>11s}"
            f"{'Avg Hold':>11s}"
        )

        print("-" * 90)

        for _, row in exit_result.iterrows():

            print(
                f"{row['variant']:18s}"
                f"{row['exit_reason']:18s}"
                f"{int(row['trades']):9d}"
                f"{row['win_rate_pct']:9.2f}%"
                f"{row['avg_net_pct']:+10.2f}%"
                f"{row['avg_holding_days']:10.2f}"
            )

    # ========================================================
    # INDICATOR BUCKETS
    # ========================================================

    bucket_results = []

    factors = [
        (
            "rsi",
            columns["rsi"],
        ),
        (
            "macd_histogram",
            columns["macd_histogram"],
        ),
        (
            "ema_separation",
            columns["ema_separation"],
        ),
        (
            "volume_ratio",
            columns["volume_ratio"],
        ),
    ]

    for metric_name, column_name in factors:

        if column_name is None:

            continue

        result = analyze_buckets(
            df,
            column_name,
            metric_name,
        )

        if not result.empty:

            bucket_results.append(
                result
            )

    if bucket_results:

        buckets = pd.concat(
            bucket_results,
            ignore_index=True,
        )

    else:

        buckets = pd.DataFrame()

    # ========================================================
    # PRINT BUCKETS
    # ========================================================

    print()
    print("=" * 100)
    print("INDICATOR BUCKET ANALYSIS")
    print("=" * 100)

    if buckets.empty:

        print(
            "No suitable indicator columns "
            "were available for bucket analysis."
        )

    else:

        for factor in buckets[
            "factor"
        ].unique():

            print()
            print(
                factor.upper()
            )

            factor_df = buckets[
                buckets["factor"]
                == factor
            ]

            print(
                f"{'Variant':18s}"
                f"{'Bucket':15s}"
                f"{'Trades':>9s}"
                f"{'Raw Mean':>11s}"
                f"{'Win %':>10s}"
                f"{'Avg Net':>11s}"
                f"{'PF':>8s}"
            )

            print("-" * 90)

            for _, row in factor_df.iterrows():

                pf = row[
                    "profit_factor"
                ]

                pf_text = (
                    "INF"
                    if np.isinf(pf)
                    else f"{pf:.2f}"
                )

                print(
                    f"{row['variant']:18s}"
                    f"{row['bucket']:15s}"
                    f"{int(row['trades']):9d}"
                    f"{row['raw_mean']:11.2f}"
                    f"{row['win_rate_pct']:9.2f}%"
                    f"{row['avg_net_pct']:+10.2f}%"
                    f"{pf_text:>8s}"
                )

    # ========================================================
    # RSI WINNER / LOSER COMPARISON
    # ========================================================

    if columns["rsi"] is not None:

        print()
        print("=" * 100)
        print("RSI WINNER VS LOSER ANALYSIS")
        print("=" * 100)

        rsi = pd.to_numeric(
            df[
                columns["rsi"]
            ],
            errors="coerce",
        )

        valid = df[
            rsi.notna()
        ].copy()

        valid[
            "rsi_analysis"
        ] = rsi[
            rsi.notna()
        ]

        valid[
            "result"
        ] = np.where(
            pd.to_numeric(
                valid["net_pnl_pct"],
                errors="coerce",
            ) > 0,
            "WIN",
            "LOSS",
        )

        for variant in [
            "CURRENT",
            "EMA_MACD",
        ]:

            subset = valid[
                valid["variant"]
                == variant
            ]

            print()
            print(
                variant
            )

            for result_name in [
                "WIN",
                "LOSS",
            ]:

                values = subset[
                    subset["result"]
                    == result_name
                ][
                    "rsi_analysis"
                ]

                if values.empty:
                    continue

                print(
                    f"{result_name:5s}"
                    f" count={len(values):4d}"
                    f" avg_RSI={values.mean():.2f}"
                    f" median_RSI={values.median():.2f}"
                    f" min={values.min():.2f}"
                    f" max={values.max():.2f}"
                )

    # ========================================================
    # MACD WINNER / LOSER
    # ========================================================

    if columns[
        "macd_histogram"
    ] is not None:

        print()
        print("=" * 100)
        print("MACD HISTOGRAM WINNER VS LOSER ANALYSIS")
        print("=" * 100)

        hist = pd.to_numeric(
            df[
                columns[
                    "macd_histogram"
                ]
            ],
            errors="coerce",
        )

        valid = df[
            hist.notna()
        ].copy()

        valid[
            "hist_analysis"
        ] = hist[
            hist.notna()
        ]

        valid[
            "result"
        ] = np.where(
            pd.to_numeric(
                valid["net_pnl_pct"],
                errors="coerce",
            ) > 0,
            "WIN",
            "LOSS",
        )

        for variant in [
            "CURRENT",
            "EMA_MACD",
        ]:

            subset = valid[
                valid["variant"]
                == variant
            ]

            print()
            print(
                variant
            )

            for result_name in [
                "WIN",
                "LOSS",
            ]:

                values = subset[
                    subset["result"]
                    == result_name
                ][
                    "hist_analysis"
                ]

                if values.empty:
                    continue

                print(
                    f"{result_name:5s}"
                    f" count={len(values):4d}"
                    f" avg_hist={values.mean():.6f}"
                    f" median={values.median():.6f}"
                )

    # ========================================================
    # SAVE
    # ========================================================

    overall.to_csv(
        OUTPUT_OVERALL,
        index=False,
    )

    if not buckets.empty:

        buckets.to_csv(
            OUTPUT_BUCKETS,
            index=False,
        )

    direction.to_csv(
        OUTPUT_DIRECTION,
        index=False,
    )

    regime.to_csv(
        OUTPUT_REGIME,
        index=False,
    )

    if not exit_result.empty:

        exit_result.to_csv(
            OUTPUT_EXIT,
            index=False,
        )

    # ========================================================
    # FINISH
    # ========================================================

    print()
    print("=" * 100)
    print("FILES SAVED")
    print("=" * 100)

    print(
        OUTPUT_OVERALL
    )

    if not buckets.empty:

        print(
            OUTPUT_BUCKETS
        )

    print(
        OUTPUT_DIRECTION
    )

    print(
        OUTPUT_REGIME
    )

    if not exit_result.empty:

        print(
            OUTPUT_EXIT
        )

    print()
    print("=" * 100)
    print("TRADE QUALITY ANALYSIS COMPLETE")
    print("=" * 100)


if __name__ == "__main__":
    main()