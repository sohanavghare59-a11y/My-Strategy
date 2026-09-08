from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

TRADES_FILE = (
    BASE_DIR
    / "output"
    / "directional_rsi_trades.csv"
)

OUTPUT_FILE = (
    BASE_DIR
    / "output"
    / "directional_rsi_bootstrap.csv"
)


# ============================================================
# SETTINGS
# ============================================================

N_BOOTSTRAPS = 10000
RANDOM_SEED = 42

CURRENT = "CURRENT"
LONG_ZONE = "LONG_RSI_ZONE"


# ============================================================
# HELPERS
# ============================================================

def bootstrap_mean_difference(
    current,
    long_zone,
    rng,
    iterations,
):

    current = np.asarray(
        current,
        dtype=float,
    )

    long_zone = np.asarray(
        long_zone,
        dtype=float,
    )

    differences = np.empty(
        iterations
    )

    for i in range(iterations):

        current_sample = rng.choice(
            current,
            size=len(current),
            replace=True,
        )

        zone_sample = rng.choice(
            long_zone,
            size=len(long_zone),
            replace=True,
        )

        differences[i] = (
            zone_sample.mean()
            - current_sample.mean()
        )

    return differences


def permutation_test(
    current,
    long_zone,
    rng,
    iterations,
):

    current = np.asarray(
        current,
        dtype=float,
    )

    long_zone = np.asarray(
        long_zone,
        dtype=float,
    )

    observed = (
        long_zone.mean()
        - current.mean()
    )

    combined = np.concatenate(
        [
            current,
            long_zone,
        ]
    )

    n_current = len(current)

    count = 0

    for _ in range(iterations):

        shuffled = rng.permutation(
            combined
        )

        shuffled_current = (
            shuffled[:n_current]
        )

        shuffled_zone = (
            shuffled[n_current:]
        )

        difference = (
            shuffled_zone.mean()
            - shuffled_current.mean()
        )

        if difference >= observed:
            count += 1

    p_value = (
        count + 1
    ) / (
        iterations + 1
    )

    return observed, p_value


def bootstrap_win_rate_difference(
    current,
    long_zone,
    rng,
    iterations,
):

    current = np.asarray(
        current,
        dtype=float,
    )

    long_zone = np.asarray(
        long_zone,
        dtype=float,
    )

    differences = np.empty(
        iterations
    )

    for i in range(iterations):

        current_sample = rng.choice(
            current,
            size=len(current),
            replace=True,
        )

        zone_sample = rng.choice(
            long_zone,
            size=len(long_zone),
            replace=True,
        )

        current_rate = (
            current_sample.mean()
            * 100
        )

        zone_rate = (
            zone_sample.mean()
            * 100
        )

        differences[i] = (
            zone_rate
            - current_rate
        )

    return differences


def confidence_interval(values):

    return (
        float(
            np.percentile(
                values,
                2.5,
            )
        ),
        float(
            np.percentile(
                values,
                97.5,
            )
        ),
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 95)
    print(
        "DIRECTIONAL RSI BOOTSTRAP / SIGNIFICANCE TEST"
    )
    print("=" * 95)

    if not TRADES_FILE.exists():

        print()
        print(
            "ERROR: Missing trade file:"
        )
        print(TRADES_FILE)
        return

    trades = pd.read_csv(
        TRADES_FILE
    )

    required = [
        "variant",
        "symbol",
        "net_pnl_pct",
    ]

    missing = [
        column
        for column in required
        if column not in trades.columns
    ]

    if missing:

        print()
        print(
            "ERROR: Missing columns:"
        )

        for column in missing:
            print(
                f"  - {column}"
            )

        return

    trades["net_pnl_pct"] = pd.to_numeric(
        trades["net_pnl_pct"],
        errors="coerce",
    )

    trades = trades.dropna(
        subset=[
            "variant",
            "net_pnl_pct",
        ]
    )

    trades = trades[
        trades["variant"].isin(
            [
                CURRENT,
                LONG_ZONE,
            ]
        )
    ].copy()

    rng = np.random.default_rng(
        RANDOM_SEED
    )

    results = []

    # ========================================================
    # FULL SAMPLE
    # ========================================================

    current_trades = trades[
        trades["variant"]
        == CURRENT
    ]["net_pnl_pct"].to_numpy()

    zone_trades = trades[
        trades["variant"]
        == LONG_ZONE
    ]["net_pnl_pct"].to_numpy()

    print()
    print(
        f"CURRENT trades      : "
        f"{len(current_trades)}"
    )

    print(
        f"LONG_RSI_ZONE trades: "
        f"{len(zone_trades)}"
    )

    observed = (
        zone_trades.mean()
        - current_trades.mean()
    )

    bootstrap = (
        bootstrap_mean_difference(
            current_trades,
            zone_trades,
            rng,
            N_BOOTSTRAPS,
        )
    )

    ci_low, ci_high = (
        confidence_interval(
            bootstrap
        )
    )

    probability_positive = (
        float(
            (bootstrap > 0).mean()
            * 100
        )
    )

    permutation_observed, permutation_p = (
        permutation_test(
            current_trades,
            zone_trades,
            rng,
            N_BOOTSTRAPS,
        )
    )

    print()
    print("=" * 95)
    print(
        "FULL SAMPLE RETURN DIFFERENCE"
    )
    print("=" * 95)

    print(
        f"CURRENT average net:"
        f" {current_trades.mean():+.4f}%"
    )

    print(
        f"LONG_RSI_ZONE average net:"
        f" {zone_trades.mean():+.4f}%"
    )

    print(
        f"Observed improvement:"
        f" {observed:+.4f}%"
    )

    print(
        f"Bootstrap 95% CI:"
        f" [{ci_low:+.4f}%, "
        f"{ci_high:+.4f}%]"
    )

    print(
        f"Probability improvement > 0:"
        f" {probability_positive:.2f}%"
    )

    print(
        f"Permutation p-value:"
        f" {permutation_p:.4f}"
    )

    results.append(
        {
            "test":
                "FULL_SAMPLE",
            "current_trades":
                len(current_trades),
            "zone_trades":
                len(zone_trades),
            "observed_delta_pct":
                observed,
            "bootstrap_ci_low_pct":
                ci_low,
            "bootstrap_ci_high_pct":
                ci_high,
            "probability_zone_better_pct":
                probability_positive,
            "permutation_p_value":
                permutation_p,
        }
    )

    # ========================================================
    # WIN RATE TEST
    # ========================================================

    current_wins = (
        current_trades > 0
    ).astype(float)

    zone_wins = (
        zone_trades > 0
    ).astype(float)

    win_bootstrap = (
        bootstrap_win_rate_difference(
            current_wins,
            zone_wins,
            rng,
            N_BOOTSTRAPS,
        )
    )

    win_observed = (
        zone_wins.mean()
        - current_wins.mean()
    ) * 100

    win_low, win_high = (
        confidence_interval(
            win_bootstrap
        )
    )

    win_probability = (
        float(
            (win_bootstrap > 0).mean()
            * 100
        )
    )

    print()
    print("=" * 95)
    print(
        "WIN RATE DIFFERENCE"
    )
    print("=" * 95)

    print(
        f"CURRENT win rate:"
        f" {current_wins.mean() * 100:.2f}%"
    )

    print(
        f"LONG_RSI_ZONE win rate:"
        f" {zone_wins.mean() * 100:.2f}%"
    )

    print(
        f"Observed difference:"
        f" {win_observed:+.2f} pp"
    )

    print(
        f"Bootstrap 95% CI:"
        f" [{win_low:+.2f}, "
        f"{win_high:+.2f}] pp"
    )

    print(
        f"Probability Zone wins:"
        f" {win_probability:.2f}%"
    )

    results.append(
        {
            "test":
                "WIN_RATE",
            "current_trades":
                len(current_wins),
            "zone_trades":
                len(zone_wins),
            "observed_delta_pct":
                win_observed,
            "bootstrap_ci_low_pct":
                win_low,
            "bootstrap_ci_high_pct":
                win_high,
            "probability_zone_better_pct":
                win_probability,
            "permutation_p_value":
                np.nan,
        }
    )

    # ========================================================
    # TOP WINNERS REMOVED
    # ========================================================

    print()
    print("=" * 95)
    print(
        "ROBUSTNESS AFTER REMOVING TOP WINNERS"
    )
    print("=" * 95)

    print(
        f"{'Removed':>10s}"
        f"{'Current':>12s}"
        f"{'Zone':>12s}"
        f"{'Delta':>12s}"
        f"{'CI Low':>12s}"
        f"{'CI High':>12s}"
        f"{'P(Zone>0)':>14s}"
    )

    print("-" * 95)

    current_sorted = np.sort(
        current_trades
    )[::-1]

    zone_sorted = np.sort(
        zone_trades
    )[::-1]

    for remove_n in [
        0,
        1,
        3,
        5,
        10,
    ]:

        current_reduced = (
            current_sorted[
                remove_n:
            ]
        )

        zone_reduced = (
            zone_sorted[
                remove_n:
            ]
        )

        if (
            len(current_reduced) == 0
            or len(zone_reduced) == 0
        ):
            continue

        observed_reduced = (
            zone_reduced.mean()
            - current_reduced.mean()
        )

        bootstrap_reduced = (
            bootstrap_mean_difference(
                current_reduced,
                zone_reduced,
                rng,
                N_BOOTSTRAPS,
            )
        )

        reduced_low, reduced_high = (
            confidence_interval(
                bootstrap_reduced
            )
        )

        reduced_probability = (
            float(
                (bootstrap_reduced > 0).mean()
                * 100
            )
        )

        print(
            f"{remove_n:10d}"
            f"{current_reduced.mean():+11.3f}%"
            f"{zone_reduced.mean():+11.3f}%"
            f"{observed_reduced:+11.3f}%"
            f"{reduced_low:+11.3f}%"
            f"{reduced_high:+11.3f}%"
            f"{reduced_probability:13.2f}%"
        )

        results.append(
            {
                "test":
                    f"REMOVE_TOP_{remove_n}",
                "current_trades":
                    len(current_reduced),
                "zone_trades":
                    len(zone_reduced),
                "observed_delta_pct":
                    observed_reduced,
                "bootstrap_ci_low_pct":
                    reduced_low,
                "bootstrap_ci_high_pct":
                    reduced_high,
                "probability_zone_better_pct":
                    reduced_probability,
                "permutation_p_value":
                    np.nan,
            }
        )

    # ========================================================
    # COMPANY-LEVEL PAIRED TEST
    # ========================================================

    print()
    print("=" * 95)
    print(
        "COMPANY-LEVEL PAIRED BOOTSTRAP"
    )
    print("=" * 95)

    company_stats = (
        trades
        .groupby(
            [
                "variant",
                "symbol",
            ]
        )["net_pnl_pct"]
        .mean()
        .unstack(
            "variant"
        )
    )

    if (
        CURRENT in company_stats.columns
        and LONG_ZONE in company_stats.columns
    ):

        company_stats = (
            company_stats
            .dropna(
                subset=[
                    CURRENT,
                    LONG_ZONE,
                ]
            )
            .copy()
        )

        company_stats[
            "delta"
        ] = (
            company_stats[
                LONG_ZONE
            ]
            - company_stats[
                CURRENT
            ]
        )

        company_deltas = (
            company_stats[
                "delta"
            ].to_numpy(
                dtype=float
            )
        )

        if len(company_deltas) > 0:

            company_bootstrap = np.empty(
                N_BOOTSTRAPS
            )

            for i in range(
                N_BOOTSTRAPS
            ):

                sample = rng.choice(
                    company_deltas,
                    size=len(
                        company_deltas
                    ),
                    replace=True,
                )

                company_bootstrap[i] = (
                    sample.mean()
                )

            company_low, company_high = (
                confidence_interval(
                    company_bootstrap
                )
            )

            company_probability = (
                float(
                    (
                        company_bootstrap
                        > 0
                    ).mean()
                    * 100
                )
            )

            print(
                f"Paired companies:"
                f" {len(company_deltas)}"
            )

            print(
                f"Average company delta:"
                f" {company_deltas.mean():+.4f}%"
            )

            print(
                f"Median company delta:"
                f" {np.median(company_deltas):+.4f}%"
            )

            print(
                f"Bootstrap 95% CI:"
                f" [{company_low:+.4f}%, "
                f"{company_high:+.4f}%]"
            )

            print(
                f"Probability average delta > 0:"
                f" {company_probability:.2f}%"
            )

            results.append(
                {
                    "test":
                        "COMPANY_LEVEL",
                    "current_trades":
                        np.nan,
                    "zone_trades":
                        np.nan,
                    "observed_delta_pct":
                        company_deltas.mean(),
                    "bootstrap_ci_low_pct":
                        company_low,
                    "bootstrap_ci_high_pct":
                        company_high,
                    "probability_zone_better_pct":
                        company_probability,
                    "permutation_p_value":
                        np.nan,
                }
            )

    # ========================================================
    # SAVE
    # ========================================================

    result_df = pd.DataFrame(
        results
    )

    result_df.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    print()
    print("=" * 95)
    print("FILES SAVED")
    print("=" * 95)

    print(
        f"Bootstrap results:"
        f" {OUTPUT_FILE}"
    )

    print()
    print("=" * 95)
    print(
        "BOOTSTRAP TEST COMPLETE"
    )
    print("=" * 95)


if __name__ == "__main__":
    main()