# ============================================================================
# run_daily.py — EOD daily runner
# ============================================================================
# Entry point for the daily signal generation pipeline.
# Intended to be run once per trading day after market close via
# Windows Task Scheduler or cron.
#
# Pipeline:
#   1. Fetch EOD rates from Federal Reserve via OpenBB
#   2. Compute rolling PCA residuals
#   3. Compute rolling ADF stationarity windows
#   4. Compute Z-scores
#   5. Fetch auction calendar
#   6. Scan signals for most recent date
#   7. Generate and save HTML report
#
# Usage:
#   python run_daily.py
# ============================================================================

import datetime as dt
import numpy as np
import pandas as pd
import config

from data.fetch_rates import FetchRates
from analytics.pca import compute_pca_residuals
from analytics.stationarity import (
    compute_rolling_adf,
    compute_rolling_kpss,
    compute_acf_summary,
    compute_hurst_exponent,
)
from signals.zscore import compute_zscore, scan_signals
from reports.daily_report import generate_daily_report
from data.auction import fetch_auction_calendar, get_auction_suppression_flag


def main():
    print(f"\n{'='*60}")
    print(f"PCA RATES ARBITRAGE — DAILY RUNNER")
    print(f"Mode: {config.MODE} | {dt.datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print(f"{'='*60}\n")

    # Step 1: Fetch rates
    print("Step 1: Fetching EOD rates...")
    rates_data = FetchRates(
        start_date=config.START_DATE,
        end_date=None,
        mode=config.MODE,
    )
    print(f"  Fetched {len(rates_data)} trading days "
          f"({rates_data['Date'].iloc[0]} → "
          f"{rates_data['Date'].iloc[-1]})")

    # Step 2: PCA residuals
    print("\nStep 2: Computing rolling PCA residuals...")
    residuals_df = compute_pca_residuals(
        rates_data,
        window=config.PCA_WINDOW,
        n_components=config.N_COMPONENTS,
        mode=config.MODE,
    )
    print(f"  Residuals computed: {residuals_df.dropna().shape[0]} valid dates")

    # Step 3: Rolling ADF stationarity
    print("\nStep 3: Computing rolling ADF stationarity windows...")
    # Level residuals require 252d and 180d windows
    # 252d: entry gate (most reliable — 10/10 tenors TRADEABLE)
    # 180d: exit vote (catches regime breaks 8-24 days before 252d)
    # 120d: computed for display/escalation only (BORDERLINE statistical power)
    rolling_adfs    = compute_rolling_adf(
        residuals_df,
        windows=config.ROLLING_ADF_WINDOWS,  # [120, 180, 252]
        mode=config.MODE,
    )
    rolling_adf_252d = rolling_adfs[252]
    rolling_adf_180d = rolling_adfs[180]

    print("\nStep 3b: Computing rolling KPSS (252d entry gate)...")
    rolling_kpss_252d = compute_rolling_kpss(
        residuals_df,
        window=252,
        mode=config.MODE,
    )

    print("Step 3c: Computing rolling KPSS (180d exit vote)...")
    rolling_kpss_180d = compute_rolling_kpss(
        residuals_df,
        window=180,
        mode=config.MODE,
    )

    # Step 4: Z-scores
    print("\nStep 4: Computing Z-scores...")
    z_score_df = compute_zscore(
        residuals_df,
        window=config.ZSCORE_WINDOW,
        mode=config.MODE,
    )

    # Step 5: ACF summary and auction calendar
    print("\nStep 5: Computing ACF summary, Hurst exponents, and auction calendar...")
    acf_summary_df    = compute_acf_summary(residuals_df)
    auction_calendar  = fetch_auction_calendar(mode=config.MODE)
    segment_regime_df = pd.DataFrame()  # dead variable — not used in live logic

    # Hurst exponent — documentation and monitoring only, not a live gate
    # H = 0.80-0.86 for level residuals: long memory, 15-25 day MR horizon
    hurst_df = compute_hurst_exponent(residuals_df)
    print(f"  Hurst exponents (H=0.5 random walk, H<0.5 MR, H>0.5 persistent):")
    for tenor in config.TENORS:
        h = hurst_df.loc[tenor, 'H'] if tenor in hurst_df.index else np.nan
        print(f"    {tenor:<8} H={h:.3f}")

    # Step 6: Signal scan for most recent date
    print("\nStep 6: Scanning signals...")
    report_date = z_score_df.index[-1]
    signal_scan = scan_signals(
        date=report_date,
        residuals_df=residuals_df,
        z_score_df=z_score_df,
        rolling_adfs=rolling_adfs,
        rolling_adf_252d=rolling_adf_252d,
        rolling_kpss_252d=rolling_kpss_252d,
        acf_summary_df=acf_summary_df,
        auction_calendar=auction_calendar,
        segment_regime_df=segment_regime_df,
        get_auction_flag_fn=get_auction_suppression_flag,
        mode=config.MODE,
    )

    # Print signal scan summary
    print(f"\n  Signal scan for {report_date.date()}:")
    for _, row in signal_scan.iterrows():
        if row['signal_direction'] != 'FLAT':
            print(f"  ★ {row['tenor']:<6} | Z={row['z_score']:>7.3f} | "
                  f"{row['signal_direction']:<5} | {row['trade_eligibility']}")
        else:
            print(f"    {row['tenor']:<6} | Z={row['z_score']:>7.3f} | FLAT")

    # Step 7: Generate HTML report
    print("\nStep 7: Generating HTML report...")
    report_path = generate_daily_report(
        signal_scan=signal_scan,
        z_score_df=z_score_df,
        rolling_adf_252d=rolling_adf_252d,
        report_date=report_date,
        output_dir=config.REPORT_OUTPUT_DIR,
        mode=config.MODE,
    )

    print(f"\n{'='*60}")
    print(f"✓ Daily run complete — {report_date.date()}")
    print(f"  Report: {report_path}")
    print(f"{'='*60}\n")


if __name__ == '__main__':
    main()
