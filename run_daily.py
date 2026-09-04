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
import os
import pickle
import numpy as np
import pandas as pd
import config

from signals.zscore import compute_zscore, scan_signals
from reports.daily_report import generate_daily_report
from data.auction import fetch_auction_calendar, get_auction_suppression_flag


def main():
    from utils.pipeline import setup_logging, build_pipeline_inputs, send_alert

    # Initialise logging
    log_path = setup_logging(log_dir='logs', log_file='daily.log')

    print(f"\n{'='*60}")
    print(f"PCA RATES ARBITRAGE — DAILY RUNNER")
    print(f"Mode: {config.MODE} | "
          f"{dt.datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print(f"{'='*60}\n")

    # Validate config at startup
    config.config_validate()

    try:
        # Steps 1-9: shared pipeline
        inputs = build_pipeline_inputs(
            start_date=config.START_DATE,
            end_date=None,
            mode=config.MODE,
        )
    except RuntimeError as e:
        send_alert(
            subject="PIPELINE FAILURE — daily runner",
            body=f"build_pipeline_inputs() failed:\n\n{e}\n\nCheck logs: {log_path}",
            mode=config.MODE,
        )
        raise

    # Unpack pipeline inputs
    rates_data              = inputs['rates_data']
    residuals_df            = inputs['residuals_df']
    cumulative_variance_s   = inputs['cumulative_variance_s']
    rolling_adfs            = inputs['rolling_adfs']
    rolling_adf_252d        = inputs['rolling_adf_252d']
    rolling_kpss_252d       = inputs['rolling_kpss_252d']
    z_score_df              = inputs['z_score_df']
    acf_summary_df          = inputs['acf_summary_df']
    auction_calendar        = inputs['auction_calendar']

    # Step 10: Signal scan for most recent date
    print("\nStep 10: Scanning signals...")
    report_date = z_score_df.index[-1]
    signal_scan = scan_signals(
        date=report_date,
        residuals_df=residuals_df,
        z_score_df=z_score_df,
        rolling_adfs=rolling_adfs,
        rolling_adf_252d=rolling_adf_252d,
        rolling_kpss_252d=rolling_kpss_252d,
        cumulative_variance_s=cumulative_variance_s,
        acf_summary_df=acf_summary_df,
        auction_calendar=auction_calendar,
        segment_regime_df=pd.DataFrame(),
        get_auction_flag_fn=get_auction_suppression_flag,
        mode=config.MODE,
    )

    # Print signal scan summary
    print(f"\n  Signal scan for {report_date.date()}:")
    for _, row in signal_scan.iterrows():
        if row['signal_direction'] != 'FLAT':
            print(f"  ★ {row['tenor']:<6} | Z={row['z_score']:>7.3f} | "
                  f"{row['signal_direction']:<5} | "
                  f"{row['trade_eligibility']}")
        else:
            print(f"    {row['tenor']:<6} | Z={row['z_score']:>7.3f} | FLAT")

    # Step 11: Load saved backtest results
    print("\nStep 11: Loading saved backtest results...")
    backtest_results = None
    backtest_pkl = os.path.join(
        config.REPORT_OUTPUT_DIR, 'backtest_results.pkl'
    )
    if os.path.exists(backtest_pkl):
        with open(backtest_pkl, 'rb') as f:
            backtest_results = pickle.load(f)
        print(f"  Loaded backtest results from {backtest_pkl}")
    else:
        print(f"  No saved backtest results found at {backtest_pkl}")
        print(f"  Run run_backtest.py to generate backtest results.")
        print(f"  Dashboard Section 2 will be empty until then.")

    # Step 12: Generate HTML report
    print("\nStep 12: Generating HTML report...")
    report_path = generate_daily_report(
        signal_scan=signal_scan,
        z_score_df=z_score_df,
        cumulative_variance_s=cumulative_variance_s,
        backtest_results=backtest_results,
        report_date=report_date,
        output_dir=config.REPORT_OUTPUT_DIR,
        mode=config.MODE,
    )

    print(f"\n{'='*60}")
    print(f"✓ Daily run complete — {report_date.date()}")
    print(f"  Report: {report_path}")
    print(f"  Log:    {log_path}")
    print(f"{'='*60}\n")


if __name__ == '__main__':
    main()
