# ============================================================================
# run_backtest.py — Manual backtest entry point
# ============================================================================
# Run this script to execute the full walk-forward backtest and print
# the strategy diagnostic report.
#
# Backtest runs are manual and on-demand — not part of the daily runner.
# All parameters are read from config.py.
#
# Usage:
#   python run_backtest.py
# ============================================================================

import datetime as dt
import numpy as np
import pandas as pd
import config

from backtest.engine import run_backtest, run_strategy_diagnostics
from data.auction import fetch_auction_calendar, get_auction_suppression_flag


def main():
    import os
    import pickle
    import sys
    from utils.pipeline import setup_logging, build_pipeline_inputs, send_alert

    # Initialise logging
    log_path = setup_logging(log_dir='logs', log_file='backtest.log')

    print(f"\n{'='*60}")
    print(f"PCA RATES ARBITRAGE — BACKTEST RUNNER")
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
            subject="PIPELINE FAILURE — backtest runner",
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
    rolling_adf_180d        = inputs['rolling_adf_180d']
    rolling_kpss_252d       = inputs['rolling_kpss_252d']
    rolling_kpss_180d       = inputs['rolling_kpss_180d']
    z_score_df              = inputs['z_score_df']
    acf_summary_df          = inputs['acf_summary_df']
    rolling_acf_horizons_df = inputs['rolling_acf_horizons_df']
    auction_calendar        = inputs['auction_calendar']

    # Step 10: Run backtest
    print("\nStep 10: Running backtest...")
    backtest_results = run_backtest(
        residuals_df=residuals_df,
        z_score_df=z_score_df,
        rolling_adf_252d=rolling_adf_252d,
        rolling_kpss_252d=rolling_kpss_252d,
        rolling_adf_180d=rolling_adf_180d,
        rolling_kpss_180d=rolling_kpss_180d,
        rolling_adfs=rolling_adfs,
        cumulative_variance_s=cumulative_variance_s,
        rolling_acf_horizons_df=rolling_acf_horizons_df,
        rates_data=rates_data,
        acf_summary_df=acf_summary_df,
        auction_calendar=auction_calendar,
        get_auction_flag_fn=get_auction_suppression_flag,
        mode=config.MODE,
    )

    # Step 11: Save backtest results to pickle
    print("\nStep 11: Saving backtest results...")
    os.makedirs(config.REPORT_OUTPUT_DIR, exist_ok=True)
    backtest_pkl = os.path.join(
        config.REPORT_OUTPUT_DIR, 'backtest_results.pkl'
    )
    with open(backtest_pkl, 'wb') as f:
        pickle.dump(backtest_results, f)
    print(f"  Saved backtest results to {backtest_pkl}")

    # Step 12: Confirm before diagnostics
    if sys.stdin.isatty():
        input("\nValidation complete. Press Enter to run diagnostics...")

    # Step 13: Strategy diagnostics
    print("\nStep 13: Running strategy diagnostics...")
    run_strategy_diagnostics(
        backtest_results=backtest_results,
        dv01_map=config.DV01_MAP,
        notional_map=config.NOTIONAL_MAP,
        z_entry_threshold=config.Z_ENTRY_THRESHOLD,
    )

    print(f"\n{'='*60}")
    print(f"✓ Backtest complete")
    print(f"  Results saved: {backtest_pkl}")
    print(f"  Log:           {log_path}")
    print(f"{'='*60}\n")


if __name__ == '__main__':
    main()
