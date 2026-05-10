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
import pandas as pd
import config

from data.fetch_rates import FetchRates
from analytics.pca import compute_pca_residuals
from analytics.stationarity import (
    compute_rolling_adf,
    compute_acf_summary,
)
from signals.zscore import compute_zscore
from backtest.engine import run_backtest, run_strategy_diagnostics
from run_daily import fetch_auction_calendar, get_auction_suppression_flag


def main():
    print(f"\n{'='*60}")
    print(f"PCA RATES ARBITRAGE — BACKTEST RUNNER")
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
          f"({rates_data['Date'].iloc[0].date()} → "
          f"{rates_data['Date'].iloc[-1].date()})")

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
    rolling_adfs = compute_rolling_adf(
        residuals_df,
        windows=config.ROLLING_ADF_WINDOWS,
        mode=config.MODE,
    )
    rolling_adf_60d = rolling_adfs[config.ADF_ENTRY_WINDOW]

    # Step 4: Z-scores
    print("\nStep 4: Computing Z-scores...")
    z_score_df = compute_zscore(
        residuals_df,
        window=config.ZSCORE_WINDOW,
        mode=config.MODE,
    )

    # Step 5: ACF summary and auction calendar
    print("\nStep 5: Computing ACF summary and fetching auction calendar...")
    acf_summary_df   = compute_acf_summary(residuals_df)
    auction_calendar = fetch_auction_calendar(mode=config.MODE)

    # Step 6: Run backtest
    print("\nStep 6: Running backtest...")
    backtest_results = run_backtest(
        residuals_df=residuals_df,
        z_score_df=z_score_df,
        rolling_adf_60d=rolling_adf_60d,
        rolling_adfs=rolling_adfs,
        acf_summary_df=acf_summary_df,
        auction_calendar=auction_calendar,
        get_auction_flag_fn=get_auction_suppression_flag,
        mode=config.MODE,
    )

    # Step 7: Confirm before diagnostics
    input("\nValidation complete. Press Enter to run strategy diagnostics...")

    # Step 8: Strategy diagnostics
    print("\nStep 7: Running strategy diagnostics...")
    run_strategy_diagnostics(
        backtest_results=backtest_results,
        dv01_map=config.DV01_MAP,
        notional_map=config.NOTIONAL_MAP,
        z_entry_threshold=config.Z_ENTRY_THRESHOLD,
    )

    print(f"\n{'='*60}")
    print(f"✓ Backtest complete")
    print(f"{'='*60}\n")


if __name__ == '__main__':
    main()
