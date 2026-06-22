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

from data.fetch_rates import FetchRates
from analytics.pca import compute_pca_residuals
from analytics.stationarity import (
    compute_rolling_adf,
    compute_rolling_kpss,
    compute_acf_summary,
    compute_hurst_exponent,
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
    # 252d: entry gate | 180d: exit vote | 120d: display/escalation only
    rolling_adfs     = compute_rolling_adf(
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
    acf_summary_df   = compute_acf_summary(residuals_df)
    auction_calendar = fetch_auction_calendar(mode=config.MODE)

    # Hurst exponent — documentation and monitoring only, not a live gate
    hurst_df = compute_hurst_exponent(residuals_df)
    print(f"  Hurst exponents:")
    for tenor in config.TENORS:
        h = hurst_df.loc[tenor, 'H'] if tenor in hurst_df.index else np.nan
        print(f"    {tenor:<8} H={h:.3f}  "
              f"{hurst_df.loc[tenor, 'Interpretation'] if tenor in hurst_df.index else ''}")

    # Step 6: Run backtest
    print("\nStep 6: Running backtest...")
    backtest_results = run_backtest(
        residuals_df=residuals_df,
        z_score_df=z_score_df,
        rolling_adf_252d=rolling_adf_252d,
        rolling_kpss_252d=rolling_kpss_252d,
        rolling_adf_180d=rolling_adf_180d,
        rolling_kpss_180d=rolling_kpss_180d,
        rolling_adfs=rolling_adfs,
        acf_summary_df=acf_summary_df,
        auction_calendar=auction_calendar,
        get_auction_flag_fn=get_auction_suppression_flag,
        mode=config.MODE,
    )

    # Step 7: Confirm before diagnostics (skip when piped / non-interactive)
    import sys
    if sys.stdin.isatty():
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
