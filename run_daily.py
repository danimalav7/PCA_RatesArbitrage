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
import pandas as pd
import requests
import warnings
import config

from data.fetch_rates import FetchRates
from analytics.pca import compute_pca_residuals
from analytics.stationarity import (
    compute_rolling_adf,
    compute_acf_summary,
)
from signals.zscore import compute_zscore, scan_signals
from reports.daily_report import generate_daily_report


# ── Auction calendar fetch ────────────────────────────────────────────────────
def fetch_auction_calendar(mode: str = config.MODE) -> pd.DataFrame:
    """
    Fetch upcoming Treasury auction calendar from TreasuryDirect.

    Returns
    -------
    pd.DataFrame
        Auction calendar with columns: cusip, securityType, securityTerm,
        announcementDate, auctionDate, issueDate.
        Returns empty DataFrame on fetch failure.
    """
    if mode != 'EOD':
        return pd.DataFrame()

    url = (
        'https://www.treasurydirect.gov/TA_WS/securities/announced'
        '?format=json&type=Bill,Note,Bond'
    )
    try:
        resp = requests.get(url, timeout=15)
        resp.raise_for_status()
        data = resp.json()
        df   = pd.DataFrame(data)
        for col in ['announcementDate', 'auctionDate', 'issueDate']:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors='coerce')
        return df
    except Exception as e:
        warnings.warn(f"Auction calendar fetch failed: {e}. "
                      f"Returning empty DataFrame.")
        return pd.DataFrame()


def get_auction_suppression_flag(
    date: pd.Timestamp,
    tenor: str,
    auction_calendar: pd.DataFrame,
    mode: str = config.MODE,
) -> tuple:
    """
    Determine auction suppression flag for a tenor on a given date.

    Returns
    -------
    tuple: (flag: str, info: dict)
        flag: 'SUPPRESS' | 'WARN' | 'CLEAR'
        info: dict with next_auction_date, days_to_next
    """
    empty_info = {'next_auction_date': None, 'days_to_next': None}

    if auction_calendar.empty or 'auctionDate' not in auction_calendar.columns:
        return 'CLEAR', empty_info

    # Tenor → security type mapping
    tenor_type_map = {
        '1Mo': 'Bill', '3Mo': 'Bill', '6Mo': 'Bill',
        '1Yr': 'Bill',
        '2Yr': 'Note', '3Yr': 'Note', '5Yr': 'Note',
        '7Yr': 'Note', '10Yr': 'Note',
        '30Yr': 'Bond',
    }
    security_type = tenor_type_map.get(tenor)
    if security_type is None:
        return 'CLEAR', empty_info

    suppress_days = config.AUCTION_SUPPRESS_DAYS.get(security_type, 5)

    # Filter to relevant security type
    if 'securityType' not in auction_calendar.columns:
        return 'CLEAR', empty_info

    relevant = auction_calendar[
        auction_calendar['securityType'].str.lower()
        == security_type.lower()
    ].dropna(subset=['auctionDate'])

    if relevant.empty:
        return 'CLEAR', empty_info

    # Find next auction on or after date
    future = relevant[relevant['auctionDate'] >= date].sort_values('auctionDate')
    if future.empty:
        return 'CLEAR', empty_info

    next_auction = future.iloc[0]['auctionDate']
    days_to_next = (next_auction - date).days

    info = {
        'next_auction_date': (
            next_auction.date()
            if hasattr(next_auction, 'date') else next_auction
        ),
        'days_to_next': days_to_next,
    }

    if days_to_next <= suppress_days:
        return 'SUPPRESS', info
    elif days_to_next <= suppress_days + 2:
        return 'WARN', info
    else:
        return 'CLEAR', info


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
    acf_summary_df    = compute_acf_summary(residuals_df)
    auction_calendar  = fetch_auction_calendar(mode=config.MODE)
    segment_regime_df = pd.DataFrame()  # dead variable — not used in live logic

    # Step 6: Signal scan for most recent date
    print("\nStep 6: Scanning signals...")
    report_date = z_score_df.index[-1]
    signal_scan = scan_signals(
        date=report_date,
        residuals_df=residuals_df,
        z_score_df=z_score_df,
        rolling_adfs=rolling_adfs,
        rolling_adf_60d=rolling_adf_60d,
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
        rolling_adf_60d=rolling_adf_60d,
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
