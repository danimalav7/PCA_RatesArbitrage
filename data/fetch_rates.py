# ============================================================================
# data/fetch_rates.py — Treasury yield curve data fetcher
# ============================================================================
# Fetches EOD Treasury rates from Federal Reserve via OpenBB.
# Returns a clean DataFrame indexed by date with standardized tenor columns.
#
# Intraday feed (IBKR) is handled in data/ibkr_feed.py — deferred to Sprint 4.
# ============================================================================

import datetime as dt
import pandas as pd
from openbb import obb


def FetchRates(
    start_date: str = None,
    end_date: str = None,
    mode: str = 'EOD'
) -> pd.DataFrame:
    """
    Fetch Treasury yield curve data from Federal Reserve via OpenBB.

    Parameters
    ----------
    start_date : str, optional
        Start date in 'YYYY-MM-DD' format. Defaults to config.START_DATE.
    end_date : str, optional
        End date in 'YYYY-MM-DD' format. Defaults to today.
    mode : str
        Data mode — 'EOD', 'intraday', or '5day'.
        Only 'EOD' is implemented; intraday deferred to Sprint 4.

    Returns
    -------
    pd.DataFrame
        Columns: Date, FedFunds, SOFR, 1Mo, 3Mo, 6Mo, 1Yr, 2Yr, 3Yr,
                 5Yr, 7Yr, 10Yr, 30Yr
        Sorted ascending by Date. Weekends and holidays dropped.
        SOFR forward-filled for missing days.

    Raises
    ------
    NotImplementedError
        If mode is not 'EOD'.
    ValueError
        If the fetched DataFrame is empty.
    """
    if mode != 'EOD':
        raise NotImplementedError(
            f"mode='{mode}' is not yet implemented. "
            f"Intraday feed via IBKR is deferred to Sprint 4. "
            f"Use mode='EOD' for now."
        )

    # Default dates
    if end_date is None:
        end_date = dt.datetime.today().strftime('%Y-%m-%d')

    # ── Fetch from Federal Reserve via OpenBB ─────────────────────────────────
    treasury_data = obb.fixedincome.government.treasury_rates(
        start_date=start_date,
        end_date=end_date,
        provider='federal_reserve'
    ).to_df()

    fed_funds = obb.fixedincome.rate.effr(
        start_date=start_date,
        end_date=end_date,
        provider='federal_reserve'
    ).to_df()[['rate']].rename(columns={'rate': 'FedFunds'})

    sofr_data = obb.fixedincome.rate.sofr(
        start_date=start_date,
        end_date=end_date,
        provider='federal_reserve'
    ).to_df()[['rate']].rename(columns={'rate': 'SOFR'})

    # ── Merge ─────────────────────────────────────────────────────────────────
    rates_data = treasury_data.join([fed_funds, sofr_data], how='outer')
    rates_data = rates_data.rename_axis('Date').reset_index()

    # ── Rename columns to standard tenor labels ───────────────────────────────
    rates_data.rename(columns={
        'month_1': '1Mo', 'month_2': '2Mo', 'month_3': '3Mo', 'month_6': '6Mo',
        'year_1':  '1Yr', 'year_2':  '2Yr', 'year_3':  '3Yr',
        'year_5':  '5Yr', 'year_7':  '7Yr', 'year_10': '10Yr', 'year_30': '30Yr'
    }, inplace=True)

    # ── Select and sort ───────────────────────────────────────────────────────
    cols = ['Date', 'FedFunds', 'SOFR',
            '1Mo', '3Mo', '6Mo', '1Yr', '2Yr', '3Yr', '5Yr', '7Yr', '10Yr', '30Yr']
    rates_data = rates_data[cols].sort_values('Date').reset_index(drop=True)

    # ── Drop weekends and holidays (rows missing primary benchmarks) ──────────
    rates_data = rates_data.dropna(subset=['FedFunds', '1Mo'])

    # ── Forward fill SOFR for missing days ────────────────────────────────────
    rates_data['SOFR'] = rates_data['SOFR'].ffill()

    # ── Validate ──────────────────────────────────────────────────────────────
    if rates_data.empty:
        raise ValueError(
            f"FetchRates returned empty DataFrame for "
            f"start_date={start_date}, end_date={end_date}, mode={mode}"
        )

    return rates_data
