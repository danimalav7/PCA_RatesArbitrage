# ============================================================================
# data/fetch_rates.py — Treasury yield curve data fetcher
# ============================================================================
# Fetches EOD Treasury rates from Federal Reserve via OpenBB.
# Falls back to direct FRED CSV download when OpenBB fixedincome extension
# is not available (e.g. CLI environments without the federal_reserve provider).
#
# Intraday feed (IBKR) is handled in data/ibkr_feed.py — deferred to Sprint 4.
# ============================================================================

import datetime as dt
import io
import requests
import pandas as pd


# ── FRED direct fallback ──────────────────────────────────────────────────────
_FRED_SERIES = {
    '1Mo':  'DGS1MO',
    '3Mo':  'DGS3MO',
    '6Mo':  'DGS6MO',
    '1Yr':  'DGS1',
    '2Yr':  'DGS2',
    '3Yr':  'DGS3',
    '5Yr':  'DGS5',
    '7Yr':  'DGS7',
    '10Yr': 'DGS10',
    '30Yr': 'DGS30',
    'FedFunds': 'DFF',
}
_FRED_BASE = 'https://fred.stlouisfed.org/graph/fredgraph.csv?id='


def _fetch_fred_series(series_id: str, start_date: str, end_date: str) -> pd.Series:
    url  = _FRED_BASE + series_id
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    df = pd.read_csv(io.StringIO(resp.text))
    df.columns = ['Date', 'value']
    df['Date']  = pd.to_datetime(df['Date'])
    df['value'] = pd.to_numeric(df['value'], errors='coerce') / 100.0  # pct → decimal
    df = df.set_index('Date')['value'].sort_index()
    if start_date:
        df = df.loc[pd.Timestamp(start_date):]
    if end_date:
        df = df.loc[:pd.Timestamp(end_date)]
    return df


def _fetch_rates_fred(start_date: str, end_date: str) -> pd.DataFrame:
    frames = {}
    for tenor, series_id in _FRED_SERIES.items():
        frames[tenor] = _fetch_fred_series(series_id, start_date, end_date)
    df = pd.DataFrame(frames).dropna(how='all')
    df = df.reset_index().rename(columns={'index': 'Date'})
    # SOFR: not yet fetched from FRED — placeholder NaN column.
    # Add SOFR fetch here when needed (FRED series: SOFR).
    df['SOFR'] = float('nan')
    # Forward fill FedFunds and SOFR for holidays and weekends where
    # FRED does not publish values (e.g. Fed Funds not published on
    # bank holidays). Use previous business day's value.
    df['FedFunds'] = df['FedFunds'].ffill()
    df['SOFR']     = df['SOFR'].ffill()
    cols = ['Date', 'FedFunds', 'SOFR',
            '1Mo', '3Mo', '6Mo', '1Yr', '2Yr', '3Yr', '5Yr', '7Yr', '10Yr', '30Yr']
    df = df[cols].sort_values('Date').reset_index(drop=True)
    df = df.dropna(subset=['1Mo'])
    return df


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

    # ── OpenBB (commented out — FRED is the active data source) ──────────────
    # OpenBB code retained for reference. To re-enable, uncomment the block
    # below and comment out the _fetch_rates_fred() call.
    # Note: OpenBB requires the fixedincome extension and federal_reserve
    # provider to be installed and configured.
    # See live trading task log — replace with Bloomberg/Refinitiv in production.
    #
    # rates_data = None
    # try:
    #     from openbb import obb
    #     if not hasattr(obb, 'fixedincome'):
    #         raise ImportError("openbb fixedincome extension not loaded")
    #
    #     treasury_data = obb.fixedincome.government.treasury_rates(
    #         start_date=start_date,
    #         end_date=end_date,
    #         provider='federal_reserve'
    #     ).to_df()
    #
    #     fed_funds = obb.fixedincome.rate.effr(
    #         start_date=start_date,
    #         end_date=end_date,
    #         provider='federal_reserve'
    #     ).to_df()[['rate']].rename(columns={'rate': 'FedFunds'})
    #
    #     sofr_data = obb.fixedincome.rate.sofr(
    #         start_date=start_date,
    #         end_date=end_date,
    #         provider='federal_reserve'
    #     ).to_df()[['rate']].rename(columns={'rate': 'SOFR'})
    #
    #     rates_data = treasury_data.join([fed_funds, sofr_data], how='outer')
    #     rates_data = rates_data.rename_axis('Date').reset_index()
    #     rates_data.rename(columns={
    #         'month_1': '1Mo', 'month_2': '2Mo', 'month_3': '3Mo', 'month_6': '6Mo',
    #         'year_1':  '1Yr', 'year_2':  '2Yr', 'year_3':  '3Yr',
    #         'year_5':  '5Yr', 'year_7':  '7Yr', 'year_10': '10Yr', 'year_30': '30Yr'
    #     }, inplace=True)
    #     cols = ['Date', 'FedFunds', 'SOFR',
    #             '1Mo', '3Mo', '6Mo', '1Yr', '2Yr', '3Yr', '5Yr', '7Yr', '10Yr', '30Yr']
    #     rates_data = rates_data[cols].sort_values('Date').reset_index(drop=True)
    #     rates_data = rates_data.dropna(subset=['FedFunds', '1Mo'])
    #     rates_data['SOFR'] = rates_data['SOFR'].ffill()
    #     print("  [FetchRates] source: OpenBB / federal_reserve")
    #
    # except Exception as obb_err:
    #     print(f"  [FetchRates] OpenBB unavailable ({obb_err}); using FRED direct CSV")
    #     rates_data = _fetch_rates_fred(start_date, end_date)
    #     print("  [FetchRates] source: FRED direct CSV")

    # ── FRED direct CSV (active data source) ─────────────────────────────────
    rates_data = _fetch_rates_fred(start_date, end_date)
    print("  [FetchRates] source: FRED direct CSV")

    # ── Validate ──────────────────────────────────────────────────────────────
    if rates_data is None or rates_data.empty:
        raise ValueError(
            f"FetchRates returned empty DataFrame for "
            f"start_date={start_date}, end_date={end_date}, mode={mode}"
        )

    return rates_data
