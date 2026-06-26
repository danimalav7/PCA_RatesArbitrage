# ============================================================================
# data/auction.py — Treasury auction calendar and suppression logic
# ============================================================================
# Provides:
#   fetch_auction_calendar()        — fetch upcoming Treasury auction calendar
#                                     from TreasuryDirect REST API
#   get_auction_suppression_flag()  — determine suppression flag for a tenor
#                                     on a given date
#
# Suppression windows (from config.AUCTION_SUPPRESS_DAYS) are in TRADING DAYS
# not calendar days. Uses numpy.busdaycalendar for business day counting.
#
# Called from both run_daily.py and run_backtest.py.
# Previously these functions lived in run_daily.py — moved here in Sprint D1.
# ============================================================================

import numpy as np
import pandas as pd
import requests
import warnings
import config


def fetch_auction_calendar(mode: str = config.MODE) -> pd.DataFrame:
    """
    Fetch upcoming Treasury auction calendar from TreasuryDirect.

    Moved from run_daily.py in Sprint D1.

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
        '?format=json'
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

    Moved from run_daily.py in Sprint D1. Suppression window is now checked
    in trading days (via numpy.busday_count) rather than calendar days.

    Returns
    -------
    tuple: (flag: str, info: dict)
        flag: 'SUPPRESS' | 'WARN' | 'CLEAR'
        info: dict with next_auction_date, days_to_next (trading days, not calendar days)
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

    # Count trading days between date and next_auction (exclusive of start,
    # inclusive of end) using numpy busday_count
    trading_days_to_next = int(np.busday_count(
        date.date(),
        next_auction.date()
    ))

    info = {
        'next_auction_date': (
            next_auction.date()
            if hasattr(next_auction, 'date') else next_auction
        ),
        'days_to_next': trading_days_to_next,  # now trading days not calendar
    }

    if trading_days_to_next <= suppress_days:
        return 'SUPPRESS', info
    elif trading_days_to_next <= suppress_days + 2:
        return 'WARN', info
    else:
        return 'CLEAR', info
