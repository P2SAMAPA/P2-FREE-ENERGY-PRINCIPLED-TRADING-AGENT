"""
us_calendar.py  —  US trading calendar helper.
"""

import pandas as pd
from pandas_market_calendars import get_calendar


def get_us_trading_days(start_date: str, end_date: str) -> pd.DatetimeIndex:
    """Get US trading days between start_date and end_date."""
    nyse = get_calendar('NYSE')
    schedule = nyse.schedule(start_date=start_date, end_date=end_date)
    return schedule.index


def is_trading_day(date: pd.Timestamp) -> bool:
    """Check if a given date is a US trading day."""
    nyse = get_calendar('NYSE')
    schedule = nyse.schedule(start_date=date, end_date=date)
    return len(schedule) > 0


def align_to_trading_days(dates: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """Align a datetime index to the nearest US trading days."""
    nyse = get_calendar('NYSE')
    return nyse.valid_days(start_date=dates.min(), end_date=dates.max())
