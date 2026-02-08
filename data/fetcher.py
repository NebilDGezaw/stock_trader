"""
Stock / Crypto / Forex data fetcher using the yfinance API.
"""

import yfinance as yf
import pandas as pd
from datetime import datetime
from typing import Optional

import config


# yfinance maximum lookback per interval
# Requests beyond these limits return empty DataFrames.
_MAX_PERIOD_FOR_INTERVAL: dict[str, str] = {
    "1m":  "7d",
    "2m":  "60d",
    "5m":  "60d",
    "15m": "60d",
    "30m": "60d",
    "60m": "2y",
    "1h":  "2y",
    "90m": "60d",
    "1d":  "max",
    "5d":  "max",
    "1wk": "max",
    "1mo": "max",
    "3mo": "max",
}

# Ordered list so we can compare which period is "larger"
_PERIOD_ORDER = ["1d", "5d", "7d", "1mo", "3mo", "6mo", "60d", "1y", "2y", "5y", "10y", "max"]

def _period_index(p: str) -> int:
    try:
        return _PERIOD_ORDER.index(p)
    except ValueError:
        return len(_PERIOD_ORDER)  # unknown → treat as very large


def clamp_period(period: str, interval: str) -> str:
    """
    If the requested period exceeds what yfinance supports for the
    given interval, clamp it down to the maximum allowed period.
    Returns the (possibly adjusted) period string.
    """
    max_period = _MAX_PERIOD_FOR_INTERVAL.get(interval)
    if max_period is None or max_period == "max":
        return period
    if _period_index(period) > _period_index(max_period):
        return max_period
    return period


class StockDataFetcher:
    """Fetches historical OHLCV data for stocks, crypto, and forex via yfinance."""

    def __init__(self, ticker: str = None):
        self.ticker = ticker or config.DEFAULT_TICKER
        self._yf_ticker = yf.Ticker(self.ticker)

    # ── public API ────────────────────────────────────────

    def fetch(
        self,
        period: str = None,
        interval: str = None,
        start: Optional[str] = None,
        end: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Download OHLCV data.

        Parameters
        ----------
        period   : yfinance period string (e.g. '6mo', '1y').  Ignored when
                   start/end are provided.
        interval : candle size ('1m','5m','15m','1h','1d','1wk','1mo').
        start    : start date string 'YYYY-MM-DD' (optional).
        end      : end date string 'YYYY-MM-DD' (optional).

        Returns
        -------
        pd.DataFrame with columns: Open, High, Low, Close, Volume
        """
        period = period or config.DEFAULT_PERIOD
        interval = interval or config.DEFAULT_INTERVAL

        # Auto-clamp period to avoid yfinance "no data" errors
        original_period = period
        period = clamp_period(period, interval)

        if start and end:
            df = self._yf_ticker.history(
                start=start, end=end, interval=interval, auto_adjust=True
            )
        else:
            df = self._yf_ticker.history(
                period=period, interval=interval, auto_adjust=True
            )

        if df.empty:
            raise ValueError(
                f"No data returned for {self.ticker} "
                f"(period={period}, interval={interval}). "
                + (
                    f"Note: period was auto-adjusted from {original_period} "
                    f"to {period} for the {interval} interval."
                    if period != original_period else
                    "Check the ticker symbol and try a shorter period."
                )
            )

        # Keep only OHLCV columns and drop timezone info for simplicity
        df = df[["Open", "High", "Low", "Close", "Volume"]].copy()
        if df.index.tz is not None:
            df.index = df.index.tz_localize(None)

        return df

    def get_info(self) -> dict:
        """Return basic ticker info (name, sector, market cap, etc.)."""
        return self._yf_ticker.info

    def get_current_price(self) -> float:
        """Return the last closing price."""
        hist = self._yf_ticker.history(period="1d")
        if hist.empty:
            raise ValueError(f"Could not fetch current price for {self.ticker}")
        return float(hist["Close"].iloc[-1])

    def fetch_multiple(
        self,
        tickers: list[str],
        period: str = None,
        interval: str = None,
    ) -> dict[str, pd.DataFrame]:
        """Fetch data for multiple tickers. Returns {ticker: DataFrame}."""
        results = {}
        for t in tickers:
            try:
                fetcher = StockDataFetcher(t)
                results[t] = fetcher.fetch(period=period, interval=interval)
            except ValueError:
                continue
        return results
