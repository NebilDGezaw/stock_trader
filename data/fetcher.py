"""
Stock data fetcher using the yfinance API.
"""

import yfinance as yf
import pandas as pd
from datetime import datetime
from typing import Optional

import config


class StockDataFetcher:
    """Fetches historical stock data via yfinance."""

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
                f"(period={period}, interval={interval})"
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
