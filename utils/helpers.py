"""
Shared utility functions for the stock trader application.
"""

import numpy as np
import pandas as pd
from datetime import datetime
from typing import List, Tuple

import config


def detect_swing_highs(df: pd.DataFrame, lookback: int = None) -> pd.Series:
    """
    Detect swing highs: a bar whose high is greater than the highs
    of `lookback` bars on each side.

    Returns a boolean Series (True at swing high bars).
    """
    lookback = lookback or config.SWING_LOOKBACK
    highs = df["High"].values
    n = len(highs)
    swing = np.zeros(n, dtype=bool)

    for i in range(lookback, n - lookback):
        window_left = highs[i - lookback : i]
        window_right = highs[i + 1 : i + lookback + 1]
        if highs[i] > window_left.max() and highs[i] > window_right.max():
            swing[i] = True

    return pd.Series(swing, index=df.index, name="swing_high")


def detect_swing_lows(df: pd.DataFrame, lookback: int = None) -> pd.Series:
    """
    Detect swing lows: a bar whose low is less than the lows
    of `lookback` bars on each side.

    Returns a boolean Series (True at swing low bars).
    """
    lookback = lookback or config.SWING_LOOKBACK
    lows = df["Low"].values
    n = len(lows)
    swing = np.zeros(n, dtype=bool)

    for i in range(lookback, n - lookback):
        window_left = lows[i - lookback : i]
        window_right = lows[i + 1 : i + lookback + 1]
        if lows[i] < window_left.min() and lows[i] < window_right.min():
            swing[i] = True

    return pd.Series(swing, index=df.index, name="swing_low")


def is_in_kill_zone(timestamp: datetime, zone: str = None) -> bool:
    """
    Check whether a given timestamp falls within an ICT kill zone.
    If `zone` is None, checks all kill zones and returns True if any match.
    """
    hour = timestamp.hour
    zones = {zone: config.KILL_ZONES[zone]} if zone else config.KILL_ZONES

    for _, (start, end) in zones.items():
        if start <= hour <= end:
            return True
    return False


def get_active_kill_zone(timestamp: datetime) -> str | None:
    """Return the name of the active kill zone, or None."""
    hour = timestamp.hour
    for name, (start, end) in config.KILL_ZONES.items():
        if start <= hour <= end:
            return name
    return None


def calculate_position_size(
    capital: float,
    entry_price: float,
    stop_loss_price: float,
    risk_pct: float = None,
) -> int:
    """
    Calculate position size (number of shares) based on risk per trade.
    """
    risk_pct = risk_pct or config.RISK_PER_TRADE
    risk_amount = capital * risk_pct
    risk_per_share = abs(entry_price - stop_loss_price)

    if risk_per_share == 0:
        return 0

    shares = int(risk_amount / risk_per_share)
    return max(shares, 0)


def pct_change(old: float, new: float) -> float:
    """Percentage change from old to new."""
    if old == 0:
        return 0.0
    return ((new - old) / abs(old)) * 100.0


def candle_body(row: pd.Series) -> float:
    """Absolute body size of a candle."""
    return abs(row["Close"] - row["Open"])


def candle_range(row: pd.Series) -> float:
    """Full range (high - low) of a candle."""
    return row["High"] - row["Low"]


def body_ratio(row: pd.Series) -> float:
    """Body-to-range ratio.  1.0 = marubozu, 0.0 = doji."""
    r = candle_range(row)
    if r == 0:
        return 0.0
    return candle_body(row) / r


def is_bullish_candle(row: pd.Series) -> bool:
    return row["Close"] > row["Open"]


def is_bearish_candle(row: pd.Series) -> bool:
    return row["Close"] < row["Open"]


def upper_wick(row: pd.Series) -> float:
    return row["High"] - max(row["Open"], row["Close"])


def lower_wick(row: pd.Series) -> float:
    return min(row["Open"], row["Close"]) - row["Low"]


def wick_to_range(row: pd.Series, side: str = "upper") -> float:
    """Ratio of upper or lower wick to total range."""
    r = candle_range(row)
    if r == 0:
        return 0.0
    w = upper_wick(row) if side == "upper" else lower_wick(row)
    return w / r


def get_equilibrium(high: float, low: float) -> float:
    """Midpoint / equilibrium of a range."""
    return (high + low) / 2.0


def is_premium(price: float, range_high: float, range_low: float) -> bool:
    """Is price in the premium zone (above equilibrium)?"""
    eq = get_equilibrium(range_high, range_low)
    return price > eq


def is_discount(price: float, range_high: float, range_low: float) -> bool:
    """Is price in the discount zone (below equilibrium)?"""
    eq = get_equilibrium(range_high, range_low)
    return price < eq
