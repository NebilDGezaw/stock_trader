"""
Shared utility functions for the stock trader application.
"""
from __future__ import annotations

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


# ──────────────────────────────────────────────────────────
#  Fakeout Detection Helpers
# ──────────────────────────────────────────────────────────

def has_displacement(df: pd.DataFrame, bar_idx: int, direction: str = "bullish",
                     min_body_ratio: float = 0.6) -> bool:
    """
    Check if the candle at bar_idx shows strong displacement (conviction).
    A displacement candle has a large body relative to its range and closes
    convincingly beyond the break level — not just a wick poke.
    """
    if bar_idx < 0 or bar_idx >= len(df):
        return False
    row = df.iloc[bar_idx]
    br = body_ratio(row)
    if direction == "bullish":
        return br >= min_body_ratio and is_bullish_candle(row)
    else:
        return br >= min_body_ratio and is_bearish_candle(row)


def has_volume_confirmation(df: pd.DataFrame, bar_idx: int,
                            lookback: int = 20, multiplier: float = 1.2) -> bool:
    """
    Check if the volume at bar_idx is above the recent average.
    Genuine breaks tend to have higher-than-average volume;
    fakeouts tend to have low volume.
    """
    if bar_idx < lookback or bar_idx >= len(df):
        return True  # can't confirm, assume OK
    if "Volume" not in df.columns:
        return True
    recent_vol = df["Volume"].iloc[bar_idx - lookback : bar_idx].mean()
    if recent_vol == 0:
        return True
    return df.iloc[bar_idx]["Volume"] >= recent_vol * multiplier


def candle_closed_beyond(df: pd.DataFrame, bar_idx: int, level: float,
                         direction: str = "above") -> bool:
    """
    Confirm candle BODY closed beyond the level, not just a wick.
    This is the key fakeout filter — wick-only breaks are traps.
    """
    if bar_idx < 0 or bar_idx >= len(df):
        return False
    close = df.iloc[bar_idx]["Close"]
    if direction == "above":
        return close > level
    else:
        return close < level


def is_strong_reversal(df: pd.DataFrame, bar_idx: int, direction: str = "bullish",
                       min_body_ratio: float = 0.5) -> bool:
    """
    After a liquidity sweep, check if the reversal candle is strong.
    A weak reversal (doji, small body) suggests the sweep may continue
    rather than reverse — likely a fakeout.
    """
    if bar_idx < 0 or bar_idx >= len(df):
        return False
    row = df.iloc[bar_idx]
    br = body_ratio(row)
    if br < min_body_ratio:
        return False
    if direction == "bullish":
        return is_bullish_candle(row)
    else:
        return is_bearish_candle(row)


def count_consecutive_breaks(df: pd.DataFrame, bar_idx: int, level: float,
                             direction: str = "above", lookforward: int = 3) -> int:
    """
    Count how many of the next `lookforward` candles also close beyond the level.
    Genuine breaks hold; fakeouts snap back within 1-2 candles.
    """
    count = 0
    for i in range(bar_idx + 1, min(bar_idx + 1 + lookforward, len(df))):
        close = df.iloc[i]["Close"]
        if direction == "above" and close > level:
            count += 1
        elif direction == "below" and close < level:
            count += 1
    return count


# ──────────────────────────────────────────────────────────
#  Stock Mode Helpers — ATR & Trend
# ──────────────────────────────────────────────────────────

def compute_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """
    Compute Average True Range (ATR).
    ATR measures volatility and is used for dynamic SL/TP sizing.
    """
    high = df["High"]
    low = df["Low"]
    close = df["Close"].shift(1)

    tr1 = high - low
    tr2 = (high - close).abs()
    tr3 = (low - close).abs()

    true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = true_range.rolling(window=period, min_periods=1).mean()
    return atr


def trend_sma_bias(df: pd.DataFrame, period: int = 20) -> str:
    """
    Determine macro trend using Simple Moving Average.
    - Price above SMA → bullish trend
    - Price below SMA → bearish trend
    - Returns 'bullish', 'bearish', or 'neutral'
    """
    if len(df) < period:
        return "neutral"

    sma = df["Close"].rolling(window=period).mean()
    current_price = df.iloc[-1]["Close"]
    current_sma = sma.iloc[-1]

    if pd.isna(current_sma):
        return "neutral"

    # Also check slope — is the SMA itself trending?
    if len(sma.dropna()) >= 3:
        sma_slope = sma.iloc[-1] - sma.iloc[-3]
    else:
        sma_slope = 0

    if current_price > current_sma and sma_slope > 0:
        return "bullish"
    elif current_price < current_sma and sma_slope < 0:
        return "bearish"
    else:
        return "neutral"


def compute_rsi_series(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """
    Compute RSI as a full Series using Wilder's smoothing (EMA with alpha=1/period).
    This matches TradingView, ThinkorSwim, and standard RSI implementations.
    """
    delta = df["Close"].diff()
    gain = delta.where(delta > 0, 0.0)
    loss = (-delta).where(delta < 0, 0.0)
    # Wilder's smoothing: equivalent to EMA with alpha = 1/period
    avg_gain = gain.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.inf)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    return rsi


def compute_bollinger_bands(df: pd.DataFrame, period: int = 20, std: float = 2.0):
    """Returns (upper, middle, lower) Bollinger Band Series."""
    middle = df["Close"].rolling(window=period, min_periods=1).mean()
    rolling_std = df["Close"].rolling(window=period, min_periods=1).std()
    upper = middle + (rolling_std * std)
    lower = middle - (rolling_std * std)
    return upper, middle, lower


def compute_ema(df: pd.DataFrame, period: int) -> pd.Series:
    """Compute Exponential Moving Average."""
    return df["Close"].ewm(span=period, adjust=False).mean()


def compute_vwap(df: pd.DataFrame) -> pd.Series:
    """
    Volume-weighted average price, resetting each trading day.
    For intraday data: accumulates from start of each day.
    For daily data: uses a 20-bar rolling window as approximation.
    """
    typical_price = (df["High"] + df["Low"] + df["Close"]) / 3.0
    vol = df["Volume"].replace(0, np.nan).fillna(1)
    tp_vol = typical_price * vol

    # Detect if intraday: if multiple bars share the same date, reset daily
    if hasattr(df.index, 'date'):
        dates = pd.Series(df.index.date, index=df.index)
        day_groups = dates.ne(dates.shift()).cumsum()
        cum_tp_vol = tp_vol.groupby(day_groups).cumsum()
        cum_vol = vol.groupby(day_groups).cumsum()
        return cum_tp_vol / cum_vol
    else:
        # Daily data: use rolling window (true VWAP isn't meaningful here)
        window = min(20, len(df))
        cum_tp_vol = tp_vol.rolling(window=window, min_periods=1).sum()
        cum_vol = vol.rolling(window=window, min_periods=1).sum()
        return cum_tp_vol / cum_vol


def bb_squeeze_detected(df: pd.DataFrame, period: int = 20, std: float = 2.0,
                        squeeze_threshold: float = 0.5) -> bool:
    """
    Detect Bollinger Band squeeze: bandwidth narrows then expands.
    Returns True if a squeeze-to-expansion just happened.
    """
    upper, middle, lower = compute_bollinger_bands(df, period, std)
    bandwidth = (upper - lower) / middle * 100
    if len(bandwidth.dropna()) < 5:
        return False
    recent_bw = bandwidth.iloc[-1]
    min_bw = bandwidth.iloc[-20:].min() if len(bandwidth) >= 20 else bandwidth.min()
    avg_bw = bandwidth.iloc[-20:].mean() if len(bandwidth) >= 20 else bandwidth.mean()
    # Squeeze: recent bandwidth was near the minimum, now expanding
    was_squeezed = min_bw < avg_bw * squeeze_threshold
    is_expanding = recent_bw > min_bw * 1.3
    return was_squeezed and is_expanding


def compute_macd(df: pd.DataFrame, fast: int = 12, slow: int = 26, signal: int = 9):
    """Compute MACD line, signal line, and histogram."""
    ema_fast = df["Close"].ewm(span=fast, adjust=False).mean()
    ema_slow = df["Close"].ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram


def compute_obv(df: pd.DataFrame) -> pd.Series:
    """Compute On-Balance Volume."""
    direction = np.sign(df["Close"].diff()).fillna(0)
    vol = df["Volume"].fillna(0)
    obv = (direction * vol).cumsum()
    return obv


def detect_rsi_divergence(df: pd.DataFrame, rsi_series: pd.Series, lookback: int = 14):
    """
    Detect RSI divergence over the lookback window.
    Returns 'bullish', 'bearish', or None.
    """
    if len(df) < lookback + 1 or len(rsi_series) < lookback + 1:
        return None

    prices = df["Close"].iloc[-lookback:]
    rsis = rsi_series.iloc[-lookback:]

    # Find two lowest price points
    price_min_idx = prices.idxmin()
    # Get the second lowest: exclude a window around the min
    remaining = prices.drop(price_min_idx, errors="ignore")
    if len(remaining) < 2:
        return None

    # Bullish: price makes lower low but RSI makes higher low
    # Compare first half vs second half
    mid = lookback // 2
    first_half_price = prices.iloc[:mid]
    second_half_price = prices.iloc[mid:]
    first_half_rsi = rsis.iloc[:mid]
    second_half_rsi = rsis.iloc[mid:]

    if len(first_half_price) == 0 or len(second_half_price) == 0:
        return None

    price_low1 = first_half_price.min()
    price_low2 = second_half_price.min()
    rsi_low1 = first_half_rsi.min()
    rsi_low2 = second_half_rsi.min()

    price_high1 = first_half_price.max()
    price_high2 = second_half_price.max()
    rsi_high1 = first_half_rsi.max()
    rsi_high2 = second_half_rsi.max()

    # Bullish divergence: price lower low, RSI higher low
    if price_low2 < price_low1 and rsi_low2 > rsi_low1:
        return "bullish"

    # Bearish divergence: price higher high, RSI lower high
    if price_high2 > price_high1 and rsi_high2 < rsi_high1:
        return "bearish"

    return None


def compute_rsi(df: pd.DataFrame, period: int = 14) -> float:
    """
    Compute RSI (Relative Strength Index) for the most recent bar.
    Uses Wilder's smoothing (EMA with alpha=1/period) to match standard
    platforms (TradingView, ThinkorSwim, etc.).
    """
    delta = df["Close"].diff()
    gain = delta.where(delta > 0, 0.0)
    loss = (-delta).where(delta < 0, 0.0)

    # Wilder's smoothing: equivalent to EMA with alpha = 1/period
    avg_gain = gain.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()

    rs = avg_gain / avg_loss.replace(0, np.inf)
    rsi = 100.0 - (100.0 / (1.0 + rs))

    return float(rsi.iloc[-1]) if not pd.isna(rsi.iloc[-1]) else 50.0
