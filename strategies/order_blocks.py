"""
Order Block Detector — identifies bullish and bearish order blocks.

ICT Concept:
- A **bullish order block** is the last bearish (down) candle before a strong
  bullish move that breaks structure.  It represents institutional buying.
- A **bearish order block** is the last bullish (up) candle before a strong
  bearish move that breaks structure.  It represents institutional selling.

The zone between the OB candle's open and close acts as a supply/demand area
that price tends to revisit.
"""

import pandas as pd
import numpy as np
from typing import List

import config
from models.signals import (
    Signal, SignalType, MarketBias, OrderBlock,
)
from utils.helpers import (
    is_bullish_candle, is_bearish_candle, body_ratio, candle_range,
)


class OrderBlockDetector:
    """Finds institutional order blocks on a price chart."""

    def __init__(self, df: pd.DataFrame, lookback: int = None):
        self.df = df.copy()
        self.lookback = lookback or config.OB_LOOKBACK
        self._order_blocks: list[OrderBlock] = []
        self._signals: list[Signal] = []

    # ── public API ────────────────────────────────────────

    def detect(self) -> "OrderBlockDetector":
        """Run detection and return self for chaining."""
        self._find_order_blocks()
        self._check_mitigation()
        self._generate_signals()
        return self

    @property
    def order_blocks(self) -> list[OrderBlock]:
        return self._order_blocks

    @property
    def signals(self) -> list[Signal]:
        return self._signals

    def active_blocks(self) -> list[OrderBlock]:
        """Return order blocks that have NOT been mitigated yet."""
        return [ob for ob in self._order_blocks if not ob.mitigated]

    # ── internals ─────────────────────────────────────────

    def _find_order_blocks(self):
        """
        Scan for order blocks by looking for strong displacement candles
        and marking the last opposite candle before them.
        """
        df = self.df
        n = len(df)

        for i in range(2, n):
            curr = df.iloc[i]
            prev = df.iloc[i - 1]

            # --- Bullish OB: bearish candle followed by strong bullish move ---
            if is_bearish_candle(prev) and is_bullish_candle(curr):
                # Check displacement: current candle body is large
                if body_ratio(curr) >= config.OB_BODY_RATIO_MIN:
                    # The bullish candle should engulf or significantly exceed
                    if curr["Close"] > prev["High"]:
                        ob = OrderBlock(
                            ob_type="bullish",
                            top=max(prev["Open"], prev["Close"]),
                            bottom=min(prev["Open"], prev["Close"]),
                            formation_index=i - 1,
                            formation_date=df.index[i - 1],
                        )
                        self._order_blocks.append(ob)

            # --- Bearish OB: bullish candle followed by strong bearish move ---
            if is_bullish_candle(prev) and is_bearish_candle(curr):
                if body_ratio(curr) >= config.OB_BODY_RATIO_MIN:
                    if curr["Close"] < prev["Low"]:
                        ob = OrderBlock(
                            ob_type="bearish",
                            top=max(prev["Open"], prev["Close"]),
                            bottom=min(prev["Open"], prev["Close"]),
                            formation_index=i - 1,
                            formation_date=df.index[i - 1],
                        )
                        self._order_blocks.append(ob)

    def _check_mitigation(self):
        """
        Mark an order block as mitigated if price has returned into the zone
        after it was formed.
        """
        if not config.OB_MITIGATION_TOUCH:
            return

        df = self.df
        for ob in self._order_blocks:
            start = ob.formation_index + 2  # skip the displacement candle
            end = min(ob.formation_index + config.OB_MAX_AGE, len(df))

            for j in range(start, end):
                row = df.iloc[j]
                # Price entered the OB zone
                if ob.ob_type == "bullish" and row["Low"] <= ob.top:
                    ob.mitigated = True
                    break
                if ob.ob_type == "bearish" and row["High"] >= ob.bottom:
                    ob.mitigated = True
                    break

    def _generate_signals(self):
        """
        Create signals for active (un-mitigated) order blocks that are
        close to the current price.
        """
        if len(self.df) == 0:
            return

        current_price = self.df.iloc[-1]["Close"]
        last_idx = len(self.df) - 1

        for ob in self.active_blocks():
            age = last_idx - ob.formation_index
            if age > config.OB_MAX_AGE:
                continue

            # Check proximity: price within 2% of the OB zone
            proximity = 0.02
            if ob.ob_type == "bullish":
                if ob.bottom * (1 - proximity) <= current_price <= ob.top * (1 + proximity):
                    self._signals.append(Signal(
                        signal_type=SignalType.BULLISH_OB,
                        timestamp=self.df.index[-1],
                        price=ob.midpoint,
                        bias=MarketBias.BULLISH,
                        score=2,
                        details=(
                            f"Bullish OB zone {ob.bottom:.2f}–{ob.top:.2f} "
                            f"(age {age} bars)"
                        ),
                    ))
            else:
                if ob.bottom * (1 - proximity) <= current_price <= ob.top * (1 + proximity):
                    self._signals.append(Signal(
                        signal_type=SignalType.BEARISH_OB,
                        timestamp=self.df.index[-1],
                        price=ob.midpoint,
                        bias=MarketBias.BEARISH,
                        score=2,
                        details=(
                            f"Bearish OB zone {ob.bottom:.2f}–{ob.top:.2f} "
                            f"(age {age} bars)"
                        ),
                    ))
