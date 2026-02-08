"""
Fair Value Gap (FVG) Detector — identifies price imbalances.

ICT Concept:
- A **bullish FVG** exists when candle-3's low is *above* candle-1's high,
  leaving a gap that wasn't "traded through".  Price tends to return to fill
  or react at this gap before continuing higher.
- A **bearish FVG** exists when candle-1's low is *above* candle-3's high.

The three-candle window is: candle_1 (left), candle_2 (middle), candle_3 (right).
"""

import pandas as pd
import numpy as np
from typing import List

import config
from models.signals import (
    Signal, SignalType, MarketBias, FairValueGap,
)


class FairValueGapDetector:
    """Finds Fair Value Gaps (price imbalances) on a chart."""

    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        self._fvgs: list[FairValueGap] = []
        self._signals: list[Signal] = []

    # ── public API ────────────────────────────────────────

    def detect(self) -> "FairValueGapDetector":
        """Run detection and return self for chaining."""
        self._find_fvgs()
        self._check_filled()
        self._generate_signals()
        return self

    @property
    def fvgs(self) -> list[FairValueGap]:
        return self._fvgs

    @property
    def signals(self) -> list[Signal]:
        return self._signals

    def active_fvgs(self) -> list[FairValueGap]:
        """Return FVGs that have not been filled."""
        return [fvg for fvg in self._fvgs if not fvg.filled]

    # ── internals ─────────────────────────────────────────

    def _find_fvgs(self):
        df = self.df
        n = len(df)

        for i in range(2, n):
            c1 = df.iloc[i - 2]  # left candle
            c3 = df.iloc[i]      # right candle

            # --- Bullish FVG: gap between c1 high and c3 low ---
            if c3["Low"] > c1["High"]:
                gap_size = c3["Low"] - c1["High"]
                mid_price = (c1["High"] + c3["Low"]) / 2
                gap_pct = (gap_size / mid_price) * 100

                if gap_pct >= config.FVG_MIN_GAP_PERCENT:
                    self._fvgs.append(FairValueGap(
                        fvg_type="bullish",
                        top=c3["Low"],
                        bottom=c1["High"],
                        formation_index=i - 1,  # middle candle
                        formation_date=df.index[i - 1],
                    ))

            # --- Bearish FVG: gap between c3 high and c1 low ---
            if c1["Low"] > c3["High"]:
                gap_size = c1["Low"] - c3["High"]
                mid_price = (c1["Low"] + c3["High"]) / 2
                gap_pct = (gap_size / mid_price) * 100

                if gap_pct >= config.FVG_MIN_GAP_PERCENT:
                    self._fvgs.append(FairValueGap(
                        fvg_type="bearish",
                        top=c1["Low"],
                        bottom=c3["High"],
                        formation_index=i - 1,
                        formation_date=df.index[i - 1],
                    ))

    def _check_filled(self):
        """Mark an FVG as filled if price has traded through it."""
        df = self.df
        for fvg in self._fvgs:
            start = fvg.formation_index + 2
            end = min(fvg.formation_index + config.FVG_MAX_AGE, len(df))

            for j in range(start, end):
                row = df.iloc[j]
                if fvg.fvg_type == "bullish":
                    # Filled when price dips into the gap
                    if row["Low"] <= fvg.bottom:
                        fvg.filled = True
                        break
                else:
                    # Filled when price rallies into the gap
                    if row["High"] >= fvg.top:
                        fvg.filled = True
                        break

    def _generate_signals(self):
        """Create signals for active FVGs near the current price."""
        if len(self.df) == 0:
            return

        current_price = self.df.iloc[-1]["Close"]
        last_idx = len(self.df) - 1

        for fvg in self.active_fvgs():
            age = last_idx - fvg.formation_index
            if age > config.FVG_MAX_AGE:
                continue

            proximity = 0.02  # within 2% of gap zone
            in_zone = (
                fvg.bottom * (1 - proximity)
                <= current_price
                <= fvg.top * (1 + proximity)
            )

            if not in_zone:
                continue

            if fvg.fvg_type == "bullish":
                self._signals.append(Signal(
                    signal_type=SignalType.BULLISH_FVG,
                    timestamp=self.df.index[-1],
                    price=(fvg.top + fvg.bottom) / 2,
                    bias=MarketBias.BULLISH,
                    score=2,
                    details=(
                        f"Bullish FVG {fvg.bottom:.2f}–{fvg.top:.2f} "
                        f"(age {age} bars)"
                    ),
                ))
            else:
                self._signals.append(Signal(
                    signal_type=SignalType.BEARISH_FVG,
                    timestamp=self.df.index[-1],
                    price=(fvg.top + fvg.bottom) / 2,
                    bias=MarketBias.BEARISH,
                    score=2,
                    details=(
                        f"Bearish FVG {fvg.bottom:.2f}–{fvg.top:.2f} "
                        f"(age {age} bars)"
                    ),
                ))
