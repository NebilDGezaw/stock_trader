"""
Market Structure Analyzer — detects Break of Structure (BOS)
and Change of Character (CHoCH) using swing highs and lows.

ICT Concept:
- BOS  = price breaks a swing high in an uptrend or a swing low in a
         downtrend, confirming continuation.
- CHoCH = price breaks in the *opposite* direction of the prevailing
          trend, signalling a potential reversal.
"""

import pandas as pd
import numpy as np
from typing import List, Tuple

import config
from models.signals import Signal, SignalType, MarketBias
from utils.helpers import detect_swing_highs, detect_swing_lows


class MarketStructureAnalyzer:
    """Analyses swing points to determine market structure."""

    def __init__(self, df: pd.DataFrame, lookback: int = None):
        self.df = df.copy()
        self.lookback = lookback or config.SWING_LOOKBACK
        self._swing_highs: list[Tuple[int, float]] = []  # (index, price)
        self._swing_lows: list[Tuple[int, float]] = []
        self._structure_signals: list[Signal] = []
        self._bias = MarketBias.NEUTRAL

    # ── public API ────────────────────────────────────────

    def analyze(self) -> "MarketStructureAnalyzer":
        """Run the full analysis pipeline and return self for chaining."""
        self._find_swings()
        self._detect_structure_breaks()
        return self

    @property
    def bias(self) -> MarketBias:
        return self._bias

    @property
    def signals(self) -> list[Signal]:
        return self._structure_signals

    @property
    def swing_highs(self) -> list[Tuple[int, float]]:
        return self._swing_highs

    @property
    def swing_lows(self) -> list[Tuple[int, float]]:
        return self._swing_lows

    def get_last_swing_high(self) -> Tuple[int, float] | None:
        return self._swing_highs[-1] if self._swing_highs else None

    def get_last_swing_low(self) -> Tuple[int, float] | None:
        return self._swing_lows[-1] if self._swing_lows else None

    # ── internals ─────────────────────────────────────────

    def _find_swings(self):
        """Identify swing highs and swing lows."""
        sh = detect_swing_highs(self.df, self.lookback)
        sl = detect_swing_lows(self.df, self.lookback)

        self._swing_highs = [
            (i, self.df.iloc[i]["High"])
            for i in range(len(self.df))
            if sh.iloc[i]
        ]
        self._swing_lows = [
            (i, self.df.iloc[i]["Low"])
            for i in range(len(self.df))
            if sl.iloc[i]
        ]

    def _detect_structure_breaks(self):
        """
        Walk through swing points chronologically and detect BOS / CHoCH.

        Logic:
        - Track the most recent Higher High (HH), Higher Low (HL),
          Lower High (LH), Lower Low (LL).
        - Uptrend = series of HH + HL.  Downtrend = series of LH + LL.
        - BOS in uptrend   = new HH.
        - BOS in downtrend = new LL.
        - CHoCH up→down    = price breaks below the most recent HL.
        - CHoCH down→up    = price breaks above the most recent LH.
        """
        if len(self._swing_highs) < 2 or len(self._swing_lows) < 2:
            return

        # Merge swings into a chronological list
        events: list[Tuple[int, float, str]] = []
        for idx, price in self._swing_highs:
            events.append((idx, price, "high"))
        for idx, price in self._swing_lows:
            events.append((idx, price, "low"))
        events.sort(key=lambda x: x[0])

        trend = "neutral"  # 'up', 'down', 'neutral'
        prev_high = None
        prev_low = None

        for idx, price, kind in events:
            ts = self.df.index[idx]

            if kind == "high":
                if prev_high is not None:
                    if price > prev_high:
                        # Higher high
                        if trend == "down":
                            # CHoCH: downtrend broken by higher high
                            self._add_signal(
                                SignalType.CHOCH, ts, price,
                                MarketBias.BULLISH, 3,
                                f"CHoCH bullish — HH at {price:.2f} breaks downtrend"
                            )
                            trend = "up"
                        elif trend == "up":
                            # BOS: continuation
                            self._add_signal(
                                SignalType.BOS, ts, price,
                                MarketBias.BULLISH, 2,
                                f"BOS bullish — HH at {price:.2f}"
                            )
                    else:
                        # Lower high
                        if trend == "up":
                            # Potential weakening — don't signal yet
                            pass
                        elif trend == "down":
                            self._add_signal(
                                SignalType.BOS, ts, price,
                                MarketBias.BEARISH, 2,
                                f"BOS bearish — LH at {price:.2f}"
                            )
                prev_high = price

            else:  # kind == "low"
                if prev_low is not None:
                    if price < prev_low:
                        # Lower low
                        if trend == "up":
                            # CHoCH: uptrend broken by lower low
                            self._add_signal(
                                SignalType.CHOCH, ts, price,
                                MarketBias.BEARISH, 3,
                                f"CHoCH bearish — LL at {price:.2f} breaks uptrend"
                            )
                            trend = "down"
                        elif trend == "down":
                            self._add_signal(
                                SignalType.BOS, ts, price,
                                MarketBias.BEARISH, 2,
                                f"BOS bearish — LL at {price:.2f}"
                            )
                    else:
                        # Higher low
                        if trend == "down":
                            pass  # potential weakening
                        elif trend == "up":
                            self._add_signal(
                                SignalType.BOS, ts, price,
                                MarketBias.BULLISH, 2,
                                f"BOS bullish — HL at {price:.2f}"
                            )
                prev_low = price

            # Initial trend determination
            if trend == "neutral" and prev_high and prev_low:
                trend = "up"  # default start assumption

        # Derive current bias from the latest structure signals
        if self._structure_signals:
            self._bias = self._structure_signals[-1].bias
        else:
            self._bias = MarketBias.NEUTRAL

    def _add_signal(self, sig_type, ts, price, bias, score, details):
        self._structure_signals.append(
            Signal(
                signal_type=sig_type,
                timestamp=ts,
                price=price,
                bias=bias,
                score=score,
                details=details,
            )
        )
