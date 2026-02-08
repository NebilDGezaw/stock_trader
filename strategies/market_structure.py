"""
Market Structure Analyzer — detects Break of Structure (BOS)
and Change of Character (CHoCH) using swing highs and lows.

ICT Concept:
- BOS  = price breaks a swing high in an uptrend or a swing low in a
         downtrend, confirming continuation.
- CHoCH = price breaks in the *opposite* direction of the prevailing
          trend, signalling a potential reversal.
"""
from __future__ import annotations

import pandas as pd
import numpy as np
from typing import List, Tuple

import config
from models.signals import Signal, SignalType, MarketBias
from utils.helpers import (
    detect_swing_highs, detect_swing_lows,
    has_displacement, has_volume_confirmation, candle_closed_beyond,
)


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

        Anti-fakeout:
        - Each break is validated for displacement (strong body close) and
          volume confirmation.  Signals that fail are penalised or dropped.
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
                            score, tag = self._fakeout_adjusted_score(
                                idx, prev_high, "bullish", base_score=3
                            )
                            if score > 0:
                                self._add_signal(
                                    SignalType.CHOCH, ts, price,
                                    MarketBias.BULLISH, score,
                                    f"CHoCH bullish — HH at {price:.2f} breaks downtrend{tag}"
                                )
                            trend = "up"
                        elif trend == "up":
                            score, tag = self._fakeout_adjusted_score(
                                idx, prev_high, "bullish", base_score=2
                            )
                            if score > 0:
                                self._add_signal(
                                    SignalType.BOS, ts, price,
                                    MarketBias.BULLISH, score,
                                    f"BOS bullish — HH at {price:.2f}{tag}"
                                )
                    else:
                        # Lower high
                        if trend == "up":
                            pass
                        elif trend == "down":
                            score, tag = self._fakeout_adjusted_score(
                                idx, prev_high, "bearish", base_score=2
                            )
                            if score > 0:
                                self._add_signal(
                                    SignalType.BOS, ts, price,
                                    MarketBias.BEARISH, score,
                                    f"BOS bearish — LH at {price:.2f}{tag}"
                                )
                prev_high = price

            else:  # kind == "low"
                if prev_low is not None:
                    if price < prev_low:
                        # Lower low
                        if trend == "up":
                            score, tag = self._fakeout_adjusted_score(
                                idx, prev_low, "bearish", base_score=3
                            )
                            if score > 0:
                                self._add_signal(
                                    SignalType.CHOCH, ts, price,
                                    MarketBias.BEARISH, score,
                                    f"CHoCH bearish — LL at {price:.2f} breaks uptrend{tag}"
                                )
                            trend = "down"
                        elif trend == "down":
                            score, tag = self._fakeout_adjusted_score(
                                idx, prev_low, "bearish", base_score=2
                            )
                            if score > 0:
                                self._add_signal(
                                    SignalType.BOS, ts, price,
                                    MarketBias.BEARISH, score,
                                    f"BOS bearish — LL at {price:.2f}{tag}"
                                )
                    else:
                        # Higher low
                        if trend == "down":
                            pass
                        elif trend == "up":
                            score, tag = self._fakeout_adjusted_score(
                                idx, prev_low, "bullish", base_score=2
                            )
                            if score > 0:
                                self._add_signal(
                                    SignalType.BOS, ts, price,
                                    MarketBias.BULLISH, score,
                                    f"BOS bullish — HL at {price:.2f}{tag}"
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

    # ── fakeout validation ────────────────────────────────

    def _fakeout_adjusted_score(
        self, bar_idx: int, level: float, direction: str, base_score: int
    ) -> Tuple[int, str]:
        """
        Validate a structure break against fakeout indicators.
        Returns (adjusted_score, detail_tag).

        Checks:
        1. Displacement — does the break candle have a strong body?
        2. Volume — is volume above average on the break?
        3. Candle close — did the body close beyond the level (not just wick)?

        Each failure reduces the score; if score reaches 0 the signal is dropped.
        """
        score = base_score
        warnings = []

        close_dir = "above" if direction == "bullish" else "below"

        # 1) Body closed beyond level (most critical — wick-only = fakeout)
        if not candle_closed_beyond(self.df, bar_idx, level, close_dir):
            score -= config.FAKEOUT_PENALTY_NO_DISPLACEMENT
            warnings.append("wick-only")

        # 2) Displacement (strong body candle)
        if not has_displacement(
            self.df, bar_idx, direction,
            min_body_ratio=config.FAKEOUT_DISPLACEMENT_MIN_BODY,
        ):
            score -= config.FAKEOUT_PENALTY_NO_DISPLACEMENT
            warnings.append("weak body")

        # 3) Volume confirmation
        if not has_volume_confirmation(
            self.df, bar_idx,
            lookback=config.FAKEOUT_VOLUME_LOOKBACK,
            multiplier=config.FAKEOUT_VOLUME_MULTIPLIER,
        ):
            score -= config.FAKEOUT_PENALTY_NO_VOLUME
            warnings.append("low vol")

        tag = ""
        if warnings:
            tag = f" ⚠ [{', '.join(warnings)}]"

        return max(score, 0), tag

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
