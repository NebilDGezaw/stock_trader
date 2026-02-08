"""
Liquidity Analyzer — detects equal highs/lows (liquidity pools) and sweeps.

ICT Concept:
- **Equal highs / equal lows** form visible liquidity pools where retail
  stop-losses cluster.  Smart money hunts these levels.
- A **liquidity sweep** occurs when price pierces past the pool level with a
  wick, then reverses, indicating institutional accumulation / distribution.
"""

import pandas as pd
import numpy as np
from typing import List

import config
from models.signals import (
    Signal, SignalType, MarketBias, LiquidityLevel,
)
from utils.helpers import (
    wick_to_range, is_strong_reversal, has_volume_confirmation,
)


class LiquidityAnalyzer:
    """Finds liquidity pools and detects sweep events."""

    def __init__(self, df: pd.DataFrame, lookback: int = None):
        self.df = df.copy()
        self.lookback = lookback or config.LIQ_LOOKBACK
        self._levels: list[LiquidityLevel] = []
        self._signals: list[Signal] = []

    # ── public API ────────────────────────────────────────

    def analyze(self) -> "LiquidityAnalyzer":
        """Run the full analysis pipeline and return self."""
        self._find_equal_levels()
        self._detect_sweeps()
        return self

    @property
    def levels(self) -> list[LiquidityLevel]:
        return self._levels

    @property
    def signals(self) -> list[Signal]:
        return self._signals

    # ── internals ─────────────────────────────────────────

    def _find_equal_levels(self):
        """
        Scan recent highs and lows for clusters at the same price level.
        Two or more touches within `tolerance` % are considered equal.
        """
        df = self.df
        n = len(df)
        start = max(0, n - self.lookback)
        tol = config.LIQ_EQUAL_HIGHS_TOLERANCE

        highs = df["High"].iloc[start:].values
        lows = df["Low"].iloc[start:].values

        # --- Equal highs ---
        self._cluster_levels(highs, "equal_highs", tol)

        # --- Equal lows ---
        self._cluster_levels(lows, "equal_lows", tol)

    def _cluster_levels(self, prices: np.ndarray, level_type: str, tol: float):
        """Group prices that are within `tol` percent of each other."""
        if len(prices) == 0:
            return

        sorted_prices = np.sort(prices)
        clusters: list[list[float]] = [[sorted_prices[0]]]

        for p in sorted_prices[1:]:
            if abs(p - clusters[-1][-1]) / clusters[-1][-1] <= tol:
                clusters[-1].append(p)
            else:
                clusters.append([p])

        for cluster in clusters:
            if len(cluster) >= 2:
                avg_price = float(np.mean(cluster))
                self._levels.append(LiquidityLevel(
                    level_type=level_type,
                    price=avg_price,
                    touch_count=len(cluster),
                ))

    def _detect_sweeps(self):
        """
        Check if the most recent candle(s) swept a liquidity level and
        reversed (wick-based).

        Anti-fakeout:
        - The sweep candle must show a strong reversal body (not a doji).
        - Volume should be above average to confirm institutional activity.
        - Weak sweeps (small body, low volume) get reduced score.
        """
        if len(self.df) < 2:
            return

        last_idx = len(self.df) - 1
        last = self.df.iloc[-1]

        for level in self._levels:
            if level.swept:
                continue

            # --- Sweep above equal highs (bearish signal) ---
            if level.level_type == "equal_highs":
                if last["High"] > level.price and last["Close"] < level.price:
                    if wick_to_range(last, "upper") >= config.LIQ_SWEEP_WICK_MIN:
                        level.swept = True
                        level.sweep_date = self.df.index[-1]

                        # Anti-fakeout checks
                        score = 3
                        warnings = []

                        # Must reverse strongly (bearish body)
                        if not is_strong_reversal(
                            self.df, last_idx, "bearish",
                            min_body_ratio=config.FAKEOUT_SWEEP_REVERSAL_BODY,
                        ):
                            score -= 1
                            warnings.append("weak reversal")

                        # Volume confirmation
                        if not has_volume_confirmation(
                            self.df, last_idx,
                            lookback=config.FAKEOUT_VOLUME_LOOKBACK,
                            multiplier=config.FAKEOUT_VOLUME_MULTIPLIER,
                        ):
                            score -= 1
                            warnings.append("low vol")

                        tag = f" ⚠ [{', '.join(warnings)}]" if warnings else ""

                        if score > 0:
                            self._signals.append(Signal(
                                signal_type=SignalType.LIQUIDITY_SWEEP_HIGH,
                                timestamp=self.df.index[-1],
                                price=level.price,
                                bias=MarketBias.BEARISH,
                                score=score,
                                details=(
                                    f"Liquidity sweep above equal highs "
                                    f"@ {level.price:.2f} "
                                    f"({level.touch_count} touches){tag}"
                                ),
                            ))

            # --- Sweep below equal lows (bullish signal) ---
            elif level.level_type == "equal_lows":
                if last["Low"] < level.price and last["Close"] > level.price:
                    if wick_to_range(last, "lower") >= config.LIQ_SWEEP_WICK_MIN:
                        level.swept = True
                        level.sweep_date = self.df.index[-1]

                        score = 3
                        warnings = []

                        if not is_strong_reversal(
                            self.df, last_idx, "bullish",
                            min_body_ratio=config.FAKEOUT_SWEEP_REVERSAL_BODY,
                        ):
                            score -= 1
                            warnings.append("weak reversal")

                        if not has_volume_confirmation(
                            self.df, last_idx,
                            lookback=config.FAKEOUT_VOLUME_LOOKBACK,
                            multiplier=config.FAKEOUT_VOLUME_MULTIPLIER,
                        ):
                            score -= 1
                            warnings.append("low vol")

                        tag = f" ⚠ [{', '.join(warnings)}]" if warnings else ""

                        if score > 0:
                            self._signals.append(Signal(
                                signal_type=SignalType.LIQUIDITY_SWEEP_LOW,
                                timestamp=self.df.index[-1],
                                price=level.price,
                                bias=MarketBias.BULLISH,
                                score=score,
                                details=(
                                    f"Liquidity sweep below equal lows "
                                    f"@ {level.price:.2f} "
                                    f"({level.touch_count} touches){tag}"
                                ),
                            ))
