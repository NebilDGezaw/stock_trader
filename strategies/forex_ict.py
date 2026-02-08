"""
Forex ICT Strategy — Kill-zone & session-bias driven ICT for 1h candles.
=========================================================================
Adapts the core SMC/ICT sub-strategies (MarketStructure, OrderBlocks,
FVG, Liquidity) with forex-specific parameters and overlays:

1. Kill zone weighting  (London 07-10, NY 12-15 UTC → 2× score)
2. Asian session bias   (00-06 UTC range sweep → directional bias)
3. London/NY overlap bonus (12-14 UTC → extra +1)
4. Asian session penalty (00-06 UTC entries → -1)
5. Tighter FVG/OB/ATR parameters for pip-scale moves

Same interface as SMCStrategy: __init__(df, ticker, stock_mode), .run(),
.trade_setup, .signals
"""

import pandas as pd
import numpy as np
from datetime import datetime
from typing import List, Tuple, Optional

import config
from models.signals import (
    Signal, SignalType, MarketBias, TradeAction, TradeSetup,
)
from strategies.market_structure import MarketStructureAnalyzer
from strategies.order_blocks import OrderBlockDetector
from strategies.fair_value_gaps import FairValueGapDetector
from strategies.liquidity import LiquidityAnalyzer
from utils.helpers import (
    is_premium, is_discount, get_equilibrium,
    calculate_position_size, compute_atr, trend_sma_bias, compute_rsi,
)


class ForexICTStrategy:
    """
    Forex-optimised ICT / Smart Money strategy for 1h candles.
    Leverages kill zones, session bias, and tighter parameters.
    """

    def __init__(self, df: pd.DataFrame, ticker: str = "UNKNOWN",
                 stock_mode: bool = False):
        self.df = df
        self.ticker = ticker
        # stock_mode accepted for interface compat but ignored
        self.cfg = config.FOREX_MODE
        self._all_signals: list[Signal] = []
        self._bias = MarketBias.NEUTRAL
        self._bullish_score = 0
        self._bearish_score = 0
        self._setup: TradeSetup | None = None
        self._session_bias: Optional[str] = None  # 'bullish', 'bearish', None

        # Sub-strategies
        self.structure: MarketStructureAnalyzer | None = None
        self.ob_detector: OrderBlockDetector | None = None
        self.fvg_detector: FairValueGapDetector | None = None
        self.liq_analyzer: LiquidityAnalyzer | None = None

    # ── public API ────────────────────────────────────────

    def run(self) -> "ForexICTStrategy":
        self._save_config()
        self._apply_forex_overrides()
        try:
            self._compute_session_bias()
            self._run_sub_strategies()
            self._apply_confluence_filters()
            self._apply_session_bias_signal()
            self._apply_kill_zone_weighting()
            self._compute_composite_score()
            self._generate_trade_setup()
        finally:
            self._restore_config()
        return self

    @property
    def signals(self) -> list[Signal]:
        return self._all_signals

    @property
    def bias(self) -> MarketBias:
        return self._bias

    @property
    def bullish_score(self) -> int:
        return self._bullish_score

    @property
    def bearish_score(self) -> int:
        return self._bearish_score

    @property
    def net_score(self) -> int:
        return self._bullish_score - self._bearish_score

    @property
    def trade_setup(self) -> TradeSetup | None:
        return self._setup

    # ── config overrides ──────────────────────────────────

    def _save_config(self):
        self._orig = {
            "SWING_LOOKBACK": config.SWING_LOOKBACK,
            "FVG_MIN_GAP_PERCENT": config.FVG_MIN_GAP_PERCENT,
            "FAKEOUT_DISPLACEMENT_MIN_BODY": config.FAKEOUT_DISPLACEMENT_MIN_BODY,
            "FAKEOUT_VOLUME_MULTIPLIER": config.FAKEOUT_VOLUME_MULTIPLIER,
            "FAKEOUT_SWEEP_REVERSAL_BODY": config.FAKEOUT_SWEEP_REVERSAL_BODY,
        }

    def _apply_forex_overrides(self):
        config.SWING_LOOKBACK = self.cfg["swing_lookback"]
        config.FVG_MIN_GAP_PERCENT = self.cfg["fvg_min_gap_pct"]
        # Relax fakeout filters for forex (smaller moves, less volume data)
        config.FAKEOUT_DISPLACEMENT_MIN_BODY = 0.35
        config.FAKEOUT_VOLUME_MULTIPLIER = 0.8
        config.FAKEOUT_SWEEP_REVERSAL_BODY = 0.30

    def _restore_config(self):
        for k, v in self._orig.items():
            setattr(config, k, v)

    # ── session bias (Asian range) ────────────────────────

    def _compute_session_bias(self):
        """
        Determine daily bias from the Asian session (00-06 UTC).
        If price sweeps Asian high → bearish bias.
        If price sweeps Asian low  → bullish bias.
        Look at the most recent complete Asian session relative to
        the last bar in the window.
        """
        df = self.df
        if len(df) < 10:
            return

        last_ts = df.index[-1]
        if not hasattr(last_ts, 'hour'):
            return

        # Find Asian session bars for today (or the most recent session)
        asian_bars = []
        post_asian_bars = []

        for i in range(len(df) - 1, -1, -1):
            ts = df.index[i]
            if not hasattr(ts, 'hour'):
                break
            hour = ts.hour
            if 0 <= hour <= 5:
                asian_bars.append(i)
            elif hour >= 7 and len(asian_bars) > 0:
                # We've found asian bars and now we're past them
                post_asian_bars.append(i)
            # Stop searching after we have enough context
            if len(asian_bars) > 0 and hour >= 7 and len(post_asian_bars) > 5:
                break
            if len(asian_bars) > 20:
                break

        if len(asian_bars) < 2:
            return

        asian_indices = sorted(asian_bars)
        asian_high = df.iloc[asian_indices]["High"].max()
        asian_low = df.iloc[asian_indices]["Low"].min()

        # Check if post-Asian price swept either level
        last_price = df.iloc[-1]["Close"]
        last_high = df.iloc[-1]["High"]
        last_low = df.iloc[-1]["Low"]

        # Check recent bars (after Asian) for sweeps
        for idx in post_asian_bars:
            bar = df.iloc[idx]
            if bar["High"] > asian_high:
                self._session_bias = "bearish"
                return
            if bar["Low"] < asian_low:
                self._session_bias = "bullish"
                return

        # Also check the current bar
        if last_high > asian_high:
            self._session_bias = "bearish"
        elif last_low < asian_low:
            self._session_bias = "bullish"

    def _apply_session_bias_signal(self):
        """Add a signal for the Asian session bias."""
        if self._session_bias is None:
            return

        ts = self.df.index[-1]
        price = self.df.iloc[-1]["Close"]

        if self._session_bias == "bullish":
            self._all_signals.append(Signal(
                signal_type=SignalType.KILL_ZONE,
                timestamp=ts,
                price=price,
                bias=MarketBias.BULLISH,
                score=2,
                details="Asian low swept → bullish session bias",
            ))
        elif self._session_bias == "bearish":
            self._all_signals.append(Signal(
                signal_type=SignalType.KILL_ZONE,
                timestamp=ts,
                price=price,
                bias=MarketBias.BEARISH,
                score=2,
                details="Asian high swept → bearish session bias",
            ))

    # ── sub-strategies ────────────────────────────────────

    def _run_sub_strategies(self):
        self.structure = MarketStructureAnalyzer(
            self.df, lookback=self.cfg["swing_lookback"]
        ).analyze()
        self.ob_detector = OrderBlockDetector(self.df).detect()
        self.fvg_detector = FairValueGapDetector(self.df).detect()
        self.liq_analyzer = LiquidityAnalyzer(self.df).analyze()

        self._all_signals = (
            self.structure.signals
            + self.ob_detector.signals
            + self.fvg_detector.signals
            + self.liq_analyzer.signals
        )

    def _apply_confluence_filters(self):
        """Premium/discount zone signals."""
        if len(self.df) == 0:
            return

        last = self.df.iloc[-1]
        current_price = last["Close"]
        ts = self.df.index[-1]

        sh = self.structure.get_last_swing_high()
        sl = self.structure.get_last_swing_low()

        if sh and sl:
            range_high = sh[1]
            range_low = sl[1]

            if is_discount(current_price, range_high, range_low):
                self._all_signals.append(Signal(
                    signal_type=SignalType.DISCOUNT_ZONE,
                    timestamp=ts,
                    price=current_price,
                    bias=MarketBias.BULLISH,
                    score=1,
                    details=f"Price in DISCOUNT zone (eq={get_equilibrium(range_high, range_low):.4f})",
                ))
            elif is_premium(current_price, range_high, range_low):
                self._all_signals.append(Signal(
                    signal_type=SignalType.PREMIUM_ZONE,
                    timestamp=ts,
                    price=current_price,
                    bias=MarketBias.BEARISH,
                    score=1,
                    details=f"Price in PREMIUM zone (eq={get_equilibrium(range_high, range_low):.4f})",
                ))

    # ── kill zone weighting (THE forex edge) ──────────────

    def _apply_kill_zone_weighting(self):
        """
        Adjust ALL signal scores based on the time of the last bar:
        - London (07-10) or NY (12-15): multiply by kill_zone_multiplier (2×)
        - London/NY overlap (12-14): additional +1 bonus
        - Asian session (00-06): penalty of -asian_penalty
        - Off-session: multiply by off_session_multiplier (0.5×)
        """
        ts = self.df.index[-1]
        if not hasattr(ts, 'hour'):
            return

        hour = ts.hour
        cfg = self.cfg

        in_london = 7 <= hour <= 10
        in_ny = 12 <= hour <= 15
        in_overlap = 12 <= hour <= 13  # 12:00-13:59 UTC
        in_asian = 0 <= hour <= 5
        in_kill_zone = in_london or in_ny

        for sig in self._all_signals:
            if sig.bias == MarketBias.NEUTRAL:
                continue

            if in_kill_zone:
                # Multiply score by kill zone multiplier
                sig.score = int(sig.score * cfg["kill_zone_multiplier"])
                if "kill-zone" not in sig.details:
                    zone_name = "London" if in_london else "NY"
                    sig.details += f" [🎯 {zone_name} kill zone]"

                # Overlap bonus
                if in_overlap:
                    sig.score += cfg["overlap_bonus"]
                    sig.details += " [🔥 LDN/NY overlap]"

            elif in_asian:
                # Penalise Asian session entries
                sig.score = max(sig.score - cfg["asian_penalty"], 0)
                if "asian" not in sig.details:
                    sig.details += " [⚠ Asian session — low vol]"

            else:
                # Off-session: reduce score
                sig.score = max(int(sig.score * cfg["off_session_multiplier"]), 0)
                if "off-session" not in sig.details:
                    sig.details += " [off-session]"

    # ── scoring ───────────────────────────────────────────

    def _compute_composite_score(self):
        for sig in self._all_signals:
            if sig.bias == MarketBias.BULLISH:
                self._bullish_score += sig.score
            elif sig.bias == MarketBias.BEARISH:
                self._bearish_score += sig.score
            else:
                self._bullish_score += sig.score
                self._bearish_score += sig.score

        net = self._bullish_score - self._bearish_score
        if net > 0:
            self._bias = MarketBias.BULLISH
        elif net < 0:
            self._bias = MarketBias.BEARISH
        else:
            self._bias = MarketBias.NEUTRAL

    # ── trade setup ───────────────────────────────────────

    def _generate_trade_setup(self):
        if len(self.df) == 0:
            return

        current_price = self.df.iloc[-1]["Close"]
        net = self.net_score
        thresholds = self.cfg["score_thresholds"]

        if net >= thresholds["strong_buy"]:
            action = TradeAction.STRONG_BUY
        elif net >= thresholds["buy"]:
            action = TradeAction.BUY
        elif net <= -thresholds["strong_sell"]:
            action = TradeAction.STRONG_SELL
        elif net <= -thresholds["sell"]:
            action = TradeAction.SELL
        else:
            action = TradeAction.HOLD

        sl, tp = self._calculate_atr_sl_tp(current_price)

        risk_per_share = abs(current_price - sl) if sl else 0
        reward_per_share = abs(tp - current_price) if tp else 0
        rr = (reward_per_share / risk_per_share) if risk_per_share > 0 else 0

        position_size = calculate_position_size(
            capital=config.INITIAL_CAPITAL,
            entry_price=current_price,
            stop_loss_price=sl or current_price,
            risk_pct=self.cfg["risk_per_trade"],
        )

        self._setup = TradeSetup(
            action=action,
            ticker=self.ticker,
            entry_price=current_price,
            stop_loss=sl or current_price,
            take_profit=tp or current_price,
            position_size=position_size,
            risk_reward=round(rr, 2),
            composite_score=net,
            signals=self._all_signals,
            bias=self._bias,
        )

    def _calculate_atr_sl_tp(self, current_price: float) -> Tuple[float, float]:
        cfg = self.cfg
        atr = compute_atr(self.df, period=cfg["atr_period"])
        current_atr = float(atr.iloc[-1])

        if current_atr == 0:
            current_atr = current_price * 0.005

        sl_mult = cfg["atr_sl_multiplier"]
        tp_mult = cfg["atr_tp_multiplier"]

        if self._bias == MarketBias.BULLISH:
            sl = current_price - (current_atr * sl_mult)
            tp = current_price + (current_atr * tp_mult)
        elif self._bias == MarketBias.BEARISH:
            sl = current_price + (current_atr * sl_mult)
            tp = current_price - (current_atr * tp_mult)
        else:
            sl = current_price - (current_atr * sl_mult)
            tp = current_price + (current_atr * tp_mult)

        return sl, tp
