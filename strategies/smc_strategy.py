"""
SMC Strategy Engine — combines all ICT / Smart Money sub-strategies
into a unified composite score and generates trade recommendations.

Scoring system:
- Each sub-strategy contributes points toward a bullish or bearish score.
- The net score determines the overall bias and signal strength.
- Premium/discount and kill-zone filters add confluence.
"""

import pandas as pd
from datetime import datetime
from typing import List, Tuple

import config
from models.signals import (
    Signal, SignalType, MarketBias, TradeAction, TradeSetup,
)
from strategies.market_structure import MarketStructureAnalyzer
from strategies.order_blocks import OrderBlockDetector
from strategies.fair_value_gaps import FairValueGapDetector
from strategies.liquidity import LiquidityAnalyzer
from utils.helpers import (
    is_premium, is_discount, is_in_kill_zone, get_equilibrium,
    calculate_position_size, get_active_kill_zone,
)


class SMCStrategy:
    """
    Orchestrates all ICT / Smart Money Concept sub-strategies and
    produces a final trade decision.
    """

    def __init__(self, df: pd.DataFrame, ticker: str = "UNKNOWN"):
        self.df = df
        self.ticker = ticker
        self._all_signals: list[Signal] = []
        self._bias = MarketBias.NEUTRAL
        self._bullish_score = 0
        self._bearish_score = 0
        self._setup: TradeSetup | None = None

        # Sub-strategy instances (populated during run)
        self.structure: MarketStructureAnalyzer | None = None
        self.ob_detector: OrderBlockDetector | None = None
        self.fvg_detector: FairValueGapDetector | None = None
        self.liq_analyzer: LiquidityAnalyzer | None = None

    # ── public API ────────────────────────────────────────

    def run(self) -> "SMCStrategy":
        """Execute the full strategy pipeline."""
        self._run_sub_strategies()
        self._apply_confluence_filters()
        self._apply_fakeout_filters()
        self._compute_composite_score()
        self._generate_trade_setup()
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

    # ── internals ─────────────────────────────────────────

    def _run_sub_strategies(self):
        """Instantiate and run each sub-strategy."""
        self.structure = MarketStructureAnalyzer(self.df).analyze()
        self.ob_detector = OrderBlockDetector(self.df).detect()
        self.fvg_detector = FairValueGapDetector(self.df).detect()
        self.liq_analyzer = LiquidityAnalyzer(self.df).analyze()

        # Collect all signals
        self._all_signals = (
            self.structure.signals
            + self.ob_detector.signals
            + self.fvg_detector.signals
            + self.liq_analyzer.signals
        )

    def _apply_confluence_filters(self):
        """Add bonus signals for premium/discount zone and kill zones."""
        if len(self.df) == 0:
            return

        last = self.df.iloc[-1]
        current_price = last["Close"]
        ts = self.df.index[-1]

        # --- Premium / Discount zone ---
        # Use the range from the most recent swing high/low
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
                    details=(
                        f"Price in DISCOUNT zone "
                        f"(eq={get_equilibrium(range_high, range_low):.2f})"
                    ),
                ))
            elif is_premium(current_price, range_high, range_low):
                self._all_signals.append(Signal(
                    signal_type=SignalType.PREMIUM_ZONE,
                    timestamp=ts,
                    price=current_price,
                    bias=MarketBias.BEARISH,
                    score=1,
                    details=(
                        f"Price in PREMIUM zone "
                        f"(eq={get_equilibrium(range_high, range_low):.2f})"
                    ),
                ))

        # --- Kill zone bonus ---
        if hasattr(ts, 'hour') and is_in_kill_zone(ts):
            self._all_signals.append(Signal(
                signal_type=SignalType.KILL_ZONE,
                timestamp=ts,
                price=current_price,
                bias=MarketBias.NEUTRAL,
                score=1,
                details="Active kill zone — higher probability window",
            ))

    def _apply_fakeout_filters(self):
        """
        Anti-fakeout layer: penalise or discard signals that lack confluence.

        Rules:
        1. **Isolated signal penalty** — if only one directional signal type
           exists (e.g. only a BOS with nothing else), reduce its weight.
           Genuine moves in ICT have *multiple* confluences (BOS + OB + FVG).
        2. **Opposing signal cancellation** — if there are roughly equal
           bullish and bearish signals, the market is choppy/ranging. Add a
           "conflicting signals" warning and reduce all scores.
        3. **Outside kill-zone penalty** — signals generated outside the
           London / NY kill zones during session-based scans are less
           reliable; reduce score.
        """
        if not self._all_signals:
            return

        # Categorise signals by direction
        bullish_types = set()
        bearish_types = set()
        for sig in self._all_signals:
            if sig.bias == MarketBias.BULLISH:
                bullish_types.add(sig.signal_type)
            elif sig.bias == MarketBias.BEARISH:
                bearish_types.add(sig.signal_type)

        # Rule 1: Isolated signal penalty
        # A single bullish signal type without any supporting confluence
        dominant_dir = None
        if bullish_types and not bearish_types:
            dominant_dir = "bullish"
            dominant_types = bullish_types
        elif bearish_types and not bullish_types:
            dominant_dir = "bearish"
            dominant_types = bearish_types
        else:
            dominant_dir = None
            dominant_types = set()

        if dominant_dir and len(dominant_types) == 1:
            # Only one signal type — potential fakeout, reduce scores
            penalty = config.FAKEOUT_PENALTY_ISOLATED
            for sig in self._all_signals:
                if sig.bias != MarketBias.NEUTRAL:
                    sig.score = max(sig.score - penalty, 0)
                    if "isolated" not in sig.details:
                        sig.details += " ⚠ [isolated — no confluence]"

        # Rule 2: Conflicting signals — when both sides are close in count
        total_bull = sum(s.score for s in self._all_signals if s.bias == MarketBias.BULLISH)
        total_bear = sum(s.score for s in self._all_signals if s.bias == MarketBias.BEARISH)
        if total_bull > 0 and total_bear > 0:
            ratio = min(total_bull, total_bear) / max(total_bull, total_bear)
            if ratio > 0.7:  # nearly equal = choppy / ranging
                self._all_signals.append(Signal(
                    signal_type=SignalType.KILL_ZONE,  # reuse as "warning"
                    timestamp=self.df.index[-1],
                    price=self.df.iloc[-1]["Close"],
                    bias=MarketBias.NEUTRAL,
                    score=0,
                    details=(
                        "⚠ Conflicting signals detected — possible ranging/choppy market. "
                        "Fakeout risk elevated."
                    ),
                ))

        # Rule 3: Outside kill-zone penalty
        ts = self.df.index[-1]
        if hasattr(ts, "hour"):
            zone = get_active_kill_zone(ts)
            if zone is None:
                # Outside any kill zone — signals are less reliable
                for sig in self._all_signals:
                    if sig.bias != MarketBias.NEUTRAL and sig.score > 1:
                        sig.score = max(sig.score - 1, 1)
                        if "off-session" not in sig.details:
                            sig.details += " [off-session]"

    def _compute_composite_score(self):
        """Tally bullish and bearish scores from all signals."""
        for sig in self._all_signals:
            if sig.bias == MarketBias.BULLISH:
                self._bullish_score += sig.score
            elif sig.bias == MarketBias.BEARISH:
                self._bearish_score += sig.score
            # NEUTRAL signals add to both (confluence bonus)
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

    def _generate_trade_setup(self):
        """
        Build a TradeSetup based on the composite score and identified
        support/resistance levels from order blocks and FVGs.
        """
        if len(self.df) == 0:
            return

        current_price = self.df.iloc[-1]["Close"]
        net = self.net_score
        thresholds = config.SIGNAL_SCORE_THRESHOLDS

        # Determine action
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

        # Determine stop loss and take profit using order blocks / structure
        stop_loss, take_profit = self._calculate_sl_tp(current_price)

        risk_per_share = abs(current_price - stop_loss) if stop_loss else 0
        reward_per_share = abs(take_profit - current_price) if take_profit else 0
        rr = (reward_per_share / risk_per_share) if risk_per_share > 0 else 0

        position_size = calculate_position_size(
            capital=config.INITIAL_CAPITAL,
            entry_price=current_price,
            stop_loss_price=stop_loss or current_price,
        )

        self._setup = TradeSetup(
            action=action,
            ticker=self.ticker,
            entry_price=current_price,
            stop_loss=stop_loss or current_price,
            take_profit=take_profit or current_price,
            position_size=position_size,
            risk_reward=round(rr, 2),
            composite_score=net,
            signals=self._all_signals,
            bias=self._bias,
        )

    def _calculate_sl_tp(
        self, current_price: float
    ) -> Tuple[float, float]:
        """
        Derive stop-loss and take-profit levels from structure,
        order blocks, and fair value gaps.
        """
        sl = current_price
        tp = current_price

        # Use swing lows for long SL, swing highs for long TP
        sh = self.structure.get_last_swing_high()
        sl_swing = self.structure.get_last_swing_low()

        if self._bias == MarketBias.BULLISH:
            # Stop below the most recent swing low
            if sl_swing:
                sl = sl_swing[1] * 0.998  # tiny buffer below

            # Target the most recent swing high (or project 2R)
            if sh:
                tp = sh[1]
            else:
                tp = current_price + 2 * abs(current_price - sl)

            # If an active bullish OB is below, tighten SL to its bottom
            for ob in self.ob_detector.active_blocks():
                if ob.ob_type == "bullish" and ob.bottom < current_price:
                    sl = max(sl, ob.bottom * 0.998)

        elif self._bias == MarketBias.BEARISH:
            # Stop above the most recent swing high
            if sh:
                sl = sh[1] * 1.002

            # Target the most recent swing low
            if sl_swing:
                tp = sl_swing[1]
            else:
                tp = current_price - 2 * abs(sl - current_price)

            for ob in self.ob_detector.active_blocks():
                if ob.ob_type == "bearish" and ob.top > current_price:
                    sl = min(sl, ob.top * 1.002)

        else:
            # Neutral — use recent range extremes as rough SL/TP
            recent = self.df.tail(20)
            sl = recent["Low"].min() * 0.998
            tp = recent["High"].max() * 1.002

        return sl, tp
