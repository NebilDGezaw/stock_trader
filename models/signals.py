"""
Data models for trading signals, setups, and market bias.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional


class MarketBias(Enum):
    BULLISH = "bullish"
    BEARISH = "bearish"
    NEUTRAL = "neutral"


class SignalType(Enum):
    BOS = "break_of_structure"
    CHOCH = "change_of_character"
    BULLISH_OB = "bullish_order_block"
    BEARISH_OB = "bearish_order_block"
    BULLISH_FVG = "bullish_fair_value_gap"
    BEARISH_FVG = "bearish_fair_value_gap"
    LIQUIDITY_SWEEP_HIGH = "liquidity_sweep_high"
    LIQUIDITY_SWEEP_LOW = "liquidity_sweep_low"
    PREMIUM_ZONE = "premium_zone"
    DISCOUNT_ZONE = "discount_zone"
    KILL_ZONE = "kill_zone"
    # Leveraged ETF strategy signals
    RSI_SIGNAL = "rsi_signal"
    BOLLINGER_BAND = "bollinger_band"
    EMA_CROSSOVER = "ema_crossover"
    VOLUME_MOMENTUM = "volume_momentum"
    BB_SQUEEZE = "bb_squeeze"
    MEAN_REVERSION = "mean_reversion"
    FAST_EMA_CROSSOVER = "fast_ema_crossover"
    # Crypto momentum strategy signals
    MACD_CROSSOVER = "macd_crossover"
    RSI_DIVERGENCE = "rsi_divergence"
    OBV_TREND = "obv_trend"
    EXHAUSTION_WARNING = "exhaustion_warning"
    EMA_RIBBON = "ema_ribbon"


class TradeAction(Enum):
    STRONG_BUY = "STRONG BUY"
    BUY = "BUY"
    HOLD = "HOLD"
    SELL = "SELL"
    STRONG_SELL = "STRONG SELL"


@dataclass
class Signal:
    """A single ICT / SMC signal detected on the chart."""

    signal_type: SignalType
    timestamp: datetime
    price: float
    bias: MarketBias
    score: int = 0                     # contribution to composite score
    details: str = ""                  # human-readable description

    def __repr__(self):
        return (
            f"Signal({self.signal_type.value}, "
            f"bias={self.bias.value}, score={self.score}, "
            f"price={self.price:.2f}, {self.details})"
        )


@dataclass
class OrderBlock:
    """Represents an institutional order block zone."""

    ob_type: str                       # 'bullish' or 'bearish'
    top: float                         # upper boundary
    bottom: float                      # lower boundary
    formation_index: int               # bar index where it formed
    formation_date: datetime = None
    mitigated: bool = False            # True once price returns to the zone

    @property
    def midpoint(self) -> float:
        return (self.top + self.bottom) / 2.0


@dataclass
class FairValueGap:
    """Represents a fair value gap (imbalance)."""

    fvg_type: str                      # 'bullish' or 'bearish'
    top: float
    bottom: float
    formation_index: int
    formation_date: datetime = None
    filled: bool = False

    @property
    def size(self) -> float:
        return abs(self.top - self.bottom)


@dataclass
class LiquidityLevel:
    """A liquidity pool (equal highs / equal lows)."""

    level_type: str                    # 'equal_highs' or 'equal_lows'
    price: float
    touch_count: int = 2
    swept: bool = False
    sweep_date: datetime = None


@dataclass
class TradeSetup:
    """A fully formed trade recommendation."""

    action: TradeAction
    ticker: str
    entry_price: float
    stop_loss: float
    take_profit: float
    position_size: int
    risk_reward: float
    composite_score: int
    signals: list[Signal] = field(default_factory=list)
    bias: MarketBias = MarketBias.NEUTRAL
    timestamp: datetime = field(default_factory=datetime.now)

    @property
    def risk_per_share(self) -> float:
        return abs(self.entry_price - self.stop_loss)

    @property
    def reward_per_share(self) -> float:
        return abs(self.take_profit - self.entry_price)

    def summary(self) -> str:
        lines = [
            f"{'═' * 50}",
            f"  {self.action.value}  —  {self.ticker}",
            f"{'═' * 50}",
            f"  Bias          : {self.bias.value.upper()}",
            f"  Entry         : ${self.entry_price:.2f}",
            f"  Stop Loss     : ${self.stop_loss:.2f}",
            f"  Take Profit   : ${self.take_profit:.2f}",
            f"  R:R           : 1:{self.risk_reward:.1f}",
            f"  Position Size : {self.position_size} shares",
            f"  Score         : {self.composite_score}",
            f"{'─' * 50}",
            f"  Signals:",
        ]
        for sig in self.signals:
            lines.append(f"    • {sig.signal_type.value} (score +{sig.score})")
        lines.append(f"{'═' * 50}")
        return "\n".join(lines)
