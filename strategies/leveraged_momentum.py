"""
Leveraged ETF Momentum / Mean-Reversion Strategy — AGGRESSIVE
==============================================================
Designed for volatile leveraged ETFs (MSTU, MSTR, MSTZ, TSLL).

Targets 25-40%+ annual returns with higher risk tolerance.
Uses: RSI, dual EMA crossovers, Bollinger Bands + extreme bands,
mean-reversion on big drops/pumps, volume momentum, trailing stops.
"""

import pandas as pd
import numpy as np
from datetime import datetime

import config
from models.signals import (
    Signal, SignalType, MarketBias, TradeAction, TradeSetup,
)
from utils.helpers import (
    compute_rsi_series, compute_bollinger_bands, compute_ema,
    compute_atr, bb_squeeze_detected, calculate_position_size,
)


class LeveragedMomentumStrategy:
    """
    Aggressive momentum / mean-reversion strategy for leveraged ETFs.
    Same interface as SMCStrategy.
    """

    def __init__(self, df: pd.DataFrame, ticker: str = "UNKNOWN",
                 stock_mode: bool = True):
        self.df = df
        self.ticker = ticker
        self.cfg = config.LEVERAGED_MODE
        self._all_signals: list[Signal] = []
        self._bias = MarketBias.NEUTRAL
        self._setup: TradeSetup | None = None
        self._bull_score = 0
        self._bear_score = 0

    def run(self) -> "LeveragedMomentumStrategy":
        if len(self.df) < 30:
            self._generate_hold()
            return self

        self._analyze_rsi()
        self._analyze_bollinger()
        self._analyze_ema_crossover()
        self._analyze_ultra_ema_crossover()
        self._analyze_volume()
        self._analyze_mean_reversion()
        self._compute_score()
        self._generate_trade_setup()
        return self

    @property
    def signals(self) -> list[Signal]:
        return self._all_signals

    @property
    def bias(self) -> MarketBias:
        return self._bias

    @property
    def trade_setup(self) -> TradeSetup | None:
        return self._setup

    @property
    def bullish_score(self) -> int:
        return self._bull_score

    @property
    def bearish_score(self) -> int:
        return self._bear_score

    @property
    def net_score(self) -> int:
        return self._bull_score - self._bear_score

    # ── signal components ─────────────────────────────────

    def _analyze_rsi(self):
        cfg = self.cfg
        rsi_series = compute_rsi_series(self.df, cfg["rsi_period"])
        rsi = float(rsi_series.iloc[-1])
        ts = self.df.index[-1]
        price = self.df.iloc[-1]["Close"]

        if rsi <= cfg["rsi_extreme_oversold"]:
            self._add_signal(SignalType.RSI_SIGNAL, ts, price,
                             MarketBias.BULLISH, 3,
                             f"🔥 RSI extreme oversold ({rsi:.1f}) — high bounce probability")
        elif rsi <= cfg["rsi_oversold"]:
            self._add_signal(SignalType.RSI_SIGNAL, ts, price,
                             MarketBias.BULLISH, 1,
                             f"RSI oversold ({rsi:.1f})")
        elif rsi >= cfg["rsi_extreme_overbought"]:
            self._add_signal(SignalType.RSI_SIGNAL, ts, price,
                             MarketBias.BEARISH, 3,
                             f"🔥 RSI extreme overbought ({rsi:.1f}) — reversal likely")
        elif rsi >= cfg["rsi_overbought"]:
            self._add_signal(SignalType.RSI_SIGNAL, ts, price,
                             MarketBias.BEARISH, 1,
                             f"RSI overbought ({rsi:.1f})")

    def _analyze_bollinger(self):
        cfg = self.cfg
        upper, middle, lower = compute_bollinger_bands(
            self.df, cfg["bb_period"], cfg["bb_std"])
        ts = self.df.index[-1]
        price = self.df.iloc[-1]["Close"]

        # BB + RSI confluence: price below lower BB AND RSI < 40 = strong bounce
        rsi_series = compute_rsi_series(self.df, cfg["rsi_period"])
        rsi = float(rsi_series.iloc[-1])

        # Standard band touch
        if price <= lower.iloc[-1]:
            self._add_signal(SignalType.BOLLINGER_BAND, ts, price,
                             MarketBias.BULLISH, 1,
                             f"Price at lower BB ({lower.iloc[-1]:.2f})")
            if rsi < cfg.get("bb_extreme_rsi_bull", 40):
                self._add_signal(SignalType.BOLLINGER_BAND, ts, price,
                                 MarketBias.BULLISH, 2,
                                 f"BB + RSI confluence ({rsi:.0f}) — strong bounce")

            # Extreme — price more than 1 extra std below lower band
            bb_width = upper.iloc[-1] - lower.iloc[-1]
            extra_std = cfg.get("bb_extreme_std", 1.0)
            extreme_lower = lower.iloc[-1] - (bb_width * extra_std / (2 * cfg["bb_std"]))
            if price <= extreme_lower:
                self._add_signal(SignalType.BOLLINGER_BAND, ts, price,
                                 MarketBias.BULLISH, 2,
                                 f"🔥 Price BELOW extreme BB — strong bounce signal")

        elif price >= upper.iloc[-1]:
            self._add_signal(SignalType.BOLLINGER_BAND, ts, price,
                             MarketBias.BEARISH, 1,
                             f"Price at upper BB ({upper.iloc[-1]:.2f})")
            if rsi > cfg.get("bb_extreme_rsi_bear", 60):
                self._add_signal(SignalType.BOLLINGER_BAND, ts, price,
                                 MarketBias.BEARISH, 2,
                                 f"BB + RSI confluence ({rsi:.0f}) — strong reversal")

            bb_width = upper.iloc[-1] - lower.iloc[-1]
            extra_std = cfg.get("bb_extreme_std", 1.0)
            extreme_upper = upper.iloc[-1] + (bb_width * extra_std / (2 * cfg["bb_std"]))
            if price >= extreme_upper:
                self._add_signal(SignalType.BOLLINGER_BAND, ts, price,
                                 MarketBias.BEARISH, 2,
                                 f"🔥 Price ABOVE extreme BB — reversal signal")

        # Squeeze breakout
        if bb_squeeze_detected(self.df, cfg["bb_period"], cfg["bb_std"]):
            if price > middle.iloc[-1]:
                self._add_signal(SignalType.BB_SQUEEZE, ts, price,
                                 MarketBias.BULLISH, 2,
                                 "BB squeeze breakout — bullish expansion")
            else:
                self._add_signal(SignalType.BB_SQUEEZE, ts, price,
                                 MarketBias.BEARISH, 2,
                                 "BB squeeze breakout — bearish expansion")

    def _analyze_ema_crossover(self):
        """Standard 9/21 EMA crossover."""
        cfg = self.cfg
        self._check_ema_pair(cfg["ema_fast"], cfg["ema_slow"], score_cross=2, score_sustain=1)

    def _analyze_ultra_ema_crossover(self):
        """Faster 5/13 EMA crossover — catches momentum earlier."""
        cfg = self.cfg
        fast = cfg.get("ema_ultra_fast", 5)
        slow = cfg.get("ema_ultra_slow", 13)
        self._check_ema_pair(fast, slow, score_cross=1, score_sustain=0)

    def _check_ema_pair(self, fast_period: int, slow_period: int,
                        score_cross: int = 2, score_sustain: int = 1):
        ema_fast = compute_ema(self.df, fast_period)
        ema_slow = compute_ema(self.df, slow_period)
        ts = self.df.index[-1]
        price = self.df.iloc[-1]["Close"]

        if len(ema_fast) < 2:
            return

        curr_above = ema_fast.iloc[-1] > ema_slow.iloc[-1]
        prev_above = ema_fast.iloc[-2] > ema_slow.iloc[-2]

        if curr_above and not prev_above:
            self._add_signal(SignalType.EMA_CROSSOVER, ts, price,
                             MarketBias.BULLISH, score_cross,
                             f"EMA {fast_period}/{slow_period} bullish cross")
        elif not curr_above and prev_above:
            self._add_signal(SignalType.EMA_CROSSOVER, ts, price,
                             MarketBias.BEARISH, score_cross,
                             f"EMA {fast_period}/{slow_period} bearish cross")
        elif curr_above and score_sustain > 0:
            self._add_signal(SignalType.EMA_CROSSOVER, ts, price,
                             MarketBias.BULLISH, score_sustain,
                             f"EMA {fast_period} above {slow_period}")
        elif not curr_above and score_sustain > 0:
            self._add_signal(SignalType.EMA_CROSSOVER, ts, price,
                             MarketBias.BEARISH, score_sustain,
                             f"EMA {fast_period} below {slow_period}")

    def _analyze_volume(self):
        cfg = self.cfg
        if "Volume" not in self.df.columns:
            return

        ts = self.df.index[-1]
        price = self.df.iloc[-1]["Close"]
        vol = self.df.iloc[-1]["Volume"]
        lookback = cfg["volume_lookback"]

        if len(self.df) < lookback + 1:
            return

        avg_vol = self.df["Volume"].iloc[-lookback - 1:-1].mean()
        if avg_vol == 0:
            return

        vol_ratio = vol / avg_vol

        if vol_ratio >= cfg["volume_multiplier"]:
            price_change = price - self.df.iloc[-2]["Close"]
            if price_change > 0:
                self._add_signal(SignalType.VOLUME_MOMENTUM, ts, price,
                                 MarketBias.BULLISH, 1,
                                 f"High volume bullish ({vol_ratio:.1f}x avg)")
            elif price_change < 0:
                self._add_signal(SignalType.VOLUME_MOMENTUM, ts, price,
                                 MarketBias.BEARISH, 1,
                                 f"High volume bearish ({vol_ratio:.1f}x avg)")

            # Extra-high volume = extra conviction
            if vol_ratio >= 2.0:
                if price_change > 0:
                    self._add_signal(SignalType.VOLUME_MOMENTUM, ts, price,
                                     MarketBias.BULLISH, 1,
                                     f"🔥 Extreme volume ({vol_ratio:.1f}x) — institutional")
                elif price_change < 0:
                    self._add_signal(SignalType.VOLUME_MOMENTUM, ts, price,
                                     MarketBias.BEARISH, 1,
                                     f"🔥 Extreme volume ({vol_ratio:.1f}x) — institutional")

    def _analyze_mean_reversion(self):
        """
        Big single-day moves on leveraged ETFs often snap back.
        A 5%+ drop on high volume = bounce buy. 5%+ pump = fade sell.
        """
        cfg = self.cfg
        if not cfg.get("mean_reversion", False):
            return

        if len(self.df) < 2:
            return

        ts = self.df.index[-1]
        price = self.df.iloc[-1]["Close"]
        prev_close = self.df.iloc[-2]["Close"]
        threshold = cfg.get("big_move_threshold", 0.05)
        score = cfg.get("big_move_score", 2)

        if prev_close == 0:
            return

        daily_return = (price - prev_close) / prev_close

        if daily_return <= -threshold:
            # Big drop — mean-reversion buy
            self._add_signal(SignalType.RSI_SIGNAL, ts, price,
                             MarketBias.BULLISH, score,
                             f"🔥 Big drop {daily_return:.1%} — mean-reversion bounce")
        elif daily_return >= threshold:
            # Big pump — mean-reversion sell/fade
            self._add_signal(SignalType.RSI_SIGNAL, ts, price,
                             MarketBias.BEARISH, score,
                             f"🔥 Big pump +{daily_return:.1%} — mean-reversion fade")

        # Consecutive days in one direction = exhaustion
        if len(self.df) >= 4:
            returns = []
            for i in range(-3, 0):
                c = self.df.iloc[i]["Close"]
                p = self.df.iloc[i - 1]["Close"]
                if p > 0:
                    returns.append((c - p) / p)
            if len(returns) == 3:
                if all(r < -0.02 for r in returns):
                    self._add_signal(SignalType.RSI_SIGNAL, ts, price,
                                     MarketBias.BULLISH, 1,
                                     "3 consecutive red days — exhaustion bounce")
                elif all(r > 0.02 for r in returns):
                    self._add_signal(SignalType.RSI_SIGNAL, ts, price,
                                     MarketBias.BEARISH, 1,
                                     "3 consecutive green days — exhaustion fade")

    # ── scoring & setup ───────────────────────────────────

    def _compute_score(self):
        for sig in self._all_signals:
            if sig.bias == MarketBias.BULLISH:
                self._bull_score += sig.score
            elif sig.bias == MarketBias.BEARISH:
                self._bear_score += sig.score

        net = self._bull_score - self._bear_score
        if net > 0:
            self._bias = MarketBias.BULLISH
        elif net < 0:
            self._bias = MarketBias.BEARISH
        else:
            self._bias = MarketBias.NEUTRAL

    def _generate_trade_setup(self):
        cfg = self.cfg
        price = self.df.iloc[-1]["Close"]
        net = self.net_score
        thresholds = cfg["score_thresholds"]

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

        # ATR-based SL/TP
        atr = compute_atr(self.df, cfg["atr_period"])
        current_atr = float(atr.iloc[-1])

        if current_atr == 0:
            current_atr = price * 0.02

        if self._bias == MarketBias.BULLISH:
            sl = price - (current_atr * cfg["atr_sl_multiplier"])
            tp = price + (current_atr * cfg["atr_tp_multiplier"])
        elif self._bias == MarketBias.BEARISH:
            sl = price + (current_atr * cfg["atr_sl_multiplier"])
            tp = price - (current_atr * cfg["atr_tp_multiplier"])
        else:
            sl = price - (current_atr * cfg["atr_sl_multiplier"])
            tp = price + (current_atr * cfg["atr_tp_multiplier"])

        risk_per_share = abs(price - sl)
        reward_per_share = abs(tp - price)
        rr = reward_per_share / risk_per_share if risk_per_share > 0 else 0

        # Use aggressive position sizing for leveraged mode
        risk_pct = cfg.get("risk_per_trade", config.RISK_PER_TRADE)
        position_size = calculate_position_size(
            capital=config.INITIAL_CAPITAL,
            entry_price=price,
            stop_loss_price=sl,
            risk_pct=risk_pct,
        )

        self._setup = TradeSetup(
            action=action,
            ticker=self.ticker,
            entry_price=price,
            stop_loss=sl,
            take_profit=tp,
            position_size=position_size,
            risk_reward=round(rr, 2),
            composite_score=net,
            signals=self._all_signals,
            bias=self._bias,
        )

    def _generate_hold(self):
        price = self.df.iloc[-1]["Close"] if len(self.df) > 0 else 0
        self._setup = TradeSetup(
            action=TradeAction.HOLD,
            ticker=self.ticker,
            entry_price=price,
            stop_loss=price,
            take_profit=price,
            position_size=0,
            risk_reward=0,
            composite_score=0,
            signals=[],
            bias=MarketBias.NEUTRAL,
        )

    def _add_signal(self, sig_type, ts, price, bias, score, details):
        self._all_signals.append(Signal(
            signal_type=sig_type,
            timestamp=ts,
            price=price,
            bias=bias,
            score=score,
            details=details,
        ))
