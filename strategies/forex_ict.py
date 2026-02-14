"""
Forex Trend-Continuation Strategy — 4H Multi-TF Confirmation
=============================================================
GROUND-UP REBUILD — replaces the over-complex ICT/session-bias approach.

Philosophy: "Simple, Robust, and Profitable"
    - Trade WITH the trend, not against it
    - Use 4H candles (less noise than 1H)
    - EMA crossover for trend direction
    - RSI for entry timing (pullback zones, not extremes)
    - MACD for momentum confirmation
    - Session timing: prefer London/NY for execution
    - Wide stops (2x ATR) so trades survive noise
    - Only 4 major pairs: EURUSD, GBPUSD, USDJPY, AUDUSD

Entry logic:
    1. 20 EMA > 50 EMA = uptrend (buy only)
    2. 20 EMA < 50 EMA = downtrend (sell only)
    3. RSI pullback to 40-50 in uptrend = buy entry
    4. RSI pullback to 50-60 in downtrend = sell entry
    5. MACD confirms direction (line above/below signal)
    6. Volume/price action: last candle confirms direction
    7. Session bonus: London or NY kill zone
    8. ATR-based SL (2x) and TP (3x) — natural 1:1.5 R:R

Same interface as the old ForexICTStrategy for drop-in replacement.
"""
from __future__ import annotations

import pandas as pd
import numpy as np
from datetime import datetime
from typing import List, Tuple, Optional

import config
from models.signals import (
    Signal, SignalType, MarketBias, TradeAction, TradeSetup,
)
from utils.helpers import (
    compute_atr, compute_ema, compute_rsi_series, compute_macd,
    calculate_position_size,
)


class ForexICTStrategy:
    """
    Forex trend-continuation strategy using 4H candles.
    Same interface as SMCStrategy for drop-in compatibility.
    """

    def __init__(self, df: pd.DataFrame, ticker: str = "UNKNOWN",
                 stock_mode: bool = False):
        self.df = df
        self.ticker = ticker
        self.cfg = config.FOREX_MODE
        self._all_signals: list[Signal] = []
        self._bias = MarketBias.NEUTRAL
        self._bullish_score = 0
        self._bearish_score = 0
        self._setup: TradeSetup | None = None

    # ── public API ────────────────────────────────────────

    def run(self) -> "ForexICTStrategy":
        if len(self.df) < 60:
            self._generate_hold()
            return self

        self._analyze_ema_trend()
        self._analyze_ema_crossover()
        self._analyze_rsi_pullback()
        self._analyze_macd()
        self._analyze_price_action()
        self._analyze_session_timing()
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

    # ── signal components ─────────────────────────────────

    def _analyze_ema_trend(self):
        """
        Core trend filter: 20 EMA vs 50 EMA alignment.
        This is the PRIMARY signal — everything else is confirmation.
        """
        ema20 = compute_ema(self.df, 20)
        ema50 = compute_ema(self.df, 50)
        ts = self.df.index[-1]
        price = self.df.iloc[-1]["Close"]

        e20 = float(ema20.iloc[-1])
        e50 = float(ema50.iloc[-1])

        # Check trend strength: how far apart are the EMAs?
        spread_pct = abs(e20 - e50) / e50 if e50 > 0 else 0

        if e20 > e50:
            # Uptrend — score based on EMA spread
            score = 3 if spread_pct > 0.002 else 2
            self._add_signal(SignalType.EMA_RIBBON, ts, price,
                             MarketBias.BULLISH, score,
                             f"Uptrend: 20 EMA ({e20:.5f}) > 50 EMA ({e50:.5f}), "
                             f"spread {spread_pct*100:.2f}%")
        elif e20 < e50:
            score = 3 if spread_pct > 0.002 else 2
            self._add_signal(SignalType.EMA_RIBBON, ts, price,
                             MarketBias.BEARISH, score,
                             f"Downtrend: 20 EMA ({e20:.5f}) < 50 EMA ({e50:.5f}), "
                             f"spread {spread_pct*100:.2f}%")
        else:
            self._add_signal(SignalType.EMA_RIBBON, ts, price,
                             MarketBias.NEUTRAL, 0,
                             "EMAs flat — no clear trend")

    def _analyze_ema_crossover(self):
        """Recent EMA crossover (within last 3 bars) = fresh momentum."""
        ema20 = compute_ema(self.df, 20)
        ema50 = compute_ema(self.df, 50)
        ts = self.df.index[-1]
        price = self.df.iloc[-1]["Close"]

        if len(ema20) < 4 or len(ema50) < 4:
            return

        # Check if crossover happened in last 3 bars
        for lookback in range(1, 4):
            prev_above = float(ema20.iloc[-lookback - 1]) > float(ema50.iloc[-lookback - 1])
            curr_above = float(ema20.iloc[-lookback]) > float(ema50.iloc[-lookback])
            if curr_above and not prev_above:
                self._add_signal(SignalType.EMA_CROSSOVER, ts, price,
                                 MarketBias.BULLISH, 2,
                                 f"Bullish EMA cross {lookback} bars ago")
                return
            elif not curr_above and prev_above:
                self._add_signal(SignalType.EMA_CROSSOVER, ts, price,
                                 MarketBias.BEARISH, 2,
                                 f"Bearish EMA cross {lookback} bars ago")
                return

    def _analyze_rsi_pullback(self):
        """
        RSI pullback entry timing.
        In uptrend: RSI pulling back to 35-50 = buying opportunity
        In downtrend: RSI bouncing to 50-65 = selling opportunity
        """
        rsi_series = compute_rsi_series(self.df, 14)
        rsi = float(rsi_series.iloc[-1])
        ts = self.df.index[-1]
        price = self.df.iloc[-1]["Close"]

        # Determine current trend from EMA
        ema20 = compute_ema(self.df, 20)
        ema50 = compute_ema(self.df, 50)
        in_uptrend = float(ema20.iloc[-1]) > float(ema50.iloc[-1])
        in_downtrend = float(ema20.iloc[-1]) < float(ema50.iloc[-1])

        if in_uptrend:
            if 35 <= rsi <= 50:
                self._add_signal(SignalType.RSI_SIGNAL, ts, price,
                                 MarketBias.BULLISH, 2,
                                 f"RSI pullback in uptrend ({rsi:.1f}) — buy zone")
            elif rsi < 35:
                self._add_signal(SignalType.RSI_SIGNAL, ts, price,
                                 MarketBias.BEARISH, 1,
                                 f"RSI too weak for uptrend ({rsi:.1f}) — trend may be reversing")
            elif rsi > 70:
                self._add_signal(SignalType.RSI_SIGNAL, ts, price,
                                 MarketBias.BEARISH, 1,
                                 f"RSI overbought ({rsi:.1f}) — avoid new longs")
        elif in_downtrend:
            if 50 <= rsi <= 65:
                self._add_signal(SignalType.RSI_SIGNAL, ts, price,
                                 MarketBias.BEARISH, 2,
                                 f"RSI bounce in downtrend ({rsi:.1f}) — sell zone")
            elif rsi > 65:
                self._add_signal(SignalType.RSI_SIGNAL, ts, price,
                                 MarketBias.BULLISH, 1,
                                 f"RSI too strong for downtrend ({rsi:.1f}) — trend may be reversing")
            elif rsi < 30:
                self._add_signal(SignalType.RSI_SIGNAL, ts, price,
                                 MarketBias.BULLISH, 1,
                                 f"RSI oversold ({rsi:.1f}) — avoid new shorts")

    def _analyze_macd(self):
        """MACD confirmation: line vs signal line alignment with trend."""
        cfg = self.cfg
        macd_fast = cfg.get("macd_fast", 12)
        macd_slow = cfg.get("macd_slow", 26)
        macd_sig = cfg.get("macd_signal", 9)

        macd_line, signal_line, histogram = compute_macd(
            self.df, macd_fast, macd_slow, macd_sig)
        ts = self.df.index[-1]
        price = self.df.iloc[-1]["Close"]

        if len(macd_line) < 2:
            return

        # MACD crossover
        prev_above = float(macd_line.iloc[-2]) > float(signal_line.iloc[-2])
        curr_above = float(macd_line.iloc[-1]) > float(signal_line.iloc[-1])

        if curr_above and not prev_above:
            self._add_signal(SignalType.MACD_CROSSOVER, ts, price,
                             MarketBias.BULLISH, 2,
                             "MACD bullish crossover — momentum confirming")
        elif not curr_above and prev_above:
            self._add_signal(SignalType.MACD_CROSSOVER, ts, price,
                             MarketBias.BEARISH, 2,
                             "MACD bearish crossover — momentum confirming")
        elif curr_above:
            # Already bullish, add minor confirmation
            self._add_signal(SignalType.MACD_CROSSOVER, ts, price,
                             MarketBias.BULLISH, 1,
                             "MACD bullish — momentum aligned")
        else:
            self._add_signal(SignalType.MACD_CROSSOVER, ts, price,
                             MarketBias.BEARISH, 1,
                             "MACD bearish — momentum aligned")

    def _analyze_price_action(self):
        """
        Last candle confirmation:
        - Bullish candle (close > open) in uptrend = +1
        - Bearish candle in downtrend = +1
        - Counter-trend candle = -1
        """
        if len(self.df) < 2:
            return

        ts = self.df.index[-1]
        last = self.df.iloc[-1]
        price = last["Close"]
        is_bullish_candle = last["Close"] > last["Open"]
        body_pct = abs(last["Close"] - last["Open"]) / (last["High"] - last["Low"]) if (last["High"] - last["Low"]) > 0 else 0

        # Strong candle (body > 50% of range)
        if body_pct > 0.5:
            if is_bullish_candle:
                self._add_signal(SignalType.VOLUME_MOMENTUM, ts, price,
                                 MarketBias.BULLISH, 1,
                                 f"Strong bullish candle (body {body_pct*100:.0f}% of range)")
            else:
                self._add_signal(SignalType.VOLUME_MOMENTUM, ts, price,
                                 MarketBias.BEARISH, 1,
                                 f"Strong bearish candle (body {body_pct*100:.0f}% of range)")

    def _analyze_session_timing(self):
        """
        Session bonus: London (07-10 UTC) and NY (12-16 UTC) get extra score.
        Asian session (00-06 UTC) gets a penalty.
        """
        ts = self.df.index[-1]
        if not hasattr(ts, 'hour'):
            return

        hour = ts.hour
        price = self.df.iloc[-1]["Close"]

        if 7 <= hour <= 10:
            self._add_signal(SignalType.KILL_ZONE, ts, price,
                             MarketBias.NEUTRAL, 0,
                             "London session — high probability window")
            # Boost all existing directional signals by 1
            for sig in self._all_signals:
                if sig.bias != MarketBias.NEUTRAL:
                    sig.score += 1
        elif 12 <= hour <= 16:
            self._add_signal(SignalType.KILL_ZONE, ts, price,
                             MarketBias.NEUTRAL, 0,
                             "NY session — high probability window")
            for sig in self._all_signals:
                if sig.bias != MarketBias.NEUTRAL:
                    sig.score += 1
        elif 0 <= hour <= 5:
            # Asian session — reduce scores
            for sig in self._all_signals:
                if sig.bias != MarketBias.NEUTRAL and sig.score > 1:
                    sig.score -= 1

    # ── scoring & setup ───────────────────────────────────

    def _compute_score(self):
        for sig in self._all_signals:
            if sig.bias == MarketBias.BULLISH:
                self._bullish_score += sig.score
            elif sig.bias == MarketBias.BEARISH:
                self._bearish_score += sig.score

        net = self._bullish_score - self._bearish_score
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

        # Trend filter: ONLY trade with the trend
        ema20 = compute_ema(self.df, 20)
        ema50 = compute_ema(self.df, 50)
        in_uptrend = float(ema20.iloc[-1]) > float(ema50.iloc[-1])
        in_downtrend = float(ema20.iloc[-1]) < float(ema50.iloc[-1])

        if action in (TradeAction.BUY, TradeAction.STRONG_BUY) and not in_uptrend:
            action = TradeAction.HOLD
        elif action in (TradeAction.SELL, TradeAction.STRONG_SELL) and not in_downtrend:
            action = TradeAction.HOLD

        # ATR-based SL/TP: WIDER stops (2x ATR SL, 3x ATR TP)
        atr = compute_atr(self.df, cfg["atr_period"])
        current_atr = float(atr.iloc[-1])
        if current_atr == 0:
            current_atr = price * 0.005

        sl_mult = cfg["atr_sl_multiplier"]  # 2.0
        tp_mult = cfg["atr_tp_multiplier"]  # 3.0

        if self._bias == MarketBias.BULLISH:
            sl = price - (current_atr * sl_mult)
            tp = price + (current_atr * tp_mult)
        elif self._bias == MarketBias.BEARISH:
            sl = price + (current_atr * sl_mult)
            tp = price - (current_atr * tp_mult)
        else:
            sl = price - (current_atr * sl_mult)
            tp = price + (current_atr * tp_mult)

        risk_per_share = abs(price - sl)
        reward_per_share = abs(tp - price)
        rr = reward_per_share / risk_per_share if risk_per_share > 0 else 0

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
