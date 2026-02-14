"""
Commodity Strategy — Gold & Silver Mean-Reversion + Trend
==========================================================
Dedicated strategy for precious metals (XAUUSD, XAGUSD).

Philosophy:
    - Gold is a "safe haven" that trends in crises but mean-reverts in calm
    - Silver follows gold but with more volatility
    - Use EMA trend for direction, RSI for mean-reversion entries
    - 4H candles for gold, daily for silver
    - Conservative sizing (this is where the $24K blowup happened)
    - 1.5x ATR SL, 3x ATR TP — wider stops to survive noise

Entry logic:
    1. 20 EMA vs 50 EMA determines trend direction
    2. RSI pullback to mean (40-50 in uptrend, 50-60 in downtrend)
    3. MACD confirmation
    4. Bollinger Band proximity for mean-reversion entries
    5. Price action confirmation (candle direction)
"""
from __future__ import annotations

import pandas as pd
import numpy as np
from datetime import datetime

import config
from models.signals import (
    Signal, SignalType, MarketBias, TradeAction, TradeSetup,
)
from utils.helpers import (
    compute_rsi_series, compute_bollinger_bands, compute_ema,
    compute_atr, calculate_position_size, compute_macd,
)


# Configuration for commodities (separate from crypto)
COMMODITY_CONFIG = {
    "ema_fast": 9,
    "ema_mid": 21,
    "ema_slow": 50,
    "rsi_period": 14,
    "rsi_oversold": 35,
    "rsi_overbought": 65,
    "macd_fast": 12,
    "macd_slow": 26,
    "macd_signal": 9,
    "bb_period": 20,
    "bb_std": 2.0,
    "atr_period": 14,
    "atr_sl_multiplier": 1.5,     # Tighter SL for commodities
    "atr_tp_multiplier": 3.0,     # Wide TP for trend runs
    "score_thresholds": {
        "strong_buy": 6,
        "buy": 4,
        "neutral": 2,
        "sell": 4,
        "strong_sell": 6,
    },
    "risk_per_trade": 0.005,       # 0.5% — conservative for commodities
}


class CommodityStrategy:
    """
    Mean-reversion + trend strategy for gold/silver.
    Same interface as other strategies.
    """

    def __init__(self, df: pd.DataFrame, ticker: str = "UNKNOWN",
                 stock_mode: bool = False):
        self.df = df
        self.ticker = ticker
        self.cfg = COMMODITY_CONFIG
        self._all_signals: list[Signal] = []
        self._bias = MarketBias.NEUTRAL
        self._setup: TradeSetup | None = None
        self._bull_score = 0
        self._bear_score = 0

    def run(self) -> "CommodityStrategy":
        if len(self.df) < 60:
            self._generate_hold()
            return self

        self._analyze_ema_trend()
        self._analyze_rsi_mean_reversion()
        self._analyze_macd()
        self._analyze_bollinger()
        self._analyze_price_action()
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

    def _analyze_ema_trend(self):
        """EMA trend structure: 9/21/50."""
        cfg = self.cfg
        ema_fast = compute_ema(self.df, cfg["ema_fast"])
        ema_mid = compute_ema(self.df, cfg["ema_mid"])
        ema_slow = compute_ema(self.df, cfg["ema_slow"])
        ts = self.df.index[-1]
        price = self.df.iloc[-1]["Close"]

        ef = float(ema_fast.iloc[-1])
        em = float(ema_mid.iloc[-1])
        es = float(ema_slow.iloc[-1])

        if ef > em > es:
            self._add_signal(SignalType.EMA_RIBBON, ts, price,
                             MarketBias.BULLISH, 3,
                             f"Full bullish EMA alignment ({ef:.1f} > {em:.1f} > {es:.1f})")
        elif ef > em and price > es:
            self._add_signal(SignalType.EMA_RIBBON, ts, price,
                             MarketBias.BULLISH, 2,
                             "Partial bullish — price above slow EMA")
        elif ef < em < es:
            self._add_signal(SignalType.EMA_RIBBON, ts, price,
                             MarketBias.BEARISH, 3,
                             f"Full bearish EMA alignment ({ef:.1f} < {em:.1f} < {es:.1f})")
        elif ef < em and price < es:
            self._add_signal(SignalType.EMA_RIBBON, ts, price,
                             MarketBias.BEARISH, 2,
                             "Partial bearish — price below slow EMA")

    def _analyze_rsi_mean_reversion(self):
        """
        Mean-reversion RSI: in trend, look for pullbacks to mean.
        This is the PRIMARY entry signal for commodities.
        """
        cfg = self.cfg
        rsi_series = compute_rsi_series(self.df, cfg["rsi_period"])
        rsi = float(rsi_series.iloc[-1])
        ts = self.df.index[-1]
        price = self.df.iloc[-1]["Close"]

        ema_fast = compute_ema(self.df, cfg["ema_fast"])
        ema_slow = compute_ema(self.df, cfg["ema_slow"])
        in_uptrend = float(ema_fast.iloc[-1]) > float(ema_slow.iloc[-1])
        in_downtrend = float(ema_fast.iloc[-1]) < float(ema_slow.iloc[-1])

        if in_uptrend:
            if rsi <= cfg["rsi_oversold"]:
                # Deep pullback in uptrend = strong mean-reversion buy
                self._add_signal(SignalType.RSI_SIGNAL, ts, price,
                                 MarketBias.BULLISH, 3,
                                 f"Deep RSI pullback in uptrend ({rsi:.1f}) — high-prob long")
            elif 35 < rsi <= 50:
                self._add_signal(SignalType.RSI_SIGNAL, ts, price,
                                 MarketBias.BULLISH, 2,
                                 f"RSI pullback in uptrend ({rsi:.1f}) — buy zone")
            elif rsi >= cfg["rsi_overbought"]:
                self._add_signal(SignalType.RSI_SIGNAL, ts, price,
                                 MarketBias.BEARISH, 1,
                                 f"RSI overbought ({rsi:.1f}) — avoid new longs")
        elif in_downtrend:
            if rsi >= cfg["rsi_overbought"]:
                self._add_signal(SignalType.RSI_SIGNAL, ts, price,
                                 MarketBias.BEARISH, 3,
                                 f"RSI bounce in downtrend ({rsi:.1f}) — high-prob short")
            elif 50 <= rsi < 65:
                self._add_signal(SignalType.RSI_SIGNAL, ts, price,
                                 MarketBias.BEARISH, 2,
                                 f"RSI bounce in downtrend ({rsi:.1f}) — sell zone")
            elif rsi <= cfg["rsi_oversold"]:
                self._add_signal(SignalType.RSI_SIGNAL, ts, price,
                                 MarketBias.BULLISH, 1,
                                 f"RSI oversold ({rsi:.1f}) — avoid new shorts")

    def _analyze_macd(self):
        """MACD momentum confirmation."""
        cfg = self.cfg
        macd_line, signal_line, histogram = compute_macd(
            self.df, cfg["macd_fast"], cfg["macd_slow"], cfg["macd_signal"])
        ts = self.df.index[-1]
        price = self.df.iloc[-1]["Close"]

        if len(macd_line) < 2:
            return

        prev_above = float(macd_line.iloc[-2]) > float(signal_line.iloc[-2])
        curr_above = float(macd_line.iloc[-1]) > float(signal_line.iloc[-1])

        if curr_above and not prev_above:
            self._add_signal(SignalType.MACD_CROSSOVER, ts, price,
                             MarketBias.BULLISH, 2,
                             "MACD bullish crossover")
        elif not curr_above and prev_above:
            self._add_signal(SignalType.MACD_CROSSOVER, ts, price,
                             MarketBias.BEARISH, 2,
                             "MACD bearish crossover")
        elif curr_above:
            self._add_signal(SignalType.MACD_CROSSOVER, ts, price,
                             MarketBias.BULLISH, 1,
                             "MACD bullish alignment")
        else:
            self._add_signal(SignalType.MACD_CROSSOVER, ts, price,
                             MarketBias.BEARISH, 1,
                             "MACD bearish alignment")

    def _analyze_bollinger(self):
        """Bollinger Bands for mean-reversion zones."""
        cfg = self.cfg
        upper, middle, lower = compute_bollinger_bands(
            self.df, cfg["bb_period"], cfg["bb_std"])
        ts = self.df.index[-1]
        price = self.df.iloc[-1]["Close"]

        # Price near lower band in uptrend = buy zone
        if price <= float(lower.iloc[-1]):
            self._add_signal(SignalType.BOLLINGER_BAND, ts, price,
                             MarketBias.BULLISH, 2,
                             f"Price at lower BB ({lower.iloc[-1]:.1f}) — mean-reversion buy")
        elif price >= float(upper.iloc[-1]):
            self._add_signal(SignalType.BOLLINGER_BAND, ts, price,
                             MarketBias.BEARISH, 2,
                             f"Price at upper BB ({upper.iloc[-1]:.1f}) — mean-reversion sell")

        # Near middle band = equilibrium (no signal)

    def _analyze_price_action(self):
        """Candlestick confirmation."""
        if len(self.df) < 2:
            return

        ts = self.df.index[-1]
        last = self.df.iloc[-1]
        price = last["Close"]
        hl_range = last["High"] - last["Low"]
        if hl_range <= 0:
            return

        body_pct = abs(last["Close"] - last["Open"]) / hl_range
        is_bullish = last["Close"] > last["Open"]

        if body_pct > 0.6:
            if is_bullish:
                self._add_signal(SignalType.VOLUME_MOMENTUM, ts, price,
                                 MarketBias.BULLISH, 1,
                                 f"Strong bullish candle ({body_pct*100:.0f}% body)")
            else:
                self._add_signal(SignalType.VOLUME_MOMENTUM, ts, price,
                                 MarketBias.BEARISH, 1,
                                 f"Strong bearish candle ({body_pct*100:.0f}% body)")

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

        # Trend filter: only trade with EMA direction
        ema_fast = compute_ema(self.df, cfg["ema_fast"])
        ema_slow = compute_ema(self.df, cfg["ema_slow"])
        in_uptrend = float(ema_fast.iloc[-1]) > float(ema_slow.iloc[-1])
        in_downtrend = float(ema_fast.iloc[-1]) < float(ema_slow.iloc[-1])

        if action in (TradeAction.BUY, TradeAction.STRONG_BUY) and not in_uptrend:
            action = TradeAction.HOLD
        elif action in (TradeAction.SELL, TradeAction.STRONG_SELL) and not in_downtrend:
            action = TradeAction.HOLD

        # ATR-based SL/TP
        atr = compute_atr(self.df, cfg["atr_period"])
        current_atr = float(atr.iloc[-1])
        if current_atr == 0:
            current_atr = price * 0.01

        sl_mult = cfg["atr_sl_multiplier"]
        tp_mult = cfg["atr_tp_multiplier"]

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

        risk_pct = cfg.get("risk_per_trade", 0.005)
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
