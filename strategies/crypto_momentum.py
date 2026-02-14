"""
Crypto Trend Momentum Strategy — 4H Candles
============================================
GROUND-UP REBUILD — simplified and more robust.

Philosophy: "Catch the trend, ride the breakout, get out early if wrong"
    - 4H candles (less noise than 1H, more timely than 1D)
    - EMA trend filter (20/50/200)
    - Bollinger Band breakout entries
    - MACD momentum confirmation
    - Volume spike confirmation
    - RSI divergence as bonus signal
    - 2x ATR SL, 3x ATR TP — natural 1:1.5 R:R
    - Focus on BTC and ETH only (highest liquidity)

Entry logic:
    1. Price above 50 EMA = only buy. Price below = only sell.
    2. BB breakout (close outside band) + volume spike = entry
    3. MACD confirms direction
    4. EMA ribbon alignment gives trend strength
    5. RSI divergence adds conviction
    6. Exhaustion detection (consecutive same-direction) = caution
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
    compute_atr, bb_squeeze_detected, calculate_position_size,
    compute_macd, compute_obv, detect_rsi_divergence,
)


class CryptoMomentumStrategy:
    """
    Trend-following momentum strategy for crypto (4H candles).
    Same interface as SMCStrategy / ForexICTStrategy.
    """

    def __init__(self, df: pd.DataFrame, ticker: str = "UNKNOWN",
                 stock_mode: bool = True):
        self.df = df
        self.ticker = ticker
        self.cfg = config.CRYPTO_MODE
        self._all_signals: list[Signal] = []
        self._bias = MarketBias.NEUTRAL
        self._setup: TradeSetup | None = None
        self._bull_score = 0
        self._bear_score = 0

    def run(self) -> "CryptoMomentumStrategy":
        if len(self.df) < 60:
            self._generate_hold()
            return self

        self._analyze_ema_trend()
        self._analyze_ema_crossover()
        self._analyze_rsi()
        self._analyze_rsi_divergence()
        self._analyze_volume()
        self._analyze_bollinger_breakout()
        self._analyze_macd()
        self._analyze_exhaustion()
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
        """EMA ribbon: 20/50/200 for trend structure."""
        cfg = self.cfg
        ema_fast = compute_ema(self.df, cfg["ema_fast"])    # 9
        ema_mid = compute_ema(self.df, cfg["ema_mid"])      # 21
        ema_slow = compute_ema(self.df, cfg["ema_slow"])    # 50
        ts = self.df.index[-1]
        price = self.df.iloc[-1]["Close"]

        ef = float(ema_fast.iloc[-1])
        em = float(ema_mid.iloc[-1])
        es = float(ema_slow.iloc[-1])

        # Full bullish alignment: 9 > 21 > 50
        if ef > em > es:
            self._add_signal(SignalType.EMA_RIBBON, ts, price,
                             MarketBias.BULLISH, 3,
                             f"EMA ribbon fully bullish ({ef:.0f} > {em:.0f} > {es:.0f})")
        elif ef > em and price > es:
            self._add_signal(SignalType.EMA_RIBBON, ts, price,
                             MarketBias.BULLISH, 2,
                             "EMA partial bullish, price above 50 EMA")
        elif ef < em < es:
            self._add_signal(SignalType.EMA_RIBBON, ts, price,
                             MarketBias.BEARISH, 3,
                             f"EMA ribbon fully bearish ({ef:.0f} < {em:.0f} < {es:.0f})")
        elif ef < em and price < es:
            self._add_signal(SignalType.EMA_RIBBON, ts, price,
                             MarketBias.BEARISH, 2,
                             "EMA partial bearish, price below 50 EMA")

    def _analyze_ema_crossover(self):
        """Recent EMA crossover for momentum timing."""
        cfg = self.cfg
        ema_fast = compute_ema(self.df, cfg["ema_fast"])
        ema_mid = compute_ema(self.df, cfg["ema_mid"])
        ts = self.df.index[-1]
        price = self.df.iloc[-1]["Close"]

        if len(ema_fast) < 2 or len(ema_mid) < 2:
            return

        prev_above = float(ema_fast.iloc[-2]) > float(ema_mid.iloc[-2])
        curr_above = float(ema_fast.iloc[-1]) > float(ema_mid.iloc[-1])
        if curr_above and not prev_above:
            self._add_signal(SignalType.EMA_CROSSOVER, ts, price,
                             MarketBias.BULLISH, 2,
                             f"EMA {cfg['ema_fast']}/{cfg['ema_mid']} bullish cross")
        elif not curr_above and prev_above:
            self._add_signal(SignalType.EMA_CROSSOVER, ts, price,
                             MarketBias.BEARISH, 2,
                             f"EMA {cfg['ema_fast']}/{cfg['ema_mid']} bearish cross")

    def _analyze_rsi(self):
        """RSI zones — wider for crypto volatility."""
        cfg = self.cfg
        rsi_series = compute_rsi_series(self.df, cfg["rsi_period"])
        rsi = float(rsi_series.iloc[-1])
        ts = self.df.index[-1]
        price = self.df.iloc[-1]["Close"]

        if rsi <= cfg["rsi_extreme_oversold"]:
            self._add_signal(SignalType.RSI_SIGNAL, ts, price,
                             MarketBias.BULLISH, 2,
                             f"RSI extreme oversold ({rsi:.1f})")
        elif rsi <= cfg["rsi_oversold"]:
            self._add_signal(SignalType.RSI_SIGNAL, ts, price,
                             MarketBias.BULLISH, 1,
                             f"RSI oversold ({rsi:.1f})")
        elif rsi >= cfg["rsi_extreme_overbought"]:
            self._add_signal(SignalType.RSI_SIGNAL, ts, price,
                             MarketBias.BEARISH, 2,
                             f"RSI extreme overbought ({rsi:.1f})")
        elif rsi >= cfg["rsi_overbought"]:
            self._add_signal(SignalType.RSI_SIGNAL, ts, price,
                             MarketBias.BEARISH, 1,
                             f"RSI overbought ({rsi:.1f})")

    def _analyze_rsi_divergence(self):
        """RSI divergence as high-conviction bonus signal."""
        cfg = self.cfg
        rsi_series = compute_rsi_series(self.df, cfg["rsi_period"])
        ts = self.df.index[-1]
        price = self.df.iloc[-1]["Close"]

        div = detect_rsi_divergence(self.df, rsi_series, lookback=14)
        if div == "bullish":
            self._add_signal(SignalType.RSI_DIVERGENCE, ts, price,
                             MarketBias.BULLISH, 2,
                             "RSI bullish divergence — price lower low, RSI higher low")
        elif div == "bearish":
            self._add_signal(SignalType.RSI_DIVERGENCE, ts, price,
                             MarketBias.BEARISH, 2,
                             "RSI bearish divergence — price higher high, RSI lower high")

    def _analyze_volume(self):
        """Volume spike confirmation."""
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

        if vol_ratio >= cfg["volume_spike_multiplier"]:
            price_change = price - self.df.iloc[-2]["Close"]
            bias = MarketBias.BULLISH if price_change > 0 else MarketBias.BEARISH
            self._add_signal(SignalType.VOLUME_MOMENTUM, ts, price,
                             bias, 2,
                             f"Volume spike {vol_ratio:.1f}x avg — institutional interest")

    def _analyze_bollinger_breakout(self):
        """BB breakout: price closing outside bands = momentum move."""
        cfg = self.cfg
        upper, middle, lower = compute_bollinger_bands(
            self.df, cfg["bb_period"], cfg["bb_std"])
        ts = self.df.index[-1]
        price = self.df.iloc[-1]["Close"]

        # BB squeeze detected = volatility expansion imminent
        if bb_squeeze_detected(self.df, cfg["bb_period"], cfg["bb_std"]):
            if price > float(middle.iloc[-1]):
                self._add_signal(SignalType.BB_SQUEEZE, ts, price,
                                 MarketBias.BULLISH, 2,
                                 "BB squeeze breakout — bullish expansion")
            else:
                self._add_signal(SignalType.BB_SQUEEZE, ts, price,
                                 MarketBias.BEARISH, 2,
                                 "BB squeeze breakout — bearish expansion")

        # Close above upper band = strong bullish
        if price >= float(upper.iloc[-1]):
            self._add_signal(SignalType.BOLLINGER_BAND, ts, price,
                             MarketBias.BULLISH, 1,
                             f"Price above upper BB ({upper.iloc[-1]:.0f}) — bullish breakout")
        elif price <= float(lower.iloc[-1]):
            self._add_signal(SignalType.BOLLINGER_BAND, ts, price,
                             MarketBias.BEARISH, 1,
                             f"Price below lower BB ({lower.iloc[-1]:.0f}) — bearish breakout")

    def _analyze_macd(self):
        """MACD crossover + histogram momentum."""
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

        # Histogram acceleration
        if len(histogram) >= 3:
            h = [float(histogram.iloc[i]) for i in range(-3, 0)]
            if h[-1] > h[-2] > h[-3] and h[-1] > 0:
                self._add_signal(SignalType.MACD_CROSSOVER, ts, price,
                                 MarketBias.BULLISH, 1,
                                 "MACD histogram accelerating bullish")
            elif h[-1] < h[-2] < h[-3] and h[-1] < 0:
                self._add_signal(SignalType.MACD_CROSSOVER, ts, price,
                                 MarketBias.BEARISH, 1,
                                 "MACD histogram accelerating bearish")

    def _analyze_exhaustion(self):
        """Consecutive same-direction candles as exhaustion warning."""
        cfg = self.cfg
        threshold = cfg["exhaustion_consecutive_days"]
        ts = self.df.index[-1]
        price = self.df.iloc[-1]["Close"]

        if len(self.df) < threshold + 1:
            return

        consecutive_green = 0
        consecutive_red = 0

        for i in range(1, min(threshold + 2, len(self.df))):
            row = self.df.iloc[-i]
            if row["Close"] > row["Open"]:
                if consecutive_red > 0:
                    break
                consecutive_green += 1
            elif row["Close"] < row["Open"]:
                if consecutive_green > 0:
                    break
                consecutive_red += 1
            else:
                break

        if consecutive_green >= threshold:
            self._add_signal(SignalType.EXHAUSTION_WARNING, ts, price,
                             MarketBias.BEARISH, 1,
                             f"{consecutive_green} consecutive green candles — exhaustion warning")
        elif consecutive_red >= threshold:
            self._add_signal(SignalType.EXHAUSTION_WARNING, ts, price,
                             MarketBias.BULLISH, 1,
                             f"{consecutive_red} consecutive red candles — exhaustion warning")

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

        # 50 EMA macro filter: don't buy below 50 EMA, don't sell above it
        ema_slow = compute_ema(self.df, cfg["ema_slow"])
        if action in (TradeAction.BUY, TradeAction.STRONG_BUY) and price < float(ema_slow.iloc[-1]):
            action = TradeAction.HOLD
        elif action in (TradeAction.SELL, TradeAction.STRONG_SELL) and price > float(ema_slow.iloc[-1]):
            action = TradeAction.HOLD

        # ATR-based SL/TP: wider for crypto
        atr = compute_atr(self.df, cfg["atr_period"])
        current_atr = float(atr.iloc[-1])
        if current_atr == 0:
            current_atr = price * 0.02

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
