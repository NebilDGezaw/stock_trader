"""
Crypto Momentum / Trend-Following Strategy
===========================================
Designed for crypto assets (BTC, ETH, SOL, XRP, BNB, etc.).
Crypto trends hard then chops — this strategy catches trends early,
rides them with trailing stops, and avoids chop via EMA ribbon filter.

Components:
1. EMA ribbon (9/21/50) — trend alignment + macro filter
2. RSI with crypto zones (40/60) + divergence detection
3. Volume spikes + OBV trend confirmation
4. Bollinger Band squeeze → breakout
5. MACD crossover + histogram momentum
6. Exhaustion detection (consecutive same-direction days)
7. ATR-based dynamic SL/TP with trailing stop
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
    compute_macd, compute_obv, detect_rsi_divergence,
)


class CryptoMomentumStrategy:
    """
    Trend-following momentum strategy tuned for crypto.
    Same interface as SMCStrategy / LeveragedMomentumStrategy.
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

        self._analyze_ema_ribbon()
        self._analyze_rsi()
        self._analyze_rsi_divergence()
        self._analyze_volume()
        self._analyze_obv()
        self._analyze_bollinger_squeeze()
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

    def _analyze_ema_ribbon(self):
        """EMA ribbon: 9/21/50. Full alignment = strong trend. 50 EMA = macro filter."""
        cfg = self.cfg
        ema_fast = compute_ema(self.df, cfg["ema_fast"])
        ema_mid = compute_ema(self.df, cfg["ema_mid"])
        ema_slow = compute_ema(self.df, cfg["ema_slow"])
        ts = self.df.index[-1]
        price = self.df.iloc[-1]["Close"]

        ef = ema_fast.iloc[-1]
        em = ema_mid.iloc[-1]
        es = ema_slow.iloc[-1]

        # Full bullish alignment: 9 > 21 > 50
        if ef > em > es:
            self._add_signal(SignalType.EMA_RIBBON, ts, price,
                             MarketBias.BULLISH, 2,
                             f"EMA ribbon fully bullish ({ef:.0f} > {em:.0f} > {es:.0f})")
        # Partial bullish: price above 50 and fast above mid
        elif ef > em and price > es:
            self._add_signal(SignalType.EMA_RIBBON, ts, price,
                             MarketBias.BULLISH, 1,
                             f"EMA partial bullish, above 50 EMA")
        # Full bearish alignment: 9 < 21 < 50
        elif ef < em < es:
            self._add_signal(SignalType.EMA_RIBBON, ts, price,
                             MarketBias.BEARISH, 2,
                             f"EMA ribbon fully bearish ({ef:.0f} < {em:.0f} < {es:.0f})")
        # Partial bearish: price below 50 and fast below mid
        elif ef < em and price < es:
            self._add_signal(SignalType.EMA_RIBBON, ts, price,
                             MarketBias.BEARISH, 1,
                             f"EMA partial bearish, below 50 EMA")

        # EMA crossovers (fast crossing mid)
        if len(ema_fast) >= 2 and len(ema_mid) >= 2:
            prev_above = ema_fast.iloc[-2] > ema_mid.iloc[-2]
            curr_above = ef > em
            if curr_above and not prev_above:
                self._add_signal(SignalType.EMA_CROSSOVER, ts, price,
                                 MarketBias.BULLISH, 1,
                                 f"EMA {cfg['ema_fast']}/{cfg['ema_mid']} bullish cross")
            elif not curr_above and prev_above:
                self._add_signal(SignalType.EMA_CROSSOVER, ts, price,
                                 MarketBias.BEARISH, 1,
                                 f"EMA {cfg['ema_fast']}/{cfg['ema_mid']} bearish cross")

    def _analyze_rsi(self):
        cfg = self.cfg
        rsi_series = compute_rsi_series(self.df, cfg["rsi_period"])
        rsi = float(rsi_series.iloc[-1])
        ts = self.df.index[-1]
        price = self.df.iloc[-1]["Close"]

        if rsi <= cfg["rsi_extreme_oversold"]:
            self._add_signal(SignalType.RSI_SIGNAL, ts, price,
                             MarketBias.BULLISH, 2,
                             f"🔥 RSI extreme oversold ({rsi:.1f})")
        elif rsi <= cfg["rsi_oversold"]:
            self._add_signal(SignalType.RSI_SIGNAL, ts, price,
                             MarketBias.BULLISH, 1,
                             f"RSI oversold ({rsi:.1f})")
        elif rsi >= cfg["rsi_extreme_overbought"]:
            self._add_signal(SignalType.RSI_SIGNAL, ts, price,
                             MarketBias.BEARISH, 2,
                             f"🔥 RSI extreme overbought ({rsi:.1f})")
        elif rsi >= cfg["rsi_overbought"]:
            self._add_signal(SignalType.RSI_SIGNAL, ts, price,
                             MarketBias.BEARISH, 1,
                             f"RSI overbought ({rsi:.1f})")

    def _analyze_rsi_divergence(self):
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
                             bias, 1,
                             f"Volume spike {vol_ratio:.1f}x avg — {'bullish' if price_change > 0 else 'bearish'}")
            if vol_ratio >= 3.0:
                self._add_signal(SignalType.VOLUME_MOMENTUM, ts, price,
                                 bias, 1,
                                 f"🔥 Extreme volume {vol_ratio:.1f}x — institutional")

    def _analyze_obv(self):
        """OBV trend as confirmation — rising OBV = bullish, falling = bearish."""
        if "Volume" not in self.df.columns:
            return

        ts = self.df.index[-1]
        price = self.df.iloc[-1]["Close"]
        obv = compute_obv(self.df)

        if len(obv) < 10:
            return

        # OBV trend: compare 5-period EMA of OBV
        obv_ema = obv.ewm(span=5, adjust=False).mean()
        if obv_ema.iloc[-1] > obv_ema.iloc[-5] and obv.iloc[-1] > obv.iloc[-5]:
            self._add_signal(SignalType.OBV_TREND, ts, price,
                             MarketBias.BULLISH, 1,
                             "OBV trending up — buying pressure")
        elif obv_ema.iloc[-1] < obv_ema.iloc[-5] and obv.iloc[-1] < obv.iloc[-5]:
            self._add_signal(SignalType.OBV_TREND, ts, price,
                             MarketBias.BEARISH, 1,
                             "OBV trending down — selling pressure")

    def _analyze_bollinger_squeeze(self):
        cfg = self.cfg
        upper, middle, lower = compute_bollinger_bands(
            self.df, cfg["bb_period"], cfg["bb_std"])
        ts = self.df.index[-1]
        price = self.df.iloc[-1]["Close"]

        if bb_squeeze_detected(self.df, cfg["bb_period"], cfg["bb_std"]):
            if price > middle.iloc[-1]:
                self._add_signal(SignalType.BB_SQUEEZE, ts, price,
                                 MarketBias.BULLISH, 2,
                                 "BB squeeze breakout — bullish expansion")
            else:
                self._add_signal(SignalType.BB_SQUEEZE, ts, price,
                                 MarketBias.BEARISH, 2,
                                 "BB squeeze breakout — bearish expansion")

        # Standard band touches
        if price <= lower.iloc[-1]:
            self._add_signal(SignalType.BOLLINGER_BAND, ts, price,
                             MarketBias.BULLISH, 1,
                             f"Price at lower BB ({lower.iloc[-1]:.2f})")
        elif price >= upper.iloc[-1]:
            self._add_signal(SignalType.BOLLINGER_BAND, ts, price,
                             MarketBias.BEARISH, 1,
                             f"Price at upper BB ({upper.iloc[-1]:.2f})")

    def _analyze_macd(self):
        cfg = self.cfg
        macd_line, signal_line, histogram = compute_macd(
            self.df, cfg["macd_fast"], cfg["macd_slow"], cfg["macd_signal"])
        ts = self.df.index[-1]
        price = self.df.iloc[-1]["Close"]

        if len(macd_line) < 2:
            return

        # MACD crossover
        prev_above = macd_line.iloc[-2] > signal_line.iloc[-2]
        curr_above = macd_line.iloc[-1] > signal_line.iloc[-1]

        if curr_above and not prev_above:
            self._add_signal(SignalType.MACD_CROSSOVER, ts, price,
                             MarketBias.BULLISH, 2,
                             "MACD bullish crossover")
        elif not curr_above and prev_above:
            self._add_signal(SignalType.MACD_CROSSOVER, ts, price,
                             MarketBias.BEARISH, 2,
                             "MACD bearish crossover")

        # Histogram momentum (growing = strengthening)
        if len(histogram) >= 3:
            h = histogram.iloc[-3:].values
            if h[-1] > h[-2] > h[-3] and h[-1] > 0:
                self._add_signal(SignalType.MACD_CROSSOVER, ts, price,
                                 MarketBias.BULLISH, 1,
                                 "MACD histogram accelerating bullish")
            elif h[-1] < h[-2] < h[-3] and h[-1] < 0:
                self._add_signal(SignalType.MACD_CROSSOVER, ts, price,
                                 MarketBias.BEARISH, 1,
                                 "MACD histogram accelerating bearish")

    def _analyze_exhaustion(self):
        """Consecutive same-direction days as exhaustion proxy."""
        cfg = self.cfg
        threshold = cfg["exhaustion_consecutive_days"]
        ts = self.df.index[-1]
        price = self.df.iloc[-1]["Close"]

        if len(self.df) < threshold + 1:
            return

        # Count consecutive green/red days ending at current bar
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
                             f"⚠️ {consecutive_green} consecutive green days — exhaustion warning")
        elif consecutive_red >= threshold:
            self._add_signal(SignalType.EXHAUSTION_WARNING, ts, price,
                             MarketBias.BULLISH, 1,
                             f"⚠️ {consecutive_red} consecutive red days — exhaustion warning")

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
        if action in (TradeAction.BUY, TradeAction.STRONG_BUY) and price < ema_slow.iloc[-1]:
            action = TradeAction.HOLD
        elif action in (TradeAction.SELL, TradeAction.STRONG_SELL) and price > ema_slow.iloc[-1]:
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
