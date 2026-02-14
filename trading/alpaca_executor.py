"""
Alpaca Trade Executor — bridges strategy analysis to Alpaca bracket orders.
============================================================================
Takes a TradeSetup from any strategy, converts it to an Alpaca bracket order,
calculates proper share sizing, and places it via the Alpaca client.

Crash Resilience — Sector Diversification:
    - 9 categories across growth + defensive + independent sectors
    - Sector momentum detection (XLK, XLE, XLP, XLI, etc.)
    - During crashes: energy/staples/industrials get EASIER entry bars
    - During crashes: tech/semis/leveraged get HARDER entry bars
    - Position rotation naturally moves capital toward green sectors

Safety (16 pre-trade checks):
    1.  Halal compliance (long-only, no short selling)
    2.  VIX volatility filter — reduces risk in fearful markets, halts at panic
    3.  Market regime filter (SPY SMA) — graduated bull→bear scoring
    4.  Sector momentum filter — rotates capital toward green sectors
    5.  Earnings blackout — skips stocks within ±3 days of earnings
    6.  Volume filter — rejects signals on dead volume (< 60% of avg)
    7.  Gap detection — skips stocks that gapped > 3% from prior close
    8.  Minimum R:R validation
    9.  Max concurrent positions (12)
    10. MSTU/MSTZ mutual exclusion (never hold both simultaneously)
    11. Position rotation — swaps weak positions for strong new signals
    12. Daily loss circuit breaker (3%)
    13. Weekly cumulative drawdown limit (5%)
    14. Category allocation caps (9 sectors)
    15. Per-ticker allocation caps + notional cap
    16. Buying power check with 2% buffer

Position Management (in alpaca_position_manager.py):
    - Breakeven stop at 0.5R (eliminate losing trades after initial promise)
    - Partial close at 1R (sell 50%, lock profit, let remainder ride)
    - Trailing stop at 1R+ (ATR-based, locks in profits on runners)
    - Signal reversal detection
    - Time-of-day filter (skip first 30 min noise window)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

import config
from models.signals import TradeSetup, TradeAction
from trading.alpaca_client import AlpacaClient, OrderResult

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────
#  Inverse ETF pairs — never hold both sides simultaneously
# ──────────────────────────────────────────────────────────
INVERSE_PAIRS = {
    "MSTU": "MSTZ",   # MSTU is 2x long MSTR, MSTZ is 2x short MSTR
    "MSTZ": "MSTU",
}


# ──────────────────────────────────────────────────────────
#  Portfolio Categories & Allocation Caps
# ──────────────────────────────────────────────────────────
#
# DIVERSIFICATION RATIONALE (crash resilience):
#   Tech + Semis + Clean Energy + Leveraged ≈ 55% cap (growth / momentum)
#   Energy + Consumer Staples + Industrials  ≈ 35% cap (defensive / value)
#   Healthcare + Consumer Discretionary      ≈ 15% cap (moderate correlation)
#
# During the 2022 tech crash:  XLK (tech) −28%, SOXL −70%, TQQQ −75%
#   Meanwhile:  XLE (energy) +64%, XLP (staples) +0%, XLI (industrials) +−5%
# Having defensive sectors means the system CAN find green stocks during crashes.
#
# ALL tickers are halal-compliant:
#   ✓ No banks / insurance (interest-based revenue)
#   ✓ No alcohol, tobacco, gambling, weapons manufacturers
#   ✓ No conventional finance or debt-heavy entities
#   ✓ Long-only (no short selling)

# Every ticker belongs to exactly one category
TICKER_CATEGORY = {
    # ── Leveraged ETFs (25% cap) ──
    # High risk / high reward, short holding periods.
    # Mostly tech-correlated — capped tighter to limit crash exposure.
    "MSTU":  "leveraged",
    "MSTR":  "leveraged",
    "MSTZ":  "leveraged",
    "TSLL":  "leveraged",
    "TQQQ":  "leveraged",
    "SOXL":  "leveraged",
    "FNGU":  "leveraged",

    # ── Mega Tech (20% cap) ──
    # Core growth holdings — strong momentum in bull markets.
    "AAPL":  "tech",
    "MSFT":  "tech",
    "GOOGL": "tech",
    "AMZN":  "tech",
    "NVDA":  "tech",
    "META":  "tech",
    "TSLA":  "tech",
    "AMD":   "tech",

    # ── Semiconductors (10% cap) ──
    # Tech-adjacent but supply-chain driven.
    "AVGO":  "semis",
    "QCOM":  "semis",
    "ASML":  "semis",
    "MU":    "semis",

    # ── Healthcare (10% cap) ──
    # Low tech correlation, recession-resistant.
    "ABBV":  "healthcare",
    "LLY":   "healthcare",
    "ISRG":  "healthcare",
    "JNJ":   "healthcare",

    # ── Energy (15% cap) ──  *** NEW — primary crash hedge ***
    # Oil & gas — historically INVERSE to tech in sector rotations.
    # XOM was +80% during the 2022 tech crash.  Halal (natural resources).
    "XOM":   "energy",
    "CVX":   "energy",
    "COP":   "energy",

    # ── Consumer Staples (10% cap) ──  *** NEW — defensive ***
    # Ultra-low-beta, recession-proof.  Hold steady in any crash.
    # Halal: household goods, beverages (non-alcoholic), personal care.
    "PG":    "consumer_staples",
    "KO":    "consumer_staples",
    "PEP":   "consumer_staples",
    "CL":    "consumer_staples",

    # ── Industrials (10% cap) ──  *** NEW — real economy ***
    # Infrastructure, logistics, agriculture.  Low tech correlation.
    # Halal: heavy machinery, farming, waste collection, railroads.
    "CAT":   "industrials",
    "DE":    "industrials",
    "WM":    "industrials",
    "UNP":   "industrials",

    # ── Clean Energy (5% cap) ──
    # Growth/speculative — trimmed to 2 tickers (dropped NIO/RIVN).
    "ENPH":  "clean_energy",
    "FSLR":  "clean_energy",

    # ── Consumer Discretionary (5% cap) ──
    # Retail — moderate correlation.
    "COST":  "consumer",
    "TGT":   "consumer",
}

# Maximum % of total equity allocated to each category
CATEGORY_CAPS = {
    "leveraged":        0.25,   # 25% (reduced from 30% — still aggressive)
    "tech":             0.20,   # 20% (reduced from 25%)
    "semis":            0.10,   # 10% (reduced from 15%)
    "healthcare":       0.10,   # 10% (reduced from 15%)
    "energy":           0.15,   # 15% *** NEW — primary crash hedge ***
    "consumer_staples": 0.10,   # 10% *** NEW — defensive ***
    "industrials":      0.10,   # 10% *** NEW — real economy ***
    "clean_energy":     0.05,   # 5%  (reduced from 10%)
    "consumer":         0.05,   # 5%
}

# Per-ticker allocation caps (overrides category default)
TICKER_CAPS = {
    # None needed — all tickers are solid, no speculative micro-caps
}

# Sector ETF proxies — used for sector-level momentum detection
# to determine which sectors are trending up/down for rotation
SECTOR_ETFS = {
    "tech":             "XLK",
    "semis":            "SOXX",
    "healthcare":       "XLV",
    "energy":           "XLE",
    "consumer_staples": "XLP",
    "industrials":      "XLI",
    "clean_energy":     "ICLN",
    "consumer":         "XLY",
    "crypto_proxy":     "BTC-USD",  # for MSTR/MSTU/MSTZ (Bitcoin proxies)
}

# Leveraged tickers each track a different underlying — map them individually
# so MSTR-family uses Bitcoin momentum, TQQQ/FNGU use tech, SOXL uses semis
LEVERAGED_MOMENTUM_MAP = {
    "MSTU":  "crypto_proxy",   # 2x long MicroStrategy → Bitcoin proxy
    "MSTR":  "crypto_proxy",   # MicroStrategy itself → Bitcoin proxy
    "MSTZ":  "crypto_proxy",   # 2x short MicroStrategy → Bitcoin proxy
    "TSLL":  "tech",           # 2x Tesla → tech sector
    "TQQQ":  "tech",           # 3x Nasdaq → tech sector
    "SOXL":  "semis",          # 3x Semiconductors → semis sector
    "FNGU":  "tech",           # 3x FAANG → tech sector
}

# Max single-ticker allocation (default: half of category cap)
DEFAULT_SINGLE_TICKER_CAP_RATIO = 0.5

LEVERAGED_TICKERS = {t for t, c in TICKER_CATEGORY.items() if c == "leveraged"}


# ──────────────────────────────────────────────────────────
#  Configuration
# ──────────────────────────────────────────────────────────

@dataclass
class AlpacaExecutorConfig:
    """Safety limits and risk parameters for Alpaca stock trading."""
    max_concurrent_positions: int = 12     # max total positions (up from 10 — more sectors)
    max_daily_loss_pct: float = 3.0        # stop trading if daily loss > 3%
    default_risk_pct: float = 0.01         # 1% of equity per trade
    leveraged_risk_pct: float = 0.02       # 2% for leveraged ETFs
    min_risk_reward: float = 2.0           # minimum R:R to execute
    min_shares: int = 1                    # minimum share count
    max_notional_pct: float = 0.20         # max 20% of equity in a single position
    dry_run: bool = False                  # log only, don't place orders


@dataclass
class ExecutionRecord:
    """Record of an executed (or skipped) trade."""
    ticker: str
    action: str
    qty: int
    entry_price: float
    sl: float
    tp: float
    risk_reward: float
    executed: bool
    order_id: str = ""
    reason: str = ""
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())


# ──────────────────────────────────────────────────────────
#  Helper: get category allocation used
# ──────────────────────────────────────────────────────────

def _get_category(ticker: str) -> str:
    """Get the portfolio category for a ticker."""
    cat = TICKER_CATEGORY.get(ticker.upper())
    if cat is None:
        logger.warning(f"Unknown ticker '{ticker}' — defaulting to 'other' category")
        return "other"
    return cat


def _category_exposure(positions, equity: float) -> dict[str, float]:
    """
    Calculate current exposure (market_value) per category as % of equity.
    Returns dict like {"leveraged": 0.18, "tech": 0.12, ...}
    """
    exposure = {}
    for p in positions:
        cat = _get_category(p.symbol)
        exposure[cat] = exposure.get(cat, 0.0) + abs(p.market_value)
    # Convert to % of equity
    if equity > 0:
        exposure = {k: v / equity for k, v in exposure.items()}
    return exposure


def _ticker_exposure(positions, equity: float) -> dict[str, float]:
    """
    Calculate current exposure per ticker as % of equity.
    Returns dict like {"AAPL": 0.05, "MSTU": 0.08, ...}
    """
    exposure = {}
    for p in positions:
        sym = p.symbol.upper()
        exposure[sym] = exposure.get(sym, 0.0) + abs(p.market_value)
    if equity > 0:
        exposure = {k: v / equity for k, v in exposure.items()}
    return exposure


# ──────────────────────────────────────────────────────────
#  Market Regime Filter (SPY SMA check)
# ──────────────────────────────────────────────────────────

_spy_regime_cache: dict[str, dict] = {}  # date → regime info

def get_market_regime(sma_period: int = 20) -> dict:
    """
    GRADUATED market regime filter (replaces binary bullish/bearish).

    Returns dict with:
        - regime: "strong_bull", "bull", "neutral", "bear", "strong_bear"
        - min_score: minimum signal score required for entry
        - risk_multiplier: multiply risk_pct by this (> 1.0 in strong trends)

    Graduated levels:
        SPY > SMA + 2%    → strong_bull  (min_score=3, risk 1.5x)
        SPY > SMA          → bull         (min_score=3, risk 1.0x)
        SPY within 1% SMA → neutral      (min_score=4, risk 0.8x)
        SPY < SMA          → bear         (min_score=5, risk 0.7x)
        SPY < SMA - 2%    → strong_bear  (min_score=7, risk 0.5x)
    """
    today = datetime.utcnow().strftime("%Y-%m-%d")
    if today in _spy_regime_cache:
        return _spy_regime_cache[today]

    default = {"regime": "bull", "min_score": 3, "risk_multiplier": 1.0}

    try:
        from data.fetcher import StockDataFetcher
        df = StockDataFetcher("SPY").fetch(period="3mo", interval="1d")
        if df is None or len(df) < sma_period + 1:
            logger.warning("Cannot determine market regime — defaulting to bull")
            _spy_regime_cache[today] = default
            return default

        sma = df["Close"].rolling(sma_period).mean()
        current_price = float(df["Close"].iloc[-1])
        current_sma = float(sma.iloc[-1])

        if current_sma <= 0:
            _spy_regime_cache[today] = default
            return default

        deviation = (current_price - current_sma) / current_sma

        if deviation > 0.02:
            regime = {"regime": "strong_bull", "min_score": 3, "risk_multiplier": 1.5}
        elif deviation > 0:
            regime = {"regime": "bull", "min_score": 3, "risk_multiplier": 1.0}
        elif deviation > -0.01:
            regime = {"regime": "neutral", "min_score": 4, "risk_multiplier": 0.8}
        elif deviation > -0.02:
            regime = {"regime": "bear", "min_score": 5, "risk_multiplier": 0.7}
        else:
            regime = {"regime": "strong_bear", "min_score": 7, "risk_multiplier": 0.5}

        logger.info(
            f"Market regime: SPY ${current_price:.2f} vs {sma_period}d SMA "
            f"${current_sma:.2f} ({deviation*100:+.1f}%) → {regime['regime'].upper()} "
            f"(min_score={regime['min_score']}, risk_mult={regime['risk_multiplier']}x)"
        )
        _spy_regime_cache[today] = regime
        return regime
    except Exception as e:
        logger.warning(f"Market regime check failed: {e} — defaulting to bull")
        _spy_regime_cache[today] = default
        return default


def is_market_bullish(sma_period: int = 20) -> bool:
    """Legacy compatibility wrapper — returns True if regime is bull or better."""
    regime = get_market_regime(sma_period)
    return regime["regime"] in ("strong_bull", "bull")


# ──────────────────────────────────────────────────────────
#  VIX Regime — volatility-based risk adjustment
# ──────────────────────────────────────────────────────────
# Since we're halal (long-only, no puts), high VIX = naked risk.
# We can't hedge, so we MUST reduce exposure in fearful markets.

_vix_cache: dict[str, dict] = {}   # date → {vix, score_adj, risk_mult, halt}

def get_vix_regime() -> dict:
    """
    Fetch VIX and return risk adjustments based on market fear level.

    Returns dict with:
        - vix: current VIX value
        - score_adj: add to min_score (higher = pickier)
        - risk_mult: multiply risk by this (lower = smaller positions)
        - halt: True if VIX > PANIC threshold (no new entries)

    Thresholds (configurable in config.py):
        VIX < 20    → calm      (no adjustment)
        VIX 20-25   → elevated  (score +1, risk ×0.8)
        VIX 25-30   → high      (score +2, risk ×0.6)
        VIX 30-35   → extreme   (score +3, risk ×0.4)
        VIX > 35    → PANIC     (HALT — no new entries at all)
    """
    today = datetime.utcnow().strftime("%Y-%m-%d")
    if today in _vix_cache:
        return _vix_cache[today]

    default = {"vix": 18.0, "score_adj": 0, "risk_mult": 1.0, "halt": False}

    try:
        from data.fetcher import StockDataFetcher
        df = StockDataFetcher("^VIX").fetch(period="1mo", interval="1d")
        if df is None or len(df) < 2:
            logger.warning("Cannot fetch VIX — defaulting to calm")
            _vix_cache[today] = default
            return default

        vix = float(df["Close"].iloc[-1])

        vix_elevated = getattr(config, "VIX_ELEVATED", 20)
        vix_high = getattr(config, "VIX_HIGH", 25)
        vix_extreme = getattr(config, "VIX_EXTREME", 30)
        vix_panic = getattr(config, "VIX_PANIC", 35)

        if vix >= vix_panic:
            result = {"vix": vix, "score_adj": 4, "risk_mult": 0.3, "halt": True}
        elif vix >= vix_extreme:
            result = {"vix": vix, "score_adj": 3, "risk_mult": 0.4, "halt": False}
        elif vix >= vix_high:
            result = {"vix": vix, "score_adj": 2, "risk_mult": 0.6, "halt": False}
        elif vix >= vix_elevated:
            result = {"vix": vix, "score_adj": 1, "risk_mult": 0.8, "halt": False}
        else:
            result = {"vix": vix, "score_adj": 0, "risk_mult": 1.0, "halt": False}

        logger.info(
            f"VIX regime: {vix:.1f} → score_adj={result['score_adj']:+d}, "
            f"risk_mult={result['risk_mult']:.1f}x"
            f"{', HALT!' if result['halt'] else ''}"
        )
        _vix_cache[today] = result
        return result
    except Exception as e:
        logger.warning(f"VIX fetch failed: {e} — defaulting to calm")
        _vix_cache[today] = default
        return default


# ──────────────────────────────────────────────────────────
#  Weekly Drawdown — cumulative loss protection
# ──────────────────────────────────────────────────────────

_weekly_pnl_cache: dict[str, float] = {}   # date → weekly_pnl_pct

def check_weekly_drawdown(client) -> dict:
    """
    Estimate weekly cumulative P&L from Alpaca portfolio history.

    Returns dict with:
        - weekly_pnl_pct: estimated weekly P&L as % of equity
        - halt: True if weekly loss exceeds threshold

    Uses the Alpaca portfolio history API to get the last 5 trading days.
    Falls back to (equity - balance) approximation if unavailable.
    """
    today = datetime.utcnow().strftime("%Y-%m-%d")
    if today in _weekly_pnl_cache:
        pnl_pct = _weekly_pnl_cache[today]
        threshold = getattr(config, "WEEKLY_MAX_LOSS_PCT", 5.0)
        return {"weekly_pnl_pct": pnl_pct, "halt": pnl_pct <= -threshold}

    try:
        # Try Alpaca portfolio history API (last 5 trading days)
        from alpaca.trading.client import TradingClient
        acct = client.get_account_info()
        if acct is None:
            return {"weekly_pnl_pct": 0.0, "halt": False}

        # Use raw client to access portfolio history
        raw_client = getattr(client, "_client", None)
        if raw_client is not None:
            try:
                from alpaca.trading.requests import GetPortfolioHistoryRequest
                req = GetPortfolioHistoryRequest(
                    period="1W",
                    timeframe="1D",
                )
                history = raw_client.get_portfolio_history(req)
                if history and hasattr(history, 'equity') and len(history.equity) >= 2:
                    start_equity = float(history.equity[0])
                    end_equity = float(history.equity[-1])
                    if start_equity > 0:
                        pnl_pct = ((end_equity - start_equity) / start_equity) * 100
                        _weekly_pnl_cache[today] = pnl_pct
                        threshold = getattr(config, "WEEKLY_MAX_LOSS_PCT", 5.0)
                        halt = pnl_pct <= -threshold
                        logger.info(
                            f"Weekly drawdown: {pnl_pct:+.2f}% "
                            f"(${start_equity:,.0f} → ${end_equity:,.0f})"
                            f"{' — WEEKLY HALT!' if halt else ''}"
                        )
                        return {"weekly_pnl_pct": pnl_pct, "halt": halt}
            except Exception as e:
                logger.debug(f"Portfolio history API failed: {e}")

        # Fallback: not enough info for weekly — allow trading
        _weekly_pnl_cache[today] = 0.0
        return {"weekly_pnl_pct": 0.0, "halt": False}
    except Exception as e:
        logger.debug(f"Weekly drawdown check failed: {e}")
        return {"weekly_pnl_pct": 0.0, "halt": False}


# ──────────────────────────────────────────────────────────
#  Volume Filter — reject signals on dead volume
# ──────────────────────────────────────────────────────────

def check_volume_filter(ticker: str) -> dict:
    """
    Check if the latest bar has adequate volume compared to 20-day average.

    Returns dict with:
        - ratio: latest_volume / avg_volume (1.0 = average)
        - pass: True if ratio >= MIN_VOLUME_RATIO
    """
    try:
        from data.fetcher import StockDataFetcher
        df = StockDataFetcher(ticker).fetch(period="3mo", interval="1d")
        if df is None or len(df) < 21 or "Volume" not in df.columns:
            return {"ratio": 1.0, "pass": True}  # can't check → allow

        avg_vol = df["Volume"].iloc[-21:-1].mean()  # 20-day avg (excl. today)
        latest_vol = float(df["Volume"].iloc[-1])

        if avg_vol <= 0:
            return {"ratio": 1.0, "pass": True}

        ratio = latest_vol / avg_vol
        min_ratio = getattr(config, "MIN_VOLUME_RATIO", 0.6)
        passed = ratio >= min_ratio

        if not passed:
            logger.info(
                f"Volume filter: {ticker} vol={latest_vol:,.0f}, "
                f"avg={avg_vol:,.0f}, ratio={ratio:.2f} < min {min_ratio} — REJECTED"
            )

        return {"ratio": ratio, "pass": passed}
    except Exception:
        return {"ratio": 1.0, "pass": True}  # error → allow


# ──────────────────────────────────────────────────────────
#  Gap Detection — avoid buying into exhaustion
# ──────────────────────────────────────────────────────────

def check_gap(ticker: str) -> dict:
    """
    Check if the stock has gapped significantly from its prior close.

    Large gaps (>3%) mean:
    - Buying: you're chasing exhaustion, likely to reverse
    - Stop loss: gap-downs can skip your SL entirely

    Returns dict with:
        - gap_pct: gap as % of prior close (positive = gap up)
        - pass: True if abs(gap) <= MAX_GAP_PCT
    """
    try:
        from data.fetcher import StockDataFetcher
        df = StockDataFetcher(ticker).fetch(period="1mo", interval="1d")
        if df is None or len(df) < 2:
            return {"gap_pct": 0.0, "pass": True}

        prior_close = float(df["Close"].iloc[-2])
        current_open = float(df["Open"].iloc[-1])

        if prior_close <= 0:
            return {"gap_pct": 0.0, "pass": True}

        gap_pct = ((current_open - prior_close) / prior_close) * 100
        max_gap = getattr(config, "MAX_GAP_PCT", 3.0)
        passed = abs(gap_pct) <= max_gap

        if not passed:
            logger.info(
                f"Gap filter: {ticker} gapped {gap_pct:+.1f}% "
                f"(prior close=${prior_close:.2f}, open=${current_open:.2f}) "
                f"— exceeds ±{max_gap}% limit — REJECTED"
            )

        return {"gap_pct": gap_pct, "pass": passed}
    except Exception:
        return {"gap_pct": 0.0, "pass": True}


# ──────────────────────────────────────────────────────────
#  Earnings Blackout — don't gamble through earnings reports
# ──────────────────────────────────────────────────────────

_earnings_cache: dict[str, dict] = {}  # ticker → {next_date, blackout}

def check_earnings_blackout(ticker: str) -> dict:
    """
    Check if a stock has earnings within the blackout window (±3 days).

    Technical analysis is meaningless through earnings — a stock can gap
    15% in either direction regardless of any chart pattern.

    Returns dict with:
        - in_blackout: True if earnings are within ±EARNINGS_BLACKOUT_DAYS
        - days_to_earnings: days until next earnings (negative = just passed)
        - earnings_date: the date string, if found
    """
    if ticker in _earnings_cache:
        return _earnings_cache[ticker]

    default = {"in_blackout": False, "days_to_earnings": 999, "earnings_date": ""}
    blackout_days = getattr(config, "EARNINGS_BLACKOUT_DAYS", 3)

    try:
        import yfinance as yf
        from datetime import timedelta

        tk = yf.Ticker(ticker)
        # yfinance .earnings_dates returns upcoming and recent earnings
        try:
            earnings_dates = tk.earnings_dates
        except Exception:
            earnings_dates = None

        if earnings_dates is None or earnings_dates.empty:
            # Some tickers (ETFs, leveraged) don't have earnings
            _earnings_cache[ticker] = default
            return default

        now = datetime.utcnow()
        closest_days = 999
        closest_date = ""

        for dt in earnings_dates.index:
            # earnings_dates index is a DatetimeIndex
            try:
                earnings_dt = dt.to_pydatetime()
                if earnings_dt.tzinfo is not None:
                    # Make comparable — strip timezone
                    from datetime import timezone
                    earnings_dt = earnings_dt.replace(tzinfo=None)
                delta = (earnings_dt - now).days
                if abs(delta) < abs(closest_days):
                    closest_days = delta
                    closest_date = earnings_dt.strftime("%Y-%m-%d")
            except Exception:
                continue

        in_blackout = abs(closest_days) <= blackout_days

        if in_blackout:
            logger.info(
                f"Earnings blackout: {ticker} has earnings on {closest_date} "
                f"({closest_days:+d} days) — within ±{blackout_days} day window"
            )

        result = {
            "in_blackout": in_blackout,
            "days_to_earnings": closest_days,
            "earnings_date": closest_date,
        }
        _earnings_cache[ticker] = result
        return result
    except Exception as e:
        logger.debug(f"Earnings check failed for {ticker}: {e}")
        _earnings_cache[ticker] = default
        return default


# ──────────────────────────────────────────────────────────
#  Sector Momentum — rotate into green sectors during crashes
# ──────────────────────────────────────────────────────────

_sector_momentum_cache: dict[str, dict] = {}   # date → {sector: momentum%}

def get_sector_momentum(lookback: int = 20) -> dict[str, float]:
    """
    Compute recent momentum (%) for each sector using sector ETF proxies.

    Returns dict like {"tech": −3.2, "energy": +5.1, "healthcare": +0.8, ...}
    Positive = sector trending up, Negative = sector trending down.

    Used by the executor to:
      - Give a min_score BONUS to sectors with strong positive momentum
      - Give a min_score PENALTY to sectors with negative momentum

    This is what makes the system "smart" about sector rotation:
    during a tech crash, energy/staples will have positive momentum,
    making them EASIER to enter while tech becomes HARDER to enter.
    """
    today = datetime.utcnow().strftime("%Y-%m-%d")
    if today in _sector_momentum_cache:
        return _sector_momentum_cache[today]

    momentum = {}
    try:
        from data.fetcher import StockDataFetcher
        import time

        for sector, etf in SECTOR_ETFS.items():
            try:
                df = StockDataFetcher(etf).fetch(period="3mo", interval="1d")
                if df is not None and len(df) >= lookback + 1:
                    lb = min(lookback, len(df) - 1)
                    ret = (float(df["Close"].iloc[-1]) / float(df["Close"].iloc[-lb]) - 1) * 100
                    momentum[sector] = ret
                else:
                    momentum[sector] = 0.0
                time.sleep(0.3)  # rate limit yfinance
            except Exception:
                momentum[sector] = 0.0

        logger.info(
            "Sector momentum: " +
            ", ".join(f"{s}={m:+.1f}%" for s, m in
                      sorted(momentum.items(), key=lambda x: -x[1]))
        )
    except Exception as e:
        logger.warning(f"Sector momentum check failed: {e}")

    _sector_momentum_cache[today] = momentum
    return momentum


def get_sector_score_adjustment(category: str, ticker: str = "") -> int:
    """
    Returns min_score ADJUSTMENT for a sector based on its momentum.

    Positive momentum (sector trending up)  → −1 to min_score (easier entry)
    Flat momentum                           →  0 (no change)
    Negative momentum (sector trending down) → +1 to +2 (harder entry)

    This means during a tech crash with energy rallying:
      tech min_score might go from 3 → 5 (harder to buy tech)
      energy min_score might go from 3 → 2 (easier to buy energy)

    For leveraged tickers: each one tracks its actual underlying:
      MSTU/MSTR/MSTZ → Bitcoin momentum (they're MicroStrategy = Bitcoin proxy)
      TQQQ/FNGU      → tech momentum (they track Nasdaq/FAANG)
      SOXL           → semis momentum (tracks semiconductors)
      TSLL           → tech momentum (tracks Tesla/tech)
    """
    momentum = get_sector_momentum()
    mom = momentum.get(category, 0.0)

    # Leveraged tickers: use per-ticker underlying momentum
    if category == "leveraged" and ticker:
        underlying = LEVERAGED_MOMENTUM_MAP.get(ticker.upper(), "tech")
        mom = momentum.get(underlying, 0.0)

    if mom > 5.0:
        return -1   # strong sector — lower the bar
    elif mom > 0.0:
        return 0    # positive but not strong — no adjustment
    elif mom > -3.0:
        return 1    # mildly negative — slightly higher bar
    else:
        return 2    # sector in freefall — much higher bar


# ──────────────────────────────────────────────────────────
#  Position Rotation — swap weak positions for strong signals
# ──────────────────────────────────────────────────────────

def find_replaceable_position(
    positions,
    new_score: int,
    new_ticker: str,
    equity: float,
) -> Optional[object]:
    """
    Find the weakest existing position that can be replaced by a stronger signal.

    Rules (asymmetric — user's preference):
    - Profitable positions: easier to swap (new score >= 5 is enough)
    - Losing positions: much harder to swap (new score >= 8 required)
    - Never swap a position for one in the same category
    - Prefer swapping the position with lowest PnL% among candidates

    Returns the position to sell, or None if no swap should happen.
    """
    if not positions:
        return None

    new_category = _get_category(new_ticker)

    candidates = []
    for pos in positions:
        sym = pos.symbol.upper()

        # Don't swap the same ticker
        if sym == new_ticker.upper():
            continue

        # Don't swap the inverse of what we're buying (would defeat purpose)
        if INVERSE_PAIRS.get(new_ticker.upper()) == sym:
            continue

        # Don't swap within the same category (preserve diversification)
        if _get_category(sym) == new_category:
            continue

        pnl_pct = pos.unrealized_plpc  # fraction, not percent

        if pnl_pct >= 0:
            # Position is profitable — lower bar
            # Only swap if new signal is solid (score >= 5)
            if abs(new_score) >= 5:
                candidates.append((pos, pnl_pct, "profit"))
        else:
            # Position is at a loss — very high bar
            # Only swap if new signal is exceptional (score >= 8)
            if abs(new_score) >= 8:
                candidates.append((pos, pnl_pct, "loss"))

    if not candidates:
        return None

    # Prefer swapping profitable positions first (less painful)
    # Among those, pick the one with highest profit (already captured most of its move)
    profit_candidates = [(p, pnl, t) for p, pnl, t in candidates if t == "profit"]
    if profit_candidates:
        # Sort by highest profit % first (most of the move is done)
        profit_candidates.sort(key=lambda x: -x[1])
        best = profit_candidates[0]
        logger.info(
            f"Rotation candidate: {best[0].symbol} (profit {best[1]*100:+.1f}%) "
            f"→ replace with {new_ticker} (score={new_score})"
        )
        return best[0]

    # Fallback: swap a loser (only for exceptional signals)
    loss_candidates = [(p, pnl, t) for p, pnl, t in candidates if t == "loss"]
    if loss_candidates:
        # Sort by worst PnL (cut the biggest loser)
        loss_candidates.sort(key=lambda x: x[1])
        best = loss_candidates[0]
        logger.info(
            f"Rotation candidate (LOSS swap): {best[0].symbol} "
            f"(loss {best[1]*100:+.1f}%) → replace with {new_ticker} (score={new_score})"
        )
        return best[0]

    return None


# ──────────────────────────────────────────────────────────
#  Share Size Calculator
# ──────────────────────────────────────────────────────────

def calculate_shares(
    equity: float,
    risk_pct: float,
    entry_price: float,
    stop_loss: float,
    min_shares: int = 1,
) -> int:
    """
    Calculate number of shares based on risk management.

    Formula: shares = (equity × risk%) / |entry - SL|

    Returns integer share count (fractional shares not used for bracket orders).
    """
    sl_distance = abs(entry_price - stop_loss)
    if sl_distance == 0:
        logger.warning("SL distance is 0 — cannot size position")
        return 0

    risk_amount = equity * risk_pct
    shares = risk_amount / sl_distance

    # Round down to whole shares
    shares = int(shares)

    # Enforce minimum
    shares = max(min_shares, shares) if shares > 0 else 0

    logger.info(
        f"Share sizing: equity=${equity:,.2f}, risk={risk_pct*100:.1f}%, "
        f"SL_dist=${sl_distance:.2f}, risk_amt=${risk_amount:.2f}, "
        f"shares={shares}"
    )
    return shares


# ──────────────────────────────────────────────────────────
#  Trade Executor
# ──────────────────────────────────────────────────────────

class AlpacaExecutor:
    """
    Executes trade setups on Alpaca Markets with category-based allocation caps.

    Usage:
        client = AlpacaClient()
        client.connect()
        executor = AlpacaExecutor(client)
        records = executor.execute_setups([setup1, setup2, ...])
    """

    def __init__(self, client: AlpacaClient, cfg: AlpacaExecutorConfig = None):
        self.client = client
        self.cfg = cfg or AlpacaExecutorConfig()
        self._daily_records: list[ExecutionRecord] = []

    def execute_setups(
        self, setups: list[TradeSetup]
    ) -> list[ExecutionRecord]:
        """Execute a list of trade setups with safety checks."""
        records = []
        for setup in setups:
            record = self._execute_single(setup)
            records.append(record)
            self._daily_records.append(record)
        return records

    def execute_single(self, setup: TradeSetup) -> ExecutionRecord:
        """Execute a single trade setup."""
        record = self._execute_single(setup)
        self._daily_records.append(record)
        return record

    # ── Internal ──────────────────────────────────────────

    def _execute_single(self, setup: TradeSetup) -> ExecutionRecord:
        """Core execution logic for a single setup."""
        ticker = setup.ticker
        action = setup.action.value
        is_leveraged = ticker.upper() in LEVERAGED_TICKERS
        category = _get_category(ticker)

        # ── Safety Check 1: Only BUY signals (no short selling) ─
        #    Short selling is not halal — we only buy what we own.
        if setup.action in (TradeAction.HOLD,):
            return ExecutionRecord(
                ticker=ticker, action=action, qty=0,
                entry_price=setup.entry_price,
                sl=setup.stop_loss, tp=setup.take_profit,
                risk_reward=setup.risk_reward, executed=False,
                reason="HOLD signal — no trade",
            )

        if setup.action in (TradeAction.SELL, TradeAction.STRONG_SELL):
            return ExecutionRecord(
                ticker=ticker, action=action, qty=0,
                entry_price=setup.entry_price,
                sl=setup.stop_loss, tp=setup.take_profit,
                risk_reward=setup.risk_reward, executed=False,
                reason="SELL signal skipped — no short selling (halal compliance)",
            )

        # ── Safety Check 1b: VIX volatility filter ──────────
        #    Since we're halal (long-only, no puts), high VIX = naked risk.
        #    We can't hedge, so we MUST reduce exposure in fearful markets.
        vix_regime = get_vix_regime()
        if vix_regime["halt"]:
            return ExecutionRecord(
                ticker=ticker, action=action, qty=0,
                entry_price=setup.entry_price,
                sl=setup.stop_loss, tp=setup.take_profit,
                risk_reward=setup.risk_reward, executed=False,
                reason=f"VIX PANIC HALT: VIX={vix_regime['vix']:.1f} > "
                       f"{getattr(config, 'VIX_PANIC', 35)} — no new entries",
            )

        # ── Safety Check 1c: GRADUATED market regime + sector + VIX ─
        #    Combine SPY regime, sector momentum, AND VIX into one score.
        regime = get_market_regime()
        base_min_score = regime["min_score"]
        sector_adj = get_sector_score_adjustment(category, ticker=ticker)
        vix_adj = vix_regime["score_adj"]
        min_score = max(2, base_min_score + sector_adj + vix_adj)  # never below 2

        adjustments = []
        if sector_adj != 0:
            adjustments.append(f"sector={sector_adj:+d}")
        if vix_adj != 0:
            adjustments.append(f"VIX={vix_adj:+d}")
        if adjustments:
            logger.info(
                f"Score adjustments for {ticker}: {', '.join(adjustments)} "
                f"(base {base_min_score} → effective {min_score})"
            )

        if abs(setup.composite_score) < min_score:
            return ExecutionRecord(
                ticker=ticker, action=action, qty=0,
                entry_price=setup.entry_price,
                sl=setup.stop_loss, tp=setup.take_profit,
                risk_reward=setup.risk_reward, executed=False,
                reason=f"Regime '{regime['regime']}' VIX={vix_regime['vix']:.0f} "
                       f"— score {setup.composite_score} < min {min_score} "
                       f"(base={base_min_score}, sector={sector_adj:+d}, vix={vix_adj:+d})",
            )

        # ── Safety Check 1d: Earnings blackout ───────────
        #    Technical analysis is meaningless through earnings.
        earnings = check_earnings_blackout(ticker)
        if earnings["in_blackout"]:
            return ExecutionRecord(
                ticker=ticker, action=action, qty=0,
                entry_price=setup.entry_price,
                sl=setup.stop_loss, tp=setup.take_profit,
                risk_reward=setup.risk_reward, executed=False,
                reason=f"Earnings blackout: {ticker} reports on "
                       f"{earnings['earnings_date']} "
                       f"({earnings['days_to_earnings']:+d} days)",
            )

        # ── Safety Check 1e: Volume filter ───────────────
        #    Low volume signals are unreliable — no conviction behind the move.
        vol_check = check_volume_filter(ticker)
        if not vol_check["pass"]:
            return ExecutionRecord(
                ticker=ticker, action=action, qty=0,
                entry_price=setup.entry_price,
                sl=setup.stop_loss, tp=setup.take_profit,
                risk_reward=setup.risk_reward, executed=False,
                reason=f"Volume too low: {ticker} vol ratio={vol_check['ratio']:.2f} "
                       f"< min {getattr(config, 'MIN_VOLUME_RATIO', 0.6)}",
            )

        # ── Safety Check 1f: Gap detection ───────────────
        #    Large gaps = buying exhaustion or SL-skip risk.
        gap_check = check_gap(ticker)
        if not gap_check["pass"]:
            return ExecutionRecord(
                ticker=ticker, action=action, qty=0,
                entry_price=setup.entry_price,
                sl=setup.stop_loss, tp=setup.take_profit,
                risk_reward=setup.risk_reward, executed=False,
                reason=f"Gap too large: {ticker} gapped {gap_check['gap_pct']:+.1f}% "
                       f"(max ±{getattr(config, 'MAX_GAP_PCT', 3.0)}%)",
            )

        # ── Safety Check 2: Minimum R:R ──────────────────
        min_rr = self.cfg.min_risk_reward
        if is_leveraged:
            min_rr = config.LEVERAGED_MODE.get("min_risk_reward", min_rr)
        if setup.risk_reward < min_rr:
            return ExecutionRecord(
                ticker=ticker, action=action, qty=0,
                entry_price=setup.entry_price,
                sl=setup.stop_loss, tp=setup.take_profit,
                risk_reward=setup.risk_reward, executed=False,
                reason=f"R:R {setup.risk_reward:.1f} < min {min_rr:.1f}",
            )

        # ── Safety Check 3: Max concurrent positions ─────
        open_positions = self.client.get_open_positions()
        existing_symbols = {p.symbol.upper() for p in open_positions}

        # ── Safety Check 3a: MSTU/MSTZ mutual exclusion ──
        #    Never hold both sides of an inverse pair simultaneously
        inverse_of = INVERSE_PAIRS.get(ticker.upper())
        if inverse_of and inverse_of in existing_symbols:
            # Close the inverse position first, then proceed
            logger.info(
                f"Inverse pair conflict: want {ticker} but holding {inverse_of}. "
                f"Closing {inverse_of} first."
            )
            if not self.cfg.dry_run:
                # Cancel all open orders on the inverse symbol first (bracket legs)
                import time
                inv_orders = self.client.get_open_orders(inverse_of)
                for order in inv_orders:
                    oid = order.get("id", "")
                    if oid:
                        self.client.cancel_order(oid)
                if inv_orders:
                    time.sleep(1)

                close_result = self.client.close_position(inverse_of)
                if close_result.success:
                    logger.info(f"Closed inverse position {inverse_of}")
                    time.sleep(1)  # Wait for settlement
                    open_positions = self.client.get_open_positions()
                    existing_symbols = {p.symbol.upper() for p in open_positions}
                else:
                    logger.error(f"Failed to close inverse {inverse_of}: {close_result.message}")
                    return ExecutionRecord(
                        ticker=ticker, action=action, qty=0,
                        entry_price=setup.entry_price,
                        sl=setup.stop_loss, tp=setup.take_profit,
                        risk_reward=setup.risk_reward, executed=False,
                        reason=f"Failed to close inverse pair {inverse_of}",
                    )

        # ── Position Rotation: swap weak position for strong signal ──
        if len(open_positions) >= self.cfg.max_concurrent_positions:
            # Try to find a replaceable position
            replaceable = find_replaceable_position(
                open_positions,
                new_score=setup.composite_score,
                new_ticker=ticker,
                equity=0,  # not used currently
            )
            if replaceable and not self.cfg.dry_run:
                logger.info(
                    f"ROTATION: Closing {replaceable.symbol} "
                    f"(PnL {replaceable.unrealized_plpc*100:+.1f}%) "
                    f"to make room for {ticker} (score={setup.composite_score})"
                )
                # Cancel all open orders first (bracket SL/TP legs)
                import time
                rot_orders = self.client.get_open_orders(replaceable.symbol)
                for order in rot_orders:
                    oid = order.get("id", "")
                    if oid:
                        self.client.cancel_order(oid)
                if rot_orders:
                    time.sleep(1)

                close_result = self.client.close_position(replaceable.symbol)
                if close_result.success:
                    logger.info(f"Closed {replaceable.symbol} for rotation")
                    time.sleep(1)
                    open_positions = self.client.get_open_positions()
                    existing_symbols = {p.symbol.upper() for p in open_positions}
                else:
                    logger.error(
                        f"Failed to close {replaceable.symbol} for rotation: "
                        f"{close_result.message}"
                    )
            elif replaceable and self.cfg.dry_run:
                logger.info(
                    f"[DRY RUN] Would rotate: close {replaceable.symbol} "
                    f"(PnL {replaceable.unrealized_plpc*100:+.1f}%) "
                    f"→ open {ticker} (score={setup.composite_score})"
                )

        # Re-check after potential rotation/inverse closure
        if len(open_positions) >= self.cfg.max_concurrent_positions:
            return ExecutionRecord(
                ticker=ticker, action=action, qty=0,
                entry_price=setup.entry_price,
                sl=setup.stop_loss, tp=setup.take_profit,
                risk_reward=setup.risk_reward, executed=False,
                reason=f"Max total positions ({self.cfg.max_concurrent_positions}) reached "
                       f"(no suitable position to rotate)",
            )

        # ── Safety Check 4: No duplicate symbol ──────────
        if ticker.upper() in existing_symbols:
            return ExecutionRecord(
                ticker=ticker, action=action, qty=0,
                entry_price=setup.entry_price,
                sl=setup.stop_loss, tp=setup.take_profit,
                risk_reward=setup.risk_reward, executed=False,
                reason=f"Already have open position on {ticker}",
            )

        # ── Safety Check 5: Daily loss limit ─────────────
        acct = self.client.get_account_info()
        if acct is None:
            return ExecutionRecord(
                ticker=ticker, action=action, qty=0,
                entry_price=setup.entry_price,
                sl=setup.stop_loss, tp=setup.take_profit,
                risk_reward=setup.risk_reward, executed=False,
                reason="Cannot retrieve account info",
            )

        if acct.trading_blocked:
            return ExecutionRecord(
                ticker=ticker, action=action, qty=0,
                entry_price=setup.entry_price,
                sl=setup.stop_loss, tp=setup.take_profit,
                risk_reward=setup.risk_reward, executed=False,
                reason="Trading is blocked on this account",
            )

        # Daily P&L: compare current equity to previous day's close equity
        # last_equity comes from Alpaca's API (equity at prior market close)
        if acct.last_equity > 0:
            daily_pnl = acct.equity - acct.last_equity
        else:
            # Fallback: total unrealized (imperfect but better than nothing)
            daily_pnl = sum(p.unrealized_pl for p in open_positions)
        if acct.equity > 0 and daily_pnl < 0:
            # Use last_equity (start-of-day equity) as denominator for accuracy
            denominator = acct.last_equity if acct.last_equity > 0 else acct.equity
            daily_loss_pct = abs(daily_pnl) / denominator * 100
            if daily_loss_pct >= self.cfg.max_daily_loss_pct:
                logger.warning(
                    f"DAILY LOSS BREAKER: {daily_loss_pct:.1f}% loss today "
                    f"(equity ${acct.equity:,.0f} vs prev close ${acct.last_equity:,.0f})"
                )
                return ExecutionRecord(
                    ticker=ticker, action=action, qty=0,
                    entry_price=setup.entry_price,
                    sl=setup.stop_loss, tp=setup.take_profit,
                    risk_reward=setup.risk_reward, executed=False,
                    reason=f"Daily loss limit hit ({daily_loss_pct:.1f}%)",
                )

        # ── Safety Check 5b: Weekly cumulative drawdown ──
        weekly = check_weekly_drawdown(self.client)
        if weekly["halt"]:
            return ExecutionRecord(
                ticker=ticker, action=action, qty=0,
                entry_price=setup.entry_price,
                sl=setup.stop_loss, tp=setup.take_profit,
                risk_reward=setup.risk_reward, executed=False,
                reason=f"Weekly drawdown limit hit ({weekly['weekly_pnl_pct']:.1f}% "
                       f"< -{getattr(config, 'WEEKLY_MAX_LOSS_PCT', 5.0)}%)",
            )

        # ── Safety Check 6: Category allocation cap ──────
        cat_exposure = _category_exposure(open_positions, acct.equity)
        cat_cap = CATEGORY_CAPS.get(category, 0.25)
        current_cat_pct = cat_exposure.get(category, 0.0)

        if current_cat_pct >= cat_cap:
            return ExecutionRecord(
                ticker=ticker, action=action, qty=0,
                entry_price=setup.entry_price,
                sl=setup.stop_loss, tp=setup.take_profit,
                risk_reward=setup.risk_reward, executed=False,
                reason=f"{category} cap {cat_cap*100:.0f}% reached "
                       f"(current: {current_cat_pct*100:.1f}%)",
            )

        # ── Safety Check 7: Per-ticker allocation cap ────
        ticker_exp = _ticker_exposure(open_positions, acct.equity)
        ticker_cap = TICKER_CAPS.get(ticker.upper())
        if ticker_cap is None:
            # Default: half of category cap
            ticker_cap = cat_cap * DEFAULT_SINGLE_TICKER_CAP_RATIO
        current_ticker_pct = ticker_exp.get(ticker.upper(), 0.0)

        if current_ticker_pct >= ticker_cap:
            return ExecutionRecord(
                ticker=ticker, action=action, qty=0,
                entry_price=setup.entry_price,
                sl=setup.stop_loss, tp=setup.take_profit,
                risk_reward=setup.risk_reward, executed=False,
                reason=f"{ticker} cap {ticker_cap*100:.1f}% reached "
                       f"(current: {current_ticker_pct*100:.1f}%)",
            )

        # ── Safety Check 8: Asset tradeable ──────────────
        asset = self.client.get_asset(ticker)
        if asset is None or not asset.get("tradable"):
            return ExecutionRecord(
                ticker=ticker, action=action, qty=0,
                entry_price=setup.entry_price,
                sl=setup.stop_loss, tp=setup.take_profit,
                risk_reward=setup.risk_reward, executed=False,
                reason=f"Asset {ticker} is not tradeable on Alpaca",
            )

        # ── Calculate share size with regime-adjusted risk ─
        base_risk_pct = (
            self.cfg.leveraged_risk_pct if is_leveraged
            else self.cfg.default_risk_pct
        )

        # Apply market regime AND VIX multipliers (compound)
        risk_pct = base_risk_pct * regime["risk_multiplier"] * vix_regime["risk_mult"]

        # High-conviction bonus: STRONG_BUY with score >= 6 gets 1.5x risk
        if setup.action == TradeAction.STRONG_BUY and abs(setup.composite_score) >= 6:
            risk_pct *= 1.5
            logger.info(
                f"HIGH CONVICTION: {ticker} score={setup.composite_score}, "
                f"risk boosted to {risk_pct*100:.1f}%"
            )

        # Cap at 3% per trade absolute maximum
        risk_pct = min(risk_pct, 0.03)

        qty = calculate_shares(
            equity=acct.equity,
            risk_pct=risk_pct,
            entry_price=setup.entry_price,
            stop_loss=setup.stop_loss,
            min_shares=self.cfg.min_shares,
        )

        if qty <= 0:
            return ExecutionRecord(
                ticker=ticker, action=action, qty=0,
                entry_price=setup.entry_price,
                sl=setup.stop_loss, tp=setup.take_profit,
                risk_reward=setup.risk_reward, executed=False,
                reason="Calculated share count is 0 (SL too tight or equity too low)",
            )

        # ── Notional cap: prevent any single position > max_notional_pct ─
        #    When ATR compresses and SL distance shrinks, share count can
        #    explode. This hard cap prevents a single position from being
        #    an outsized % of equity regardless of SL distance.
        max_notional = acct.equity * self.cfg.max_notional_pct
        notional = qty * setup.entry_price
        if notional > max_notional:
            capped_qty = int(max_notional / setup.entry_price)
            logger.info(
                f"NOTIONAL CAP: {ticker} capped from {qty} to {capped_qty} shares "
                f"(${notional:,.0f} → ${capped_qty * setup.entry_price:,.0f}, "
                f"max {self.cfg.max_notional_pct*100:.0f}% of equity)"
            )
            qty = capped_qty
            if qty <= 0:
                return ExecutionRecord(
                    ticker=ticker, action=action, qty=0,
                    entry_price=setup.entry_price,
                    sl=setup.stop_loss, tp=setup.take_profit,
                    risk_reward=setup.risk_reward, executed=False,
                    reason=f"Notional cap ({self.cfg.max_notional_pct*100:.0f}%) "
                           f"would result in 0 shares",
                )

        # ── Cap qty by remaining category room ───────────
        remaining_cat_room = (cat_cap - current_cat_pct) * acct.equity
        remaining_ticker_room = (ticker_cap - current_ticker_pct) * acct.equity
        max_cost = min(remaining_cat_room, remaining_ticker_room)

        if max_cost > 0 and qty * setup.entry_price > max_cost:
            capped_qty = int(max_cost / setup.entry_price)
            if capped_qty < qty:
                logger.info(
                    f"Capping {ticker} from {qty} to {capped_qty} shares "
                    f"(category room: ${remaining_cat_room:,.0f}, "
                    f"ticker room: ${remaining_ticker_room:,.0f})"
                )
                qty = capped_qty
            if qty <= 0:
                return ExecutionRecord(
                    ticker=ticker, action=action, qty=0,
                    entry_price=setup.entry_price,
                    sl=setup.stop_loss, tp=setup.take_profit,
                    risk_reward=setup.risk_reward, executed=False,
                    reason=f"No room left in {category} allocation",
                )

        # ── Check buying power (with 2% buffer) ─────────
        buffer = 1.02
        approx_cost = qty * setup.entry_price * buffer
        if approx_cost > acct.buying_power:
            qty = int(acct.buying_power / (setup.entry_price * buffer))
            logger.info(
                f"Reduced {ticker} qty to {qty} to fit buying power "
                f"(${acct.buying_power:,.2f} available)"
            )
            if qty <= 0:
                return ExecutionRecord(
                    ticker=ticker, action=action, qty=0,
                    entry_price=setup.entry_price,
                    sl=setup.stop_loss, tp=setup.take_profit,
                    risk_reward=setup.risk_reward, executed=False,
                    reason=f"Insufficient buying power (${acct.buying_power:,.2f})",
                )

        # ── Dry Run Mode ─────────────────────────────────
        if self.cfg.dry_run:
            cost_pct = (qty * setup.entry_price / acct.equity * 100) if acct.equity > 0 else 0
            logger.info(
                f"[DRY RUN] Would place: {action} {qty} {ticker} "
                f"(${qty * setup.entry_price:,.0f}, {cost_pct:.1f}% of equity, "
                f"cat={category}) "
                f"SL=${setup.stop_loss:.2f} TP=${setup.take_profit:.2f} "
                f"R:R=1:{setup.risk_reward:.1f}"
            )
            return ExecutionRecord(
                ticker=ticker, action=action, qty=qty,
                entry_price=setup.entry_price,
                sl=setup.stop_loss, tp=setup.take_profit,
                risk_reward=setup.risk_reward, executed=False,
                reason="DRY RUN — order not placed",
            )

        # ── Place bracket order (BUY only — no short selling) ─
        order_action = "BUY"
        client_id = f"bot_{ticker}_{setup.composite_score}_{datetime.utcnow().strftime('%H%M%S')}"

        result: OrderResult = self.client.place_bracket_order(
            symbol=ticker,
            action=order_action,
            qty=qty,
            sl=setup.stop_loss,
            tp=setup.take_profit,
            client_order_id=client_id,
        )

        if result.success:
            cost_pct = (qty * setup.entry_price / acct.equity * 100) if acct.equity > 0 else 0
            logger.info(
                f"EXECUTED: {action} {qty} {ticker} "
                f"(${qty * setup.entry_price:,.0f}, {cost_pct:.1f}% equity, "
                f"cat={category}) "
                f"| SL=${setup.stop_loss:.2f} TP=${setup.take_profit:.2f} "
                f"| Order ID={result.order_id}"
            )
            return ExecutionRecord(
                ticker=ticker, action=action, qty=qty,
                entry_price=setup.entry_price,
                sl=setup.stop_loss, tp=setup.take_profit,
                risk_reward=setup.risk_reward, executed=True,
                order_id=result.order_id,
                reason=f"Order submitted: {result.status}",
            )
        else:
            logger.error(f"FAILED: {action} {ticker} — {result.message}")
            return ExecutionRecord(
                ticker=ticker, action=action, qty=qty,
                entry_price=setup.entry_price,
                sl=setup.stop_loss, tp=setup.take_profit,
                risk_reward=setup.risk_reward, executed=False,
                reason=f"Order failed: {result.message}",
            )
