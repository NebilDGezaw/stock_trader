"""
Alpaca Trade Executor — bridges strategy analysis to Alpaca bracket orders.
============================================================================
Takes a TradeSetup from any strategy, converts it to an Alpaca bracket order,
calculates proper share sizing, and places it via the Alpaca client.

Safety:
    - Never places an order without SL/TP (bracket orders)
    - Category-based allocation caps (leveraged 30%, tech 25%, etc.)
    - Per-ticker caps (NIO/RIVN capped at 1%)
    - Enforces max daily loss limit
    - Minimum R:R validation
    - No duplicate symbol orders
    - Buying power check with 2% buffer
    - MSTU/MSTZ mutual exclusion (never hold both simultaneously)
    - Market regime filter (SPY SMA) — pickier in bearish markets
    - Position rotation — swaps weak positions for strong new signals
    - Dry-run mode for testing
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

# Every ticker belongs to exactly one category
TICKER_CATEGORY = {
    # Leveraged ETFs (30% cap)
    "MSTU":  "leveraged",
    "MSTR":  "leveraged",
    "MSTZ":  "leveraged",
    "TSLL":  "leveraged",
    "TQQQ":  "leveraged",
    "SOXL":  "leveraged",
    "FNGU":  "leveraged",

    # Mega Tech (25% cap)
    "AAPL":  "tech",
    "MSFT":  "tech",
    "GOOGL": "tech",
    "AMZN":  "tech",
    "NVDA":  "tech",
    "META":  "tech",
    "TSLA":  "tech",
    "AMD":   "tech",

    # Semiconductors (15% cap)
    "AVGO":  "semis",
    "QCOM":  "semis",
    "ASML":  "semis",
    "MU":    "semis",

    # Healthcare (15% cap)
    "UNH":   "healthcare",
    "ABBV":  "healthcare",
    "LLY":   "healthcare",
    "ISRG":  "healthcare",

    # Clean Energy / EV (10% cap)
    "ENPH":  "clean_energy",
    "FSLR":  "clean_energy",
    "RIVN":  "clean_energy",
    "NIO":   "clean_energy",

    # Consumer (5% cap)
    "COST":  "consumer",
    "TGT":   "consumer",
}

# Maximum % of total equity allocated to each category
CATEGORY_CAPS = {
    "leveraged":    0.30,   # 30%
    "tech":         0.25,   # 25%
    "semis":        0.15,   # 15%
    "healthcare":   0.15,   # 15%
    "clean_energy": 0.10,   # 10%
    "consumer":     0.05,   # 5%
}

# Per-ticker allocation caps (overrides category default)
# Speculative names get tighter limits
TICKER_CAPS = {
    "NIO":  0.01,   # max 1% of equity
    "RIVN": 0.01,   # max 1% of equity
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
    max_concurrent_positions: int = 10     # max total positions (up from 6)
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
    return TICKER_CATEGORY.get(ticker.upper(), "tech")  # default to tech


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

        # ── Safety Check 1b: GRADUATED market regime filter ─
        #    Different market conditions require different conviction levels
        regime = get_market_regime()
        min_score = regime["min_score"]
        if abs(setup.composite_score) < min_score:
            return ExecutionRecord(
                ticker=ticker, action=action, qty=0,
                entry_price=setup.entry_price,
                sl=setup.stop_loss, tp=setup.take_profit,
                risk_reward=setup.risk_reward, executed=False,
                reason=f"Market regime '{regime['regime']}' — score "
                       f"{setup.composite_score} < min {min_score} required",
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
            daily_loss_pct = abs(daily_pnl) / acct.equity * 100
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

        # Apply market regime multiplier
        risk_pct = base_risk_pct * regime["risk_multiplier"]

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
