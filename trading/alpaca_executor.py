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

        if len(open_positions) >= self.cfg.max_concurrent_positions:
            return ExecutionRecord(
                ticker=ticker, action=action, qty=0,
                entry_price=setup.entry_price,
                sl=setup.stop_loss, tp=setup.take_profit,
                risk_reward=setup.risk_reward, executed=False,
                reason=f"Max total positions ({self.cfg.max_concurrent_positions}) reached",
            )

        # ── Safety Check 4: No duplicate symbol ──────────
        existing_symbols = {p.symbol.upper() for p in open_positions}
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

        daily_pnl = sum(p.unrealized_pl for p in open_positions)
        if acct.equity > 0 and daily_pnl < 0:
            daily_loss_pct = abs(daily_pnl) / acct.equity * 100
            if daily_loss_pct >= self.cfg.max_daily_loss_pct:
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

        # ── Calculate share size ─────────────────────────
        risk_pct = (
            self.cfg.leveraged_risk_pct if is_leveraged
            else self.cfg.default_risk_pct
        )

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
