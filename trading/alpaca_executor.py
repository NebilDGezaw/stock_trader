"""
Alpaca Trade Executor — bridges strategy analysis to Alpaca bracket orders.
============================================================================
Takes a TradeSetup from any strategy, converts it to an Alpaca bracket order,
calculates proper share sizing, and places it via the Alpaca client.

Safety:
    - Never places an order without SL/TP (bracket orders)
    - Enforces max concurrent positions (global + per-type)
    - Enforces max daily loss limit
    - Minimum R:R validation
    - No duplicate symbol orders
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
#  Configuration
# ──────────────────────────────────────────────────────────

@dataclass
class AlpacaExecutorConfig:
    """Safety limits and risk parameters for Alpaca stock trading."""
    max_concurrent_positions: int = 6      # max total positions
    max_leveraged_positions: int = 3       # max leveraged ETF positions
    max_regular_positions: int = 3         # max regular stock positions
    max_daily_loss_pct: float = 5.0        # stop trading if daily loss > 5%
    default_risk_pct: float = 0.02         # 2% of equity per trade
    leveraged_risk_pct: float = 0.03       # 3% for leveraged ETFs (more aggressive)
    min_risk_reward: float = 2.0           # minimum R:R to execute
    min_shares: int = 1                    # minimum share count
    dry_run: bool = False                  # log only, don't place orders


LEVERAGED_TICKERS = set(getattr(config, "LEVERAGED_TICKERS", ["MSTU", "MSTR", "MSTZ", "TSLL"]))


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
    Executes trade setups on Alpaca Markets.

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

        # ── Safety Check 1: Only actionable signals ───────
        if setup.action in (TradeAction.HOLD,):
            return ExecutionRecord(
                ticker=ticker, action=action, qty=0,
                entry_price=setup.entry_price,
                sl=setup.stop_loss, tp=setup.take_profit,
                risk_reward=setup.risk_reward, executed=False,
                reason="HOLD signal — no trade",
            )

        # ── Safety Check 2: Minimum R:R ──────────────────
        min_rr = self.cfg.min_risk_reward
        if is_leveraged:
            # Leveraged ETFs can use slightly lower R:R from config
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

        # 3a: Global limit
        if len(open_positions) >= self.cfg.max_concurrent_positions:
            return ExecutionRecord(
                ticker=ticker, action=action, qty=0,
                entry_price=setup.entry_price,
                sl=setup.stop_loss, tp=setup.take_profit,
                risk_reward=setup.risk_reward, executed=False,
                reason=f"Max total positions ({self.cfg.max_concurrent_positions}) reached",
            )

        # 3b: Per-type limit (leveraged vs regular)
        leveraged_count = sum(
            1 for p in open_positions if p.symbol.upper() in LEVERAGED_TICKERS
        )
        regular_count = len(open_positions) - leveraged_count

        if is_leveraged and leveraged_count >= self.cfg.max_leveraged_positions:
            return ExecutionRecord(
                ticker=ticker, action=action, qty=0,
                entry_price=setup.entry_price,
                sl=setup.stop_loss, tp=setup.take_profit,
                risk_reward=setup.risk_reward, executed=False,
                reason=f"Max leveraged positions ({self.cfg.max_leveraged_positions}) reached",
            )
        if not is_leveraged and regular_count >= self.cfg.max_regular_positions:
            return ExecutionRecord(
                ticker=ticker, action=action, qty=0,
                entry_price=setup.entry_price,
                sl=setup.stop_loss, tp=setup.take_profit,
                risk_reward=setup.risk_reward, executed=False,
                reason=f"Max regular positions ({self.cfg.max_regular_positions}) reached",
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

        # ── Safety Check 6: Asset tradeable ──────────────
        asset = self.client.get_asset(ticker)
        if asset is None or not asset.get("tradable"):
            return ExecutionRecord(
                ticker=ticker, action=action, qty=0,
                entry_price=setup.entry_price,
                sl=setup.stop_loss, tp=setup.take_profit,
                risk_reward=setup.risk_reward, executed=False,
                reason=f"Asset {ticker} is not tradeable on Alpaca",
            )

        # ── Safety Check 7: Can't short non-shortable assets ─
        is_sell = setup.action in (TradeAction.SELL, TradeAction.STRONG_SELL)
        if is_sell and not asset.get("shortable", False):
            return ExecutionRecord(
                ticker=ticker, action=action, qty=0,
                entry_price=setup.entry_price,
                sl=setup.stop_loss, tp=setup.take_profit,
                risk_reward=setup.risk_reward, executed=False,
                reason=f"{ticker} cannot be sold short on Alpaca",
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

        # Check we have enough buying power (with 2% buffer for price movement)
        buffer = 1.02  # 2% safety margin for market price > strategy price
        approx_cost = qty * setup.entry_price * buffer
        if approx_cost > acct.buying_power:
            # Reduce qty to fit buying power (with buffer)
            qty = int(acct.buying_power / (setup.entry_price * buffer))
            logger.info(
                f"Reduced {ticker} qty to {qty} to fit buying power "
                f"(${acct.buying_power:,.2f} available, "
                f"~${qty * setup.entry_price:,.2f} needed)"
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
            logger.info(
                f"[DRY RUN] Would place: {action} {qty} {ticker} "
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

        # ── Place bracket order ──────────────────────────
        is_buy = setup.action in (TradeAction.BUY, TradeAction.STRONG_BUY)
        order_action = "BUY" if is_buy else "SELL"
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
            logger.info(
                f"EXECUTED: {action} {qty} {ticker} "
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
