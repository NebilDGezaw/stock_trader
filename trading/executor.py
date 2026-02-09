"""
Trade Executor — bridges strategy analysis to live MT5 orders.
==============================================================
Takes a TradeSetup from any strategy, converts it to an MT5 order,
calculates proper lot sizing, and places it via the MT5 client.

Safety:
    - Never places an order without SL/TP
    - Enforces max concurrent positions
    - Enforces max daily loss limit
    - Dry-run mode for testing
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

import config
from models.signals import TradeSetup, TradeAction
from trading.mt5_client import MT5Client, OrderResult, MAGIC_NUMBER
from trading.symbols import to_mt5, get_asset_type

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────
#  Configuration
# ──────────────────────────────────────────────────────────

@dataclass
class ExecutorConfig:
    """Safety limits and risk parameters."""
    max_concurrent_positions: int = 3
    max_daily_loss_pct: float = 5.0       # stop trading if daily loss > 5%
    default_risk_pct: float = 0.02        # 2% of equity per trade
    min_risk_reward: float = 2.0          # minimum R:R to execute
    dry_run: bool = False                 # log only, don't place orders


@dataclass
class ExecutionRecord:
    """Record of an executed (or skipped) trade."""
    ticker: str
    mt5_symbol: str
    action: str
    volume: float
    entry_price: float
    sl: float
    tp: float
    risk_reward: float
    executed: bool
    ticket: int = 0
    reason: str = ""
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())


# ──────────────────────────────────────────────────────────
#  Lot Size Calculator
# ──────────────────────────────────────────────────────────

def calculate_lot_size(
    equity: float,
    risk_pct: float,
    entry_price: float,
    stop_loss: float,
    symbol_info: dict,
    asset_type: str,
) -> float:
    """
    Calculate the lot size based on risk management.

    Formula: lot_size = (equity × risk%) / (SL distance in price × contract size)

    Returns the lot size rounded to the broker's volume step.
    """
    sl_distance = abs(entry_price - stop_loss)
    if sl_distance == 0:
        logger.warning("SL distance is 0 — cannot size position")
        return 0.0

    risk_amount = equity * risk_pct
    contract_size = symbol_info.get("trade_contract_size", 100000)

    # For forex: contract_size is typically 100,000 (standard lot)
    # For metals: contract_size is typically 100 (oz of gold)
    # For crypto: contract_size is typically 1

    lot_size = risk_amount / (sl_distance * contract_size)

    # Clamp to broker limits
    vol_min = symbol_info.get("volume_min", 0.01)
    vol_max = symbol_info.get("volume_max", 100.0)
    vol_step = symbol_info.get("volume_step", 0.01)

    # Round to step
    lot_size = max(vol_min, min(vol_max, round(lot_size / vol_step) * vol_step))

    # Final precision fix
    lot_size = round(lot_size, 2)

    logger.info(
        f"Lot sizing: equity={equity:.2f}, risk={risk_pct*100:.1f}%, "
        f"SL_dist={sl_distance:.5f}, contract={contract_size}, "
        f"lot={lot_size}"
    )
    return lot_size


# ──────────────────────────────────────────────────────────
#  Trade Executor
# ──────────────────────────────────────────────────────────

class TradeExecutor:
    """
    Executes trade setups on the HFM MT5 account.

    Usage:
        client = MT5Client()
        client.connect(...)
        executor = TradeExecutor(client)
        records = executor.execute_setups([setup1, setup2, ...])
    """

    def __init__(self, client: MT5Client, cfg: ExecutorConfig = None):
        self.client = client
        self.cfg = cfg or ExecutorConfig()
        self._daily_records: list[ExecutionRecord] = []

    def execute_setups(
        self, setups: list[TradeSetup]
    ) -> list[ExecutionRecord]:
        """
        Execute a list of trade setups with safety checks.
        Returns a list of execution records.
        """
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
        mt5_sym = to_mt5(ticker)
        action = setup.action.value

        # ── Safety Check 1: Only actionable signals ───────
        if setup.action in (TradeAction.HOLD,):
            return ExecutionRecord(
                ticker=ticker, mt5_symbol=mt5_sym, action=action,
                volume=0, entry_price=setup.entry_price,
                sl=setup.stop_loss, tp=setup.take_profit,
                risk_reward=setup.risk_reward, executed=False,
                reason="HOLD signal — no trade",
            )

        # ── Safety Check 2: Minimum R:R ──────────────────
        if setup.risk_reward < self.cfg.min_risk_reward:
            return ExecutionRecord(
                ticker=ticker, mt5_symbol=mt5_sym, action=action,
                volume=0, entry_price=setup.entry_price,
                sl=setup.stop_loss, tp=setup.take_profit,
                risk_reward=setup.risk_reward, executed=False,
                reason=f"R:R {setup.risk_reward:.1f} < min {self.cfg.min_risk_reward:.1f}",
            )

        # ── Safety Check 3: Max concurrent positions ─────
        open_positions = self.client.get_open_positions()
        if len(open_positions) >= self.cfg.max_concurrent_positions:
            return ExecutionRecord(
                ticker=ticker, mt5_symbol=mt5_sym, action=action,
                volume=0, entry_price=setup.entry_price,
                sl=setup.stop_loss, tp=setup.take_profit,
                risk_reward=setup.risk_reward, executed=False,
                reason=f"Max positions ({self.cfg.max_concurrent_positions}) reached",
            )

        # ── Safety Check 4: No duplicate symbol ──────────
        existing_symbols = {p.symbol for p in open_positions}
        if mt5_sym in existing_symbols:
            return ExecutionRecord(
                ticker=ticker, mt5_symbol=mt5_sym, action=action,
                volume=0, entry_price=setup.entry_price,
                sl=setup.stop_loss, tp=setup.take_profit,
                risk_reward=setup.risk_reward, executed=False,
                reason=f"Already have open position on {mt5_sym}",
            )

        # ── Safety Check 5: Daily loss limit ─────────────
        acct = self.client.get_account_info()
        if acct is None:
            return ExecutionRecord(
                ticker=ticker, mt5_symbol=mt5_sym, action=action,
                volume=0, entry_price=setup.entry_price,
                sl=setup.stop_loss, tp=setup.take_profit,
                risk_reward=setup.risk_reward, executed=False,
                reason="Cannot retrieve account info",
            )

        daily_pnl = sum(p.profit for p in open_positions)
        if acct.balance > 0:
            daily_loss_pct = abs(min(0, daily_pnl)) / acct.balance * 100
            if daily_loss_pct >= self.cfg.max_daily_loss_pct:
                return ExecutionRecord(
                    ticker=ticker, mt5_symbol=mt5_sym, action=action,
                    volume=0, entry_price=setup.entry_price,
                    sl=setup.stop_loss, tp=setup.take_profit,
                    risk_reward=setup.risk_reward, executed=False,
                    reason=f"Daily loss limit hit ({daily_loss_pct:.1f}%)",
                )

        # ── Get symbol info and calculate lot size ───────
        sym_info = self.client.get_symbol_info(mt5_sym)
        if sym_info is None:
            return ExecutionRecord(
                ticker=ticker, mt5_symbol=mt5_sym, action=action,
                volume=0, entry_price=setup.entry_price,
                sl=setup.stop_loss, tp=setup.take_profit,
                risk_reward=setup.risk_reward, executed=False,
                reason=f"Symbol {mt5_sym} not available on broker",
            )

        # Determine risk per trade
        asset_type = get_asset_type(ticker)
        if asset_type == "crypto":
            risk_pct = config.CRYPTO_MODE.get("risk_per_trade", self.cfg.default_risk_pct)
        elif asset_type == "forex":
            risk_pct = config.FOREX_MODE.get("risk_per_trade", self.cfg.default_risk_pct)
        else:
            risk_pct = self.cfg.default_risk_pct

        volume = calculate_lot_size(
            equity=acct.equity,
            risk_pct=risk_pct,
            entry_price=setup.entry_price,
            stop_loss=setup.stop_loss,
            symbol_info=sym_info,
            asset_type=asset_type,
        )

        if volume <= 0:
            return ExecutionRecord(
                ticker=ticker, mt5_symbol=mt5_sym, action=action,
                volume=0, entry_price=setup.entry_price,
                sl=setup.stop_loss, tp=setup.take_profit,
                risk_reward=setup.risk_reward, executed=False,
                reason="Calculated lot size is 0 (SL too tight or equity too low)",
            )

        # ── Dry Run Mode ─────────────────────────────────
        if self.cfg.dry_run:
            logger.info(
                f"[DRY RUN] Would place: {action} {volume} {mt5_sym} "
                f"SL={setup.stop_loss} TP={setup.take_profit} R:R=1:{setup.risk_reward:.1f}"
            )
            return ExecutionRecord(
                ticker=ticker, mt5_symbol=mt5_sym, action=action,
                volume=volume, entry_price=setup.entry_price,
                sl=setup.stop_loss, tp=setup.take_profit,
                risk_reward=setup.risk_reward, executed=False,
                reason="DRY RUN — order not placed",
            )

        # ── Place the order ──────────────────────────────
        is_buy = setup.action in (TradeAction.BUY, TradeAction.STRONG_BUY)
        order_action = "BUY" if is_buy else "SELL"
        comment = f"bot_{ticker}_{setup.composite_score}"

        result: OrderResult = self.client.place_order(
            symbol=mt5_sym,
            action=order_action,
            volume=volume,
            sl=setup.stop_loss,
            tp=setup.take_profit,
            comment=comment,
        )

        if result.success:
            logger.info(
                f"EXECUTED: {action} {volume} {mt5_sym} @ {result.price} "
                f"| SL={setup.stop_loss} TP={setup.take_profit} "
                f"| Ticket={result.ticket}"
            )
            return ExecutionRecord(
                ticker=ticker, mt5_symbol=mt5_sym, action=action,
                volume=volume, entry_price=result.price,
                sl=setup.stop_loss, tp=setup.take_profit,
                risk_reward=setup.risk_reward, executed=True,
                ticket=result.ticket,
                reason=f"Order filled @ {result.price}",
            )
        else:
            logger.error(f"FAILED: {action} {mt5_sym} — {result.message}")
            return ExecutionRecord(
                ticker=ticker, mt5_symbol=mt5_sym, action=action,
                volume=volume, entry_price=setup.entry_price,
                sl=setup.stop_loss, tp=setup.take_profit,
                risk_reward=setup.risk_reward, executed=False,
                reason=f"Order failed: {result.message}",
            )
