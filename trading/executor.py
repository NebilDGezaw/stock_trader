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
from trading.symbols import to_mt5, get_asset_type, get_asset_type_from_mt5

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────
#  Configuration
# ──────────────────────────────────────────────────────────

@dataclass
class ExecutorConfig:
    """Safety limits and risk parameters."""
    max_concurrent_positions: int = 9     # absolute cap across all classes
    max_positions_per_class: dict = None  # per-class limits
    max_daily_loss_pct: float = 4.0       # stop trading if daily loss > 4%
    default_risk_pct: float = 0.015       # 1.5% of equity per trade
    min_risk_reward: float = 1.5          # minimum R:R to execute
    max_risk_per_trade: float = 0.0       # 0 = no hard cap; lot caps handle this
    min_lot_size: float = 0.01            # minimum lot size (broker floor)
    dry_run: bool = False                 # log only, don't place orders

    def __post_init__(self):
        if self.max_positions_per_class is None:
            self.max_positions_per_class = {
                "forex": 2,
                "crypto": 2,
                "commodity": 2,
                "stock": 3,
            }


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

# ── Per-asset lot size caps ───────────────────────────────
# These prevent catastrophic exposure even when SL is tight.
# Values are conservative for a $100K account.
MAX_LOT_BY_ASSET = {
    "forex": 2.0,       # max 2 standard lots ($200K notional) per trade
    "metal": 0.20,      # max 0.20 lots gold (~20 oz ≈ $100K notional)
    "commodity": 0.50,  # max 0.50 lots energy/commodity
    "crypto": 0.50,     # max 0.50 lots crypto
    "stock": 1.0,       # fallback
}

# ── Minimum SL distances (reject trades with impossibly tight stops) ──
# Prevents lot size explosion from sub-pip SL distances.
MIN_SL_DISTANCE = {
    "forex": 0.0010,    # at least 10 pips for forex
    "metal": 5.0,       # at least $5 for gold
    "commodity": 0.50,  # at least $0.50 for silver/energy
    "crypto": 50.0,     # at least $50 for BTC, $0.50 for alts (checked dynamically)
    "stock": 0.10,      # fallback
}


def calculate_lot_size(
    equity: float,
    risk_pct: float,
    entry_price: float,
    stop_loss: float,
    symbol_info: dict,
    asset_type: str,
    max_risk_amount: float = 1.0,
) -> float:
    """
    Calculate the lot size based on risk management.

    Formula: lot_size = (equity × risk%) / (SL distance in price × contract size)

    Safety layers:
        1. risk_amount capped by max_risk_amount (if > 0)
        2. Minimum SL distance enforced per asset type
        3. Hard lot size cap per asset type (prevents 10x+ leverage blowups)
        4. Broker volume min/max/step constraints

    Returns the lot size rounded to the broker's volume step.
    """
    sl_distance = abs(entry_price - stop_loss)
    if sl_distance == 0:
        logger.warning("SL distance is 0 — cannot size position")
        return 0.0

    # ── Safety: minimum SL distance ──────────────────────
    min_sl = MIN_SL_DISTANCE.get(asset_type, 0.0)
    # For crypto alts (price < $100), relax the minimum
    if asset_type == "crypto" and entry_price < 100:
        min_sl = entry_price * 0.01  # at least 1% of price
    if sl_distance < min_sl:
        logger.warning(
            f"SL distance {sl_distance:.5f} is below minimum {min_sl:.5f} "
            f"for {asset_type} — rejecting trade (too tight)"
        )
        return 0.0

    risk_amount = equity * risk_pct
    # Hard cap: if max_risk_amount > 0, enforce it
    if max_risk_amount > 0:
        risk_amount = min(risk_amount, max_risk_amount)
    contract_size = symbol_info.get("trade_contract_size", 100000)

    # For forex: contract_size is typically 100,000 (standard lot)
    # For metals: contract_size is typically 100 (oz of gold)
    # For crypto: contract_size is typically 1

    lot_size = risk_amount / (sl_distance * contract_size)

    # ── Safety: hard lot size cap per asset type ─────────
    max_lot = MAX_LOT_BY_ASSET.get(asset_type, 2.0)
    if lot_size > max_lot:
        logger.warning(
            f"Lot size {lot_size:.2f} exceeds max {max_lot:.2f} for {asset_type} "
            f"— capping to {max_lot:.2f}"
        )
        lot_size = max_lot

    # Clamp to broker limits
    vol_min = symbol_info.get("volume_min", 0.01)
    vol_max = symbol_info.get("volume_max", 100.0)
    vol_step = symbol_info.get("volume_step", 0.01)

    # Round to step
    lot_size = max(vol_min, min(vol_max, round(lot_size / vol_step) * vol_step))

    # Final precision fix
    lot_size = round(lot_size, 2)

    # ── Log with notional exposure for visibility ────────
    notional = lot_size * contract_size * (entry_price if asset_type != "forex" else 1.0)
    logger.info(
        f"Lot sizing: equity={equity:.2f}, risk={risk_pct*100:.1f}%, "
        f"SL_dist={sl_distance:.5f}, contract={contract_size}, "
        f"lot={lot_size}, notional≈${notional:,.0f}"
    )
    return lot_size


# ──────────────────────────────────────────────────────────
#  Trade Executor
# ──────────────────────────────────────────────────────────

def _classify_position(mt5_symbol: str) -> str:
    """Classify an open position's MT5 symbol into an asset class.
    Metals are grouped under 'commodity' for position counting."""
    asset_type = get_asset_type_from_mt5(mt5_symbol)
    if asset_type == "metal":
        return "commodity"
    return asset_type


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

        # ── Safety Check 3: Max concurrent positions (global + per-class) ─
        open_positions = self.client.get_open_positions()

        # 3a: Global hard cap
        if len(open_positions) >= self.cfg.max_concurrent_positions:
            return ExecutionRecord(
                ticker=ticker, mt5_symbol=mt5_sym, action=action,
                volume=0, entry_price=setup.entry_price,
                sl=setup.stop_loss, tp=setup.take_profit,
                risk_reward=setup.risk_reward, executed=False,
                reason=f"Max total positions ({self.cfg.max_concurrent_positions}) reached",
            )

        # 3b: Per-asset-class limit (diversification)
        this_asset_class = get_asset_type(ticker)
        # metals count as commodity for position limits
        if this_asset_class == "metal":
            this_asset_class = "commodity"

        class_limit = self.cfg.max_positions_per_class.get(this_asset_class, 3)
        class_count = sum(
            1 for p in open_positions
            if _classify_position(p.symbol) == this_asset_class
        )
        if class_count >= class_limit:
            return ExecutionRecord(
                ticker=ticker, mt5_symbol=mt5_sym, action=action,
                volume=0, entry_price=setup.entry_price,
                sl=setup.stop_loss, tp=setup.take_profit,
                risk_reward=setup.risk_reward, executed=False,
                reason=f"Max {this_asset_class} positions ({class_limit}) reached "
                       f"({class_count} open)",
            )

        # ── Safety Check 4: No duplicate symbol ──────────
        # Prevent stacking (e.g., BUY EURUSD in London + BUY EURUSD in Overlap)
        is_buy = setup.action in (TradeAction.BUY, TradeAction.STRONG_BUY)
        for p in open_positions:
            if p.symbol == mt5_sym:
                pos_is_buy = (p.type == "BUY")
                if pos_is_buy == is_buy:
                    return ExecutionRecord(
                        ticker=ticker, mt5_symbol=mt5_sym, action=action,
                        volume=0, entry_price=setup.entry_price,
                        sl=setup.stop_loss, tp=setup.take_profit,
                        risk_reward=setup.risk_reward, executed=False,
                        reason=f"Already have same-direction position on {mt5_sym}",
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
            max_risk_amount=self.cfg.max_risk_per_trade,
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
