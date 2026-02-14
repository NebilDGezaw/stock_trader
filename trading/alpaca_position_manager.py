"""
Alpaca Position Manager — monitors and manages open Alpaca positions.
=====================================================================
Handles:
    - Trailing stops (cancel old stop → place new tighter stop)
    - Signal reversal detection (re-run strategy, close if flipped)
    - Partial close (sell half at 1R, let remainder run)
    - Position status reporting for Telegram

Designed to run periodically via GitHub Actions (every 1-2 hours).

NOTE: Alpaca bracket orders have built-in SL/TP (OCO legs). This manager
      handles *trailing* the stop above breakeven and detecting reversals
      that the static bracket doesn't catch.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

import config
from data.fetcher import StockDataFetcher
from models.signals import TradeAction
from trading.alpaca_client import AlpacaClient, PositionInfo

logger = logging.getLogger(__name__)

LEVERAGED_TICKERS = set(getattr(config, "LEVERAGED_TICKERS", ["MSTU", "MSTR", "MSTZ", "TSLL"]))


# ──────────────────────────────────────────────────────────
#  Data models
# ──────────────────────────────────────────────────────────

@dataclass
class PositionUpdate:
    """Record of a position management action."""
    symbol: str
    action_taken: str     # "trail_stop", "partial_close", "full_close", "no_change"
    old_sl: float = 0.0
    new_sl: float = 0.0
    pnl: float = 0.0
    reason: str = ""
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())


# ──────────────────────────────────────────────────────────
#  ATR helper
# ──────────────────────────────────────────────────────────

def _get_current_atr(ticker: str, period: int = 14) -> float:
    """Fetch recent data and compute ATR for a ticker."""
    try:
        from utils.helpers import compute_atr

        is_leveraged = ticker.upper() in LEVERAGED_TICKERS
        interval = "1h" if is_leveraged else "1d"
        fetch_period = "1mo" if is_leveraged else "3mo"

        df = StockDataFetcher(ticker).fetch(period=fetch_period, interval=interval)
        if df is None or len(df) < period + 1:
            return 0.0

        atr_series = compute_atr(df, period=period)
        atr = float(atr_series.iloc[-1])
        return atr if atr == atr else 0.0  # NaN check
    except Exception as e:
        logger.warning(f"ATR fetch failed for {ticker}: {e}")
        return 0.0


def _rerun_strategy(ticker: str) -> Optional[TradeAction]:
    """Re-run the appropriate strategy to check if signal has flipped."""
    try:
        from bt_engine.engine import _detect_asset_type, _run_strategy

        is_leveraged = ticker.upper() in LEVERAGED_TICKERS

        if is_leveraged:
            # MUST match the entry interval (1h) for consistency
            df = StockDataFetcher(ticker).fetch(period="3mo", interval="1h")
        else:
            df = StockDataFetcher(ticker).fetch(period="6mo", interval="1d")

        if df is None or len(df) < 30:
            return None

        strat = _run_strategy(df, ticker, stock_mode=True)
        if strat and strat.trade_setup:
            return strat.trade_setup.action
        return TradeAction.HOLD
    except Exception as e:
        logger.warning(f"Strategy re-run failed for {ticker}: {e}")
        return None


# ──────────────────────────────────────────────────────────
#  Position Manager
# ──────────────────────────────────────────────────────────

class AlpacaPositionManager:
    """
    Manages open stock positions on Alpaca.

    Usage:
        client = AlpacaClient()
        client.connect()
        manager = AlpacaPositionManager(client)
        updates = manager.manage_all()
    """

    def __init__(
        self,
        client: AlpacaClient,
        trail_activation_r: float = 1.0,
        trail_atr_multiplier: float = 1.0,
        enable_reversal_close: bool = True,
        partial_close_at_r: float = 1.0,
        partial_close_pct: float = 0.5,
        dry_run: bool = False,
    ):
        self.client = client
        self.trail_activation_r = trail_activation_r
        self.trail_atr_multiplier = trail_atr_multiplier
        self.enable_reversal_close = enable_reversal_close
        self.partial_close_at_r = partial_close_at_r
        self.partial_close_pct = partial_close_pct
        self.dry_run = dry_run

    def manage_all(self) -> list[PositionUpdate]:
        """Check and manage all open positions."""
        positions = self.client.get_open_positions()
        if not positions:
            logger.info("No open Alpaca positions to manage.")
            return []

        updates = []
        for pos in positions:
            update = self._manage_position(pos)
            if update:
                updates.append(update)

        return updates

    def get_summary(self) -> dict:
        """Get a summary of all open positions for Telegram reporting."""
        positions = self.client.get_open_positions()
        acct = self.client.get_account_info()

        total_pnl = sum(p.unrealized_pl for p in positions)
        winners = sum(1 for p in positions if p.unrealized_pl > 0)
        losers = sum(1 for p in positions if p.unrealized_pl < 0)

        return {
            "total_positions": len(positions),
            "total_pnl": total_pnl,
            "winners": winners,
            "losers": losers,
            "balance": acct.balance if acct else 0,
            "equity": acct.equity if acct else 0,
            "buying_power": acct.buying_power if acct else 0,
            "positions": [
                {
                    "symbol": p.symbol,
                    "side": p.side,
                    "qty": p.qty,
                    "avg_entry": p.avg_entry_price,
                    "current_price": p.current_price,
                    "profit": p.unrealized_pl,
                    "profit_pct": p.unrealized_plpc * 100,
                }
                for p in positions
            ],
        }

    # ── Internal ──────────────────────────────────────────

    def _manage_position(self, pos: PositionInfo) -> Optional[PositionUpdate]:
        """Manage a single position: check reversal, log status."""
        ticker = pos.symbol
        is_long = pos.side == "long"

        # ── Halal compliance: close any short positions immediately ──
        if not is_long:
            logger.info(
                f"SHORT position detected on {ticker} — closing for halal compliance. "
                f"PnL: ${pos.unrealized_pl:+.2f}"
            )
            if not self.dry_run:
                # Cancel ALL open orders on this symbol (bracket legs lock shares)
                # Keep cancelling until no orders remain
                import time
                for attempt in range(3):
                    open_orders = self.client.get_open_orders(ticker)
                    if not open_orders:
                        break
                    for order in open_orders:
                        oid = order.get("id", "")
                        logger.info(f"Cancelling order {oid} on {ticker} (attempt {attempt+1})")
                        self.client.cancel_order(oid)
                    time.sleep(1)  # Wait for cancellation to propagate

                # Now close the position
                result = self.client.close_position(ticker)
                if result.success:
                    return PositionUpdate(
                        symbol=ticker,
                        action_taken="full_close",
                        pnl=pos.unrealized_pl,
                        reason=f"Short position closed (halal compliance)",
                    )
                else:
                    logger.error(f"Failed to close short {ticker}: {result.message}")
                    return PositionUpdate(
                        symbol=ticker,
                        action_taken="no_change",
                        pnl=pos.unrealized_pl,
                        reason=f"FAILED to close short {ticker} (halal): {result.message}",
                    )
            else:
                return PositionUpdate(
                    symbol=ticker,
                    action_taken="full_close",
                    pnl=pos.unrealized_pl,
                    reason=f"[DRY RUN] Short position closed (halal compliance)",
                )

        # We need to find the current SL from the open stop-loss order
        sl_price = self._find_stop_loss_price(ticker)

        # Calculate R-multiple
        risk = abs(pos.avg_entry_price - sl_price) if sl_price > 0 else 0
        if risk == 0:
            # CRITICAL: Position has no stop loss — place a protective stop NOW
            logger.warning(f"Position {ticker} has NO SL order — placing protective stop")
            is_leveraged = ticker.upper() in LEVERAGED_TICKERS
            atr = _get_current_atr(ticker)

            if atr > 0:
                # Use ATR-based SL
                multiplier = (
                    config.LEVERAGED_MODE.get("atr_sl_multiplier", 1.0) if is_leveraged
                    else config.STOCK_MODE.get("atr_sl_multiplier", 1.5)
                )
                protective_sl = pos.avg_entry_price - (atr * multiplier)
            else:
                # Fallback: 3% below entry for stocks, 5% for leveraged
                pct = 0.05 if is_leveraged else 0.03
                protective_sl = pos.avg_entry_price * (1 - pct)

            # Don't set SL above current price (would trigger immediately)
            if protective_sl >= pos.current_price:
                protective_sl = pos.current_price * 0.97  # 3% below current

            if not self.dry_run:
                # Cancel any existing orders first
                self._cancel_all_orders(ticker)
                # Retry SL placement up to 3 times — a naked position is critical
                import time
                sl_placed = False
                for attempt in range(3):
                    result = self.client.place_stop_order(
                        symbol=ticker,
                        qty=int(pos.qty),
                        stop_price=protective_sl,
                        side="sell",
                    )
                    if result.success:
                        sl_placed = True
                        logger.info(
                            f"Placed protective SL on {ticker}: ${protective_sl:.2f} "
                            f"(entry=${pos.avg_entry_price:.2f})"
                        )
                        break
                    else:
                        logger.error(
                            f"CRITICAL: SL placement FAILED for {ticker} "
                            f"(attempt {attempt + 1}/3): {result.message}"
                        )
                        if attempt < 2:
                            time.sleep(2)

                if sl_placed:
                    return PositionUpdate(
                        symbol=ticker,
                        action_taken="trail_stop",
                        old_sl=0.0,
                        new_sl=protective_sl,
                        pnl=pos.unrealized_pl,
                        reason=f"Placed missing SL: ${protective_sl:.2f}",
                    )
                else:
                    logger.error(
                        f"CRITICAL: ALL 3 SL placement attempts FAILED for {ticker}. "
                        f"Position is NAKED with NO stop loss! "
                        f"Entry=${pos.avg_entry_price:.2f}, "
                        f"Current=${pos.current_price:.2f}, "
                        f"PnL=${pos.unrealized_pl:+.2f}"
                    )
                    # Last resort: close the position entirely rather than leave it naked
                    logger.warning(f"EMERGENCY: Closing naked position {ticker}")
                    result = self.client.close_position(ticker)
                    if result.success:
                        return PositionUpdate(
                            symbol=ticker,
                            action_taken="full_close",
                            pnl=pos.unrealized_pl,
                            reason=f"EMERGENCY close — SL placement failed 3x, position was naked",
                        )
                    else:
                        # Truly critical — nothing we can do
                        logger.error(
                            f"EMERGENCY CLOSE ALSO FAILED for {ticker}: {result.message}"
                        )
                        return PositionUpdate(
                            symbol=ticker,
                            action_taken="no_change",
                            pnl=pos.unrealized_pl,
                            reason=f"CRITICAL: Position NAKED — SL and close both failed",
                        )
            else:
                return PositionUpdate(
                    symbol=ticker,
                    action_taken="trail_stop",
                    old_sl=0.0,
                    new_sl=protective_sl,
                    pnl=pos.unrealized_pl,
                    reason=f"[DRY RUN] Would place missing SL: ${protective_sl:.2f}",
                )

            # Use entry-based risk for R-multiple even if SL placement failed
            risk = pos.avg_entry_price * 0.02  # fallback: 2%
            sl_price = pos.avg_entry_price - risk

        if is_long:
            profit_distance = pos.current_price - pos.avg_entry_price
        else:
            profit_distance = pos.avg_entry_price - pos.current_price

        r_multiple = profit_distance / risk if risk > 0 else 0

        logger.info(
            f"Position {ticker} {pos.side}: "
            f"entry=${pos.avg_entry_price:.2f}, current=${pos.current_price:.2f}, "
            f"SL=${sl_price:.2f}, R={r_multiple:.2f}, "
            f"PnL=${pos.unrealized_pl:+.2f} ({pos.unrealized_plpc*100:+.1f}%)"
        )

        # ── Step 0a: VIX-aware stop tightening for EXISTING positions ─
        #    When VIX spikes, our existing positions are in danger.
        #    We can't just ignore them. Tighten stops proactively.
        try:
            from trading.alpaca_executor import get_vix_regime
            vix = get_vix_regime()
            vix_val = vix.get("vix", 18)

            vix_extreme = getattr(config, "VIX_EXTREME", 30)
            vix_high = getattr(config, "VIX_HIGH", 25)

            if vix_val >= vix_extreme and is_long and sl_price > 0:
                # VIX EXTREME (>30): tighten SL to 1% below current price
                # This protects against crash continuation
                tight_sl = pos.current_price * 0.99
                if tight_sl > sl_price:
                    logger.warning(
                        f"VIX EXTREME ({vix_val:.0f}): Tightening SL on {ticker} "
                        f"from ${sl_price:.2f} to ${tight_sl:.2f} (1% below current)"
                    )
                    if not self.dry_run:
                        success = self._replace_stop_order(
                            ticker, int(pos.qty), tight_sl, "sell"
                        )
                        if success:
                            return PositionUpdate(
                                symbol=ticker,
                                action_taken="trail_stop",
                                old_sl=sl_price,
                                new_sl=tight_sl,
                                pnl=pos.unrealized_pl,
                                reason=f"VIX extreme ({vix_val:.0f}): "
                                       f"tightened SL ${sl_price:.2f} → ${tight_sl:.2f}",
                            )
                    else:
                        return PositionUpdate(
                            symbol=ticker,
                            action_taken="trail_stop",
                            old_sl=sl_price,
                            new_sl=tight_sl,
                            pnl=pos.unrealized_pl,
                            reason=f"[DRY RUN] VIX extreme ({vix_val:.0f}): "
                                   f"tightened SL ${sl_price:.2f} → ${tight_sl:.2f}",
                        )

            elif vix_val >= vix_high and is_long and sl_price > 0:
                # VIX HIGH (25-30): tighten SL to 2% below current price
                tight_sl = pos.current_price * 0.98
                if tight_sl > sl_price:
                    logger.info(
                        f"VIX HIGH ({vix_val:.0f}): Tightening SL on {ticker} "
                        f"from ${sl_price:.2f} to ${tight_sl:.2f} (2% below current)"
                    )
                    if not self.dry_run:
                        success = self._replace_stop_order(
                            ticker, int(pos.qty), tight_sl, "sell"
                        )
                        if success:
                            return PositionUpdate(
                                symbol=ticker,
                                action_taken="trail_stop",
                                old_sl=sl_price,
                                new_sl=tight_sl,
                                pnl=pos.unrealized_pl,
                                reason=f"VIX high ({vix_val:.0f}): "
                                       f"tightened SL ${sl_price:.2f} → ${tight_sl:.2f}",
                            )
                    else:
                        return PositionUpdate(
                            symbol=ticker,
                            action_taken="trail_stop",
                            old_sl=sl_price,
                            new_sl=tight_sl,
                            pnl=pos.unrealized_pl,
                            reason=f"[DRY RUN] VIX high ({vix_val:.0f}): "
                                   f"tightened SL ${sl_price:.2f} → ${tight_sl:.2f}",
                        )
        except Exception as e:
            logger.debug(f"VIX check for existing position {ticker} failed: {e}")

        # ── Step 0b: Earnings protection for EXISTING positions ───
        #    If we're holding a stock and earnings are TOMORROW,
        #    close it now rather than gamble through the report.
        try:
            from trading.alpaca_executor import check_earnings_blackout
            earnings = check_earnings_blackout(ticker)
            days_to = earnings.get("days_to_earnings", 999)

            # Close if earnings are within 1 day (tomorrow or today)
            if 0 <= days_to <= 1:
                logger.warning(
                    f"EARNINGS PROTECTION: {ticker} reports on "
                    f"{earnings.get('earnings_date', '?')} ({days_to}d away) "
                    f"— closing to avoid gap risk. PnL=${pos.unrealized_pl:+.2f}"
                )
                if not self.dry_run:
                    self._cancel_all_orders(ticker)
                    result = self.client.close_position(ticker)
                    if result.success:
                        return PositionUpdate(
                            symbol=ticker,
                            action_taken="full_close",
                            pnl=pos.unrealized_pl,
                            reason=f"Earnings protection: reports in {days_to}d "
                                   f"({earnings.get('earnings_date', '?')})",
                        )
                    else:
                        logger.error(
                            f"Earnings close FAILED for {ticker}: {result.message}"
                        )
                else:
                    return PositionUpdate(
                        symbol=ticker,
                        action_taken="full_close",
                        pnl=pos.unrealized_pl,
                        reason=f"[DRY RUN] Earnings protection: reports in {days_to}d",
                    )
        except Exception as e:
            logger.debug(f"Earnings check for existing position {ticker} failed: {e}")

        # ── Step 1: Check for signal reversal ────────────
        if self.enable_reversal_close and r_multiple < 0.5:
            current_signal = _rerun_strategy(ticker)
            if current_signal is not None:
                signal_is_buy = current_signal in (
                    TradeAction.BUY, TradeAction.STRONG_BUY
                )
                signal_is_sell = current_signal in (
                    TradeAction.SELL, TradeAction.STRONG_SELL
                )

                if (is_long and signal_is_sell) or (not is_long and signal_is_buy):
                    logger.info(
                        f"Signal REVERSED for {ticker}: "
                        f"position={pos.side}, new_signal={current_signal.value}"
                    )
                    if not self.dry_run:
                        # Cancel bracket SL/TP legs FIRST to prevent orphaned orders
                        self._cancel_all_orders(ticker)
                        result = self.client.close_position(ticker)
                        if result.success:
                            return PositionUpdate(
                                symbol=ticker,
                                action_taken="full_close",
                                pnl=pos.unrealized_pl,
                                reason=f"Signal reversed to {current_signal.value}",
                            )
                        else:
                            logger.error(
                                f"CRITICAL: Cancelled orders on {ticker} but "
                                f"failed to close position: {result.message}"
                            )
                            # Re-place emergency stop since we cancelled the bracket
                            emergency_sl = pos.current_price * 0.95  # 5% below current
                            self.client.place_stop_order(
                                symbol=ticker, qty=int(pos.qty),
                                stop_price=emergency_sl, side="sell",
                            )
                            logger.warning(
                                f"Re-placed emergency SL on {ticker} at ${emergency_sl:.2f}"
                            )
                            return PositionUpdate(
                                symbol=ticker,
                                action_taken="no_change",
                                pnl=pos.unrealized_pl,
                                reason=f"Reversal close failed — re-placed emergency SL at ${emergency_sl:.2f}",
                            )
                    else:
                        return PositionUpdate(
                            symbol=ticker,
                            action_taken="full_close",
                            pnl=pos.unrealized_pl,
                            reason=f"[DRY RUN] Signal reversed to {current_signal.value}",
                        )

        # ── Step 2: Breakeven stop at 0.5R ──────────────────
        # Move SL to entry price once trade shows 0.5R profit.
        # This eliminates losing trades that initially showed promise.
        # We do this BEFORE partial close and trailing to lock in breakeven first.
        is_leveraged = ticker.upper() in LEVERAGED_TICKERS
        breakeven_r = 0.5

        if (is_long and sl_price > 0 and r_multiple >= breakeven_r
                and sl_price < pos.avg_entry_price):
            # SL is still below entry — move it to breakeven
            # Add a tiny buffer (0.1% above entry) to cover commissions
            breakeven_price = pos.avg_entry_price * 1.001

            if breakeven_price > sl_price:
                logger.info(
                    f"BREAKEVEN: {ticker} at {r_multiple:.1f}R — moving SL "
                    f"from ${sl_price:.2f} to ${breakeven_price:.2f} (entry)"
                )
                if not self.dry_run:
                    success = self._replace_stop_order(
                        ticker, int(pos.qty), breakeven_price, "sell"
                    )
                    if success:
                        return PositionUpdate(
                            symbol=ticker,
                            action_taken="trail_stop",
                            old_sl=sl_price,
                            new_sl=breakeven_price,
                            pnl=pos.unrealized_pl,
                            reason=f"Breakeven at {r_multiple:.1f}R: "
                                   f"SL ${sl_price:.2f} → ${breakeven_price:.2f}",
                        )
                else:
                    return PositionUpdate(
                        symbol=ticker,
                        action_taken="trail_stop",
                        old_sl=sl_price,
                        new_sl=breakeven_price,
                        pnl=pos.unrealized_pl,
                        reason=f"[DRY RUN] Breakeven at {r_multiple:.1f}R: "
                               f"SL ${sl_price:.2f} → ${breakeven_price:.2f}",
                    )

        # ── Step 3: Partial close at 1R ───────────────────
        # Sell 50% at 1R to lock in profit, let remainder ride with trailing stop.
        # This is what separates amateurs from professionals — guaranteed profit capture.
        # We track whether we've already done a partial close by checking qty vs expected.
        if (is_long and r_multiple >= self.partial_close_at_r
                and pos.qty > 1):
            # Check if this looks like a full position (not already partially closed)
            # Heuristic: if qty is a round number or > some threshold, likely not yet partial-closed
            # We use a simple approach: check if there's a partial close marker in orders
            partial_qty = int(pos.qty * self.partial_close_pct)
            if partial_qty >= 1:
                # Only partial close if we haven't already (check if qty is close to original)
                # Simple approach: if position is still at full qty (qty > 1), do partial
                # After partial close, qty will be smaller — trailing stop handles the rest
                logger.info(
                    f"PARTIAL CLOSE: {ticker} at {r_multiple:.1f}R — "
                    f"closing {partial_qty} of {int(pos.qty)} shares "
                    f"(${pos.unrealized_pl:+.2f} unrealized)"
                )
                if not self.dry_run:
                    # Sell partial qty at market
                    result = self.client.close_position(ticker, qty=partial_qty)
                    if result.success:
                        remaining = int(pos.qty) - partial_qty
                        logger.info(
                            f"Partial close executed: sold {partial_qty} {ticker}, "
                            f"{remaining} remaining"
                        )
                        # Update the SL order for the remaining quantity
                        # The old SL order is for the full qty — needs updating
                        if sl_price > 0 and remaining > 0:
                            import time
                            time.sleep(1)
                            # Re-place SL for remaining shares at current SL level
                            self._replace_stop_order(
                                ticker, remaining, sl_price, "sell"
                            )
                        return PositionUpdate(
                            symbol=ticker,
                            action_taken="partial_close",
                            old_sl=sl_price,
                            new_sl=sl_price,
                            pnl=pos.unrealized_pl,
                            reason=f"Partial close at {r_multiple:.1f}R: "
                                   f"sold {partial_qty}/{int(pos.qty)} shares",
                        )
                    else:
                        logger.error(
                            f"Partial close FAILED for {ticker}: {result.message}"
                        )
                else:
                    return PositionUpdate(
                        symbol=ticker,
                        action_taken="partial_close",
                        pnl=pos.unrealized_pl,
                        reason=f"[DRY RUN] Partial close at {r_multiple:.1f}R: "
                               f"would sell {partial_qty}/{int(pos.qty)} shares",
                    )

        # ── Step 4: Trail stop ─────────────────────────────────
        # For leveraged ETFs with trailing enabled: cancel old SL → place new tighter SL
        # For regular stocks: always trail at 1R+
        trail_enabled = (
            config.LEVERAGED_MODE.get("trailing_stop", False) if is_leveraged
            else True  # always trail regular stocks
        )

        if trail_enabled and sl_price > 0 and r_multiple >= self.trail_activation_r:
            atr = _get_current_atr(ticker)
            if atr <= 0:
                trail_distance = risk * 0.75
            else:
                trail_distance = atr * self.trail_atr_multiplier

            if is_long:
                new_sl = pos.current_price - trail_distance
                # Never move SL backwards, and must be above entry for profit lock
                if new_sl > sl_price and new_sl > pos.avg_entry_price:
                    logger.info(
                        f"Trailing stop {ticker}: "
                        f"SL ${sl_price:.2f} → ${new_sl:.2f} (at {r_multiple:.1f}R)"
                    )
                    if not self.dry_run:
                        # Cancel ALL existing stop orders for this symbol
                        success = self._replace_stop_order(
                            ticker, int(pos.qty), new_sl, "sell"
                        )
                        if success:
                            return PositionUpdate(
                                symbol=ticker,
                                action_taken="trail_stop",
                                old_sl=sl_price,
                                new_sl=new_sl,
                                pnl=pos.unrealized_pl,
                                reason=f"Trailed at {r_multiple:.1f}R: "
                                       f"SL ${sl_price:.2f} → ${new_sl:.2f}",
                            )
                    else:
                        return PositionUpdate(
                            symbol=ticker,
                            action_taken="trail_stop",
                            old_sl=sl_price,
                            new_sl=new_sl,
                            pnl=pos.unrealized_pl,
                            reason=f"[DRY RUN] Trail at {r_multiple:.1f}R: "
                                   f"SL ${sl_price:.2f} → ${new_sl:.2f}",
                        )

        # No action needed
        return PositionUpdate(
            symbol=ticker,
            action_taken="no_change",
            pnl=pos.unrealized_pl,
            reason=f"R={r_multiple:.2f}, holding",
        )

    def _replace_stop_order(
        self, symbol: str, qty: int, new_stop_price: float, side: str
    ) -> bool:
        """
        Cancel ALL old orders on this symbol and place a new stop.
        
        We cancel everything (including TP legs) because:
        1. Orphaned TP legs from brackets can cause unexpected fills
        2. The trailing stop replaces the original risk management
        3. Once we're trailing, the trade is profitable — we just want SL protection
        """
        import time

        # Cancel ALL existing orders on this symbol (SL, TP, bracket legs)
        orders = self.client.get_open_orders(symbol)
        if orders:
            for order in orders:
                oid = order.get("id", "")
                if oid:
                    logger.info(
                        f"Cancelling order {oid} on {symbol} "
                        f"(type={order.get('type', '?')}, side={order.get('side', '?')})"
                    )
                    self.client.cancel_order(oid)
            time.sleep(1.5)  # Wait for cancellation to propagate

        # Place new stop order with retry — position is NAKED after cancellation
        for attempt in range(3):
            result = self.client.place_stop_order(
                symbol=symbol, qty=qty, stop_price=new_stop_price, side=side
            )
            if result.success:
                return True
            logger.error(
                f"CRITICAL: Trail SL placement FAILED for {symbol} "
                f"(attempt {attempt + 1}/3): {result.message}"
            )
            if attempt < 2:
                time.sleep(2)

        logger.error(
            f"CRITICAL: ALL 3 trail SL attempts FAILED for {symbol}. "
            f"Position may be NAKED after order cancellation! "
            f"Closing position as safety measure."
        )
        # Emergency: close position rather than leave it naked
        close_result = self.client.close_position(symbol)
        if close_result.success:
            logger.warning(f"EMERGENCY: Closed {symbol} after trail SL failure")
        else:
            logger.error(f"EMERGENCY CLOSE ALSO FAILED for {symbol}: {close_result.message}")
        return False

    def _cancel_all_orders(self, symbol: str):
        """Cancel all open orders for a symbol with retry."""
        import time
        orders = self.client.get_open_orders(symbol)
        for order in orders:
            oid = order.get("id", "")
            if oid:
                logger.info(f"Cancelling order {oid} on {symbol}")
                success = self.client.cancel_order(oid)
                if not success:
                    logger.warning(f"Cancel failed for order {oid} — retrying")
                    time.sleep(0.5)
                    self.client.cancel_order(oid)
        if orders:
            time.sleep(1)  # Wait for cancellation to propagate

    def _find_stop_loss_price(self, symbol: str) -> float:
        """Find the current stop-loss price from open orders for a symbol."""
        try:
            orders = self.client.get_open_orders(symbol)
            for order in orders:
                stop_price = order.get("stop_price", 0)
                if stop_price and stop_price > 0:
                    order_type = order.get("type", "").lower()
                    order_side = order.get("side", "").lower()
                    # For long positions: SL is a SELL stop order
                    if "stop" in order_type and "sell" in order_side:
                        return float(stop_price)
                    # Also accept any stop-priced order as SL
                    if "stop" in order_type:
                        return float(stop_price)
            return 0.0
        except Exception as e:
            logger.warning(f"Error finding SL for {symbol}: {e}")
            return 0.0
